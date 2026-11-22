"""
Continuous batching with a preallocated KV pool (no PagedAttention).
Compatible with transformers v4 and v5.

Engine: owns the model, tokenizer, the KV pool and slot management, and
implements prefill and batched decode. Scheduling lives in scheduler/batcher.py.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cache_compat import TF_MAJOR, make_cache, to_legacy
from .request import Request


class Engine:
    def __init__(self, model_name: str, max_slots: int = 4,
                 max_seq_len: int = 2048,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_chat_template: bool = True):
        self.device = device
        self.use_chat_template = use_chat_template
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        dtype_kw = {"dtype": torch.float16} if TF_MAJOR >= 5 else {"torch_dtype": torch.float16}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, **dtype_kw
        ).eval()

        cfg = self.model.config
        self.num_layers = cfg.num_hidden_layers
        n_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads

        self.max_slots = max_slots
        self.max_seq_len = max_seq_len

        # All KV blocks are allocated once, here.
        # shape: (layers, slots, kv_heads, max_seq_len, head_dim)
        shape = (self.num_layers, self.max_slots,
                 self.num_kv_heads, self.max_seq_len, self.head_dim)
        self.k_pool = torch.zeros(shape, dtype=torch.float16, device=device)
        self.v_pool = torch.zeros_like(self.k_pool)

        # Slot management
        self.free_slots: list[int] = list(range(max_slots))
        self.slot_of: dict[int, int] = {}
        self._req_counter = 0

        self.total_tokens = 0
        self.start_time: Optional[float] = None  # set on first token
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop):
        self._loop = loop

    # ------------------------------------------------------------------
    # Slot lifecycle
    # ------------------------------------------------------------------
    def claim_slot(self, req_id: int) -> int | None:
        if not self.free_slots:
            return None
        slot = self.free_slots.pop()
        self.slot_of[req_id] = slot
        return slot

    def release_slot(self, req_id: int) -> None:
        slot = self.slot_of.pop(req_id, None)
        if slot is not None:
            self.free_slots.append(slot)
        # No need to zero the block: masks never expose columns >= past_len.

    # ------------------------------------------------------------------
    # Metrics / helpers
    # ------------------------------------------------------------------
    def log_token(self):
        if self.start_time is None:
            self.start_time = time.time()
        self.total_tokens += 1

    def throughput(self) -> float:
        if self.start_time is None:
            return 0.0
        return self.total_tokens / max(1e-6, time.time() - self.start_time)

    def new_request(self, prompt: str, max_tokens: int = 64) -> Request:
        self._req_counter += 1
        return Request(
            req_id=self._req_counter,
            prompt=prompt,
            max_tokens=max_tokens,
            token_queue=asyncio.Queue(),
        )

    def _encode(self, prompt: str) -> torch.Tensor:
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        else:
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return ids.to(self.device)

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_prefill(self, request: Request) -> bool:
        slot = self.claim_slot(request.req_id)
        if slot is None:
            return False

        try:
            input_ids = self._encode(request.prompt)
            # Keep at least one free position so one decode step is possible.
            limit = self.max_seq_len - 1
            if input_ids.size(1) > limit:
                input_ids = input_ids[:, -limit:]  # keep the most recent tokens
            prompt_len = input_ids.size(1)

            outputs = self.model(input_ids, use_cache=True)
            legacy_kv = to_legacy(outputs.past_key_values)

            for layer in range(self.num_layers):
                k, v = legacy_kv[layer]  # (1, H_kv, L, D)
                self.k_pool[layer, slot, :, :prompt_len, :] = k[0]
                self.v_pool[layer, slot, :, :prompt_len, :] = v[0]

            request.slot = slot
            request.past_len = prompt_len

            next_id = int(torch.argmax(outputs.logits[0, -1, :]).item())
            request.next_token_id = next_id
            self._emit_token(request, next_id)
            del outputs, legacy_kv
        except Exception:
            self.release_slot(request.req_id)
            raise
        return True

    # ------------------------------------------------------------------
    # Batched decode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def batch_decode(self, requests: list[Request]):
        # Finish anything that has no room left BEFORE building the batch.
        for r in requests:
            if r.past_len >= self.max_seq_len:
                r.finished = True
        requests = [r for r in requests if not r.finished]
        if not requests:
            return

        B = len(requests)
        dev = self.device
        slots = torch.tensor([r.slot for r in requests], device=dev)
        pos = torch.tensor([r.past_len for r in requests], device=dev)  # (B,)
        max_past = int(pos.max().item())

        # ── 1. Gather each slot's history in one op, right-padded to max_past.
        # Columns >= past_len hold stale/zero data but are masked out below.
        k_batch = self.k_pool[:, slots, :, :max_past, :]  # (L, B, H, max_past, D)
        v_batch = self.v_pool[:, slots, :, :max_past, :]
        batched_kv = tuple((k_batch[l], v_batch[l]) for l in range(self.num_layers))

        # Mask length = max_past + 1: the model appends the new token at the
        # LAST column for every row, so unmask [0:cur] and the last column.
        cols = torch.arange(max_past + 1, device=dev).unsqueeze(0)  # (1, max_past+1)
        attn_mask = ((cols < pos.unsqueeze(1)) | (cols == max_past)).long()  # (B, max_past+1)

        # RoPE uses the true position, not the cache index.
        pos_ids = pos.unsqueeze(1)  # (B, 1)
        input_ids = torch.tensor(
            [[r.next_token_id] for r in requests], device=dev
        )  # (B, 1)

        # ── 2. Forward ──────────────────────────────────────────────────
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=make_cache(batched_kv),
            attention_mask=attn_mask,
            position_ids=pos_ids,
            use_cache=True,
        )
        new_kv = to_legacy(outputs.past_key_values)

        # ── 3. Write the new token's K/V (last column) back to each slot ─
        for layer in range(self.num_layers):
            k_full, v_full = new_kv[layer]  # (B, H, max_past+1, D)
            self.k_pool[layer, slots, :, pos, :] = k_full[:, :, -1, :]
            self.v_pool[layer, slots, :, pos, :] = v_full[:, :, -1, :]

        next_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1).tolist()
        for r, nid in zip(requests, next_ids):
            r.past_len += 1
            r.next_token_id = nid
            self._emit_token(r, nid)

        del outputs, new_kv, batched_kv, k_batch, v_batch

    # ------------------------------------------------------------------
    # Token emission (thread-safe, incremental detokenization)
    # ------------------------------------------------------------------
    def _emit_token(self, request: Request, token_id: int):
        request.generated_ids.append(token_id)
        self.log_token()

        if token_id == self.tokenizer.eos_token_id or \
           len(request.generated_ids) >= request.max_tokens:
            request.finished = True

        ids = request.generated_ids
        prefix_text = self.tokenizer.decode(
            ids[request._prefix_offset:request._read_offset], skip_special_tokens=True
        )
        new_text = self.tokenizer.decode(
            ids[request._prefix_offset:], skip_special_tokens=True
        )

        delta = ""
        if len(new_text) > len(prefix_text):
            # Hold back while the tail is an incomplete UTF-8 sequence,
            # unless this is the last token (then flush what we have).
            if not new_text.endswith("\ufffd") or request.finished:
                delta = new_text[len(prefix_text):]
                request._prefix_offset = request._read_offset
                request._read_offset = len(ids)

        if delta:
            loop = self._loop or asyncio.get_event_loop()
            loop.call_soon_threadsafe(request.token_queue.put_nowait, delta)
