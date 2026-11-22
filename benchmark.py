"""
Benchmark three decoding strategies on the SAME model, prompts and greedy decoding:

  1. no_cache      : recompute the full sequence every step (use_cache=False), one request at a time
  2. kv_cache      : standard HF KV cache, one request at a time (batch size 1)
  3. batched_pool  : the continuous batcher with the preallocated KV pool

All requests "arrive" at t=0, so TTFT/latency include queueing time.

Run from the project root:
    python -m mini_vllm.benchmark --model Qwen/Qwen2.5-0.5B-Instruct \
        --num-requests 1 4 8 --max-tokens 64 --max-slots 8
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import torch

from mini_vllm import ContinuousBatcher, Engine

perf = time.perf_counter


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
BASE_PROMPTS = [
    "Explain what a KV cache is in one paragraph.",
    "Write a short poem about the ocean at night.",
    "List three differences between TCP and UDP.",
    "What is the capital of France, and what is it famous for?",
    "Summarize the plot of Romeo and Juliet in a few sentences.",
    "Give me a Python function that checks whether a number is prime.",
    "Why is the sky blue? Explain simply.",
    "Describe how continuous batching improves LLM serving throughput.",
]


def make_prompts(n: int) -> list[str]:
    # Cycle through the base prompts; vary length a bit so batches are ragged.
    out = []
    for i in range(n):
        p = BASE_PROMPTS[i % len(BASE_PROMPTS)]
        if i % 3 == 1:
            p = "Please answer carefully and in detail. " * 3 + p
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Shared helpers (mirror Engine's prompt handling so comparisons are fair)
# ---------------------------------------------------------------------------
def is_eos(engine, tok: int) -> bool:
    ids = getattr(engine, "eos_ids", None)  # present if you add a generation_config-based EOS set
    if ids:
        return tok in ids
    return tok == engine.tokenizer.eos_token_id


def prepare_ids(engine, prompt: str) -> torch.Tensor:
    ids = engine._encode(prompt)
    limit = engine.max_seq_len - 1
    if ids.size(1) > limit:
        ids = ids[:, -limit:]
    return ids


# ---------------------------------------------------------------------------
# Mode 1: no KV cache
# ---------------------------------------------------------------------------
@torch.no_grad()
def gen_no_cache(engine, prompt: str, max_tokens: int):
    ids = prepare_ids(engine, prompt)
    toks, t_first = [], None
    for _ in range(max_tokens):
        logits = engine.model(ids, use_cache=False).logits[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        tok = int(nxt.item())  # forces sync
        if t_first is None:
            t_first = perf()
        toks.append(tok)
        if is_eos(engine, tok) or ids.size(1) + 1 >= engine.max_seq_len:
            break
        ids = torch.cat([ids, nxt], dim=1)
    return toks, t_first


# ---------------------------------------------------------------------------
# Mode 2: KV cache, batch size 1
# ---------------------------------------------------------------------------
@torch.no_grad()
def gen_kv_cache(engine, prompt: str, max_tokens: int):
    ids = prepare_ids(engine, prompt)
    out = engine.model(ids, use_cache=True)
    past = out.past_key_values
    cur_len = ids.size(1)

    toks, t_first = [], None
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    while True:
        tok = int(nxt.item())
        if t_first is None:
            t_first = perf()
        toks.append(tok)
        if is_eos(engine, tok) or len(toks) >= max_tokens or cur_len >= engine.max_seq_len:
            break
        out = engine.model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        cur_len += 1
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return toks, t_first


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_sequential(fn, engine, prompts, max_tokens):
    sync()
    t0 = perf()
    ttfts, lats, outs = [], [], []
    for p in prompts:
        toks, t_first = fn(engine, p, max_tokens)
        sync()
        ttfts.append(t_first - t0)
        lats.append(perf() - t0)
        outs.append(toks)
    return perf() - t0, ttfts, lats, outs


async def _run_batched_async(engine, prompts, max_tokens):
    batcher = ContinuousBatcher(engine)
    await batcher.start()
    t0 = perf()
    reqs = [await batcher.submit(p, max_tokens) for p in prompts]

    async def consume(r):
        first = None
        while True:
            delta = await r.token_queue.get()
            if delta is None:
                break
            if first is None:
                first = perf()
        return first, perf()

    results = await asyncio.gather(*(consume(r) for r in reqs))
    wall = perf() - t0
    await batcher.stop()

    # first can be None if a request produced no visible text (e.g. failed)
    ttfts = [(f if f is not None else e) - t0 for f, e in results]
    lats = [e - t0 for _, e in results]
    outs = [list(r.generated_ids) for r in reqs]
    return wall, ttfts, lats, outs


def run_batched(engine, prompts, max_tokens):
    sync()
    return asyncio.run(_run_batched_async(engine, prompts, max_tokens))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(name, res, peak_gb):
    wall, ttfts, lats, outs = res
    total = sum(len(o) for o in outs)
    p95 = sorted(lats)[max(0, int(0.95 * len(lats)) - 1)]
    return {
        "mode": name,
        "wall_s": wall,
        "tokens": total,
        "tok/s": total / wall if wall > 0 else 0.0,
        "ttft_mean_s": statistics.mean(ttfts),
        "lat_mean_s": statistics.mean(lats),
        "lat_p95_s": p95,
        "peak_gb": peak_gb,
    }


def print_table(rows):
    cols = ["mode", "wall_s", "tokens", "tok/s", "ttft_mean_s", "lat_mean_s", "lat_p95_s", "peak_gb"]
    widths = {c: max(len(c), 12) for c in cols}
    print(" ".join(c.ljust(widths[c]) for c in cols))
    print(" ".join("-" * widths[c] for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append((f"{v:.3f}" if isinstance(v, float) else str(v)).ljust(widths[c]))
        print(" ".join(cells))


def match_rate(a_outs, b_outs):
    same = sum(1 for a, b in zip(a_outs, b_outs) if a == b)
    return same / max(1, len(a_outs))


def measured(fn):
    """Run fn() and return (result, peak GPU memory in GB)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    res = fn()
    sync()
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return res, peak


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--num-requests", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-slots", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--skip-nocache", action="store_true", help="no-cache mode is O(n^2); skip for long runs")
    ap.add_argument("--no-chat-template", action="store_true")
    args = ap.parse_args()

    engine = Engine(
        args.model,
        max_slots=args.max_slots,
        max_seq_len=args.max_seq_len,
        use_chat_template=not args.no_chat_template,
    )
    pool_gb = (engine.k_pool.numel() + engine.v_pool.numel()) * engine.k_pool.element_size() / 1e9
    print(f"model={args.model} device={engine.device} pool={pool_gb:.3f} GB "
          f"(slots={args.max_slots}, seq={args.max_seq_len})")

    # Warmup (CUDA kernels, allocator, tokenizer caches)
    warm = make_prompts(2)
    gen_kv_cache(engine, warm[0], 8)
    gen_no_cache(engine, warm[0], 8)
    run_batched(engine, warm, 8)

    for n in args.num_requests:
        prompts = make_prompts(n)
        print(f"\n=== {n} concurrent request(s), max_tokens={args.max_tokens} ===")
        rows, results = [], {}

        if not args.skip_nocache:
            res, peak = measured(lambda: run_sequential(gen_no_cache, engine, prompts, args.max_tokens))
            results["no_cache"] = res
            rows.append(summarize("no_cache", res, peak))

        res, peak = measured(lambda: run_sequential(gen_kv_cache, engine, prompts, args.max_tokens))
        results["kv_cache"] = res
        rows.append(summarize("kv_cache", res, peak))

        res, peak = measured(lambda: run_batched(engine, prompts, args.max_tokens))
        results["batched_pool"] = res
        rows.append(summarize("batched_pool", res, peak))

        print_table(rows)

        # Greedy outputs should (nearly) agree. fp16 + padding can cause late divergence.
        ref = results["kv_cache"][3]
        for name, res in results.items():
            if name != "kv_cache":
                print(f"exact token match vs kv_cache [{name}]: {match_rate(ref, res[3]) * 100:.0f}%")

        base = next(r for r in rows if r["mode"] == "kv_cache")["tok/s"]
        for r in rows:
            print(f"speedup of {r['mode']} vs kv_cache: {r['tok/s'] / base:.2f}x")


if __name__ == "__main__":
    main()
