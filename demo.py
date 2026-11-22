"""Minimal demo: submit a few prompts and stream the results.

Run from the project root:
    python -m mini_vllm.demo --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio

from mini_vllm import ContinuousBatcher, Engine

PROMPTS = [
    "What is a KV cache?",
    "Write a haiku about GPUs.",
    "Explain TCP vs UDP.",
]


async def main(args):
    engine = Engine(args.model, max_slots=args.max_slots, max_seq_len=args.max_seq_len)
    batcher = ContinuousBatcher(engine)
    await batcher.start()

    reqs = [await batcher.submit(p, max_tokens=args.max_tokens) for p in PROMPTS]

    async def stream(r):
        text = ""
        while True:
            delta = await r.token_queue.get()
            if delta is None:  # sentinel: request finished
                break
            text += delta
        print(f"\n--- request {r.req_id} ---\n{text}")

    await asyncio.gather(*(stream(r) for r in reqs))
    print(f"\nthroughput: {engine.throughput():.1f} tok/s")
    await batcher.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-slots", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=64)
    asyncio.run(main(ap.parse_args()))
