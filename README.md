# mini_vllm

**A small, readable LLM inference engine that shows how continuous batching and a preallocated KV cache work, in about 500 lines of Python.**

Built on Hugging Face `transformers`. Inspired by [vLLM](https://github.com/vllm-project/vllm), but **not affiliated with it**. This is an educational project, not a production server.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Why this project exists

Serving LLMs efficiently comes down to two ideas:

1. **KV caching**: don't recompute attention keys/values for tokens you've already processed.
2. **Continuous batching**: don't wait for a whole batch to finish. Admit new requests and evict finished ones at every decode step.

`mini_vllm` implements both from scratch so you can read, change and benchmark them. It keeps the scheduler, the cache and the model call in three small, separate modules.

## Features

- **Continuous batching**: requests join and leave the running batch at every step
- **Preallocated KV pool**: all KV memory is allocated once at startup, with one slot per concurrent request
- **Vectorised pool-to-batch gather**: one tensor op, with no per-request or per-layer Python loops on the read path
- **Async streaming API**: tokens arrive through per-request `asyncio.Queue`s
- **Incremental detokenizer**: O(1) per token, and it handles multibyte characters
- **Non-blocking event loop**: model work runs in a worker thread, and prefills per step are capped so decode isn't starved
- **Failure isolation**: an error in prefill or decode finishes the affected requests instead of hanging consumers
- **transformers v4 and v5 compatible** through a small cache shim
- **Built-in benchmark**: no cache vs KV cache vs batched KV pool

## Architecture

```
            submit()                      ┌──────────────────────────────┐
 client ───────────────►  ContinuousBatcher│   waiting ──► running        │
   ▲                      (scheduler/)     │      (prefill)   (decode)    │
   │ token_queue                           └──────────┬───────────────────┘
   │ (streamed text)                                  │ run_prefill / batch_decode
   │                                       ┌──────────▼───────────────────┐
   └───────────────────────────────────────┤ Engine (engine/)             │
                                           │  model + tokenizer           │
                                           │  KV pool: (layers, slots,    │
                                           │   kv_heads, max_seq, head_dim)│
                                           └──────────────────────────────┘
```

**Scheduler loop** (one iteration):
1. Evict finished requests and release their slots
2. Admit waiting requests into free slots (capped per step) and prefill them
3. Run **one batched decode step** for every running request

**Batched decode** (the interesting part):
- Gather each request's history from its pool slot, right-padded to the longest history in the batch
- Build an attention mask of length `max_past + 1` that unmasks `[0 : cur_len]` plus the final column, which is where the model appends the new token
- Pass `position_ids = cur_len` so RoPE sees each request's true position
- Write the new token's K/V (the last column) back into each request's slot

## Project structure

```
mini_vllm/
├── mini_vllm/
│   ├── __init__.py
│   ├── engine/
│   │   ├── cache_compat.py   # transformers v4/v5 cache shims
│   │   ├── request.py        # Request dataclass
│   │   └── engine.py         # model, KV pool, slots, prefill, batched decode
│   ├── scheduler/
│   │   └── batcher.py        # ContinuousBatcher
│   ├── demo.py               # streaming demo
│   └── benchmark.py          # no-cache vs kv-cache vs batched pool
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/mini_vllm.git
cd mini_vllm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended. CPU works but is slow, and fp16 may not be supported for every op.

## Quickstart

```bash
python -m mini_vllm.demo --model Qwen/Qwen2.5-0.5B-Instruct
```

Or use it as a library:

```python
import asyncio
from mini_vllm import Engine, ContinuousBatcher

async def main():
    engine = Engine("Qwen/Qwen2.5-0.5B-Instruct", max_slots=4, max_seq_len=1024)
    batcher = ContinuousBatcher(engine)
    await batcher.start()

    reqs = [await batcher.submit(p, max_tokens=64)
            for p in ["What is a KV cache?", "Write a haiku about GPUs."]]

    async def stream(r):
        while (delta := await r.token_queue.get()) is not None:
            print(f"[{r.req_id}] {delta}", end="", flush=True)

    await asyncio.gather(*(stream(r) for r in reqs))
    await batcher.stop()

asyncio.run(main())
```

Run all commands from the project root, using `python -m ...`.

## Benchmark

Compares three strategies on the same model, prompts and greedy decoding. All requests arrive at t=0, so latency and time-to-first-token include queueing.

| Mode | What it does |
|---|---|
| `no_cache` | Recomputes the full sequence every step, one request at a time |
| `kv_cache` | Standard HF KV cache, batch size 1, one request at a time |
| `batched_pool` | This project: continuous batching over the preallocated KV pool |

```bash
python -m mini_vllm.benchmark \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --num-requests 1 4 8 16 \
    --max-tokens 64 --max-slots 8
```

It reports throughput (tok/s), mean TTFT, mean and p95 latency, peak GPU memory, and the exact-token match rate against the `kv_cache` baseline as a correctness check.

### Results

# Mini-vLLM Benchmark Results

## Overview

This benchmark evaluates the performance of a custom mini-vLLM inference engine implemented with:

- KV Cache
- Request queue and scheduling
- Continuous batching / batched request pool
- Concurrent request execution

The benchmark compares three inference modes:

| Mode | Description |
|---|---|
| `no_cache` | Autoregressive generation without KV cache |
| `kv_cache` | Autoregressive generation using KV cache |
| `batched_pool` | KV cache with a batching pool and request scheduler |

---

## 1. Single Request Benchmark

**Configuration:** 1 request, 64 maximum generated tokens.

| Mode | Wall Time (s) | Tokens | Throughput (tok/s) | Mean TTFT (s) | Mean Latency (s) | P95 Latency (s) | Peak GPU Memory (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_cache | 171.457 | 64 | 0.373 | 1.577 | 171.457 | 171.457 | 0.000 |
| kv_cache | 10.321 | 64 | 6.201 | 1.854 | 10.321 | 10.321 | 0.000 |
| batched_pool | 10.816 | 64 | 5.917 | 1.920 | 10.816 | 10.816 | 0.000 |

### Single Request Observations

- KV cache improved throughput from 0.373 tok/s to 6.201 tok/s.
- KV cache achieved approximately **16.6× higher throughput** than no-cache generation.
- The batching pool was slightly slower than direct KV-cache inference for a single request.
- This is expected because batching introduces scheduling and request-management overhead when there is no concurrent workload to batch.

---

## 2. Concurrent Request Benchmark

**Configuration:** 4 concurrent requests, 64 maximum generated tokens per request.

Total expected generated tokens:

`4 requests × 64 tokens = 256 tokens`

| Mode | Wall Time (s) | Tokens | Throughput (tok/s) | Mean TTFT (s) | Mean Latency (s) | P95 Latency (s) | Peak GPU Memory (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_cache | 785.180 | 256 | 0.326 | 303.138 | 497.443 | 592.650 | 0.000 |
| kv_cache | 31.107 | 256 | 8.230 | 13.393 | 19.423 | 23.282 | 0.000 |
| batched_pool | 19.817 | 256 | 12.918 | 4.322 | 19.762 | 19.817 | 0.000 |

### Concurrent Request Observations

- KV cache achieved approximately **25.2× higher throughput** than no-cache generation.
- The batching pool achieved approximately **39.6× higher throughput** than no-cache generation.
- The batching pool achieved approximately **57% higher throughput** than direct KV-cache inference.
- Total benchmark wall time decreased from 31.107 seconds to 19.817 seconds when using the batching pool.
- Mean TTFT decreased from 13.393 seconds to 4.322 seconds with the batching pool.
- Mean request latency remained approximately similar between KV cache and the batching pool.

These results demonstrate the benefit of batching multiple concurrent requests instead of processing them independently.

---

## 3. Throughput Comparison

| Configuration | no_cache | kv_cache | batched_pool |
|---|---:|---:|---:|
| 1 request | 0.373 tok/s | 6.201 tok/s | 5.917 tok/s |
| 4 concurrent requests | 0.326 tok/s | 8.230 tok/s | 12.918 tok/s |

### Throughput Improvement

| Comparison | Improvement |
|---|---:|
| KV cache vs no_cache, 1 request | 16.6× |
| KV cache vs no_cache, 4 requests | 25.2× |
| Batched pool vs no_cache, 4 requests | 39.6× |
| Batched pool vs KV cache, 4 requests | 1.57× |

The batching pool performs slightly worse for a single request because of scheduler overhead. Under concurrent load, however, it provides significantly better aggregate throughput.

This behavior is consistent with the goal of continuous batching: improving GPU utilization by processing multiple active requests together.

---

## 4. Key Findings

### KV Cache

KV cache significantly reduces repeated computation during autoregressive generation by reusing previously computed attention keys and values.

The single-request benchmark shows a substantial throughput improvement compared with no-cache generation.

### Continuous Batching

The batching pool demonstrates a clear throughput advantage under concurrent load.

For four concurrent requests:

- Direct KV-cache inference: 8.230 tok/s
- Batched pool: 12.918 tok/s

This represents approximately a 57% throughput improvement.

### Scheduler Behavior

The batching pool provides better aggregate throughput and lower mean TTFT in the concurrent benchmark.

The next step is to measure active batch size, queue wait time, prefill time, and decode time to understand scheduler behavior in more detail.

---

## 5. Metrics to Improve

The following metrics currently require further validation:

- **Peak GPU Memory:** All results show 0.000 GB. GPU memory measurement may not be implemented or may not be capturing CUDA allocations correctly.
- **P95 Latency:** The single-request P95 equals total wall time. This suggests the percentile may be calculated from a single sample.
- **No-cache TTFT:** The concurrent no-cache TTFT is unusually high and should be checked to ensure it measures request arrival to first token correctly.

Future benchmarks should include:

- Queue wait time
- Prefill duration
- Decode duration
- Average decode latency per token
- Average active batch size
- Maximum active batch size
- GPU utilization
- Peak GPU memory allocation

---

## Conclusion

The benchmark demonstrates two important improvements in the mini-vLLM inference engine:

1. **KV cache** significantly improves autoregressive generation performance by avoiding repeated processing of the entire sequence.
2. **Continuous batching** improves aggregate throughput when multiple requests are active concurrently.

The current implementation achieves:

- 16.6× higher throughput with KV cache for a single request.
- 25.2× higher throughput with KV cache for four concurrent requests.
- 57% higher throughput with the batching pool compared with direct KV-cache inference under four concurrent requests.

The results indicate that the KV-cache and batching components are providing measurable performance benefits. Further work will focus on scheduler metrics, GPU memory tracking, and benchmarking with larger numbers of concurrent requests.


## Limitations

This is a learning project. Know what it does **not** do:

- **Full-attention models only** (for example Llama, Qwen2, Phi). Sliding-window and hybrid-attention models break the cache-shape assumptions.
- **Greedy decoding only**: no temperature, top-k or top-p yet.
- **No PagedAttention**: each request reserves a full `max_seq_len` slot, so memory is wasted on short requests.
- **Copy-heavy decode step**: history is gathered from the pool and then concatenated again by `DynamicCache`, so throughput drops as sequences grow. Real engines avoid this with custom attention kernels.
- **Unbatched prefill**: long prompts stall decode for other requests.
- **EOS** uses `tokenizer.eos_token_id`, which is wrong for some chat models (they stop on a different token).
- **fp16** model and pool, which can overflow on some models.
- **No request cancellation**, and no HTTP server.

## Roadmap

- [ ] EOS handling from `generation_config`
- [ ] bf16 support, with the pool dtype matching the model
- [ ] Request cancellation (`abort(req_id)`)
- [ ] Sampling: temperature, top-k, top-p
- [ ] Batched / chunked prefill
- [ ] Block-based (paged) KV allocation
- [ ] OpenAI-compatible HTTP server (FastAPI + SSE)
- [ ] Tests comparing output against `model.generate`

## Acknowledgements

- [vLLM](https://github.com/vllm-project/vllm) and the PagedAttention paper (Kwon et al., 2023) for the ideas this project explores
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the models and tokenizers

## License

MIT. See [LICENSE](LICENSE).