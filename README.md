# mini_vllm

A toy vLLM-style inference engine: continuous batching over a preallocated KV
pool (no PagedAttention). Works with `transformers` v4 and v5, greedy decoding only.

## Layout

```
mini_vllm/
├── mini_vllm/
│   ├── __init__.py
│   ├── engine/
│   │   ├── cache_compat.py   # transformers v4/v5 cache shims
│   │   ├── request.py        # Request dataclass
│   │   └── engine.py         # model, KV pool, slots, prefill, batched decode
│   ├── scheduler/
│   │   └── batcher.py        # ContinuousBatcher (admit / decode / evict)
│   ├── demo.py               # streaming demo
│   └── benchmark.py          # no-cache vs kv-cache vs batched pool
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

A GPU is strongly recommended.

## Run (from the project root)

```bash
python -m mini_vllm.demo --model Qwen/Qwen2.5-0.5B-Instruct

python -m mini_vllm.benchmark --model Qwen/Qwen2.5-0.5B-Instruct \
    --num-requests 1 4 8 --max-tokens 64 --max-slots 8
```

Use `--skip-nocache` on the benchmark for long generations (no-cache is O(n^2)).

## Use as a library

```python
from mini_vllm import Engine, ContinuousBatcher

engine = Engine("Qwen/Qwen2.5-0.5B-Instruct", max_slots=4, max_seq_len=1024)
batcher = ContinuousBatcher(engine)
await batcher.start()
req = await batcher.submit("What is a KV cache?", max_tokens=64)
while (delta := await req.token_queue.get()) is not None:
    print(delta, end="")
await batcher.stop()
```

## Known limitations

- Full-attention models only (Llama, Qwen2, Phi). Sliding-window/hybrid models break the cache-shape assumptions.
- Greedy decoding only; EOS uses `tokenizer.eos_token_id`.
- fp16 pool and model; some models need bf16/fp32.
- Each decode step copies history (pool gather + `DynamicCache` concat), so long sequences slow down.
