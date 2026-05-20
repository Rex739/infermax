# InferMax

InferMax is being built incrementally. The first milestone is a plain vLLM
baseline benchmark for `Qwen/Qwen3-4B` inference.

This milestone does not include InferMax optimization logic, scheduling, or
custom batching behavior.

## Setup

Use Python 3.10 or newer with a CUDA-capable environment supported by vLLM.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Baseline Benchmark

```bash
python benchmarks/run_baseline.py
```

The script loads prompts from `data/prompts.jsonl`, runs inference with
`Qwen/Qwen3-4B`, and writes benchmark metrics to `results/baseline.json`.

You can override the defaults:

```bash
python benchmarks/run_baseline.py \
  --model Qwen/Qwen3-4B \
  --prompts data/prompts.jsonl \
  --max-new-tokens 128 \
  --batch-size 8 \
  --output results/baseline.json
```

## Metrics

The baseline result includes:

- Total runtime
- Total generated tokens
- Tokens per second
- Requests per second
- Average latency
- P95 latency
