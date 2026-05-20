#!/usr/bin/env bash
set -euo pipefail

python benchmarks/run_baseline.py \
  --model Qwen/Qwen3-4B \
  --prompts data/prompts.jsonl \
  --max-new-tokens 128 \
  --batch-size 8 \
  --output results/baseline.json
