#!/usr/bin/env bash
# GPU benchmarks for Tasks 3-5. Smoke versions run first; any failure stops before the long runs.
# Usage: bash scripts/kaggle_benchmarks.sh [adapter_repo_or_path]
set -euo pipefail
ADAPTER="${1:-umairishuman/trialspec-qwen2.5-1.5b-lora}"
echo "=== benchmarks commit=$(git rev-parse --short HEAD) adapter=$ADAPTER ==="
step() { echo; echo "##### $1 #####"; date -u +%H:%M:%S; }

step "tests";                      python -m pytest -q tests
step "smoke: attention";           python -m src.attention.benchmark --smoke
step "smoke: kv cache";            python -m src.decoding.benchmark_kv --smoke
echo; echo "===== ALL BENCHMARK SMOKE TESTS PASSED ====="

step "Task 4: attention fp16";     python -m src.attention.benchmark --dtype float16
step "Task 4: attention fp32";     python -m src.attention.benchmark --dtype float32
step "Task 3: KV cache (1.5B fine-tuned, fp16)"
python -m src.decoding.benchmark_kv --adapter "$ADAPTER"
echo; echo "===== BENCHMARKS FINISHED ====="
