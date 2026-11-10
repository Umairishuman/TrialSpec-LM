#!/usr/bin/env bash
# Final evaluation (Task 2, D-024): tests -> smoke (4 val examples per model, no test) -> full val + one test pass.
# Usage (via the notebook): CONFIG = 'final_eval'   or   bash scripts/kaggle_final_eval.sh [--allow-test-rerun]
set -euo pipefail
echo "=== final evaluation commit=$(git rev-parse --short HEAD) ==="
step() { echo; echo "##### $1 #####"; date -u +%H:%M:%S; }

step "tests";                       python -m pytest -q tests
step "smoke: final evaluation";     python -m src.eval.final_eval --config configs/final_eval.yaml --smoke
echo; echo "===== FINAL-EVAL SMOKE TEST PASSED ====="
step "final evaluation (val all models, then test once)"
python -m src.eval.final_eval --config configs/final_eval.yaml "$@"
cat results/final/summary.md
