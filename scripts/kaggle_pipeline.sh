#!/usr/bin/env bash
# Full Kaggle experiment for one config: tests -> smoke -> baseline -> train -> fine-tuned eval -> zip.
# Usage: bash scripts/kaggle_pipeline.sh configs/run2.yaml [--smoke-only] [--skip-baseline]
#        bash scripts/kaggle_pipeline.sh benchmarks
#        bash scripts/kaggle_pipeline.sh final_eval
set -euo pipefail

CONFIG="$1"; shift
if [ "$CONFIG" = "benchmarks" ]; then exec bash scripts/kaggle_benchmarks.sh "$@"; fi
if [ "$CONFIG" = "final_eval" ]; then exec bash scripts/kaggle_final_eval.sh "$@"; fi
SMOKE_ONLY=0; SKIP_BASELINE=0
for a in "$@"; do
  case "$a" in
    --smoke-only) SMOKE_ONLY=1 ;;
    --skip-baseline) SKIP_BASELINE=1 ;;
  esac
done

cfg() { python -c "import yaml; v=yaml.safe_load(open('$CONFIG')).get('$1', '$2'); print('' if v is None else v)"; }
RUN=$(cfg run_name)
PV=$(cfg prompt_version)
MODEL=$(cfg model_id)
RP=$(cfg eval_repetition_penalty 1.0)      # runs 1-3 used Qwen's implicit 1.1 (D-020)
SELECT_BY=$(cfg select_by "")
[ "$(cfg run_baseline True)" = "False" ] && SKIP_BASELINE=1
EV=(--model "$MODEL" --prompt-version "$PV" --repetition-penalty "$RP")
echo "=== run=$RUN prompt=$PV model=$MODEL eval_repetition_penalty=$RP select_by=${SELECT_BY:-eval_loss} commit=$(git rev-parse --short HEAD) ==="

step() { echo; echo "##### $1 #####"; date -u +%H:%M:%S; }

step "tests";          python -m pytest -q tests
step "smoke train";    python -m src.train.run --config "$CONFIG" --smoke
step "smoke predict";  python -m src.eval.predict --smoke --split val "${EV[@]}" --out "results/smoke/${RUN}_baseline_val"
step "smoke predict from a raw training checkpoint"
python -m src.eval.predict --smoke --split val "${EV[@]}" --adapter "outputs/${RUN}-smoke/checkpoint-10" --out "results/smoke/${RUN}_ckpt_val"
echo; echo "===== ALL SMOKE TESTS PASSED ====="
[ "$SMOKE_ONLY" = 1 ] && exit 0

if [ "$SKIP_BASELINE" = 0 ]; then
  step "zero-shot baseline (val)"
  python -m src.eval.predict --split val "${EV[@]}" --out "results/baseline/${MODEL##*/}_prompt-${PV}_rp${RP}_val"
fi
step "full training";  python -m src.train.run --config "$CONFIG"
if [ "$SELECT_BY" = "val_macro_f1" ]; then
  for CK in outputs/$RUN/checkpoint-*; do
    step "fine-tuned predict (val) $(basename "$CK")"
    python -m src.eval.predict --split val "${EV[@]}" --adapter "$CK" --out "results/finetuned/${RUN}_$(basename "$CK")_val"
  done
  step "select checkpoint by val macro-F1"
  python -m src.eval.select_checkpoint --config "$CONFIG" --push
else
  step "fine-tuned predict (val)"
  python -m src.eval.predict --split val "${EV[@]}" --adapter "outputs/$RUN/final" --out "results/finetuned/${RUN}_val"
fi
mkdir -p results/training
cp "outputs/$RUN/final/training_meta.json" "results/training/${RUN}_training_meta.json"
echo; echo "===== FULL RUN FINISHED: $RUN ====="
