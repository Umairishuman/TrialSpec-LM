# Run Log

One entry per execution attempt (local or Kaggle), **including failed ones**, numbered in order.
Each entry records: what was run (config, commit, platform), what happened, why it happened, what was changed in response, and the results.
Design reasoning lives in `docs/decisions.md` (D-XXX); this file is the chronological lab notebook.

| # | Date | Platform | Config / commit | Outcome | Main result or failure |
|---|---|---|---|---|---|
| 0 | 2026-11-05 | Laptop CPU | run1 (0.5B override) / 834516d | Failed (killed) | Out of RAM after step 5 |
| 1 | 2026-11-05 | Kaggle (no accelerator) | run1 / 316e638 | Failed in 2 min | No GPU: CPU-only torch |
| 2 | 2026-11-05 | Kaggle T4 | run1 / 965d874 | Failed in 1.5 min | torchao 0.10 vs peft 0.21 |
| 3 | 2026-11-05 | Kaggle T4 | run1 / 29b1596 | **Success** | Val acc 0.743, macro-F1 0.432 |
| 4 | 2026-11-05 | Kaggle T4 | run2 / (pending) | Pending | — |

---

## Attempt 0: local CPU smoke test
- **Run:** `python -m src.train.run --smoke --model Qwen/Qwen2.5-0.5B-Instruct` on the laptop (16 GB RAM, Intel Iris Xe, no CUDA).
- **First try:** crashed immediately with `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'`. **Why:** transformers 5.22 removed `warmup_ratio`; a float `warmup_steps` in [0,1) is now read as a ratio. **Fix:** pass the ratio via `warmup_steps` (D-012).
- **Second try:** trained 5 steps (loss 0.27 → 0.23), ran eval and saved `checkpoint-5`, then the process was killed (exit 255) with about 7.5 GB RAM free. **Why:** fp32 0.5B model + LoRA + activations on CPU exceeded available memory.
- **Response:** all model runs moved to Kaggle. The laptop is used only for code, tests and data scripts.
- **Side bug found later:** the `warmup_ratio` edit accidentally commented out `weight_decay=...`. No effect on run 1, since the config value 0.0 equals the default. Fixed before run 2.

## Attempt 1: Kaggle, first commit
- **Run:** notebook "Save & Run All", commit 316e638.
- **What happened:** `AssertionError: Torch not compiled with CUDA enabled` in the GPU check cell after about 2 min. The log showed `torch 2.11.0+cpu`.
- **Why:** the notebook version was saved with Accelerator = None, so Kaggle used its CPU image.
- **Also observed:** pip dependency conflicts. Our pins `numpy==2.5.4` and `pandas==3.1.1` overrode Kaggle's builds (numba needs numpy<2.3; google-colab needs pandas==2.2.3).
- **Fix:** select GPU T4 x2 before saving. Added an in-kernel GPU check with a fix-it message. Stopped pinning numpy/pandas/torch and log their versions per run instead (D-013, commit e8ec91e).

## Attempt 2: Kaggle T4
- **Run:** commit 965d874. GPU detected (Tesla T4, 14.6 GiB), 13/13 tests passed, data loaded.
- **What happened:** the smoke training crashed in `get_peft_model` with `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`.
- **Why:** Kaggle preinstalls torchao 0.10. peft 0.21 checks torchao whenever it is installed, even though we never use it.
- **Fix:** uninstall torchao. Moved setup into `scripts/kaggle_setup.sh`, which installs deps, uninstalls torchao and runs a tiny LoRA self-check, so that future environment fixes arrive via `git pull` (D-014, commit b156f63).
- **Process lesson:** the smoke stage caught this after 90 s instead of after a long run.

## Attempt 3 = Run 1: first successful full run
- **Run:** `configs/run1.yaml`, commit 29b1596, Kaggle Tesla T4, fp16 AMP. Torch 2.11.0+cu130, transformers 5.22.1, peft 0.21.2, numpy 2.1.3, pandas 2.3.3, datasets 5.1.0.
- **Config:** Qwen2.5-1.5B-Instruct, prompt **v1** (label names only), LoRA r=16/α=32/dropout 0.05 on q,k,v,o,gate,up,down; lr 2e-4 cosine, warmup 5%, 3 epochs, effective batch 16, no oversampling.
- **Training:** val loss 0.1315 → **0.1146 (epoch 2, kept)** → 0.1201. Train loss 0.29 → 0.05. No NaN/inf.
- **Artifacts:** adapter `huggingface.co/umairishuman/trialspec-qwen2.5-1.5b-lora`; `results/baseline/qwen2.5-1.5b_val.*`, `results/finetuned/run1_qwen2.5-1.5b-lora_val.*`, `results/training/run1_training_meta.json`.

| Val (n=206) | Acc | Macro-F1 | Incl. Macro-F1 | Excl. Macro-F1 | Evid. micro-F1 | Parse OK | Time |
|---|---|---|---|---|---|---|---|
| Zero-shot (prompt v1) | 0.068 | 0.054 | 0.064 | 0.074 | 0.000 | 100% | 27 s |
| Run 1 LoRA | 0.743 | 0.432 | 0.360 | 0.401 | 0.554 | 100% | 36 s |

- **Problems found:**
  1. The zero-shot baseline is degenerate: no evidence cited in any of the 206 cases, and "not applicable" predicted 138 times. **Why:** prompt v1 never defines the labels.
  2. Rare classes collapse after fine-tuning. F1 = 0 for "not included" (3), "excluded" (2) and inclusion "not applicable" (1). Exclusion "not enough information" recall is 0.19. **Why:** class imbalance (D-006); "not excluded" is 48% of train.
  3. Bug: output files were named `qwen2.5-1.jsonl` because `Path.with_suffix` treated `.5b_val` as the extension. Fixed (commit a8b846d).
- **Response:** run 2 (D-016).

## Attempt 4 = Run 2: label definitions + oversampling (pending)
- **Config:** `configs/run2.yaml`. Identical to run 1 except:
  - **prompt v2:** one-line definition per label, written to match the expert convention measured on train. For example, 75% of "not excluded" labels cite no evidence, i.e. experts use "not excluded" when the excluding condition is not reported.
  - **oversampling:** each (type, label) group is repeated up to 60 rows, at most 4×. Train goes from 603 to 727 rows: excluded 8→32, not included 13→52, incl. not applicable 5→20, excl. not applicable 23→69; the other groups are unchanged.
- **Also re-run:** the zero-shot baseline with prompt v2, so the before/after comparison uses identical prompts.
- **Hypotheses:** (a) v2 raises zero-shot accuracy well above 0.068; (b) oversampling raises fine-tuned macro-F1 without a large accuracy drop.
- **Caveat:** two changes at once, so the fine-tuned difference can't be attributed to just one of them. The zero-shot v1 vs v2 comparison isolates the prompt effect. If time allows, run 3 = v2 without oversampling.
- **Results:** (to be filled in)
