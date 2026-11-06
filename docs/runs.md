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
| 4 | 2026-11-06 | Kaggle T4 | run2 / 6f67820 | Completed, **worse than run 1** | Val acc 0.704, macro-F1 0.335; exclusion collapsed to "not excluded" |
| 5 | 2026-11-05 | Laptop CPU | attention benchmark (fp32) / d7008ea | **Success** | Tiled = reference within ~1e-6; 4096× smaller score buffer; slower than fused SDPA |
| 6 | 2026-11-06 | Kaggle T4 | run3 / 7080257 | Planned | — |

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
- **Results** (Kaggle T4, commit 6f67820, whole notebook 1,307 s: training 1,021 s, the rest model download, smoke tests and about 64 s of val predictions; the earlier "~30 min" was only an estimate):

| Model | Prompt | Acc | Macro-F1 | Incl. mF1 | Excl. mF1 | Evid. micro-F1 | Evid. macro-F1 | Parse OK |
|---|---|---|---|---|---|---|---|---|
| zero-shot | v1 | 0.068 | 0.054 | 0.064 | 0.074 | 0.000 | 0.563 | 100% |
| zero-shot | v2 | 0.175 | 0.070 | 0.111 | 0.097 | 0.000 | 0.563 | 100% |
| run 1 LoRA | v1 | **0.743** | **0.432** | **0.360** | **0.401** | **0.554** | **0.745** | 100% |
| run 2 LoRA | v2 + oversample | 0.704 | 0.335 | 0.343 | 0.210 | 0.477 | 0.693 | 100% |

- **Training:** val loss 0.1222 (epoch 1, **selected**) → 0.1313 → 0.1368. Run 1 was 0.1315 → 0.1146 → 0.1201.
- **Hypothesis (a), the prompt helps zero-shot: partly confirmed.** Accuracy 0.068 → 0.175, and inclusion "not enough information" F1 0.22 → 0.44. But the zero-shot model *still never cites evidence* (all 206 predictions empty) and now over-predicts "not enough information" (72 of 101 "not excluded" cases).
- **Hypothesis (b), oversampling improves macro-F1: rejected.** Macro-F1 fell 0.432 → 0.335. The exclusion confusion matrix shows **every one of the 140 exclusion criteria predicted "not excluded"**. Rare inclusion classes still get F1 = 0.
- **Why it happened** (diagnosis):
  1. **Checkpoint selection by val loss picked epoch 1.** With duplicated rare examples, val loss rises from epoch 2 on (memorising the copies). `load_best_model_at_end` therefore restored the epoch-1 adapter, taken while the learning rate was still high and the model still answered with the majority label.
  2. **Val loss is a poor proxy for the metric we report.** Most target tokens are the fixed JSON wrapper and the evidence list; the label itself is only 2–5 tokens. A lower loss doesn't mean a better macro-F1.
  3. **Confound:** prompt and oversampling changed together, so run 2 can't say which one hurt.
- **No user mistake.** The notebook ran every stage correctly; the problem is in the experiment design.
- **Response:** run 3 (attempt 6). Prompt v2 *without* oversampling, so the prompt effect is isolated against run 1, and the checkpoint is selected by **val macro-F1** from generated predictions at every epoch (D-018).
- **Artifacts:** `results/baseline/Qwen2.5-1.5B-Instruct_prompt-v2_val.*`, `results/finetuned/run2-v2prompt-oversample_val.*`, `results/training/run2-v2prompt-oversample_training_meta.json`, adapter `huggingface.co/umairishuman/trialspec-qwen2.5-1.5b-lora-run2`.

## Attempt 5: Task 4 attention benchmark (laptop CPU, fp32)
- **Run:** `python -m src.attention.benchmark --dtype float32` (blocks 64×64; 3 warm-up + 10 timed repeats; median ± std). Correctness first: `pytest tests/test_attention.py` gives **52 passed**. Tiled matches the reference to 1e-10 in fp64 across 6 shapes × causal/non-causal × 3 block sizes (16×16, 64×32, 7×13), including Lq=1 decode against a cache and chunked prefill (Lq < Lk). fp16 max error is under 2e-3, and logits around 1e4 stay finite.
- **What happened:** completed. Errors are measured against an fp64 ground truth.

| Shape (B,H,Lq,Lk,D=128) | Causal | Reference ms | Tiled ms | SDPA ms | Max abs err (tiled) | Score buffer ref / tiled (MiB) |
|---|---|---|---|---|---|---|
| 1,12,512,512 | no | 18.3 | 39.0 | 11.4 | 5.4e-7 | 12 / 0.19 |
| 1,12,1024,1024 | yes | 86.5 | 87.8 | 33.3 | 1.2e-6 | 48 / 0.19 |
| 1,12,2048,2048 | yes | 971 | 947 | 255 | 1.8e-6 | 192 / 0.19 |
| 1,12,4096,4096 | no | 2833 | 7048 | 1529 | 3.1e-7 | 768 / 0.19 |
| 1,12,4096,4096 | yes | 3208 | 3554 | 873 | 1.2e-6 | 768 / 0.19 |
| 8,12,1,2048 (decode) | yes | 14.9 | 38.5 | 11.3 | 4.7e-8 | 0.75 / 0.05 |

- **Interpretation:** all three implementations are numerically equivalent at fp32 precision. Tiling cuts the score buffer from O(Lq·Lk) to O(bq·bk), so it is 4096× smaller at L=4096. The Python tiled version is **slower** without causal masking, because of a Python loop with one small matmul per tile (kernel launches and allocations each time). With causal masking it skips fully masked tiles (about half the work) and roughly ties the reference. Fused SDPA is 2–4× faster than both: it runs in a single kernel, keeps tiles in on-chip memory and uses optimised GEMMs. This is the expected result for an educational implementation (assignment Task 4).
- **Noise:** some runs have large std (e.g. tiled at L=2048 non-causal: ±574 ms) from laptop CPU frequency scaling and background load. The **GPU benchmark on Kaggle** (fp16, synchronised timing, CUDA peak memory) is still to run; it will be the reported one.
- **Results file:** `results/attention/benchmark_cpu_float32.json`.

## Attempt 6 = Run 3: prompt v2, no oversampling, checkpoint chosen by val macro-F1 (planned)
- **Config:** `configs/run3.yaml`. Identical to run 1 except: **prompt v2** (label definitions); `save_total_limit: null` so all 3 epoch checkpoints are kept; `select_by: val_macro_f1`.
- **New pipeline steps:** the smoke stage now also loads a *raw training checkpoint* as an adapter (`checkpoint-10`), so checkpoint loading is tested before the long run. After training, every `checkpoint-*` is evaluated on val by greedy generation. `src/eval/select_checkpoint.py` then picks the highest val macro-F1 (ties: accuracy, then earlier step), copies it to `outputs/run3-v2prompt-f1select/selected/`, writes `results/finetuned/run3-v2prompt-f1select_selection.json` and pushes only that adapter to `huggingface.co/umairishuman/trialspec-qwen2.5-1.5b-lora-run3`. Unit-tested in `tests/test_select_checkpoint.py`.
- **Test-set discipline:** selection uses val only. The test split is still untouched.
- **Hypotheses:** (a) run 3 vs run 1 isolates the prompt effect under fine-tuning; (b) at least one checkpoint has higher val macro-F1 than its loss-selected counterpart would have had (the per-checkpoint table will show whether loss and F1 disagree).
- **Expected time:** about 22 min (as run 2) minus the oversampling overhead, plus 3 × ~36 s of checkpoint evaluation.
- **Results:** (to be filled in)