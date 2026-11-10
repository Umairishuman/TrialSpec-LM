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
| 6 | 2026-11-06 | Kaggle T4 | run3 / 1a1e11d | Completed, still below run 1 | Selected epoch 3: val macro-F1 0.379; exclusion still collapses |
| 7 | 2026-11-06 | Laptop CPU | Task 3 KV-cache verification / 0d8f427 | **Success** | Manual decoder = HF to 3e-5 (fp32); all 4 decoding paths token-identical |
| 8 | 2026-11-06 | Laptop CPU | Task 5 speculative verification / 3a2e7a6 | **Success** (after 2 test-design fixes) | 1.5B/0.5B bf16: tokens identical to greedy, acceptance 50–58%, 0.7× speed on CPU |
| 9 | 2026-11-06 | Kaggle T4 | benchmarks (Tasks 3–5) / 78cdfab | **Success** (2,983 s) | KV cache 9.3× decode speed-up; tiled attn correct but slow; speculative exact, 0.61–0.80× |
| 10 | 2026-11-10 | Kaggle T4 | run4 / 14eb0b5 | Failed in 1 min | `kaggle_pipeline.sh: line 19: $2: unbound variable` |
| 11 | 2026-11-10 | Kaggle T4 | run4 / 326900a | **Success**, best val so far | Selected epoch 3 (ckpt-138): val acc 0.743, macro-F1 0.604 (rp 1.1); gain mostly from "excluded" (n=2) and exclusion NEI |
| 12 | 2026-11-10 | Kaggle T4 | final_eval / e13346a | **Success**, test locked | Selected run4 (val mF1 0.604); **test acc 0.738, macro-F1 0.660**, evidence micro-F1 0.544 (zero-shot: 0.136 / 0.131 / 0.000) |
| 13 | 2026-11-10 | Laptop CPU | app `--smoke` (tiny models) / uncommitted | Failed after splitting | `UnicodeEncodeError: 'charmap' codec` printing random model output |
| 14 | 2026-11-10 | Laptop CPU | app `--smoke` + `pytest` / uncommitted | **Success** (8.6 s; 41/41 tests after 1 fix) | Retry → fallback path exercised; all 4 decoding modes token-identical |
| 15 | 2026-11-10 | Laptop CPU | app real-model check (scratch script) | Failed at import | `ModuleNotFoundError: No module named 'src'` (script outside repo) |
| 16 | 2026-11-10 | Laptop CPU | app real-model check, run 1 adapter, bf16, offline | Correct but **too slow** | ~15 s/criterion (prefill ≈ 15 s); uncached benchmark 382 s for 16 tokens; all modes token-identical |
| 17 | 2026-11-10 | Laptop CPU | bf16 vs fp32 prefill/decode timing (scratch) | **Success** | fp32 prefill 5.4 s vs bf16 12.6 s (284 tokens); decode 2.2 vs 3.0 tok/s → app switched to fp32 (D-031) |
| 18 | 2026-11-10 | Laptop CPU | app real-model check, **run 4**, fp32, offline | **Success** | ~10.8 s/criterion after the first (25.8 s warm-up); uncached benchmark 47.6 s for 8 tokens; 4/6 labels match experts; all modes token-identical |
| 19 | 2026-11-10 | Laptop CPU | `python -m src.app` (real server, run 4, fp32, offline) driven through the Gradio API | **Success** | Starts in ~23 s; validation errors shown; unseen 4-criterion note screened in 46.7 s; live benchmark 4/4 modes identical |
| 20 | 2026-11-10 | Laptop | report figures + LaTeX build (MiKTeX) | **Success** after 1 fix | First PDF showed a foreign warning-sign image for `attention.png`; renamed figures `fig_*`; 6-page PDF |

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
- **Results** (Kaggle T4, commit 1a1e11d, so evaluated with Qwen's implicit repetition_penalty 1.1 like runs 1–2, see D-020; training 907 s):

| Checkpoint | Val loss | Acc | Macro-F1 | Incl. mF1 | Excl. mF1 | Evid. micro-F1 |
|---|---|---|---|---|---|---|
| epoch 1 (step 38) | 0.1338 | 0.709 | 0.342 | 0.352 | 0.210 | 0.384 |
| epoch 2 (step 76), **lowest loss** | **0.1149** | 0.718 | 0.351 | 0.367 | 0.210 | **0.570** |
| epoch 3 (step 114), **selected (best F1)** | 0.1175 | **0.728** | **0.379** | **0.374** | **0.252** | 0.556 |
| *run 1 (prompt v1), for reference* | 0.1146 | 0.743 | 0.432 | 0.360 | 0.401 | 0.554 |

- **Hypothesis (b) confirmed:** loss and F1 disagree. The loss-optimal checkpoint (epoch 2) has lower macro-F1 than epoch 3, so F1-based selection changed the choice. This directly supports the run 2 diagnosis.
- **Hypothesis (a), the prompt effect, answered negatively:** with prompt v2 and no oversampling, exclusion **still collapses**. 137 of 140 exclusion criteria are predicted "not excluded" (all 26 gold "not enough information" and 10 of 11 "not applicable" included). Inclusion is slightly *better* than run 1 (0.374 vs 0.360; "included" F1 0.78).
- **Why (most likely):** the v2 definition of *not excluded*, "the patient does not have the excluding condition (the note contradicts it **or gives no sign of it**)", tells the model to default to "not excluded" whenever the note is silent. That absorbs the "not enough information" cases, which are exactly the silent-note cases. The definition matches the expert majority convention (D-016) but erases the boundary with "not enough information". Run 1's names-only prompt left that boundary to be learned from data, and the model did learn it partly (NEI F1 0.29).
- **Conclusion:** **run 1 remains the best model on validation** (macro-F1 0.432). The prompt-definition idea helped zero-shot (0.068 → 0.175 accuracy) but hurts fine-tuning for exclusion criteria. This goes in the report as a documented negative result.
- **Artifacts:** `results/finetuned/run3-v2prompt-f1select_*` (all 3 checkpoints + selection), `results/training/run3-v2prompt-f1select_training_meta.json`, adapter `huggingface.co/umairishuman/trialspec-qwen2.5-1.5b-lora-run3` (epoch 3).
## Attempt 7: Task 3 KV-cache verification (laptop CPU, fp32)
- **Run:** a scratch check script; `pytest tests/test_kv_cache.py` (14 passed + 1 real-model test with `RUN_MODEL_TESTS=1`); `python -m src.decoding.benchmark_kv --smoke --modes no_cache hf_cache manual_cache manual_no_cache` (Qwen2.5-0.5B, fp32).
- **What was built:**
  - `src/decoding/manual_decoder.py`: a hand-written Qwen2 forward pass that uses only the HF *weights*, with explicit RMSNorm, RoPE at absolute positions, grouped-query attention (2 KV heads → 14 query heads), and a preallocated `KVCache` with `write` / `advance` / `crop`. Attention goes through the Task 4 code (reference or tiled).
  - `src/decoding/generate.py`: three explicit loops. `greedy_no_cache` re-runs the full sequence each step; `greedy_hf_cache` passes a `DynamicCache` with explicit `position_ids` and `attention_mask`; `greedy_manual` uses the hand-written decoder, cached or not.
- **Correctness results:**
  - Manual forward vs HF forward, 0.5B fp32: max abs logit difference **3.1e-5**. Cached incremental (prefill T−1, then 1 step) vs full: **3.0e-5**.
  - Tiny random Qwen2 in fp64: manual = HF to 1e-9, both full and incremental (prefill 20 + 13 single steps), with reference and tiled attention.
  - Greedy tokens are **identical across all paths** for input lengths {1, 7, 40} × output lengths {1, 5, 24} (tiny model), on the real 0.5B model (16 tokens), and in every smoke benchmark config. **No mismatches to investigate yet.** The GPU fp16 benchmark will be the real test, since reduction order differs between cached and uncached kernels there.
- **CPU smoke timings (0.5B fp32, 1 repeat, indicative only):**

| Input | Output | Mode | Prefill ms | Decode ms | Decode tok/s |
|---|---|---|---|---|---|
| 48 | 8 | no_cache | 294 | 2274 | 3.1 |
| 48 | 8 | hf_cache | 317 | 510 | 13.7 |
| 48 | 8 | manual_cache | 341 | 519 | 13.5 |
| 48 | 8 | manual_no_cache | 336 | 2489 | 2.8 |

- **Interpretation:** prefill costs about the same in every mode (it's the same computation). Decoding with a cache is about 4.5× faster at 48+8 tokens, and the gap grows with length because uncached steps reprocess the whole sequence. The manual decoder is about as fast as HF's cached path, so the hand-written loop adds no measurable overhead.
- **Next:** the GPU benchmark (`CONFIG = 'benchmarks'` in the Kaggle notebook) on the fine-tuned 1.5B in fp16 with inputs {128, 512, 1024} × outputs {32, 128, 256}, 3 repeats.
## Attempt 8: Task 5 speculative decoding verification (laptop CPU)
- **Run:** `pytest tests/test_speculative.py`; a scratch check with Qwen2.5-1.5B-Instruct (target) + 0.5B-Instruct (draft) in bf16, 40 tokens; `python -m src.decoding.benchmark_spec --smoke`.
- **Problem 1 (test design):** `test_unrelated_draft_still_correct` failed, because a *random, unrelated* draft got 100% acceptance. **Why:** tiny random Qwen2 models with **tied** input/output embeddings put most of the logit mass on the input token, so every such model just repeats the last prompt token (`6, 6, 6, …`), and two random models always agree. The KV-cache equality tests had the same weakness. **Fix:** untied embeddings in both test files (outputs now vary, e.g. `1, 59, 22, 59, …`).
- **Problem 2 (test design):** after untying, the "noisy draft" (target weights + 0.05 absolute noise, one layer fewer) got **0%** acceptance. **Why:** the weights are initialised with std ≈ 0.02, so 0.05 noise swamps them. **Fix:** noise = 20% of each tensor's std, same depth. Measured acceptance at k=4 over 5 prompts: 11–54%. Added `test_noisy_draft_partially_accepted`, which asserts 0 < acceptance < 1, so both the accept and reject paths are always exercised.
- **Problem 3 (real finding, decoding settings):** in the smoke benchmark, HF assisted generation did **not** match our greedy output on the free-text prompt. **Why:** Qwen2.5's shipped `generation_config.json` sets `do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1`. Passing `do_sample=False` disables sampling, but **the repetition penalty stays active**, so `model.generate` was not pure greedy. **Impact:** `src/eval/predict.py` called `generate(do_sample=False)`, so **runs 1–3 val predictions used greedy + repetition_penalty 1.1**. Run-to-run comparisons stay valid (identical settings), but the logged decoding description was incomplete. **Fix:** an explicit `GenerationConfig` (`greedy_config`) with `repetition_penalty=1.0` by default and a `--repetition-penalty` flag to reproduce the old setting. All settings are now written to the metrics metadata (D-020). After the fix, HF assisted generation matches on both prompt sets.
- **Problem 4 (benchmark design):** to force fixed-length free-text output, HF needs `min_new_tokens`, which *suppresses* EOS, while our loops just ignore it, so they would diverge exactly where the model wants to stop. **Fix:** both prompt sets are EOS-terminated in every mode (task ≤ 64 tokens, free ≤ 128).
- **Real-model result (bf16 CPU, 40 tokens, prompt "what is chronic stable angina?"):**

| Mode | Time | Tokens equal to greedy | Acceptance | Mean accepted/round | Target forwards |
|---|---|---|---|---|---|
| Target greedy (cached) | 7.4 s | — | — | — | 40 |
| Speculative k=2 | 10.0 s | yes | 0.58 | 1.17 | 19 |
| Speculative k=4 | 11.2 s | yes | 0.50 | 2.00 | 14 |

- **Interpretation:** correct (identical tokens), and the target forward passes drop from 40 to 14–19. It's still slower on CPU: compute-bound CPU decoding makes a 0.5B draft step cost about 1/3 of a target step, so k draft steps plus the extra verification work outweigh the passes saved. On a GPU, single-token decoding is memory-bandwidth-bound, so verifying k+1 tokens costs about the same as one step; that's where a speedup is possible. To be measured in the Kaggle benchmark.
## Attempt 9: GPU benchmarks for Tasks 3–5
- **Run:** Kaggle notebook with `CONFIG = 'benchmarks'` → `scripts/kaggle_benchmarks.sh`, commit 78cdfab, adapter `umairishuman/trialspec-qwen2.5-1.5b-lora` (run 1). Order: tests → attention/KV/speculative smoke → attention fp16 + fp32 → KV cache (inputs {128, 512, 1024} × outputs {32, 128, 256}, 3 repeats) → speculative (k ∈ {2,4,6,8}, 8 task + 8 free-text prompts, 3 repeats).
- **Outcome:** completed in 2,983 s (about 50 min) on a Tesla T4, commit 78cdfab. All tests and the 3 benchmark smoke runs passed first. Files: `results/attention/benchmark_cuda_float{16,32}.json`, `results/kv_cache/benchmark_cuda_float16.json`, `results/speculative/benchmark_cuda_float16.json`. (The downloaded zip was named ``results-benchmark-v`.zip``: a backtick typed instead of "1". Harmless, but PowerShell needs `-LiteralPath` to open it.)

### Task 4: attention (T4; error vs fp64 truth; 3 warm-up + 10 repeats, median)
| B,H,Lq,Lk (D=128) | Causal | dtype | Ref ms | Tiled ms | SDPA ms | Max err (all 3) | Peak extra MiB ref / tiled / SDPA |
|---|---|---|---|---|---|---|---|
| 1,12,512,512 | yes | fp16 | 1.46 | 13.4 | 0.16 | 1.4e-3 | 36 / 3.8 / 1.5 |
| 1,12,1024,1024 | yes | fp16 | 4.95 | 45.8 | 0.34 | 1.7e-3 | 144 / 5.3 / 3.0 |
| 1,12,2048,2048 | yes | fp16 | 17.7 | 168 | 1.26 | 1.7e-3 | 576 / 8.3 / 6.0 |
| 1,12,4096,4096 | no | fp16 | 55.2 | 1233 | 9.0 | 1.7e-4 | **2304** / 14.3 / 12.0 |
| 1,12,4096,4096 | yes | fp16 | 67.9 | 652 | 5.06 | 1.6e-3 | **2304** / 14.3 / 12.0 |
| 1,12,4096,4096 | yes | fp32 | 68.4 | 550 | 18.1 | 1.2e-6 | 2304 / 25.9 / 24.0 |
| 8,12,1,2048 (decode) | yes | fp16 | 2.82 | 9.96 | 0.67 | 8.6e-5 | 97.5 / 3.3 / 0.0 |

- **Numerics:** all three are equivalent. fp32 max error ≤ 1.7e-6; fp16 ≤ 1.7e-3, which is fp16 output rounding at |values| ~ 1 (fp16 spacing near 1 is 9.8e-4), the same for every implementation.
- **Memory:** the reference materialises the B·H·L² score matrix (2.3 GB at L=4096). The tiled version's peak grows only linearly (output + one tile), matching fused SDPA (FlashAttention backend available on T4: `flash=True`).
- **Speed:** on the GPU the tiled Python version is **10–22× slower than the reference and about 70–135× slower than SDPA**, far worse than on CPU (about 1×). Each tile issues several separate CUDA kernels (matmul, max, exp, sum, rescale) from a Python loop: (L/64)² tiles × about 8 launches, e.g. 4096 tiles at L=4096 → about 30k launches. Launch overhead (~5–10 µs each) and HBM round-trips for every intermediate dominate. A fused kernel does the same online softmax inside one kernel, keeps tiles in SRAM and uses tensor cores. Causal tile skipping halves the tiled time (1233 → 652 ms), as expected.

### Task 3: KV cache (fine-tuned 1.5B merged, fp16, greedy, EOS ignored; 1 warm-up + 3 repeats)
| Input | Output | no_cache tok/s | hf_cache tok/s | manual_cache tok/s | Speed-up (manual vs none) | Peak extra MiB none / manual | KV cache MiB | Tokens equal |
|---|---|---|---|---|---|---|---|---|
| 128 | 32 | 23.7 | 26.2 | 25.4 | 1.07× | 92 / 44 | 4.4 | ✔ ✔ |
| 128 | 128 | 20.1 | 26.9 | 26.0 | 1.29× | 149 / 48 | 7.0 | ✔ / ✘ at step 76 |
| 128 | 256 | 15.1 | 26.8 | 25.9 | 1.72× | 223 / 51 | 10.5 | ✔ / ✘ at step 76 |
| 512 | 128 | 6.4 | 26.7 | 25.8 | 4.0× | 373 / 178 | 17.5 | ✔ ✔ |
| 512 | 256 | 5.7 | 26.5 | 25.4 | 4.5× | 447 / 182 | 21.0 | ✔ ✔ |
| 1024 | 128 | 3.0 | 26.4 | 25.5 | 8.5× | 671 / 350 | 31.5 | ✔ ✔ |
| 1024 | 256 | 2.8 | 26.5 | 25.9 | **9.3×** | 745 / 353 | 35.0 | ✔ ✔ |

- **Prefill** is the same in every mode (41 / 120 / 285 ms for 128 / 512 / 1024 tokens), since it's the same computation.
- **Decode:** uncached steps cost O(current length) and grow during generation (1024+256: 314 ms → 396 ms per step). Cached steps are flat at about 37–39 ms regardless of length, so caching turns decoding from quadratic into linear. The manual decoder is within about 3–4% of HF's cached path (Python-level RMSNorm/RoPE).
- **Memory:** the KV cache matches the formula 28,672 B/token: 1,280 tokens → 35.0 MiB. The uncached path has a *higher* peak (it recomputes activations for the whole sequence each step).
- **Mismatches: 2 of 27 configs** (manual_cache, input 128, outputs 128 and 256; the same event). The first difference is at step 76. **Diagnosis:** the uncached reference's top-1 and top-2 logits at that step are **exactly equal in fp16** (margin 0.000); the cached path's logits differ in the last bits (different kernels and reduction order for a 1×L vs L×L attention), so the tie breaks the other way. After that the sequences diverge, as greedy decoding does. This is the numerical effect predicted in D-019, not a logic error: the same code is token-exact in fp32/fp64 (attempt 7, tests).

### Task 5: speculative decoding (target = fine-tuned 1.5B, draft = base 0.5B, fp16, EOS-terminated; 8 val prompts per set; 1 warm-up + 3 repeats)
| Set | k | Acceptance | Mean speed-up | tok/s (target greedy: task 24.2, free 25.6) | Tokens equal (8/8) |
|---|---|---|---|---|---|
| task (JSON, ~15–23 tokens) | 2 | 0.818 | 0.72× | 17.4 | ✔ |
| task | 4 | 0.792 | 0.74× | 17.8 | ✔ |
| task | 6 | 0.827 | **0.80×** | 19.3 | ✔ |
| task | 8 | 0.723 | 0.75× | 18.1 | ✔ |
| free (≤128 tokens) | 2 | 0.762 | 0.76× | 19.4 | ✔ |
| free | 4 | 0.634 | 0.73× | 18.6 | ✔ |
| free | 6 | 0.555 | 0.67× | 17.1 | ✔ |
| free | 8 | 0.490 | 0.61× | 15.6 | ✔ |

- **Correctness:** 64/64 speculative runs are token-identical to target greedy. HF `assistant_model=` greedy output is identical too (16/16).
- **Acceptance** is high on the task prompts (0.72–0.83) even though the draft is *not* fine-tuned: the JSON scaffold (`{"evidence_ids": [`, `"label": "`) is easy to predict. Free text has lower acceptance, falling with k as expected.
- **Why it's slower: launch-bound decoding.** A cached 1.5B step takes about 37 ms at batch 1 on the T4, which is too slow to be explained by memory bandwidth (3 GB of weights / 320 GB/s ≈ 9 ms). Most of the time is per-layer kernel-launch overhead, so cost scales with **layer count**, not parameter count. The 0.5B draft has 24 layers vs the target's 28, so a draft step costs about c ≈ 24/28 ≈ 0.86 of a target step, instead of the ~0.33 its parameter count suggests.
- **Quantitative check:** the expected speed-up (Leviathan et al., 2023) is S = (1 − α^(k+1)) / ((1 − α)(k·c + 1)). With α = 0.79, k = 4, c = 0.86 this gives **0.77×**; measured: **0.74×**. With α = 0.83, k = 6 it predicts 0.73× vs 0.80× measured. So the slowdown is explained by the draft-to-target cost ratio, not by a bug. The same formula shows a speed-up needs c ≲ 0.3 at this acceptance rate.
- **What would make it faster** (not done; out of scope): a shallower draft (few layers), CUDA graphs / `torch.compile` to remove launch overhead (which would bring c toward the parameter ratio), or a larger batch.
- **Memory:** extra peak 116–155 MiB (task) and 60–96 MiB (free) on top of both models' weights (2.9 GB + 0.9 GB), mostly the two KV caches plus the k+1-token verification activations.

## Attempt 10: Run 4, first launch (failed)
- **Run:** Kaggle notebook, `CONFIG = 'configs/run4.yaml'`, cloned commit 14eb0b5. Setup and GPU check passed (Tesla T4, torch 2.11.0+cu130, transformers 5.22.1, peft 0.21.2).
- **What happened:** the pipeline script stopped at once with `scripts/kaggle_pipeline.sh: line 19: $2: unbound variable` (exit 1, about 70 s into the notebook, before tests or any GPU work).
- **Why:** commit 1144c1c gave the `cfg` helper a default-value argument (`'$2'`), but the script runs with `set -u`, and calls such as `cfg run_name` pass only one argument, so `$2` is unbound. Runs 1–3 used an older version of the script without that helper, and this shell code was never executed locally (only the Python modules are covered by pytest).
- **Fix:** `${2:-}` in the helper (empty default when not given). Verified locally by running the config-parsing part of the script under `set -euo pipefail` for `run4.yaml` (prompt v1, rp 1.1, select_by val_macro_f1) and `run1.yaml`; `bash -n` syntax check on all four Kaggle scripts.
- **Lesson:** shell glue is code too: after editing a Kaggle script, run its non-GPU part locally before launching.

## Attempt 11 = Run 4: prompt v1 + oversampling + F1 selection (planned re-launch of attempt 10)
- **Config:** `configs/run4.yaml`. Same as run 1 except: oversampling (target 60, max 4×, as in run 2); `select_by: val_macro_f1`; all checkpoints kept. `eval_repetition_penalty: 1.1`, so its val numbers are directly comparable with runs 1–3; zero-shot baseline skipped (it already exists for v1 at 1.1).
- **Why:** completes a 2×2 ablation, with prompt {v1, v2} × oversampling {off, on} = run 1, run 4, run 3, run 2. Run 2 was the only oversampling run, and it was loss-selected with prompt v2, so oversampling has not been fairly tested yet.
- **Hypothesis:** with the v1 prompt (which keeps the exclusion NEI boundary) and F1 selection, oversampling raises rare-class recall and macro-F1 above run 1's 0.432. If it doesn't, oversampling is rejected for this data.
- **Caveat:** run 1 was selected by loss, runs 3–4 by F1. Run 1's checkpoints weren't kept, so this can't be equalised without retraining. It will be stated in the report.
- **Outcome:** completed on a Tesla T4, commit 326900a (fp16 AMP). Results downloaded as `results-v4.zip`; the new files were copied to `results/finetuned/run4-*`, `results/smoke/run4-*` and `results/training/run4-*_training_meta.json`. The other files in the zip are older copies of what the repo already had (Kaggle clones the repo), so they were not copied. The selected adapter (checkpoint-138) is on `huggingface.co/umairishuman/trialspec-qwen2.5-1.5b-lora-run4` (Hub commit "checkpoint-138 selected by val macro-F1 (326900a3)"). `training_meta.json` lists `best_checkpoint = checkpoint-46`: that is the Trainer's own loss-based choice, not the one used.
- **Results (val, n=206, greedy + repetition_penalty 1.1, same as runs 1–3):**

| Checkpoint | Val loss | Acc | Macro-F1 | Incl. mF1 | Excl. mF1 | Evid. micro-F1 | Evid. macro-F1 | Parse OK |
|---|---|---|---|---|---|---|---|---|
| epoch 1 (step 46), lowest loss | **0.1211** | 0.733 | 0.589 | 0.344 | 0.598 | 0.500 | — | 100% |
| epoch 2 (step 92) | 0.1309 | 0.728 | 0.542 | 0.354 | 0.597 | **0.537** | — | 100% |
| epoch 3 (step 138), **selected (best F1)** | 0.1355 | **0.743** | **0.604** | **0.368** | **0.676** | 0.520 | 0.717 | 100% |
| *run 1 (v1, no oversampling), for reference* | 0.1146 | 0.743 | 0.432 | 0.360 | 0.401 | 0.554 | 0.745 | 100% |

Per-class F1 (support), run 4 epoch 3 vs run 1:

| Class | Support | Run 1 | Run 4 |
|---|---|---|---|
| incl. included | 35 | 0.742 | 0.750 |
| incl. not included | 3 | 0.000 | 0.000 |
| incl. not enough information | 27 | 0.697 | 0.721 |
| incl. not applicable | 1 | 0.000 | 0.000 |
| excl. excluded | **2** | 0.000 | **1.000** |
| excl. not excluded | 101 | 0.867 | 0.860 |
| excl. not enough information | 26 | 0.294 | **0.410** |
| excl. not applicable | 11 | 0.444 | 0.435 |

- **Hypothesis confirmed, with a caveat.** With prompt v1 and F1 selection, oversampling raises val macro-F1 from 0.432 to 0.604, and exclusion macro-F1 from 0.401 to 0.676. Most of the jump, though, is the class "excluded" going from F1 0 to 1.0 **on 2 validation examples**. In a macro average over 8 classes that one class alone adds about 0.12. The more robust gain is exclusion "not enough information" (n=26: 0.294 → 0.410). Accuracy is unchanged (0.743), and evidence micro-F1 drops slightly (0.554 → 0.520: recall up, precision down).
- **Loss vs F1 again:** val loss is lowest at epoch 1 and rises to epoch 3 (memorisation of the repeated rare examples, as in run 2), yet epoch 3 has the best macro-F1. Loss-based selection would have picked epoch 1 (0.589). This is a second independent case for D-018 (see `results/training/wandb_export/training_curves.png`, now including run 4).
- **2×2 ablation at rp 1.1 (val macro-F1):** v1 = 0.432 (run 1), v1 + oversampling = **0.604** (run 4), v2 = 0.379 (run 3), v2 + oversampling = 0.335 (run 2, loss-selected). Oversampling helps with v1 and not with v2; the v2 "not excluded" definition remains the problem (D-022).
- **W&B export re-run** (`python -m src.train.export_wandb`, key loaded from the git-ignored `.env` into the process only): 4 runs exported, run 4 = `wandb.ai/.../trialspec-lm/runs/ns8kvh36` (31 history rows).
- **Next:** attempt 12 (final evaluation) re-scores runs 1–4 on val at pure greedy and makes the final choice (D-024). Run 4 is the favourite, but the decision rests on that run.

## Attempt 12: final evaluation, val selection + test once (completed 2026-11-10)
- **Run:** Kaggle notebook with `CONFIG = 'final_eval'` → `scripts/kaggle_final_eval.sh`: tests → smoke (`final_eval --smoke`: 4 val examples per model, every adapter downloaded, no test access) → full final evaluation. Protocol: D-024.
- **Precondition:** run 4 finished and its selected adapter is on `huggingface.co/umairishuman/trialspec-qwen2.5-1.5b-lora-run4`. If the push failed, the smoke stage stops within minutes.
- **Config** (`configs/final_eval.yaml`): pure greedy (repetition_penalty 1.0), max 64 new tokens, batch 8, fp16. Stage A on val: zeroshot-v1, zeroshot-v2, run1–run4. Selection: max val macro-F1 → val accuracy → config order. Stage B on test: the selected run and the zero-shot baseline with its prompt version.
- **Hypotheses:** (a) removing the 1.1 repetition penalty changes val macro-F1 only slightly (≤ 0.03), because answers are short JSON whose repeated tokens (`"`, `,`) are mostly scaffold; (b) the val ranking among runs 1–4 is the same as at 1.1; (c) val and test macro-F1 of the selected model differ by less than 0.1. With 8 patients per split the variance is large, and "excluded" has no test examples (D-006).
- **Expected time:** about 20 min (6 model loads for smoke + 6 val passes of ~40 s + 2 test passes + model loads).
- **After the run:** download as `results-final.zip`, copy `results/final/` into the repo and **commit `test_lock.json`**, which locks the test split. Then write the "why" for each error case in `error_analysis_test.md`.
- **Outcome:** completed on a Tesla T4, commit e13346a. Selection written 04:24:08 UTC; test evaluated once, `test_lock.json` written 04:25:16 UTC. The downloaded zip is `results-final-eval.zip` (not `results-final.zip` as planned). `results/final/` (37 files) and `results/smoke/final/` were copied into the repo. The rest of the zip holds older copies of files the repo already has, so those were not copied. **`test_lock.json` is committed, so the test split is now locked** (D-024). This evaluation does not log to W&B; the W&B export was re-run anyway and returned the same 4 training runs with byte-identical files.
- **Stage A (val, n=206, pure greedy):**

| Model | Prompt | Acc | Macro-F1 | Incl. mF1 | Excl. mF1 | Evid. micro-F1 | Evid. macro-F1 | Parse OK |
|---|---|---|---|---|---|---|---|---|
| zeroshot-v1 | v1 | 0.068 | 0.054 | 0.064 | 0.074 | 0.000 | 0.563 | 100% |
| zeroshot-v2 | v2 | 0.175 | 0.070 | 0.111 | 0.097 | 0.000 | 0.563 | 100% |
| run1 | v1 | 0.743 | 0.432 | 0.360 | 0.401 | 0.554 | 0.745 | 100% |
| run2 | v2 | 0.704 | 0.335 | 0.343 | 0.210 | 0.468 | 0.688 | 100% |
| run3 | v2 | 0.728 | 0.379 | 0.374 | 0.252 | 0.556 | 0.740 | 100% |
| **run4 (selected)** | v1 | **0.743** | **0.604** | 0.368 | **0.676** | 0.520 | 0.717 | 100% |

- **Selection:** run4 (`umairishuman/trialspec-qwen2.5-1.5b-lora-run4`) by val macro-F1; its zero-shot counterpart is zeroshot-v1.
- **Stage B (test, n=206, pure greedy, run once):**

| Model | Acc | Macro-F1 | Incl. mF1 | Excl. mF1 | Evid. micro P / R / F1 | Evid. macro-F1 | Invalid IDs | Parse OK |
|---|---|---|---|---|---|---|---|---|
| zeroshot-v1 | 0.136 | 0.131 | 0.158 | 0.132 | 0 / 0 / 0.000 | 0.704 | 0 | 100% |
| **run4** | **0.738** | **0.660** | **0.664** | **0.514** | 0.463 / 0.660 / **0.544** | **0.778** | 0 | 100% |

Test per-class F1 for run4 (support): included 0.863 (27), not included 0.429 (7), inclusion NEI 0.864 (38), inclusion N/A 0.500 (3); exclusion "excluded" N/A (0 examples), not excluded 0.816 (91), **exclusion NEI 0.154 (28; recall 0.107)**, exclusion N/A 0.571 (12). Test macro-F1 averages 7 classes, because "excluded" has no test support (D-011). 145 of 206 test gold evidence sets are empty, which inflates evidence macro-F1, so micro-F1 is the main evidence metric (D-015).
- **Hypotheses:**
  - (a) Confirmed, and more strongly than predicted. Pure greedy changes **1 of 1,236** val outputs relative to repetition_penalty 1.1 (one evidence list in run 2; 0 label changes in any model). Every val metric except run 2's evidence (micro-F1 0.477 → 0.468) is identical to the rp 1.1 numbers. The hidden penalty (D-020) did not affect any earlier conclusion.
  - (b) Confirmed: the ranking is identical (run4 > run1 > run3 > run2).
  - (c) Confirmed: val 0.604 vs test 0.660 (difference 0.056 < 0.1). The two averages use different class sets (8 vs 7 classes), so the comparison is loose.
- **Error analysis:** test has 54 wrong labels out of 206 (26.2%). By category: missed_information 28, unsupported_evidence 30 (correct label), spurious_nei 16, applicability 6, polarity_flip 4, hallucinated IDs 0, parse failures 0. 12 cases are printed per split in `error_analysis_{val,test}.md`. The "why it went wrong" text is still to be written by hand (report task).
- **Main weakness:** exclusion "not enough information". The model mostly answers "not excluded" when the note is silent (recall 0.107 on test), which is the D-016 tension between expert convention and the no-evidence rule. The app flags these decisions as "no supporting evidence cited".
- **Before/after (B4, same 206 test examples, identical decoding):** accuracy 0.136 → 0.738, macro-F1 0.131 → 0.660, evidence micro-F1 0.000 → 0.544.

## Attempt 13: app smoke test, first try (failed)
- **Run:** `python -m src.app --smoke` on the laptop (tiny random 2-layer Qwen2 target + 1-layer draft with the real Qwen tokenizer; one free-text note with 3 criteria at the time).
- **What happened:** validation, sentence splitting and generation for all 3 criteria worked. The script then crashed while printing the results: `UnicodeEncodeError: 'charmap' codec can't encode characters in position 55-56`.
- **Why:** random models emit arbitrary Unicode tokens, and the Windows console uses cp1252. The same crash could hit the real app if a model ever printed an unusual character to the console.
- **Fix:** `src/app/__main__.py` reconfigures stdout/stderr with `errors="replace"` at start-up. The smoke input was also reduced to the planned 1 note with 2 criteria.
- **Observed before the crash:** the free-text splitter correctly kept "500 mg b.i.d. and…" and "Dr. Smith…" inside their sentences (4 sentences, IDs 0–3).

## Attempt 14: app smoke test and unit tests (success)
- **Smoke run:** `python -m src.app --smoke` passed in 8.6 s.
  - The empty note and empty criteria were both rejected with friendly messages.
  - Both criteria went through the full malformed-output path: the 8-token answer was cut off → retry with 12 tokens → still unparseable → constrained label scoring (fallback). Labels "included" and "not excluded" were flagged 🔁, and the outcome was "Needs review" because fallback rows count as unresolved (D-029).
  - Markdown and HTML reports were written. The benchmark ran all 4 modes (uncached, HF DynamicCache, hand-written cache, speculative k=2), all token-identical in fp32.
  - The Gradio UI was built with 5 examples.
- **Unit tests:** `pytest tests/test_app.py`: 40/41 on the first run. The failure was real: the Markdown report escaped `_` inside the decoding JSON (`repetition\_penalty`), which makes the provenance hard to read. Fix: render JSON as code spans. 41/41 after the fix. Full suite: 173 passed, 1 skipped (gated real-model test).
- **Prefetch:** `python -m src.app --prefetch` downloaded the run 1–4 adapters (17–35 s each); the base and draft models were already cached.

## Attempt 15: real-model app check, first try (failed)
- **Run:** a scratch script (outside the repo) that loads the real backend and screens one val example headless.
- **What happened:** `ModuleNotFoundError: No module named 'src'`, at once.
- **Why:** Python puts the *script's* directory on `sys.path`, not the working directory. **Fix:** relaunched with `PYTHONPATH=<repo root>`. No code change.

## Attempt 16: real-model app check on the laptop (CPU, bf16, offline)
- **Run:** same script, `HF_HUB_OFFLINE=1` (proves the demo can start without a network). Model: base 1.5B + **run 1** adapter (the final selection did not exist yet when it started) + 0.5B draft, bf16 on the i5-1145G7. Example: the first val example (6 criteria). Then a benchmark of all 4 modes on criterion I1, 16 new tokens, k=4, warm-up on.
- **Hypothesis:** about 5–8 s per criterion (D-026); uncached decoding much slower than the cached modes; speculative exact but not faster on CPU (attempt 8).
- **Outcome:** functionally correct. It started fully offline; target + adapter loaded and merged in 19.1 s, draft in 2.7 s, process RSS 4.44 GiB. All 6 criteria parsed on the first attempt (no retry/fallback), reports were written, and all 4 benchmark modes were token-identical. **But it was about 2× slower than planned:**

| Criterion | Prompt tokens | Model label | Expert label | Evidence | Time |
|---|---|---|---|---|---|
| I1 | 284 | not enough information | not included | [0] | 15.0 s |
| I2 | 287 | not enough information | included | [0] | 15.1 s |
| I3 | 299 | not enough information | not applicable | [] | 15.4 s |
| E1 | 294 | not excluded | not excluded | [] | 14.7 s |
| E2 | 294 | not enough information | not enough information | [0–5] | 18.2 s |
| E3 | 283 | not excluded | not enough information | [] | 14.2 s |

Outcome card: **Needs review** (unresolved I1, I2, I3, E2; no inclusion criterion met; E1 and E3 flagged "no supporting sentence"). 2/6 labels agree with the experts on this example. This is run 1's adapter on one val pair, so it says nothing about overall accuracy; the app now uses run 4 (D-030).

Benchmark, criterion I1 (284-token prompt, 16 new tokens, EOS-terminated, warm-up, 1 repeat, bf16):

| Mode | Prefill | Decode | Total | Decode tok/s | Speed-up vs uncached | Target forwards | Acceptance | KV cache | Same tokens |
|---|---|---|---|---|---|---|---|---|---|
| Uncached reference | 15.3 s | 367.2 s | 382.4 s | 0.04 | 1.0× | 16 | — | — | ✔ |
| KV cache (HF DynamicCache) | 14.2 s | 3.5 s | 17.7 s | 4.25 | 21.6× | 16 | — | 8.2 MiB | ✔ |
| KV cache (hand-written) | 15.0 s | 3.7 s | 18.8 s | 4.03 | 20.4× | 16 | — | 8.2 MiB | ✔ |
| Speculative (k=4) | 19.6 s | 3.8 s | 23.5 s | 3.91 | 16.3× | 5 | 0.81 | 11.7 MiB | ✔ |

- **Problem:** prefill dominates. About 14–15 s for ~290 tokens (≈ 20 tok/s), while cached decoding runs at a reasonable ~4 tok/s. Uncached decoding repeats that prefill at every step, so 16 tokens took 6.4 minutes, which is far too slow for a live demo.
- **Suspected cause:** the i5-1145G7 (Tiger Lake) has no native bf16 matrix instructions (no AVX512-BF16/AMX), so large bf16 matmuls (prefill) run through a slow conversion path. Small decode steps are memory-bound, so they are hurt less. The 7.4 s in attempt 8 had a short prompt, which hid this. **Next:** measure fp32 vs bf16 prefill on the same prompt (attempt 17).
- **Positive findings:** 284-token prompts, 1/3 of the training maximum. Speculative decoding needed only 5 target forwards for 16 tokens (acceptance 0.81) and was exact, but slower overall: the 0.5B draft's own prefill adds 4–5 s.
## Attempt 17: bf16 vs fp32 speed on the laptop CPU
- **Run:** scratch script: base Qwen2.5-1.5B-Instruct (no adapter), 284 random token IDs (the length of the app prompts), `greedy_hf_cache` for 16 tokens; warm-up, then the best of 2 runs, for fp32 and bf16. 4 threads, offline.
- **Results:** fp32 prefill **5.44 s**, decode 2.19 tok/s; bf16 prefill **12.59 s**, decode 2.99 tok/s.
- **Interpretation:** bf16 prefill is 2.3× slower because this CPU has no native bf16 matmul. Decode is memory-bound, so bf16 is a little faster there. Per criterion, fp32 wins (~13 s vs ~18 s), and the uncached benchmark (one prefill per token) gains the full 2.3×. Decode tok/s here is lower than in attempt 16 (4.0–4.2): laptop clock speed varies with temperature, so CPU timings carry ±30% noise. **Decision:** D-031, app dtype → fp32.

## Attempt 18: real-model app check with the final model in fp32
- **Run:** the attempt 16 script, now picking **run 4** automatically from `results/final/selection.json`, fp32, offline; benchmark at 8 new tokens.
- **Hypothesis:** ~10–15 s per criterion; uncached benchmark ≈ 8 × 5 s ≈ 45 s; RAM ≈ 8–9 GiB.
- **Outcome:** success, offline. Target + run 4 adapter loaded in 18.7 s, draft in 8.8 s. Reported RSS 1.83 GiB: on Windows, memory-mapped safetensors weights are not fully counted in RSS, so this understates real use (fp32 weights ≈ 8.2 GB by size); D-031's estimate stands.

| Criterion | Model label (run 4) | Expert label | Evidence | Time |
|---|---|---|---|---|
| I1 | not enough information | not included | [0] | 25.8 s (first call: warm-up) |
| I2 | included | included ✓ | [0] | 12.8 s |
| I3 | not enough information | not applicable | [0, 4, 5] | 9.6 s |
| E1 | not excluded | not excluded ✓ | [0] | 9.5 s |
| E2 | not enough information | not enough information ✓ | [0–5] | 13.4 s |
| E3 | not excluded | not enough information | [6] | 8.6 s |

Total 79.6 s (bf16 run 1 in attempt 16: 92.7 s); **mean 10.8 s per criterion after the first.** Outcome card: Needs review (unresolved I1, I3, E2). 4/6 labels agree with the experts (run 1: 2/6 on the same pair). This is a single val pair, not an accuracy estimate.

Benchmark, criterion I1, 8 new tokens, fp32:

| Mode | Prefill | Decode | Total | Decode tok/s | Speed-up vs uncached | Target forwards | Acceptance | KV cache | Same tokens |
|---|---|---|---|---|---|---|---|---|---|
| Uncached reference | 5.8 s | 41.8 s | 47.6 s | 0.17 | 1.0× | 8 | — | — | ✔ |
| KV cache (HF DynamicCache) | 5.1 s | 1.6 s | 6.7 s | 4.41 | 7.1× | 8 | — | 16.0 MiB | ✔ |
| KV cache (hand-written) | 7.8 s | 2.8 s | 10.6 s | 2.47 | 4.5× | 8 | — | 16.0 MiB | ✔ |
| Speculative (k=4) | 11.9 s | 2.2 s | 14.0 s | 3.20 | 3.4× | 3 | 0.63 | 22.8 MiB | ✔ |

- **Hypotheses:** ~10–15 s per criterion ✔; uncached ≈ 45 s ✔ (47.6 s).
- **Interpretation:** fp32 prefill is ~5–6 s, as in attempt 17. Uncached decode costs one full re-run of the prompt per token (~5.2 s each), so the KV cache is the difference between ~5 s and ~0.2–0.4 s per token. The hand-written decoder's slower prefill (7.8 s vs 5.1 s) comes from Python-level RMSNorm/RoPE and our reference attention instead of fused SDPA on CPU; single runs on a laptop vary by ±30% (attempt 17). Speculative decoding is exact, needs 3 instead of 8 target passes, but pays the draft's prefill (≈ +6 s), so it is slower end to end, as on the T4 (D-023).

## Attempt 19: the real app server, end to end through its own API
- **Run:** `HF_HUB_OFFLINE=1 python -m src.app` (the documented one-line start), then the UI's own event handlers called through `gradio_client` against `http://127.0.0.1:7860`. The page config contains the disclaimer.
- **Start-up:** target + run 4 adapter 14.1 s, draft 8.5 s; serving after ~23 s, fully offline.
- **Input validation (G1/G4):** an empty note and empty criteria returned both friendly messages; a 10,000-character note returned "The note is too long (10,000 characters; the limit is 6,000)…". No model call was made in either case.
- **Unseen note (G2):** a free-text note written for this test (not from the dataset; 4 sentences) with 2 inclusion + 2 exclusion criteria. Screened in 46.7 s (5.8 / 7.3 / 9.2 / 8.9 s per criterion). The table shows label, highlighted evidence sentence and explanation for each criterion; Markdown and HTML reports were offered for download.
  - I1 "Type 2 diabetes mellitus" → included, cites [0] ✔
  - I2 "HbA1c between 7% and 10%" → not enough information, cites [1] ("HbA1c last month was 8.1%") ✘: the value is inside the range, so a clinician would say "included". This is a genuine model error (numeric range reasoning) and a useful example for the report and demo; the rule turns it into **Needs review** with 🟠 missing information, not a wrong final verdict.
  - E1 "Myocardial infarction within the past 6 months" → not excluded, cites [2] (MI in 2021) ✔
  - E2 "Pregnancy" → not excluded, cites [3] ("She is not pregnant") ✔
- **Live benchmark (G3):** custom prompt "Name one symptom of diabetes." (35 tokens), 4 new tokens, k=2, all 4 modes, 24 s for the whole request: uncached 4.43 s, HF cache 1.89 s (2.34×), hand-written cache 1.94 s (2.29×), speculative 3.33 s (1.33×; acceptance 0.50, 3 target passes). Tokens identical in every mode; KV cache 2.1 MiB (target) / 3.1 MiB (target + draft).
- **Note:** the `gradio_client` test script first crashed printing 🟠 to the cp1252 console (the same issue as attempt 13, in the test script, not the app); re-run with `PYTHONIOENCODING=utf-8`.

## Attempt 20: report figures and LaTeX build
- **Run:** `python -m src.report.figures --smoke`, then without `--smoke` (6 PNGs); headless Edge screenshot of the exported HTML report from attempt 19; `latexmk -pdf main.tex` with MiKTeX.
- **What happened:** the first build compiled without errors (7 pages), but reading the rendered PDF showed three problems: (1) Fig. 5 was a large red warning-sign bitmap instead of our attention plot; (2) Tables II and IV were wider than the column; (3) Eq. (1) ran into the next column.
- **Why:** (1) `\includegraphics{attention.png}` searches the TeX tree as well as `\graphicspath`, and MiKTeX ships an unrelated `attention.png`, so a generic file name was silently resolved to the wrong image. No warning is printed, which is why the rendered PDF has to be checked. (2–3) Layout.
- **Fix:** all figures renamed `fig_*.png` (script and `main.tex`); tables wrapped in `\resizebox{\columnwidth}`; the update equations split over three aligned lines; `--` in `\texttt` escaped. **Result:** 6-page PDF, no undefined references, no overfull boxes reported; every figure checked visually.
