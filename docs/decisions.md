# Decision Log

Two parts:
1. **Changelog:** what changed, at which project stage, and why (one row per change, newest at the bottom).
2. **Decisions:** the full reasoning for each architectural or technical decision (D-XXX), referenced from the changelog.

Stages follow `ASSIGNMENT_PLAN.md`: S1 Setup · S2 Dataset/split · S3 Research/model choice · S4 Baseline · S5 Fine-tuning · S6 Eval (Task 2) · S7 KV cache (Task 3) · S8 Tiled attention (Task 4) · S9 Speculative decoding (Task 5) · S10 App (Task 6) · S11 Report · S12 Submission.

---

## Changelog

| Date | Stage | Change | Reason / trigger | Ref | Commit |
|---|---|---|---|---|---|
| 2026-11-05 | S1 | Created plan, CLAUDE.md and decision log | Project kickoff | D-001–D-003 | 51ef949 |
| 2026-11-05 | S2 | Dataset loader drops GPT-4 columns + `explanation_correctness` + `training` | Leakage rule (B2) | D-005 | 51ef949 |
| 2026-11-05 | S2 | Fixed 37/8/8 split + manifest + 9 leakage/split tests | Prescribed protocol (B1) | D-006 | 51ef949 |
| 2026-11-05 | S3 | Model candidates narrowed to Qwen2.5-1.5B target / 0.5B draft; 3B rejected | 3B license is not Apache; identical tokenizer verified | D-004 → D-008 | 834516d |
| 2026-11-05 | S4 | Output schema `{"evidence_ids", "label"}`; no free-text explanation training | Dataset has no expert explanations; GPT-4 ones forbidden | D-009 | 834516d |
| 2026-11-05 | S5 | `max_len=1024`, no truncation (longest example 625 tokens) | Measured token lengths | D-008 | 834516d |
| 2026-11-05 | S5 | `warmup_ratio` → float `warmup_steps` | Local CPU smoke test crashed: removed in transformers 5.22 | D-012 | 834516d |
| 2026-11-05 | S5 | Local CPU smoke training abandoned; all model runs moved to Kaggle | Laptop has 16 GB RAM, no NVIDIA GPU; CPU run was killed after step 5 | D-012 | — |
| 2026-11-05 | S5 | Kaggle notebook changed to run the full training automatically once smoke tests pass | One "Save & Run All" per experiment | D-012 | 316e638 |
| 2026-11-05 | S5 | First Kaggle run failed: no GPU selected (`torch+cpu`). Added in-kernel GPU check with fix-it message | Accelerator not set before commit | D-013 | e8ec91e |
| 2026-11-05 | S5 | Stopped pinning numpy/pandas; versions now logged per run | pip conflicts with Kaggle's preinstalled packages | D-013 | e8ec91e |
| 2026-11-05 | S10 | Planned app hosting changed from laptop to free HF Space (CPU, 16 GB) | Laptop RAM too small for 1.5B + 0.5B draft | to be written at S10 | — |
| 2026-11-05 | S5 | 2nd Kaggle run: GPU + tests OK, smoke failed at LoRA injection. Added `scripts/kaggle_setup.sh` (installs deps, uninstalls torchao, self-checks peft); notebook calls it | peft 0.21 rejects Kaggle's preinstalled torchao 0.10 | D-014 | b156f63 |
| 2026-11-05 | S4/S5 | **Run 1 complete** (Kaggle T4, commit 29b1596): zero-shot baseline + LoRA fine-tune + val predictions; adapter on HF Hub; results copied to `results/` | First full pipeline run | D-015 | a8b846d |
| 2026-11-05 | S4 | Fixed output filenames: `with_suffix` truncated `qwen2.5-1.5b_val` → `qwen2.5-1.jsonl` | Found in run 1 outputs | D-015 | a8b846d |
| 2026-11-05 | S4 | Prompt versioning: v1 (names only, run 1) kept; v2 adds a one-line definition per label | Degenerate zero-shot baseline in run 1 | D-016 | 800548a |
| 2026-11-05 | S5 | Optional per-group oversampling (`train.oversample`: target 60, max 4×) | Rare-class collapse in run 1 | D-016 | 800548a |
| 2026-11-05 | S5 | Restored `weight_decay` argument accidentally commented out by the D-012 edit | Found while reviewing `run.py`; no effect on run 1 (0.0 = default) | D-016 | 800548a |
| 2026-11-05 | S1/S5 | Configs are now per run (`configs/run1.yaml`, `configs/run2.yaml`) with `run_name`; Kaggle steps moved to `scripts/kaggle_pipeline.sh`; notebook reduced to choosing `CONFIG` | A new experiment shouldn't need notebook edits | D-016 | 800548a |
| 2026-11-05 | S1 | Added `docs/runs.md`: numbered log of every run attempt, including failures | User request | — | 800548a |
| 2026-11-05 | S8 | Task 4: `src/attention/attention.py` (reference SDPA + tiled online-softmax), 52 correctness tests, `src/attention/benchmark.py` with `--smoke` | Task 4 requirements (E1–E3) | D-017 | d7008ea |
| 2026-11-06 | S5 | Run 2 results recorded: worse than run 1 (macro-F1 0.335 vs 0.432); oversampling + loss-based selection picked an under-trained epoch-1 adapter | Run 2 analysis | D-018 | 7080257 |
| 2026-11-06 | S5 | Checkpoint selection by **val macro-F1** from generation (`src/eval/select_checkpoint.py`, `select_by` in config, `save_total_limit` configurable); pipeline evaluates each epoch checkpoint; smoke stage now loads a raw checkpoint | Val loss disagrees with the reported metric | D-018 | 7080257 |
| 2026-11-06 | S5 | `configs/run3.yaml`: prompt v2, no oversampling, F1-based selection | Remove run 2's confound; fix selection | D-018 | 7080257 |
| 2026-11-06 | S6/S11 | `src/eval/compare.py`: markdown comparison table of metrics files | Report and run-log tables | — | 7080257 |

---

## Decisions

## D-001: Compute platform: Kaggle (2026-11-05)
**Context:** Only free compute is allowed; training needs a GPU.
**Options considered:** Kaggle (T4 x2 / P100, ~30 GPU-h/week, 12 h sessions), Google Colab free (T4, session limits less predictable), local hardware.
**Decision:** Kaggle for training and GPU benchmarks; local machine for development, CPU tests and the app.
**Why:** Predictable weekly quota, background "Save & Run All" runs, Secrets for the W&B key.
**Consequences/risks:** T4/P100 have no bf16 and no FlashAttention-2. See D-002.
**Status:** accepted

## D-002: Training precision on T4 (2026-11-05)
**Context:** T4 (Turing) supports fp16 and fp32 but not bf16. fp16 has a narrower range, so it can overflow to NaN/inf on some models.
**Options considered:**
1. fp16 mixed precision: LoRA adapters in fp32, base in fp16, loss scaling.
2. Full fp32 base + LoRA: about 6 GB of weights for 1.5B, fits in 16 GB with gradient checkpointing. Slower, but numerically safest.
3. QLoRA: 4-bit base with fp16 compute. Lowest memory, but quantization affects the before/after comparison and is slower per step.
4. A bf16-capable free GPU: not reliably available for free.
**Decision:** Option 1 is the default. If the smoke test shows NaN/inf loss, fall back to option 2.
**Why:** Fastest on T4 tensor cores. A 1.5B model leaves memory headroom, so option 2 is a cheap fallback.
**Consequences/risks:** Report the precision used. Watch for loss spikes in W&B.
**Status:** proposed (confirm after smoke test)

## D-003: Experiment tracking: Weights & Biases (2026-11-05)
**Context:** Need training curves and logs for the report (B3, H1).
**Decision:** W&B. The API key lives in Kaggle Secrets. Smoke runs are logged to a separate group.
**Why:** Already familiar to the author; free; curves export easily.
**Status:** accepted

## D-004: Target/draft model candidates (2026-11-05)
**Context:** Need an open-weight decoder that fits a T4, plus a smaller draft model with an identical tokenizer for speculative decoding.
**Options considered:** Qwen2.5-1.5B-Instruct + Qwen2.5-0.5B-Instruct; Llama-3.2-3B + 1B (gated license, 3B is tighter on a T4); Qwen3 family.
**Decision:** Leaning towards Qwen2.5-1.5B-Instruct (target) and Qwen2.5-0.5B-Instruct (draft).
**Why:** Shared tokenizer, Apache-2.0 (verify on the model card), fits a T4 in fp16 with LoRA.
**Status:** proposed (verify licenses, context length and memory before accepting)

## D-005: Dataset loading and leakage guard (2026-11-05)
**Context:** The dataset has 14 columns. Five of them carry previous GPT-4 outputs, or judgments of those outputs, plus the original authors' split flag.
**Decision:** `src/data/load.py` drops `gpt4_explanation`, `gpt4_sentences`, `gpt4_eligibility`, `explanation_correctness` and `training` at load time. A test asserts that no `gpt4*` column survives. `explanation_correctness` is dropped too, because it grades GPT-4's explanation and could leak label information. The built-in `training` flag is ignored in favour of the prescribed split.
**Why:** The assignment forbids GPT-4 predictions as inputs or in evaluation (rubric B2).
**Status:** accepted

## D-006: Prescribed split and observed class imbalance (2026-11-05)
**Context:** 1,015 annotations from 53 patients (all `sigir-*`). The split is sorted IDs → `default_rng(42).shuffle` → 37/8/8.
**Result:** train 37 patients / 603 annotations, val 8 / 206, test 8 / 206. Manifest: `data/splits/split_manifest.json`. Tests cover disjointness, one split per patient, determinism and manifest match.
**Observed issue:** Severe imbalance. "excluded" has 10 examples in total (train 8, val 2, **test 0**). "not included" has 23. "not applicable" (inclusion) has 9.
**Consequences:** Accuracy is dominated by "not excluded". Macro-F1 must be the headline metric. The "excluded" per-class metrics are undefined on test and must be reported as N/A, with no sample made up to fill the gap. Weighting the training loss by class could be considered, and must be tuned only on val.
**Status:** accepted

## D-007: Note parsing and truncation (2026-11-05)
**Context:** Each note line is `"<id>. <sentence>"`. All 1,015 notes parse, sentence IDs are sequential from 0, and all expert evidence IDs exist in their note. Note length is 370–1,447 characters (median 579).
**Decision:** Keep the original sentence IDs from the note text and present the note to the model with those IDs. Truncation is expected to be unnecessary, because the longest note is about 400 tokens. Confirm with the tokenizer when the model is chosen.
**Empty evidence:** 610/1,015 annotations (60%) have no gold evidence (train 349, val 116, test 145). The empty-set rule for the evidence metrics is decided in the Task 2 entry.
**Status:** accepted (token count to be confirmed)

## D-008: Model choice confirmed (2026-11-05) (finalises D-004)
**Evidence (from HF configs and model cards):**
| Model | License | Layers | Hidden | Q heads / KV heads | Context | Vocab |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | apache-2.0 | 24 | 896 | 14 / 2 | 32,768 | 151,936 |
| Qwen2.5-1.5B-Instruct | apache-2.0 | 28 | 1536 | 12 / 2 | 32,768 | 151,936 |
| Qwen2.5-3B-Instruct | **other (Qwen Research)** | 36 | 2048 | 16 / 2 | 32,768 | 151,936 |

The 0.5B and 1.5B tokenizers are identical: same vocab and same special tokens, checked programmatically.
**Decision:** Target = Qwen2.5-1.5B-Instruct, draft = Qwen2.5-0.5B-Instruct. The 3B model is rejected because of its non-Apache license and tighter T4 memory. Llama-3.2 is rejected because it is gated.
**Token lengths (Qwen tokenizer + chat template):** prompts are 227–597 tokens (median 297, p95 455), and targets are 12–38 tokens. With `max_len=1024`, **no example is truncated**. The code raises an error instead of silently truncating.
**Status:** accepted

## D-009: Output schema, and why the model does not generate free-text explanations (2026-11-05)
**Context:** Task 2 asks for an explanation. The dataset has **no expert explanation**; the only explanations are GPT-4's, which are forbidden as inputs or targets.
**Options considered:** (a) train on GPT-4 explanations: forbidden; (b) generate free-text explanations zero-shot: unsupervised, may hallucinate; (c) train on label + evidence IDs only, and build the explanation deterministically from the cited sentences and the label.
**Decision:** The target is `{"evidence_ids": [...], "label": "..."}`, with evidence listed **before** the label so the label is generated conditioned on the cited sentences. The loss covers answer tokens only (prompt tokens masked to -100). Explanations are composed from the extracted evidence text, so every explanation traces back to source sentences (rubric C4).
**Consequences:** Explanations are faithful but templated. Mention this in Limitations.
**Status:** accepted

## D-010: Single GPU on Kaggle T4 x2 (2026-11-05)
**Decision:** Set `CUDA_VISIBLE_DEVICES=0`. Plain `python` with HF Trainer on 2 GPUs would use DataParallel, which is inefficient and error-prone with LoRA + gradient checkpointing. A 1.5B model with LoRA fits on one 16 GB T4.
**Status:** accepted

## D-011: Metric definitions (2026-11-05)
- **Labels:** inclusion and exclusion are evaluated separately, each with its own 4-label set. "Overall" pools them over the union of labels. Unparseable outputs count as the extra class `<invalid>`; they are always wrong and reported as `invalid_rate`.
- **Macro-F1** is averaged over classes present in the gold labels of that subset. Absent classes (e.g. "excluded" on test) are shown as N/A.
- **Evidence:** set-based P/R/F1 over sentence IDs. Per record: both sets empty → P=R=F1=1; one side empty → 0. Macro averages per-record scores; micro pools TP/FP/FN, so both-empty records don't contribute. Cited IDs that are not in the note are dropped and counted (`invalid_ids`).
- Covered by `tests/test_metrics.py`.
**Status:** accepted

## D-012: Smoke-test protocol and first bug caught (2026-11-05)
**Protocol:** `--smoke` trains on 16 examples (always including the longest) for 10 steps, with eval and save every 5 steps. It then reloads the saved adapter, generates on 2 val examples and reports peak GPU memory.
**Bug caught locally:** transformers 5.22 removed `warmup_ratio`. A float `warmup_steps` in [0,1) is now read as a ratio. Fixed before any GPU time was spent.
**Status:** accepted

## D-013: First Kaggle run: no GPU, and dependency pins conflicted (2026-11-05)
**What happened:** The first committed Kaggle version ran on the CPU image (`torch 2.11.0+cpu`) because no accelerator was selected. The GPU check in the notebook failed after about 2 minutes, before any training, which is what it is for. pip also reported conflicts: pinning `numpy==2.5.4`/`pandas==3.1.1` replaced Kaggle's preinstalled builds (numba requires numpy<2.3; google-colab requires pandas==2.2.3).
**Decision:** Pin only the ML stack (`transformers`, `peft`, `accelerate`, `datasets`). Use the platform's `torch`/`numpy`/`pandas`, and record their exact versions in every run's metadata instead. The notebook now checks for a GPU in the kernel and fails with a clear message telling you how to fix it.
**Why:** Overriding the platform's numerical libraries risks binary incompatibilities. Logging the versions keeps reproducibility.
**Status:** accepted

## D-014: torchao conflict, and moving environment setup into the repo (2026-11-05)
**What happened:** The 2nd Kaggle run reached `get_peft_model` and failed with `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`. peft probes torchao whenever it is installed.
**Options considered:** (a) upgrade torchao: it must match the torch build (2.11+cu130), which risks another conflict; (b) pin an older peft: loses fixes and diverges from the locally tested version; (c) uninstall torchao, which this project never uses.
**Decision:** (c). The setup now lives in `scripts/kaggle_setup.sh`, which installs the requirements, uninstalls torchao and runs a tiny LoRA injection as a self-check. The notebook only calls this script.
**Why:** It's the smallest change. Future environment fixes reach Kaggle through `git pull` with no notebook re-import, and the self-check fails within seconds, before any model download.
**Status:** accepted

## D-015: Run 1 results (validation split only) and what they reveal (2026-11-05)
**Setup:** Qwen2.5-1.5B-Instruct, LoRA r=16/α=32 on all attention+MLP projections, lr 2e-4 cosine, 3 epochs, effective batch 16, fp16 AMP on a Tesla T4, commit 29b1596. Greedy decoding, max 64 new tokens, batch 8. Adapter: `huggingface.co/umairishuman/trialspec-qwen2.5-1.5b-lora`.
**Training:** val loss 0.1315 (epoch 1) → **0.1146 (epoch 2, selected)** → 0.1201 (epoch 3; mild overfitting). Train loss 0.29 → 0.05. fp16 was stable, so the D-002 fallback was not needed.

| Val (n=206) | Acc | Macro-F1 | Incl. Macro-F1 | Excl. Macro-F1 | Evid. micro-F1 | Evid. macro-F1 | Parse OK |
|---|---|---|---|---|---|---|---|
| Zero-shot | 0.068 | 0.054 | 0.064 | 0.074 | 0.000 | 0.563 | 100% |
| LoRA run 1 | **0.743** | **0.432** | 0.360 | 0.401 | **0.554** | 0.745 | 100% |

**Findings:**
1. **The baseline is degenerate:** it cites no evidence for any of the 206 examples and predicts "not applicable" 138 times. The prompt names the labels but never *defines* them, so the untuned model can't tell, for example, "excluded" (meets the exclusion criterion) from "not excluded". This is a prompt-design weakness that makes the before/after comparison look lopsided.
2. **The empty-set rule inflates macro evidence F1:** the baseline gets macro-F1 0.563 purely by predicting empty sets, since 116 of 206 gold sets are empty. **Report micro-F1 as the main evidence metric**, and macro alongside it with this caveat.
3. **Fine-tuned minority classes collapse:** F1 is 0 for "not included" (3 examples), "excluded" (2) and inclusion "not applicable" (1). Exclusion "not enough information" recall is only 0.19; 19 of 26 are predicted as the majority class "not excluded". Imbalance (D-006) is the main limitation.
4. 0 parse failures and 0 hallucinated sentence IDs after fine-tuning. Inference takes 36 s for 206 examples on a T4.
**Status:** accepted (results); improvements proposed in the next entry

## D-016: Run 2 design: label definitions + rare-label oversampling (2026-11-05)
**Context:** Run 1 had a degenerate zero-shot baseline and rare-class collapse (D-015).
**Evidence used for the label definitions (train split):** fraction of annotations with *no* gold evidence: exclusion "not excluded" 0.75, "not applicable" 0.83, "not enough information" 0.69, "excluded" 0.00; inclusion "included" 0.06, "not included" 0.23, "not enough information" 0.50, "not applicable" 0.80. So experts label "not excluded" when the excluding condition is *not reported*, and "not applicable" covers criteria about other populations or study procedures (e.g. "MRI images unable to be processed").
**Options considered for imbalance:** (a) class-weighted token loss: awkward with generative targets, since the label is spread over several tokens; (b) oversampling rare (type, label) groups; (c) focal loss: same problem as (a); (d) do nothing and report it.
**Decision:** prompt v2 with definitions that match that convention. Oversample to 60 rows per group, capped at 4×, so a 5-example group isn't repeated 12 times (memorisation risk). Validation is the only selection signal; the test set stays untouched.
**Tension to discuss in the report:** the assignment says absent evidence must never be treated as proof. The expert labels, however, map "condition not reported" to "not excluded". We follow the gold convention for the metrics, and the app will flag such decisions as "no supporting evidence cited" rather than presenting them as confirmed.
**Status:** accepted (results in docs/runs.md, attempt 4)

## D-017: Task 4 attention design (2026-11-05)
**Context:** Rubric E1–E3 asks for a correct reference SDPA, a tiled online-softmax variant that doesn't call a fused backend, multi-shape causal tests with max/mean error, and an honest speed/memory comparison.
**Decisions:**
- **Causal convention with an offset:** queries are the *last* Lq positions of the keys (offset = Lk − Lq), so query i sees keys j ≤ i + offset. One code path covers prefill (Lq = Lk), single-token cached decode (Lq = 1) and chunked prefill. This is the same masking the Task 3 KV-cached decoder needs.
- **Accumulation dtype:** fp32 for fp16/fp32 inputs, and fp64 for fp64 inputs, so tests can demand 1e-10 agreement. The output is cast back to the input dtype, as fused kernels do.
- **Online softmax:** per query block, keep a running max m, normaliser l and unnormalised accumulator; rescale by exp(m_old − m_new) whenever the max increases. NaN is impossible because every row sees key 0 in its first tile, so m is finite from the first tile on.
- **Causal tile skipping:** stop the key loop at the last visible key of the query block, and mask only tiles that cross the diagonal.
- **Ground truth:** benchmark errors are computed against an fp64 reference, so reference, tiled and SDPA are compared against the same truth.
**Options rejected:** writing a Triton/CUDA kernel (not required by the assignment; not portable to CPU and P100); measuring only against SDPA (SDPA's own fp16 error would be hidden).
**Status:** accepted

## D-018: Select checkpoints by validation macro-F1, not validation loss (2026-11-06)
**Context:** Run 2 (prompt v2 + oversampling) scored below run 1 on every metric (val macro-F1 0.335 vs 0.432). Its exclusion predictions collapsed to "not excluded" for all 140 cases. `load_best_model_at_end` with `metric_for_best_model=eval_loss` picked epoch 1, because val loss rose afterwards (0.122 → 0.131 → 0.137) as the duplicated rare examples were memorised.
**Analysis:** loss is averaged over all answer tokens. In `{"evidence_ids": [...], "label": "..."}` the label is only 2–5 of 12–38 tokens, so loss mostly measures the JSON wrapper and the evidence IDs. Loss and the reported metric (macro-F1 over labels) can disagree, as they did in run 2.
**Options considered:** (a) keep loss-based selection: reproduces run 2's failure; (b) a Trainer callback that generates on val each epoch: complex, and holds a second copy of state in GPU memory during training; (c) keep every epoch checkpoint and evaluate each *after* training with the same `predict.py` used for reporting, then choose the best val macro-F1; (d) a loss restricted to label tokens: still a proxy.
**Decision:** (c). Selection rule: maximum val macro-F1; ties go to higher accuracy, then the earlier step. Implemented in `src/eval/select_checkpoint.py` (unit-tested), turned on with `select_by: val_macro_f1`. All checkpoints' val metrics are kept, so the loss-vs-F1 disagreement can be shown in the report. The test split is never used.
**Run 3 design:** prompt v2 *without* oversampling, removing run 2's confound so run 3 vs run 1 isolates the prompt. Oversampling stays off for now; if run 3 still shows rare-class collapse, a later run can re-test oversampling with F1 selection.
**Consequences:** 3 × ~36 s extra evaluation per run, and about 3 × 150 MB of checkpoints on Kaggle disk (fine).
**Status:** accepted (results in docs/runs.md, attempt 6)