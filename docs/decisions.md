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
| 2026-11-06 | S5 | Fixed `results/smoke` committed as a file (copy into a missing folder); ignore `results-*.zip` | Noticed in `git status` | — | fea194a |
| 2026-11-06 | S7 | Task 3: hand-written Qwen2 decoder with explicit `KVCache` (`src/decoding/manual_decoder.py`), three explicit greedy loops (`src/decoding/generate.py`), 15 tests (tiny random Qwen2 in fp64 + gated real-model test), benchmark with mismatch diagnosis (`src/decoding/benchmark_kv.py`) | Task 3 requirements (D1–D3) | D-019 | 0d8f427 |
| 2026-11-06 | S7/S8 | `scripts/kaggle_benchmarks.sh` (smoke first, then attention fp16/fp32 and KV benchmarks); notebook runs it with `CONFIG = 'benchmarks'` | GPU numbers for Tasks 3–5 in one Kaggle run | D-019 | 0d8f427 |
| 2026-11-06 | S7/S9 | Tiny test models now use untied embeddings; noisy draft calibrated (20% of each tensor's std) + partial-acceptance test | Tied random models just copy the last token, making equality tests near-trivial | D-021 | 3a2e7a6 |
| 2026-11-06 | S4/S6 | **Explicit decoding config** (`greedy_config`): `repetition_penalty=1.0` by default, `--repetition-penalty` flag, full settings in metrics metadata. Runs 1–3 used Qwen's implicit 1.1 | Qwen2.5 `generation_config.json` adds `repetition_penalty=1.1` to every `generate()` call | D-020 | 3a2e7a6 |
| 2026-11-06 | S9 | Task 5: `src/decoding/speculative.py` (greedy speculative decoding, explicit cache rollback), 40 tests, `src/decoding/benchmark_spec.py` (task + free-text prompt sets, k ∈ {2,4,6,8}, HF assisted reference); added to `kaggle_benchmarks.sh` | Task 5 requirements (F1–F3) | D-021 | 3a2e7a6 |
| 2026-11-06 | S5 | Run 3 results recorded: prompt v2 still collapses exclusion (macro-F1 0.379 < run 1's 0.432); F1 selection chose epoch 3 over the loss-optimal epoch 2. **Run 1 remains the best model** | Run 3 analysis | D-022 | 1144c1c |
| 2026-11-06 | S5 | Pipeline: per-config `eval_repetition_penalty` (default 1.0), `run_baseline`, and `select_by` read from YAML; baseline filename includes the penalty; `final_eval` entry point reserved | Comparable ablation decoding; D-020 | D-022 | 1144c1c |
| 2026-11-06 | S5 | `configs/run4.yaml`: prompt v1 + oversampling + F1 selection, eval at repetition_penalty 1.1 | Complete the 2×2 ablation | D-022 | 1144c1c |
| 2026-11-06 | S7/S8/S9 | GPU benchmark results for Tasks 3–5 recorded (`results/{attention,kv_cache,speculative}/*cuda*.json`, run log attempt 9): KV cache 9.3× decode speed-up at 1024+256; 2/27 token mismatches diagnosed as exact fp16 logit ties; tiled attention exact but launch-bound; speculative decoding exact but 0.61–0.80× | Kaggle benchmark run | D-023 | a82c600 |
| 2026-11-10 | S6 | Final-evaluation stage: `src/eval/final_eval.py`, `configs/final_eval.yaml`, `scripts/kaggle_final_eval.sh` (notebook `CONFIG = 'final_eval'`). Val for all models at pure greedy → select by val macro-F1 → test once, guarded by `results/final/test_lock.json` | Task 2 (C1–C4), B4 before/after at identical decoding; D-020 promised a pure-greedy re-evaluation | D-024 | dc1dfb5 |
| 2026-11-10 | S6 | Every prediction record now carries `pred_evidence_text` and a deterministic evidence-grounded `explanation` (`src/eval/explain.py`); metrics count hallucinated sentence IDs (`n_invalid_ids`) | C1 output fields, C3 validity check | D-024, D-009 | dc1dfb5 |
| 2026-11-10 | S6 | Error categorisation and case files (`src/eval/error_analysis.py`), confusion-matrix figures (`src/eval/plots.py`); `matplotlib` added to requirements | C4 (≥8 analysed errors), report figures | D-024 | dc1dfb5 |
| 2026-11-10 | S6 | `compare.row` shows N/A for a missing inclusion/exclusion block | New orchestrator test found a crash: the 4-example smoke sample contains no exclusion criteria | D-024 | dc1dfb5 |
| 2026-11-10 | S5/S11 | `src/train/export_wandb.py`: pulls every non-smoke W&B run's history to `results/training/wandb_export/*.csv` (+ `runs.json` with run URLs and configs) and draws `training_curves.png` (train/val loss, LR, grad norm). Runs 1–3 exported; run 4 to be added when it finishes | Report needs training curves (B3, H1); user provided W&B access | D-025 | 520691a |
| 2026-11-10 | S5 | `kaggle_pipeline.sh`: `cfg` helper uses `${2:-}` instead of `$2` | Run 4 launch failed after ~1 min: `$2: unbound variable` under `set -u` (runs.md attempt 10) | D-022 | b8a687a |
| 2026-11-10 | S10 | App hosting: run locally on the laptop (CPU, bf16, target 1.5B + draft 0.5B); optional HF Space only as a public backup. Reverses the 2026-11-05 "HF Space" plan | Measured: laptop ran both models in bf16 (runs.md attempt 8, 5.4 tok/s); free Space has 2 shared vCPUs | D-026 | e8197a1 |
| 2026-11-10 | S5 | **Run 4 results recorded** (runs.md attempt 11): val macro-F1 0.604 vs run 1's 0.432 at rp 1.1; run 4 files copied from `results-v4.zip` into `results/` | Run 4 finished | D-027 | 1b45c79 |
| 2026-11-10 | S5/S11 | W&B export re-run: run 4 history added, `training_curves.png` redrawn with 4 runs | D-025 said to add run 4 when it finished | D-025, D-027 | 1b45c79 |

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
## D-019: Task 3 KV-cache design (2026-11-06)
**Context:** Rubric D1 requires model forward passes with explicit cache reuse and update, causal masking and positional handling, "not only `use_cache=True`". D2 requires greedy-token equivalence with an uncached reference over at least 3 input and output lengths. D3 requires prefill/decode/tokens-per-second/peak-memory measurements with warm-up and synchronised repeats.
**Options considered:** (a) `model.generate(use_cache=True/False)`: explicitly insufficient; (b) an explicit loop over the HF forward with a `DynamicCache`: shows the decoding flow, but the cache internals stay inside HF; (c) a hand-written decoder forward with our own cache tensors.
**Decision:** implement both (b) and (c), and check both against the uncached reference.
- **Positions:** RoPE is applied to each new key at its *absolute* position (`cache.length + t`) before it is written, so cached keys are never re-rotated. In (b) the same thing is done by passing `position_ids = [[cache_len]]` explicitly.
- **Masking:** prefill uses the causal mask. A single decode query may see every cached key, so no mask entries are needed. This is the Task 4 offset convention (offset = Lk − Lq), so the same attention function serves both phases. In (b) the attention mask covers past + new tokens.
- **Cache growth:** preallocated `[B, n_kv, capacity, head_dim]` per layer for K and V; `write` fills `[length, length+T)`, `advance` commits, `crop` rolls back (needed for Task 5). Memory = 2 × layers × kv_heads × head_dim × bytes per token, so Qwen2.5-1.5B in fp16 uses 2·28·2·128·2 = **28 KiB/token**. GQA cuts this 6× (2 KV heads instead of 12).
- **Numerical differences:** in fp16 on GPU, a cached decode step computes 1×L attention while the uncached path computes the last row of an L×L matrix with different kernels and reduction orders. Logits can differ in the last bits, and a greedy argmax can flip when the top-2 logits nearly tie. The benchmark records the reference top-1/top-2 margin at the first mismatch to tell a near-tie from a bug.
- **Tests:** a tiny random Qwen2 (3 layers, GQA 4/2) in fp64 runs in seconds with no download, so it runs on Kaggle and in CI. A real-model test is gated by `RUN_MODEL_TESTS=1`.
**Limitations:** batch size 1 (no padding) for the manual decoder; RoPE implemented for `rope_type=default` only (true for Qwen2.5), with an assertion guarding it.
**Status:** accepted (CPU verification: docs/runs.md attempt 7; GPU benchmark pending)
## D-020: Decoding settings must be explicit (Qwen's hidden repetition penalty) (2026-11-06)
**Context:** While cross-checking Task 5 against HF assisted generation, outputs differed on free-text prompts. Qwen2.5-Instruct ships `generation_config.json` with `do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1`. `generate(do_sample=False)` switches off sampling, but `repetition_penalty=1.1` **still rescales logits**.
**Impact:** the val predictions of runs 1, 2 and 3 (`src/eval/predict.py`) were greedy **with repetition_penalty 1.1**, not pure greedy as described. The comparisons among those runs remain valid (same settings for every model, baseline included), but rubric B4 requires the decoding settings to be documented exactly.
**Options considered:** (a) keep 1.1 everywhere: then Task 3/5 equivalence tests against our manual loops (pure argmax) wouldn't be comparable; (b) pure greedy everywhere, with the earlier runs documented as 1.1; (c) report both.
**Decision:** (b). `greedy_config()` builds an explicit `GenerationConfig` (`do_sample=False, num_beams=1, repetition_penalty=1.0`, sampling fields cleared, EOS = `<|im_end|>`/`<|endoftext|>`), used by `predict.py` and the Task 5 HF reference. `--repetition-penalty 1.1` reproduces runs 1–3. All decoding fields are stored in each metrics file. The **final val and test evaluation will be re-run with pure greedy for every compared model** (zero-shot and fine-tuned), so the before/after comparison uses identical, fully specified settings.
**Lesson:** never rely on a model's shipped generation defaults; pass the full config.
**Status:** accepted

## D-021: Task 5 speculative decoding design (2026-11-06)
**Draft/target pair (F1):** target = Qwen2.5-1.5B-Instruct + our LoRA (merged); draft = Qwen2.5-0.5B-Instruct (base). Same tokenizer: identical vocab of 151,936 and identical special tokens, asserted at benchmark start (D-008). The draft has 3× fewer parameters (494M vs 1.54B), so fp16 memory is about 1.0 GB + 3.1 GB, which fits a T4 together, plus two KV caches (draft 12 KiB/token, target 28 KiB/token). The draft is *not* fine-tuned, so acceptance on our JSON answers may be lower than on free text. Both sets are measured. Fine-tuning the draft is a possible follow-up.
**Algorithm (F2):** see `src/decoding/speculative.py`. Invariant: both caches hold all committed tokens except the newest (`last`). Each round: the draft proposes k tokens one at a time; the target verifies `[last, d1..dk]` in one forward pass; accept the longest prefix where the draft token equals the target's argmax; append the target's token at the first mismatch (or a bonus token if all k matched); crop the target cache to `base+1+n`. For the draft, crop to `base+1+n`, or, when n == k, feed `d_k`, which the draft never processed. EOS and the length budget are applied while committing.
**Correctness evidence (F3):** 40 tests on tiny fp64 models: k ∈ {1,2,4,7} × input {1,9,30} × output {1,6,25}, equal to target greedy; a noisy draft with 11–54% acceptance; an identical draft with 100% acceptance and exactly 1 + ⌈20/(k+1)⌉ target passes; an unrelated draft; EOS handling. On real models, bf16 CPU: identical tokens at k=2 and 4. The HF `assistant_model=` output matches too, once pure greedy is enforced (D-020).
**Test-design lesson:** tied-embedding random models copy the input token, so draft and target agree trivially. Tests now use untied embeddings and assert partial acceptance.
**Expected performance:** a speedup is not guaranteed. On CPU (compute-bound) it measured 0.7× despite 50–58% acceptance. On GPU, decoding is memory-bound, so k+1-token verification costs about the same as a 1-token step; the benchmark measures k ∈ {2,4,6,8}, latency, tokens/s, acceptance and peak memory.
**Status:** accepted (GPU benchmark pending)
## D-022: Prompt v2 rejected for fine-tuning; run 1 stays the best model; 2×2 ablation (2026-11-06)
**Evidence (val, all at repetition_penalty 1.1):** run 1 (v1, no oversampling) macro-F1 **0.432**; run 2 (v2, oversampling, loss-selected) 0.335; run 3 (v2, no oversampling, F1-selected) 0.379. Exclusion macro-F1: 0.401 / 0.210 / 0.252. In run 3, 137 of 140 exclusion criteria were predicted "not excluded" at the selected checkpoint.
**Interpretation:** the v2 definition of "not excluded" ("…or gives no sign of it") encodes the expert majority convention but erases the boundary with "not enough information", the other silent-note label. It improves zero-shot behaviour (accuracy 0.068 → 0.175) and slightly improves inclusion after fine-tuning (0.360 → 0.374), but hurts exclusion.
**Decision:** prompt v1 remains the production prompt, and **run 1's adapter (`umairishuman/trialspec-qwen2.5-1.5b-lora`) is the current best model**. Prompt v2 is kept in the code as a documented negative result. Run 4 (v1 + oversampling + F1 selection) completes a 2×2 ablation. If run 4 beats run 1 on val macro-F1 it becomes the final model; otherwise run 1 does. The choice is made **only on validation**; the test split is used once afterwards.
**Rejected for now:** a v3 prompt that rewrites the "not excluded" definition. It would mean tuning prompt wording on 8 validation patients (overfitting risk), and there's no time for another round of iteration before the deadline.
**Status:** accepted
## D-023: Interpretation of the GPU benchmarks; no further optimisation (2026-11-06)
**Context:** the T4 benchmarks (run log attempt 9) show the cache and tiled-attention mechanisms working as intended. Speculative decoding is exact but slower.
**Finding (shared root cause):** at batch size 1 the T4 is **kernel-launch-bound**, not compute- or bandwidth-bound. A cached 1.5B decode step takes about 37 ms, roughly 4× the weight-read time (3 GB / 320 GB/s ≈ 9 ms). This one effect explains (i) the Python tiled attention being 70–135× slower than fused SDPA (thousands of small kernels), and (ii) the speculative slowdown: the 24-layer draft costs about 0.86 of a 28-layer target step. Leviathan et al.'s speed-up formula with the measured acceptance predicts 0.77× at k=4, and we measured 0.74×.
**Options considered:** (a) CUDA graphs / `torch.compile(mode="reduce-overhead")` for the decoders; (b) a smaller, shallower draft (would need training a new draft model); (c) report as is, with the quantitative explanation.
**Decision:** (c). The rubric explicitly allows a slower speculative decoder when correctness and measurements are documented (grading rule iv). (a) and (b) carry real risk with 4 days left and are listed in the report as future work.
**Mismatch policy (Task 3):** fp16 greedy mismatches are accepted only when the reference top-1/top-2 margin at the first differing step is ~0 (here exactly 0.000). Any mismatch with a clear margin would be treated as a bug.
**Status:** accepted
## D-024: Final evaluation protocol (Task 2): val-only model selection at pure greedy, test exactly once (2026-11-10)
**Context:** Rubric C1–C4 needs per-prediction output (label, cited IDs, extracted evidence text, explanation), classification metrics overall/inclusion/exclusion, evidence P/R/F1 micro+macro, a hallucinated-ID check, and ≥8 analysed errors. B4 needs before/after on the *same* examples with *identical, documented* decoding. All val numbers so far (runs 1–4) used Qwen's implicit repetition_penalty 1.1 (D-020), and the test split is still untouched.
**Options considered:**
1. Pick the final model from the existing rp 1.1 val numbers, then evaluate only that model at rp 1.0. Cheap, but the selection setting would differ from the reported setting.
2. Re-run val at pure greedy (rp 1.0) for every candidate in one Kaggle run, select on those numbers, then run test once for the selected model and its zero-shot counterpart. About 10 extra minutes of GPU time.
3. Run test for every candidate and report all. Rejected: it turns the test split into a selection signal ("never tune on test").
**Decision:** option 2, implemented as `src/eval/final_eval.py` + `configs/final_eval.yaml` + `scripts/kaggle_final_eval.sh` (notebook `CONFIG = 'final_eval'`).
- **Stage A (val, rp 1.0):** zero-shot with prompts v1 and v2, and the selected adapters of runs 1–4.
- **Selection rule (fixed before seeing any rp 1.0 numbers):** highest val macro-F1 among the fine-tuned candidates; ties → higher val accuracy → earlier run in the config order. The zero-shot baseline reported against it is the one with the **same prompt version**.
- **Stage B (test, rp 1.0, once):** the selected model and its zero-shot baseline only.
- **Test-once guard:** after Stage B the script writes `results/final/test_lock.json` (commit, UTC time, selected model). Once that file is committed, the script refuses to touch test again unless `--allow-test-rerun` is passed, and any such re-run must get its own decision entry.
- **Smoke mode:** 4 val examples for every model (so every adapter download is checked) and no test access at all.
- **Per-prediction output (C1):** each record also stores `pred_evidence_text` (looked up from the original sentence IDs) and a deterministic `explanation` built from the label and the cited sentences (D-009). When nothing is cited, the explanation says so explicitly ("no sentence cited; label reflects absence of reported information"), which addresses the absent-evidence tension in D-016.
- **Artifacts:** metrics JSON + predictions JSONL + CSV per (model, split), confusion-matrix PNGs (inclusion and exclusion separately), `summary.md` (before/after tables, per-class tables), and `error_analysis_<split>.md`.
- **Error analysis (C4):** errors are categorised automatically: *polarity flip* (included ↔ not included, excluded ↔ not excluded), *missed information* (gold NEI, predicted definite), *spurious NEI* (gold definite, predicted NEI), *applicability* (not applicable confused with anything), *unsupported evidence* (correct label, but no overlap with non-empty gold evidence, or citing sentences when gold cites none), *hallucinated IDs*, *parse failure*. At least 8 are sampled round-robin across categories (deterministic, ordered by annotation_id), with the cited and gold sentences printed. The "why it went wrong" text is written by hand afterwards, from those printed cases, for the report.
**Why:** selection and reporting then use one fully specified decoding setting, and test is used exactly once, by construction. Re-running val for runs 1–4 also gives the 2×2 ablation table at pure greedy, as D-020 promised.
**Consequences/risks:** the rp 1.0 ranking could differ from the rp 1.1 ranking in D-022; whatever happens, the rp 1.0 choice is final. Runs 1 and 2 were loss-selected, runs 3 and 4 F1-selected (stated in the report). If a hub adapter is missing (e.g. run 4's push failed), the smoke stage fails within minutes.
**Status:** accepted
## D-025: Training curves exported by script from the W&B API (2026-11-10)
**Context:** the report needs training curves (B3) and run links. Exporting PNGs panel by panel from the W&B web UI is manual, gives inconsistent styling and can't be re-run when run 4 finishes.
**Options considered:** (a) manual PNG/CSV export from the UI; (b) a script using `wandb.Api()` that saves each run's full history as CSV and draws one consistent figure.
**Decision:** (b), `python -m src.train.export_wandb`. Smoke runs (group `smoke`) are skipped. The API key is read from `WANDB_API_KEY` in the environment only, never written to code or files. `runs.json` stores run IDs, URLs and configs; Trainer already masks the hub token as `<HUB_TOKEN>` (checked).
**Why:** reproducible and raw data committed, so the figures can be redrawn without W&B access. The first export (runs 1–3) visibly confirms D-018: run 2's val loss is lowest at epoch 1 and rises afterwards, while runs 1 and 3 bottom out at epoch 2.
**Consequences/risks:** W&B run names differ from config run names for run 1 (`qwen2.5-1.5b-lora`); a name map gives the figure labels. Runs 1 and 3 have identical LR schedules, so their lines overlap in the LR panel.
**Status:** accepted
## D-026: Task 6 app runs locally on the laptop; HF Space is only a backup (2026-11-10)
**Context:** the 2026-11-05 changelog planned a free HF Space for the app because the laptop "can't fit 1.5B + 0.5B". That conclusion came from the failed fp32 *training* run (runs.md attempt 0: weights + gradients + optimizer state + activations). Inference needs only weights.
**Evidence:** laptop = Intel i5-1145G7 (4 cores / 8 threads), 15.7 GB RAM. Runs.md attempt 8 already ran target 1.5B + draft 0.5B together in bf16 on this laptop: target greedy 40 tokens in 7.4 s (about 5.4 tok/s), speculative output identical. A free CPU Space has 2 shared vCPUs (estimated 2–3× slower), sleeps when idle and re-downloads models on wake-up, and depends on the network during the demo.
**Options considered:** (a) free HF Space as the main host; (b) laptop with the target only, speculative disabled; (c) laptop with target + draft in bf16, all three decoding modes live.
**Decision:** (c). Launched with one command (`python -m src.app`, Gradio on `http://localhost:7860`). The same code can be pushed to a Space afterwards as a public backup link, not as the demo host.
**Consequences/risks:** about 5–8 s per criterion on CPU (estimate: ~350-token prompt + ~20 output tokens), so the app processes criteria sequentially with a progress bar and limits input length. Models must be downloaded before the demo (offline-safe afterwards). Supersedes the 2026-11-05 hosting note (changelog row "Planned app hosting changed from laptop to free HF Space").
**Status:** accepted
## D-027: Run 4 results: oversampling works with prompt v1; macro-F1 gain is partly a 2-example effect (2026-11-10)
**Context:** run 4 (prompt v1 + oversampling to 60/group, max 4×, F1-based checkpoint selection) completes the 2×2 ablation planned in D-022 (runs.md attempt 11).
**Evidence (val, n=206, repetition_penalty 1.1, comparable with runs 1–3):** selected epoch 3: accuracy 0.743 (= run 1), macro-F1 **0.604** (run 1: 0.432), inclusion 0.368 (0.360), exclusion **0.676** (0.401), evidence micro-F1 0.520 (0.554). Per class: exclusion "excluded" F1 0 → 1.0 (**support 2**), exclusion "not enough information" 0.294 → 0.410 (support 26); the other classes moved by ≤ 0.03. Ablation (val macro-F1): v1 0.432 · v1+oversample 0.604 · v2 0.379 · v2+oversample 0.335.
**Interpretation:**
1. Oversampling helps only when the prompt keeps the "not excluded" vs "not enough information" boundary (v1). With v2 the exclusion collapse dominates (D-022).
2. The size of the macro-F1 gain is fragile. The "excluded" class has 2 val examples, so getting both right adds about 0.12 to an 8-class macro average. The robust part of the gain is exclusion NEI recall, so the report must show per-class F1 with supports next to the macro-F1, and must not present 0.604 as a 40% improvement without that caveat.
3. Val loss is again lowest at epoch 1 (0.121) and highest at the selected epoch 3 (0.135). This is a second independent confirmation of D-018: loss-based selection would have picked a worse model (macro-F1 0.589).
**Decision:** no new model choice is made here. Under D-022's rule run 4 would replace run 1, but the binding choice is the pure-greedy val re-scoring in the final evaluation (D-024, runs.md attempt 12), which is already running. The app (Task 6) takes its adapter from config, so whichever model `results/final/selection.json` names can be used without code changes.
**Consequences/risks:** with 8 val patients and supports of 1–3 for four classes, rankings between runs are noisy. This is stated as a limitation, and is not fixed by more tuning on val (overfitting risk; no time).
**Status:** accepted
