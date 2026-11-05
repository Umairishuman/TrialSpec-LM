# Decision Log

Architectural and technical decisions, with alternatives and reasoning. Newest at the bottom.

---

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
