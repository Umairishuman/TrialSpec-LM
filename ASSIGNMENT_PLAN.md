# GenAI Assignment #2 — Step-by-Step Plan

**Project:** Transformer-based clinical-trial criterion matcher (patient note + criterion → label + evidence sentence IDs + explanation)
**Deadline:** November 10, 2026 (2-hour grace period; after that you get ZERO). **Demo is mandatory.** If you miss it, you get no marks.
**Rubric total:** 135 marks (sections A–H)
**Compute:** Kaggle (free). This is allowed. The PDF says any free environment works, and nothing may *require* paid GPUs or APIs.

---

## 0. Reality check: time budget (5 days)

| Day | Date | Goal |
|---|---|---|
| 1 | Nov 5 | Repo skeleton, data loading + split + manifest, model choice, zero-shot baseline, Kaggle smoke test of training |
| 2 | Nov 6 | Full fine-tune run on Kaggle (runs on its own while you build Tasks 3 & 4 locally/CPU) |
| 3 | Nov 7 | Task 2 evaluation on val/test, Task 5 speculative decoding, benchmarks |
| 4 | Nov 8 | Task 6 web app (Gradio/FastAPI), integrate everything, record numbers |
| 5 | Nov 9 | IEEE LaTeX report, demo video, README, upload adapter. Submit by Nov 9 night, not Nov 10 |

Tasks 3, 4 and 5 **don't depend on the fine-tune**. Build them while training runs.

---

## Step 1: Project setup (supports H2 and B3)

**What to do**
1. Create this repo layout:
   ```
   configs/            # YAML configs for training/eval (no magic numbers in code)
   data/splits/        # split_manifest.json (committed)
   src/
     data/             # load, split, format prompts
     train/            # fine-tuning script
     eval/             # metrics, confusion matrices, error analysis
     kv_cache/         # Task 3
     attention/        # Task 4
     speculative/      # Task 5
     app/              # Task 6
   notebooks/          # Kaggle notebooks (thin wrappers that call src/)
   tests/              # pytest: split leakage, attention equality, cache equality
   results/            # JSON/CSV logs, figures (committed)
   report/             # LaTeX IEEE paper
   ```
2. Add `requirements.txt` with **pinned versions** (`torch`, `transformers`, `peft`, `datasets`, `accelerate`, `bitsandbytes`, `wandb`, `scikit-learn`, `gradio`).
3. Add a `.gitignore` covering checkpoints, `wandb/`, `.env` and `*.safetensors`.
4. Kaggle workflow: push the repo to GitHub, then run `git clone` in the Kaggle notebook and call `python -m src.train.run --config configs/...`. The notebook only *calls* the code. This counts toward "not a notebook-only demonstration".
5. Put your W&B API key in **Kaggle Secrets** (Add-ons → Secrets). Never put it in the code.

---

## Step 2: Dataset and fixed split (B1, B2: 10 marks)

**Dataset:** `ncbi/TrialGPT-Criterion-Annotations` (Hugging Face). Load it and print the columns and label values first. Don't assume them.

**What to do**
1. Load the dataset and inspect the columns. Expect roughly: `patient_id`, `note` (sentences prefixed with IDs), `trial_id`, `criterion_type`, `criterion_text`, `expert_eligibility`, `expert_sentences`, plus **`gpt4_*` columns** (predictions and explanations).
2. **Drop every `gpt4_*` column immediately** in the loader. Add an assert that no `gpt4` key reaches the prompt builder. This is B2 and the anti-leakage rule. Ignore any built-in `training` column too, because you must use the prescribed split.
3. Build the split exactly as specified:
   ```python
   ids = np.array(sorted(df.patient_id.unique()))
   rng = np.random.default_rng(42)
   rng.shuffle(ids)
   train, val, test = ids[:37], ids[37:45], ids[45:53]
   ```
   Assert that there are 53 unique patients. If the count differs, stop and investigate.
4. Save `data/splits/split_manifest.json` with the patient IDs and annotation counts for each split, plus label distributions.
5. Write `tests/test_split.py`. It should check that the split sets are disjoint, that every annotation lands in exactly one split, that the counts match the manifest, and that re-running gives an identical split.
6. Parse the sentence IDs out of the note so you keep the **original IDs**. C3 requires the original sentence IDs.
7. Record the label sets for inclusion criteria (e.g. included / not included / not enough information / not applicable) and for exclusion criteria (excluded / not excluded / not enough information / not applicable). **Don't collapse them.**

**Deliverables:** manifest JSON, split test, a label-distribution table for the report.

---

## Step 3: Research and model selection (Task 1, A1–A3: 15 marks)

**What to do**
1. Read and cite **at least 5 sources**. Strong picks:
   - TrialGPT (Jin et al., Nature Communications 2024): the task source
   - Attention Is All You Need (Vaswani 2017)
   - LoRA (Hu 2021) and QLoRA (Dettmers 2023)
   - FlashAttention (Dao 2022) and the online-softmax paper (Milakov & Gimelshein 2018)
   - Speculative decoding (Leviathan 2023; Chen 2023)
   - Model tech report for your chosen model (e.g. Qwen2.5 / Qwen3)
2. Compare **at least 2 decoder candidates** in a table with these columns: params, layers, heads (GQA?), context window, tokenizer/vocab, license, fp16 memory, and LoRA training memory on a T4.
   - **Suggested target:** `Qwen2.5-1.5B-Instruct` (Apache-2.0). It fits a Kaggle T4 in fp16 with LoRA.
   - **Suggested draft (Task 5):** `Qwen2.5-0.5B-Instruct`. It uses the **same tokenizer**, which Task 5 needs.
   - **Alternative candidate:** Llama-3.2-3B / 1B (gated license), or the Qwen3 family. Document why you rejected it.
   - Check the exact license on each model card. For example, Qwen2.5-**3B** uses a different license than 1.5B.
3. Write a "decisions log" (`docs/decisions.md`) as you go. Record what you tried, what failed, and why you switched. A3 grades this directly.

---

## Step 4: Zero-shot baseline, "before" model (B4)

**What to do**
1. Define the **output schema** (JSON):
   ```json
   {"label": "not enough information", "evidence_ids": [3, 7], "explanation": "..."}
   ```
2. Write a prompt template that takes the patient note (with sentence IDs), the criterion type and the criterion text.
3. Run the base model on **validation** with greedy decoding. Use the same settings you'll use after fine-tuning.
4. Track the **JSON parse-failure rate** too. It's a real metric.
5. Save the predictions to `results/baseline_val.jsonl`. Run the test set *only at the very end*, once.

---

## Step 5: Fine-tuning (Task 1, B3: 5 marks + feeds everything)

**What to do**
1. Tokenize prompt + target. Compute the **loss only on the answer tokens** by masking prompt labels to -100, and document this.
2. **Truncation:** measure the token-length distribution of the notes. If you truncate, log how many examples lost a gold-evidence sentence. The PDF explicitly asks for this.
3. Use LoRA (or QLoRA) with r=16, alpha=32, dropout 0.05 on the attention + MLP projections. Use LR around 2e-4, 2–3 epochs, gradient accumulation and **fp16** (the T4 has no bf16).
4. Log to W&B: train/val loss, LR and grad norm. Save the run URL for the report (training curves).
5. Evaluate on validation each epoch and keep the best checkpoint.
6. Push the adapter to the **Hugging Face Hub** (free). This gives you the "weights download link" requirement.
7. Save the exact config YAML + git commit hash + seed alongside the checkpoint.

---

## Step 6: Eligibility and evidence evaluation (Task 2, C1–C4: 20 marks)

**What to do**
1. The output for each prediction is: criterion, predicted label, cited sentence IDs, **extracted evidence text** (looked up from the IDs) and an explanation.
2. Use scikit-learn for the classification metrics: accuracy, macro-F1, per-class P/R/F1 and confusion matrices. Report them **overall, inclusion-only and exclusion-only**.
3. Compute evidence metrics as set-based P/R/F1 over sentence IDs, both **micro and macro**. State the empty-set rule clearly. For example, gold empty and predicted empty counts as a match (P=R=1). Gold empty with something predicted gives P=0. Report how many records have empty gold.
4. Do a validity check: reject cited IDs that don't exist in the note (hallucinated IDs). Count them.
5. Error analysis needs **at least 8 distinct errors**. Include wrong labels, unsupported explanations, missing-information cases and contradictory cases. Show the note snippet, the gold, the prediction and why it went wrong.
6. Produce before-vs-after tables on the same val/test examples (B4).

---

## Step 7: KV-cached generation (Task 3, D1–D3: 15 marks)

**What to do**
1. **Reference path:** for each new token, run the **full sequence** through `model(input_ids)` with no cache, take the argmax and append.
2. **Cached path:** write your own loop, not `generate()`. First, prefill runs the prompt once and returns `past_key_values`. Then each decode step feeds **only the last token** plus the past cache, together with the correct `position_ids` and `attention_mask`, and updates the cache.
3. Explain in the report:
   - **Positions:** RoPE position = cache length + step.
   - **Masking:** causal masking is implicit for 1 query token against all past keys.
   - **Cache growth:** the memory formula is 2 × layers × kv_heads × head_dim × seq × bytes. Mention how GQA shrinks it.
   - **Numerical differences:** fp16 reduction-order differences can flip an argmax.
4. Test token equality at **at least 3 input lengths × 3 output lengths**. If tokens mismatch, show where and why.
5. Benchmark prefill time, decode time, tokens/s and peak memory. Use warm-up runs, call `torch.cuda.synchronize()`, repeat at least 5 times and report mean ± std. Use `torch.cuda.reset_peak_memory_stats()`.

---

## Step 8: Tiled attention with online softmax (Task 4, E1–E3: 15 marks)

**What to do**
1. **Reference SDPA:** `softmax(QKᵀ/√d + mask) V`, with max-subtraction for numerical stability.
2. **Tiled version:** loop over Q blocks and K/V blocks. Keep a running max `m`, a running normalizer `l` and an accumulator `O`, and rescale with `exp(m_old − m_new)`. Apply the causal mask per tile, and skip fully-masked tiles.
3. Test across shapes (batch, heads, seq 1…2048, d 32/64/128), block sizes, causal and non-causal masks, and fp32 and fp16. Report **max and mean absolute error** against the reference and against `torch.nn.functional.scaled_dot_product_attention`.
4. Benchmark speed and memory for all three. Explain honestly why your Python loop is slower than the fused kernel: kernel launch overhead, no SRAM control and Python loops.
5. Optional bonus: plug it into the model's attention to show it gives the same logits.
6. **These can run on CPU locally**, so no Kaggle quota is needed except for the GPU benchmarks.

---

## Step 9: Speculative decoding (Task 5, F1–F3: 15 marks)

**What to do**
1. Use the draft model (e.g. Qwen2.5-0.5B) and the target (your fine-tuned 1.5B). **Verify that the tokenizers are identical** (same vocab, same special tokens) and include that check in the code.
2. The algorithm, done by hand:
   1. The draft proposes `k` tokens greedily, using its own KV cache.
   2. The target runs **one forward pass** over those `k` tokens.
   3. Accept the longest prefix where the target's argmax equals the draft token.
   4. At the first mismatch, take the target's token. If all `k` tokens are accepted, take the target's bonus token.
   5. **Roll back both caches** to the accepted length (crop `past_key_values`).
3. Check that the output is token-for-token equal to plain greedy decoding with the target. This is the key correctness test.
4. Report acceptance rate, tokens accepted per step, latency, tokens/s, speedup or slowdown, and memory, for k = 2, 4, 6, 8. A slowdown is fine if you explain it.
5. Optionally compare against HF `assistant_model=` as a reference only.

---

## Step 10: Browser application (Task 6, G1–G4: 20 marks)

**What to do**
1. Use **Gradio** (fastest) or FastAPI + a simple HTML frontend. It must start with one documented command, e.g. `python -m src.app` → `http://localhost:7860`.
2. Features:
   - Paste a patient note. It gets auto-split into numbered sentences.
   - Enter criteria, or load an example trial.
   - Run it. Show a table of criterion → label → highlighted evidence sentences → explanation.
   - Show an **overall preliminary outcome** from an **explicit rule**. For example: "eligible" only if every inclusion criterion is met and no exclusion criterion is met; otherwise "not eligible" or "needs review". Label it as an aggregation, not ground truth.
   - Missing-information criteria are clearly flagged.
   - A **benchmark tab** that lets you choose a decoding mode (uncached / KV-cached / speculative) and run it live with timings. Static charts are not enough.
   - **Export the report** as HTML, PDF or Markdown, with provenance: model name, adapter version, decoding settings and timestamp.
   - A **disclaimer banner**: "Educational screening prototype — not a clinical enrollment decision."
   - **Input validation:** handle an empty note, an empty criterion, overly long input and malformed model JSON (retry or fallback). Show friendly errors.
3. Hosting: run it locally on your laptop (a 1.5B model can run on CPU, though slowly) or on a free **Hugging Face Space**. Test it on your demo machine **before** the demo day.

---

## Step 11: IEEE report (H1: 5 marks)

**What to do**
Use the IEEEtran LaTeX template (Overleaf works). It needs these sections: Title/Author, Abstract + Keywords, Introduction, Related Work, Materials & Methods (dataset, split, model, fine-tuning, **architecture diagrams**, attention and decoding algorithms, app workflow), Experimental Setup & Metrics, Results & Discussion, Limitations, Conclusion, References, and an **AI-use appendix**.
- Include figures: training curves (export them from W&B), confusion matrices, KV benchmark plots, attention error tables, speculative acceptance plots and app screenshots.
- **Every figure and table must be referenced and interpreted in the text.**

---

## Step 12: Submission checklist (H2, H3)

- [ ] Report PDF
- [ ] GitHub URL (public, README with setup + reproduction commands)
- [ ] Adapter/weights download link (HF Hub)
- [ ] `split_manifest.json`
- [ ] Benchmark logs (`results/*.json`)
- [ ] 5–7 minute demo video link (YouTube unlisted / Drive)
- [ ] AI-use appendix
- [ ] Practice the defense: be able to explain the cache, online softmax and spec-decoding by hand, and modify a parameter live

---

## Kaggle smoke-test protocol (do this before every long run)

The goal is that a 6-hour run never dies at hour 5.

1. **Add a `--smoke` flag to every script.** It should:
   - use about 16 training examples, 4 validation examples and `max_steps=10`
   - run eval every 5 steps
   - save a checkpoint at step 10, then **reload it** and generate one prediction
   - run the metric code on that output
   - log to W&B in a separate `smoke` project or group
2. **What a smoke run must prove:**
   - The data loads, and the split matches the manifest.
   - The loss is finite and decreasing (overfitting 16 examples to near-zero loss is a great sanity check).
   - GPU memory at your *real* max sequence length fits. Test with the **longest** example, not a random one.
   - Checkpoint save → load → inference works.
   - The eval/metrics code runs end-to-end on model output.
   - W&B logging shows up.
   - Outputs land in `/kaggle/working` (only that directory persists).
3. **For long runs:**
   - Use "Save & Run All (Commit)". It runs in the background even after you close the browser.
   - Save checkpoints every N steps, and **push them to the HF Hub during training** (`push_to_hub=True`, `hub_strategy="checkpoint"`). If the session dies, you can resume.
   - Use `resume_from_checkpoint` support.
   - Keep in mind the Kaggle limits: about 30 GPU-hours per week, a 12-hour max session and 20 GB of output.
   - Pick **T4 x2 or P100**. The T4 needs fp16 (no bf16) and has no FlashAttention-2, so use SDPA.
4. **Smoke-test the non-training steps too:** run the KV, attention and speculative benchmarks with tiny lengths first. For the app, run one note with 2 criteria first.

---

## Professional practices (suggestions)

- **Configs, not hardcoding:** a YAML file per experiment, saved next to its results.
- **Seed everything** (`random`, `numpy`, `torch`, `transformers.set_seed`), and log the seed.
- **Log provenance** with every result file: git commit hash, model ID, adapter version, decoding parameters.
- **Unit tests** (`pytest`) for the things that are *provably* correct: split leakage, attention equality, KV-cache token equality, spec-decoding equality and the metric functions on hand-made examples. These double as evidence for the D2, E3 and F3 marks.
- **Use W&B** for training curves and an eval table (log the predictions table so you can browse errors). Link it in the report.
- **Test-set discipline:** keep a `--split test` flag that prints a warning. Run it exactly once at the end and record the date in the decisions log.
- **Commit often** with meaningful messages. The git history itself shows the work is genuinely yours (useful for the defense).
- **Decisions log** (`docs/decisions.md`): a dated entry per decision or failure. It goes almost directly into the report's A3 and Limitations sections.
- **Back up early:** push the adapter to the HF Hub as soon as you have *any* decent checkpoint.
- **Record the demo video on Nov 9.** Don't leave it for submission day.
