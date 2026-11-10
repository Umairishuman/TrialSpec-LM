# Demo video: recording plan and script (5–7 minutes)

Target length **6:00–6:45**. The rubric needs the app to be shown working end to end (G1–G4), the inference modes live
(G3), recovery from invalid input (G4), and a pointer to the reproducible repository (H2).

---

## A. Preparation (do this 10 minutes before recording)

1. **Free memory.** Close Chrome tabs, Teams, games. The app needs about 8 GB of RAM (fp32 models).
2. **Plug in the charger** and set Windows power mode to *Best performance* (CPU speed affects the timings).
3. Open a terminal in the project folder and activate the venv:
   ```powershell
   cd "D:\7th Semester\GenAI\Assignment no. 2"
   .venv\Scripts\activate
   $env:HF_HUB_OFFLINE = "1"      # proves it runs offline; models are already downloaded
   python -m src.app
   ```
   Wait for `Running on local URL: http://127.0.0.1:7860` (about 25 s).
4. Open <http://localhost:7860> in Edge or Chrome, at **90% zoom** so the results table fits the screen.
5. **Warm-up run (important):** the first criterion after start-up takes ~25 s instead of ~8 s. Load any example, click
   *Run screening* once, wait for it to finish, then click *Clear*. Do the same once in the Benchmark tab
   (custom prompt, 2 tokens).
6. Keep these ready in a text file to paste (Section C):
   - unseen note + criteria
   - one invalid input
7. Recorder: OBS Studio or Xbox Game Bar (Win+Alt+R), 1920×1080, microphone on. Record the browser window, and the
   terminal for segment 1.
8. Optional second window: VS Code with `docs/decisions.md` and `results/final/summary.md` open.

**Waiting times.** Screening runs at ~6–13 s per criterion. While the progress bar runs, keep talking (the narration below
is timed for that), or cut / speed up the wait in editing and say so ("sped up 4×").

---

## B. Timeline and narration

| Time | Show | Say (roughly) |
|---|---|---|
| **0:00–0:30** | Title: the app page with the red disclaimer banner | "This is TrialSpec-LM. It checks a patient note against clinical-trial criteria one at a time, gives a label with the exact supporting sentences, and combines them into a preliminary outcome. It's an educational screening prototype, not a clinical decision tool; the banner says so on every page and every report." |
| **0:30–1:00** | Terminal: the `python -m src.app` command and the start-up log (model, adapter `…-lora-run4`, source `results/final/selection.json`, fp32) | "It starts with one command and runs locally on this laptop's CPU, offline. The model is Qwen2.5-1.5B with my LoRA adapter from run 4, which was selected on validation. The app reads that choice from the final-evaluation file, so no code changes when the model changes." |
| **1:00–1:40** | Screen tab → choose a **val** example in *Example trial* → *Load example*. Point at the numbered-sentence preview | "Examples come only from the training and validation patients, never the test set. The note is shown with its original sentence IDs; that's exactly what the model sees." |
| **1:40–2:40** | *Run screening*; progress bar; then the outcome card and results table. Point at one row with highlighted evidence, one 🟠 row, one ⚪ "no supporting evidence" row. Show the *Expert label* column | "Each criterion gives a label from the dataset's four categories, the cited sentences highlighted, and an explanation built only from those sentences. Orange means missing information. White means the model decided without citing anything, which follows how experts label silent notes, so I flag it instead of presenting it as confirmed. For these example trials the expert labels are shown for comparison." Then on the outcome card: "The overall result is an explicit rule, not a label from the data: any failing criterion means not eligible, any unknown means needs review, otherwise likely eligible." |
| **2:40–3:00** | *Highlight evidence for* → pick one criterion; the note view highlights its sentence | "I can trace any decision back to the note." |
| **3:00–3:45** | *Clear* → paste the **unseen note** and criteria from Section C → *Run screening* (4 criteria, ~35 s) | "Now a note the model has never seen, written by me in free text. It's split into sentences automatically." When the results appear: "Diabetes is confirmed from sentence 0, the old infarction doesn't trigger the 6-month exclusion, and pregnancy is ruled out by sentence 3. HbA1c 8.1% is inside 7 to 10%, but the model says not enough information. That's a real weakness with numeric ranges, and the rule turns it into *needs review*, not a wrong verdict." |
| **3:45–4:05** | *Download report (HTML)* → open it in a new tab, scroll to Provenance | "The report can be downloaded as HTML or Markdown, with the model, adapter, prompt version, decoding settings, git commit and timestamp." |
| **4:05–4:35** | Invalid input: *Clear*, click *Run screening* with nothing → error box. Then paste the huge note (Section C) → error | "Invalid inputs are caught with readable messages before the model runs: an empty note, no criteria, or a note longer than the limit. Malformed model output is retried and then resolved by ranking the allowed labels, and those rows are flagged." |
| **4:35–5:40** | Benchmark tab → *Custom text prompt* (default) → all 4 modes, max new tokens 8, k=4 → *Run benchmark* (~30 s). Show the table, the two charts, the "identical tokens" note | "This runs the decoders I implemented, live. Uncached recomputes the whole sequence for every token; the two KV-cache versions, Hugging Face's cache object and my own hand-written cache, reuse stored keys and values, and are several times faster. Speculative decoding lets the 0.5B model propose tokens that the 1.5B model checks in one pass. All four give identical tokens. Speculative decoding needs fewer target passes but isn't faster here, because the draft isn't cheap enough; on the GPU it was 0.6 to 0.8 times the speed, and the report explains why with the measured cost ratio." |
| **5:40–6:20** | Optional: switch prompt to *Criterion from the Screen tab*, 4 tokens, run only *Uncached* and *KV cache (hand-written)*: the gap is bigger on a 300-token prompt | "On a real 300-token prompt the uncached path is far slower, because every step reprocesses the whole prompt." |
| **6:20–6:45** | VS Code or GitHub: `README.md`, `docs/decisions.md`, `results/final/summary.md` | "Everything is reproducible: the split manifest, configs, logs of every run including the failed ones, and a decision log. On the held-out test set the fine-tuned model reaches 73.8% accuracy and 0.66 macro-F1, up from 13.6% and 0.13 before fine-tuning. Thank you." |

If you are running long, drop the 5:40–6:20 segment.

---

## C. Text to paste

**Unseen note** (Screen tab → Patient note):
```
A 52-year-old woman with type 2 diabetes for 8 years, on metformin. HbA1c last month was 8.1%. She had a myocardial infarction in 2021. She is not pregnant.
```
**Inclusion criteria:**
```
Type 2 diabetes mellitus
HbA1c between 7% and 10%
```
**Exclusion criteria:**
```
Myocardial infarction within the past 6 months
Pregnancy
```
Tested on 2026-11-10 (docs/runs.md attempt 19): I1 included [0], I2 not enough information [1] (model error), E1 not
excluded [2], E2 not excluded [3] → **Needs review**. Greedy decoding is deterministic, so you should get the same result
(fp32, same model).

**Invalid input:** click *Run screening* with everything empty. For the "too long" error, paste any text over 6,000
characters (e.g. this note copied 40 times).

**Optional second unseen note** (try it before recording; the result is not pre-verified):
```
A 61-year-old man with type 2 diabetes treated with insulin. His eGFR is 24 mL/min. He has no history of heart failure.
```
Inclusion: `Type 2 diabetes mellitus`. Exclusion: `Severe renal impairment (eGFR below 30 mL/min)`. If the model answers
*excluded*, this shows the **Not eligible** outcome.

---

## D. If something goes wrong while recording

| Problem | Fix |
|---|---|
| Port already in use | `python -m src.app --port 7861` |
| "Could not load the model" offline | Run once without `HF_HUB_OFFLINE`, or `python -m src.app --prefetch` while online |
| Very slow (> 20 s per criterion) | Close other apps, plug in the charger; the first criterion after start-up is always slow (warm-up) |
| Out of memory | Close the browser tabs you don't need; in `configs/app.yaml` set `preload_draft: false` (the draft then loads when the benchmark first needs it) |
| Need a UI without the real model | `python -m src.app --smoke --launch` (tiny random models: the outputs are nonsense, but every screen works) |

---

## E. Defense preparation (likely questions)

- **Why Qwen2.5-1.5B?** Apache-2.0, fits a free T4 with LoRA in fp16, and the 0.5B has an identical tokenizer for the
  draft. 3B: non-Apache licence and tight memory; Llama-3.2: gated. (D-004, D-008)
- **How is leakage prevented?** GPT-4 columns are dropped at load time and a test checks it; patient-grouped split with
  seed 42; test evaluated once, with a lock file. (D-005, D-006, D-024)
- **KV cache by hand:** prefill fills positions 0..T-1; each step feeds one token at position = cache length; RoPE is
  applied before the key is stored; no mask is needed for one query; memory = 2·layers·kv_heads·head_dim·bytes per token
  = 28 KiB/token for this model in fp16 (GQA: 2 KV heads instead of 12). Mismatch at step 76 = exact fp16 tie. (D-019, D-023)
- **Online softmax:** running max m, normaliser l and output O, rescaled by exp(m_old − m_new) when the max grows; causal
  tiles above the diagonal are skipped. It is slower than SDPA because of thousands of small kernel launches from Python. (D-017)
- **Speculative decoding:** draft k tokens → one target pass → accept the longest matching prefix → the target's token →
  crop both caches. Exact by construction; slower because the 24-layer draft costs ~0.86 of a 28-layer target step on a
  launch-bound GPU, and the formula predicts 0.77× vs 0.74× measured. (D-021, D-023)
- **Why macro-F1 and F1-based checkpoint selection?** Strong class imbalance; validation loss picked worse checkpoints in
  runs 2 and 4. (D-006, D-018)
- **Main weakness?** Exclusion "not enough information" (test F1 0.154): experts label silent notes inconsistently, and the
  model learned "not excluded". (D-016, D-030)
- **"Change something live":** e.g. switch the app to run 1 by setting `use_final_selection: false` in `configs/app.yaml`
  and restarting; change `k` in the benchmark tab; change `max_new_tokens` in `configs/app.yaml`; run
  `pytest tests/test_app.py -k aggregation`.
