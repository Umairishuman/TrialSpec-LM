# TrialSpec-LM

**Clinical-trial criterion matching with a fine-tuned small Transformer, plus hand-written KV caching, tiled attention and speculative decoding.**
GenAI Assignment #2.

> ⚠ **Educational screening prototype — not a clinical enrollment decision.**

Given a patient note split into numbered sentences, a criterion type (inclusion/exclusion) and the criterion text, the model
returns one of the TrialGPT labels and the IDs of the note sentences that support it:

```json
{"evidence_ids": [2, 3], "label": "not enough information"}
```

An explanation is then built deterministically from the cited sentences, so every statement points back to the note.

## Results (held-out test set, 8 patients, 206 criteria, evaluated once)

| Model | Accuracy | Macro-F1 | Inclusion mF1 | Exclusion mF1 | Evidence micro-F1 | Parse failures | Invalid IDs |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct, zero-shot | 0.136 | 0.131 | 0.158 | 0.132 | 0.000 | 0 | 0 |
| **+ LoRA, run 4 (final)** | **0.738** | **0.660** | **0.664** | **0.514** | **0.544** | 0 | 0 |

The final model was chosen on validation only (val macro-F1: run 1 0.432, run 2 0.335, run 3 0.379, **run 4 0.604**).
Full tables: [results/final/summary.md](results/final/summary.md). Error analysis of 12 cases:
[results/final/error_analysis_test.md](results/final/error_analysis_test.md).

| Mechanism | Result |
|---|---|
| KV cache (Task 3) | Up to **9.3×** faster decoding than the uncached reference (T4, 1024+256 tokens); token-identical except 2 diagnosed fp16 exact-tie flips |
| Tiled online-softmax attention (Task 4) | Matches reference/SDPA (fp32 max error 1.2e-6); memory linear instead of quadratic; slower than fused SDPA (launch-bound Python loop) |
| Speculative decoding (Task 5) | 64/64 runs token-identical to greedy; acceptance 0.49–0.83; **0.61–0.80×** speed on T4, explained by the draft/target cost ratio |

**Links**
- Report (IEEE format): [report/main.pdf](report/main.pdf) (source: [report/main.tex](report/main.tex))
- Final adapter: <https://huggingface.co/umairishuman/trialspec-qwen2.5-1.5b-lora-run4> (runs 1–3: `…-lora`, `…-lora-run2`, `…-lora-run3`)
- Demo video (screening app walkthrough): <https://drive.google.com/file/d/1bRcu7PCX7ZL3rSs98bDm_kMLhnFrL_SC/view?usp=sharing>
- Training curves: W&B project `trialspec-lm`; exported histories and figure in [results/training/wandb_export/](results/training/wandb_export/)
- Split manifest: [data/splits/split_manifest.json](data/splits/split_manifest.json)
- Decision log (every design choice, failure and reversal): [docs/decisions.md](docs/decisions.md)
- Run log (every execution attempt, including failures): [docs/runs.md](docs/runs.md)

---

## 1. Setup (local, Windows/Linux/macOS)

Requires Python 3.11+ (tested with 3.13) and ~10 GB of free disk space for the models.

```bash
git clone https://github.com/Umairishuman/TrialSpec-LM.git
cd TrialSpec-LM
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # or the CUDA wheel for your GPU
pip install -r requirements.txt
```

No API keys are needed to run the app or the tests. Keys are only needed to re-train (`WANDB_API_KEY`, `HF_TOKEN`), and
they go in a git-ignored `.env` file or Kaggle Secrets, never in code.

## 2. Run the application (Task 6)

```bash
python -m src.app --prefetch      # once, while online: downloads base model, adapters and draft (~5 GB)
python -m src.app                 # loads the models (~25 s), then open http://localhost:7860
```

- **Screen tab:** paste a note (free text is split into numbered sentences automatically) and enter criteria one per line,
  or pick an example trial (taken from the train/val split only). Click **Run screening**. Each criterion takes about
  6–13 s on a 4-core laptop CPU.
- **Results:** label, highlighted evidence sentences, explanation and flags per criterion, plus an **overall preliminary
  outcome** from an explicit rule (any blocking criterion → *Not eligible*; any unresolved / no inclusion met →
  *Needs review*; otherwise *Likely eligible*; "not applicable" ignored). It is labelled as an aggregation, not ground truth.
- **Download** the report as Markdown or HTML (printable to PDF) with provenance: model, adapter, prompt version,
  decoding settings, git commit and timestamp.
- **Benchmark tab:** runs uncached, HF-cache, hand-written-cache and speculative decoding live on the same prompt, with
  timings, speed-up, KV-cache size, acceptance rate and token equality.

Configuration is in [configs/app.yaml](configs/app.yaml). The app automatically uses the model chosen by the final
evaluation (`results/final/selection.json`), and falls back to run 1 if that file is absent. It runs in fp32 on CPU (about
8 GB RAM): this CPU has no native bf16, which made bf16 2.3× slower (D-031). On an NVIDIA GPU it switches to fp16
automatically.

Other options: `--port 7861`, `--open` (opens the browser), `--smoke` (see below).

## 3. Test it

```bash
pytest -q                                   # 173 tests, ~35 s on CPU, no downloads beyond the cached tokenizer
python -m src.app --smoke                   # whole app pipeline on tiny random models, ~6 s, no server
python -m src.app --smoke --launch          # same tiny models, but serves the UI (UI check without the real model)
```

What the tests cover:

| File | Checks |
|---|---|
| `tests/test_data.py` | split disjointness, determinism, manifest match, GPT-4 columns never reach the prompt |
| `tests/test_prompt.py`, `tests/test_metrics.py` | prompt format, output parser, metrics and the empty-evidence rule on hand-made cases |
| `tests/test_attention.py` | tiled online softmax = reference = PyTorch SDPA (6 shapes × masks × block sizes, fp64/fp32/fp16) |
| `tests/test_kv_cache.py` | KV-cached decoding is token-identical to the uncached reference (≥3 input × 3 output lengths) |
| `tests/test_speculative.py` | speculative decoding is token-identical to target greedy; acceptance statistics; cache rollback |
| `tests/test_select_checkpoint.py`, `tests/test_final_eval.py` | selection rule, test-lock guard |
| `tests/test_app.py` | sentence splitting, input validation, aggregation rule, retry/fallback on malformed JSON, report export, config resolution, no test patients in examples |

The real-model check is gated: `RUN_MODEL_TESTS=1 pytest tests/test_kv_cache.py` (downloads Qwen2.5).

**Manual checks to try in the app:** empty note (error message); a 7,000-character note (error); the example
trials; an unseen note of your own; the benchmark tab with "Custom text prompt" and 4–8 tokens.

## 4. Reproduce the experiments

All GPU work ran on Kaggle (Tesla T4). The notebook [notebooks/kaggle_train.ipynb](notebooks/kaggle_train.ipynb) only clones this
repository and calls a script. To run an experiment, set `CONFIG`, then Save Version → Save & Run All. Every script runs its
`--smoke` mode first.

| `CONFIG` | What it runs | Output |
|---|---|---|
| `configs/run1.yaml` … `configs/run4.yaml` | tests → smoke → zero-shot baseline → LoRA training → per-epoch val evaluation → select by val macro-F1 → push adapter | `results/{baseline,finetuned,training}/` |
| `benchmarks` | Task 3/4/5 GPU benchmarks | `results/{kv_cache,attention,speculative}/*cuda*.json` |
| `final_eval` | all models on val at pure greedy → select → test **once** (writes `results/final/test_lock.json`) | `results/final/` |

The same steps can be run locally on any GPU:

```bash
python -m src.data.split                                   # rebuild the split manifest (37/8/8, seed 42)
python -m src.train.run --config configs/run4.yaml --smoke  # then without --smoke
python -m src.eval.select_checkpoint --config configs/run4.yaml
python -m src.eval.predict --adapter umairishuman/trialspec-qwen2.5-1.5b-lora-run4 --prompt-version v1 --split val --out results/tmp/run4_val
python -m src.eval.final_eval --smoke                      # the full run refuses to touch test again (lock file)
python -m src.decoding.benchmark_kv --adapter umairishuman/trialspec-qwen2.5-1.5b-lora --smoke
python -m src.attention.benchmark --smoke
python -m src.decoding.benchmark_spec --smoke
python -m src.train.export_wandb                           # needs WANDB_API_KEY in the environment
python -m src.report.figures                               # report figures from the committed results
cd report && latexmk -pdf main.tex                         # or upload report/ to Overleaf
```

Model: `Qwen/Qwen2.5-1.5B-Instruct` with LoRA r=16, α=32, dropout 0.05 on all attention and MLP projections; lr 2e-4 cosine,
3 epochs, effective batch 16, fp16 on a T4, max length 1024 (no example truncated), loss on answer tokens only. Run 4 adds
oversampling of rare (type, label) groups to 60 rows (max 4×). Decoding everywhere: greedy, repetition penalty 1.0, ≤64 new
tokens. Qwen's shipped default of 1.1 is explicitly overridden (D-020).

## 5. Data rules

- Dataset: [`ncbi/TrialGPT-Criterion-Annotations`](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations).
  The `gpt4_*` columns, `explanation_correctness` and the original `training` flag are dropped at load time
  ([src/data/load.py](src/data/load.py)).
- Patient-grouped split exactly as prescribed: sorted patient IDs, `numpy.random.default_rng(42)` shuffle, 37/8/8.
- Model selection and all tuning used validation only. The test split was evaluated once (2026-11-10) and is locked by
  [results/final/test_lock.json](results/final/test_lock.json).
- The app's example trials come from train/val only. This is checked at export time, at start-up and in tests.

## 6. Repository layout

```
configs/      run1-4.yaml (training), final_eval.yaml, app.yaml
data/splits/  split_manifest.json            data/app_examples.json (app examples, train/val only)
src/data/     loading, leakage guard, split, prompt + parser
src/train/    LoRA fine-tuning, W&B export
src/eval/     batched prediction, metrics, checkpoint selection, final evaluation, error analysis, explanations
src/decoding/ hand-written Qwen2 decoder + KV cache, greedy loops, speculative decoding, benchmarks
src/attention/ reference and tiled online-softmax attention, benchmark
src/app/      browser application (UI in ui.py; logic in pipeline/aggregate/validate/text/bench/report)
src/report/   report figures
scripts/      Kaggle setup and pipelines        notebooks/  Kaggle notebook (calls scripts only)
results/      all metrics, predictions, benchmarks, figures (committed)
report/       IEEE LaTeX report + figures       docs/       decisions.md, runs.md, demo_script.md
tests/        pytest suite
```

## 7. Limitations

Small evaluation splits (8 patients each; "excluded" has no test examples), inconsistent expert labelling of notes that
are silent about a criterion (the main source of errors: exclusion "not enough information" F1 0.154), weak numeric and
compound-criterion reasoning in a 1.5B model, and templated explanations. See Section VII of the report.

## AI use

Claude Code (Anthropic) was used as a coding assistant for code, tests, debugging and documentation. All runs were launched
and checked by the author, and all reported numbers come from committed result files. See the report's AI-use appendix.
