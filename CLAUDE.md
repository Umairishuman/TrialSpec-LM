# CLAUDE.md

## Git commits
- Commit using the user's own git account (UmairisHuman). Do NOT add Claude as a co-author: no `Co-Authored-By: Claude ...` trailer and no "Generated with Claude Code" lines in commits or PRs.

## Project context
- GenAI Assignment #2: fine-tuned Transformer for TrialGPT criterion-level eligibility + evidence. See `ASSIGNMENT_PLAN.md`.
- Training runs on Kaggle (T4/P100: use fp16, no bf16, no FlashAttention-2). Experiment tracking uses Weights & Biases (key stored in Kaggle Secrets, never in code).
- Every training/benchmark script must support a `--smoke` mode for a quick Kaggle test before long runs.
- Never use `gpt4_*` dataset columns as inputs. Never tune on the test split.

## Decision log (mandatory)
- Every architectural, modelling, data, training, evaluation or tooling decision goes in `docs/decisions.md` as a new dated entry, written at the time the decision is made, without waiting to be asked.
- Also log failures, rejected alternatives and changes of direction. A reversed decision gets a new entry that references the old one; never delete old entries.
- Entry format: `## D-XXX: <title> (YYYY-MM-DD)` with **Context**, **Options considered**, **Decision**, **Why** (evidence: numbers, links, sources), **Consequences/risks**, **Status** (proposed / accepted / superseded by D-YYY).
- Every change to code, config, data processing, infrastructure or plan also gets a row in the **Changelog** table at the top of `docs/decisions.md`: date, stage (S1–S12 from `ASSIGNMENT_PLAN.md`), what changed, the reason or trigger (a bug, a failed run, a measurement, a user request), the linked D-XXX entry, and the commit hash (fill it in after committing). Add the row in the same commit as the change whenever possible.
- Every execution attempt (local or Kaggle, **including failed ones**) gets a numbered entry in `docs/runs.md` and a row in its summary table: platform, config, commit, what happened (exact error text), why it happened, what was changed to fix it (with commit hash), and the results with a metrics table. Plan the next run's entry (config diff + hypothesis) before it is launched, then fill in the results afterwards.
- This file feeds the report's Related Work, Methods, Limitations and rubric item A3, so keep the reasoning concrete.
