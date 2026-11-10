# Final evaluation summary (SMOKE: 4 val examples per model)

Protocol: docs/decisions.md D-024. Pure greedy (repetition_penalty 1.0). Selected by val macro-F1: **run3**; zero-shot counterpart: **zeroshot-v2**.

## Val: labels

| Model | Prompt | Acc | Macro-F1 | Incl. mF1 | Excl. mF1 | Evid. micro-F1 | Evid. macro-F1 | Parse OK | Commit |
|---|---|---|---|---|---|---|---|---|---|
| zeroshot-v1 | v1 | 0.000 | 0.000 | 0.000 | N/A | 0.000 | 0.000 | 100% | e13346a |
| zeroshot-v2 | v2 | 0.000 | 0.000 | 0.000 | N/A | 0.000 | 0.000 | 100% | e13346a |
| run1 | v1 | 0.750 | 0.733 | 0.733 | N/A | 0.667 | 0.750 | 100% | e13346a |
| run2 | v2 | 0.750 | 0.733 | 0.733 | N/A | 0.667 | 0.750 | 100% | e13346a |
| run3 | v2 | 1.000 | 1.000 | 1.000 | N/A | 0.667 | 0.750 | 100% | e13346a |
| run4 | v1 | 1.000 | 1.000 | 1.000 | N/A | 0.667 | 0.750 | 100% | e13346a |

## Val: evidence (set-based over sentence IDs; both-empty records score 1 in macro, nothing in micro)

| Model | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | Gold empty | Pred empty | Both empty | Invalid IDs |
|---|---|---|---|---|---|---|---|---|---|---|
| zeroshot-v1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 4 | 0 | 0 |
| zeroshot-v2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 4 | 0 | 0 |
| run1 | 1.000 | 0.500 | 0.667 | 0.750 | 0.750 | 0.750 | 0 | 1 | 0 | 0 |
| run2 | 1.000 | 0.500 | 0.667 | 0.750 | 0.750 | 0.750 | 0 | 1 | 0 | 0 |
| run3 | 1.000 | 0.500 | 0.667 | 0.750 | 0.750 | 0.750 | 0 | 1 | 0 | 0 |
| run4 | 1.000 | 0.500 | 0.667 | 0.750 | 0.750 | 0.750 | 0 | 1 | 0 | 0 |

## Val: per-class, zeroshot-v2

| Type | Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| inclusion | included | 0.000 | 0.000 | 0.000 | 3 |
| inclusion | not included | N/A | N/A | N/A | 0 |
| inclusion | not enough information | 0.000 | 0.000 | 0.000 | 1 |
| inclusion | not applicable | N/A | N/A | N/A | 0 |

## Val: per-class, run3

| Type | Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| inclusion | included | 1.000 | 1.000 | 1.000 | 3 |
| inclusion | not included | N/A | N/A | N/A | 0 |
| inclusion | not enough information | 1.000 | 1.000 | 1.000 | 1 |
| inclusion | not applicable | N/A | N/A | N/A | 0 |
