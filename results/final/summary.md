# Final evaluation summary

Protocol: docs/decisions.md D-024. Pure greedy (repetition_penalty 1.0). Selected by val macro-F1: **run4**; zero-shot counterpart: **zeroshot-v1**.

## Val: labels

| Model | Prompt | Acc | Macro-F1 | Incl. mF1 | Excl. mF1 | Evid. micro-F1 | Evid. macro-F1 | Parse OK | Commit |
|---|---|---|---|---|---|---|---|---|---|
| zeroshot-v1 | v1 | 0.068 | 0.054 | 0.064 | 0.074 | 0.000 | 0.563 | 100% | e13346a |
| zeroshot-v2 | v2 | 0.175 | 0.070 | 0.111 | 0.097 | 0.000 | 0.563 | 100% | e13346a |
| run1 | v1 | 0.743 | 0.432 | 0.360 | 0.401 | 0.554 | 0.745 | 100% | e13346a |
| run2 | v2 | 0.704 | 0.335 | 0.343 | 0.210 | 0.468 | 0.688 | 100% | e13346a |
| run3 | v2 | 0.728 | 0.379 | 0.374 | 0.252 | 0.556 | 0.740 | 100% | e13346a |
| run4 | v1 | 0.743 | 0.604 | 0.368 | 0.676 | 0.520 | 0.717 | 100% | e13346a |

## Val: evidence (set-based over sentence IDs; both-empty records score 1 in macro, nothing in micro)

| Model | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | Gold empty | Pred empty | Both empty | Invalid IDs |
|---|---|---|---|---|---|---|---|---|---|---|
| zeroshot-v1 | 0.000 | 0.000 | 0.000 | 0.563 | 0.563 | 0.563 | 116 | 206 | 116 | 0 |
| zeroshot-v2 | 0.000 | 0.000 | 0.000 | 0.563 | 0.563 | 0.563 | 116 | 206 | 116 | 0 |
| run1 | 0.528 | 0.583 | 0.554 | 0.741 | 0.765 | 0.745 | 116 | 127 | 102 | 0 |
| run2 | 0.436 | 0.504 | 0.468 | 0.692 | 0.706 | 0.688 | 116 | 119 | 97 | 0 |
| run3 | 0.500 | 0.626 | 0.556 | 0.738 | 0.763 | 0.740 | 116 | 117 | 97 | 0 |
| run4 | 0.449 | 0.617 | 0.520 | 0.712 | 0.742 | 0.717 | 116 | 111 | 94 | 0 |

## Val: per-class, zeroshot-v1

| Type | Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| inclusion | included | 0.000 | 0.000 | 0.000 | 35 |
| inclusion | not included | 0.000 | 0.000 | 0.000 | 3 |
| inclusion | not enough information | 0.444 | 0.148 | 0.222 | 27 |
| inclusion | not applicable | 0.018 | 1.000 | 0.034 | 1 |
| exclusion | excluded | 0.056 | 1.000 | 0.105 | 2 |
| exclusion | not excluded | 0.000 | 0.000 | 0.000 | 101 |
| exclusion | not enough information | 0.087 | 0.077 | 0.082 | 26 |
| exclusion | not applicable | 0.062 | 0.455 | 0.109 | 11 |

## Val: per-class, run4

| Type | Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| inclusion | included | 0.828 | 0.686 | 0.750 | 35 |
| inclusion | not included | 0.000 | 0.000 | 0.000 | 3 |
| inclusion | not enough information | 0.647 | 0.815 | 0.721 | 27 |
| inclusion | not applicable | 0.000 | 0.000 | 0.000 | 1 |
| exclusion | excluded | 1.000 | 1.000 | 1.000 | 2 |
| exclusion | not excluded | 0.814 | 0.911 | 0.860 | 101 |
| exclusion | not enough information | 0.615 | 0.308 | 0.410 | 26 |
| exclusion | not applicable | 0.417 | 0.455 | 0.435 | 11 |

## Test: labels

| Model | Prompt | Acc | Macro-F1 | Incl. mF1 | Excl. mF1 | Evid. micro-F1 | Evid. macro-F1 | Parse OK | Commit |
|---|---|---|---|---|---|---|---|---|---|
| zeroshot-v1 | v1 | 0.136 | 0.131 | 0.158 | 0.132 | 0.000 | 0.704 | 100% | e13346a |
| run4 | v1 | 0.738 | 0.660 | 0.664 | 0.514 | 0.544 | 0.778 | 100% | e13346a |

## Test: evidence (set-based over sentence IDs; both-empty records score 1 in macro, nothing in micro)

| Model | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | Gold empty | Pred empty | Both empty | Invalid IDs |
|---|---|---|---|---|---|---|---|---|---|---|
| zeroshot-v1 | 0.000 | 0.000 | 0.000 | 0.704 | 0.704 | 0.704 | 145 | 206 | 145 | 0 |
| run4 | 0.463 | 0.660 | 0.544 | 0.771 | 0.799 | 0.778 | 145 | 135 | 122 | 0 |

## Test: per-class, zeroshot-v1

| Type | Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| inclusion | included | 1.000 | 0.148 | 0.258 | 27 |
| inclusion | not included | 0.000 | 0.000 | 0.000 | 7 |
| inclusion | not enough information | 0.538 | 0.184 | 0.275 | 38 |
| inclusion | not applicable | 0.052 | 1.000 | 0.098 | 3 |
| exclusion | excluded | N/A | N/A | N/A | 0 |
| exclusion | not excluded | 0.000 | 0.000 | 0.000 | 91 |
| exclusion | not enough information | 0.263 | 0.179 | 0.213 | 28 |
| exclusion | not applicable | 0.103 | 0.750 | 0.182 | 12 |

## Test: per-class, run4

| Type | Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| inclusion | included | 0.917 | 0.815 | 0.863 | 27 |
| inclusion | not included | 0.429 | 0.429 | 0.429 | 7 |
| inclusion | not enough information | 0.814 | 0.921 | 0.864 | 38 |
| inclusion | not applicable | 1.000 | 0.333 | 0.500 | 3 |
| exclusion | excluded | N/A | N/A | N/A | 0 |
| exclusion | not excluded | 0.745 | 0.901 | 0.816 | 91 |
| exclusion | not enough information | 0.273 | 0.107 | 0.154 | 28 |
| exclusion | not applicable | 0.667 | 0.500 | 0.571 | 12 |
