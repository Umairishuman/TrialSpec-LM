# Error analysis: run3_val.jsonl (val)

4 predictions, 0 wrong labels (0.0%). One record can fall into several categories.

| Category | Count | Meaning |
|---|---|---|
| polarity_flip | 0 | Contradictory label: the opposite decision (e.g. included vs not included) |
| missed_information | 0 | Gold says the note is insufficient, but the model made a definite decision |
| spurious_nei | 0 | The model claimed missing information, but the experts made a definite decision |
| applicability | 0 | Confusion involving "not applicable" |
| unsupported_evidence | 1 | Correct label, but the cited evidence does not support it (no overlap with the gold sentences, or citations where gold cites none) |
| hallucinated_ids | 0 | Cited sentence IDs that do not exist in the note |
| parse_failure | 0 | Output was not valid JSON with an allowed label |

Sentence tags: `G` = cited by the experts, `P` = cited by the model.

## Case 1: unsupported_evidence (annotation 0, sigir-20141, NCT01397994)

- **Criterion (inclusion):** Patients of chronic stable angina with abnormal Exercise Myocardial Perfusion Spect Scan with reversible and partially reversible ischemic changes.
- **Gold:** `not enough information`, evidence [0, 1, 2]
- **Predicted:** `not enough information`, evidence []
- **All categories:** unsupported_evidence
- **Raw output:** `{"evidence_ids": [], "label": "not enough information"}`
- **Note snippet:**

```
    [G-] 0. A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.
    [G-] 1. The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.
    [G-] 2. The latest episode of pain ended half an hour prior to her arrival.
```

- **Why it went wrong:** _to be written from this case_
