# Error analysis: run4_test.jsonl (test)

206 predictions, 54 wrong labels (26.2%). One record can fall into several categories.

| Category | Count | Meaning |
|---|---|---|
| polarity_flip | 4 | Contradictory label: the opposite decision (e.g. included vs not included) |
| missed_information | 28 | Gold says the note is insufficient, but the model made a definite decision |
| spurious_nei | 16 | The model claimed missing information, but the experts made a definite decision |
| applicability | 6 | Confusion involving "not applicable" |
| unsupported_evidence | 30 | Correct label, but the cited evidence does not support it (no overlap with the gold sentences, or citations where gold cites none) |
| hallucinated_ids | 0 | Cited sentence IDs that do not exist in the note |
| parse_failure | 0 | Output was not valid JSON with an allowed label |

Sentence tags: `G` = cited by the experts, `P` = cited by the model.

## Case 1: polarity_flip (annotation 164, sigir-201410, NCT02097186)

- **Criterion (inclusion):** Patients undergoing surgical lower limb revascularisation (suprainguinal or infrainguinal)
- **Gold:** `not included`, evidence []
- **Predicted:** `included`, evidence [0, 2]
- **All categories:** polarity_flip
- **Raw output:** `{"evidence_ids": [0, 2], "label": "included"}`
- **Note snippet:**

```
    [-P] 0. A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
    [-P] 2. Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
```

- **Why it went wrong:** _to be written from this case_

## Case 2: missed_information (annotation 174, sigir-201410, NCT02097186)

- **Criterion (exclusion):** Patients with previous axillary surgery
- **Gold:** `not enough information`, evidence []
- **Predicted:** `not excluded`, evidence []
- **All categories:** missed_information
- **Raw output:** `{"evidence_ids": [], "label": "not excluded"}`
- **Note snippet:**

```
    (neither side cites a sentence; first sentences of the note shown)
    [--] 0. A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
    [--] 1. She is now complaining of a cool right foot.
    [--] 2. Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
```

- **Why it went wrong:** _to be written from this case_

## Case 3: spurious_nei (annotation 177, sigir-201410, NCT02264964)

- **Criterion (inclusion):** Maintenance hemodialysis after central vena catheterization.
- **Gold:** `not included`, evidence []
- **Predicted:** `not enough information`, evidence []
- **All categories:** spurious_nei
- **Raw output:** `{"evidence_ids": [], "label": "not enough information"}`
- **Note snippet:**

```
    (neither side cites a sentence; first sentences of the note shown)
    [--] 0. A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
    [--] 1. She is now complaining of a cool right foot.
    [--] 2. Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
```

- **Why it went wrong:** _to be written from this case_

## Case 4: applicability (annotation 322, sigir-201417, NCT01745731)

- **Criterion (exclusion):** Pregnant women or women of childbearing age who do not have adequate contraception.
- **Gold:** `not excluded`, evidence [0]
- **Predicted:** `not applicable`, evidence []
- **All categories:** applicability
- **Raw output:** `{"evidence_ids": [], "label": "not applicable"}`
- **Note snippet:**

```
    [G-] 0. A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.
```

- **Why it went wrong:** _to be written from this case_

## Case 5: unsupported_evidence (annotation 161, sigir-201410, NCT02097186)

- **Criterion (inclusion):** Patients undergoing elective carotid endarterectomy
- **Gold:** `not included`, evidence []
- **Predicted:** `not included`, evidence [0]
- **All categories:** unsupported_evidence
- **Raw output:** `{"evidence_ids": [0], "label": "not included"}`
- **Note snippet:**

```
    [-P] 0. A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
```

- **Why it went wrong:** _to be written from this case_

## Case 6: polarity_flip (annotation 184, sigir-201410, NCT02264964)

- **Criterion (exclusion):** Other inappropriate situation
- **Gold:** `not excluded`, evidence []
- **Predicted:** `excluded`, evidence [0]
- **All categories:** polarity_flip
- **Raw output:** `{"evidence_ids": [0], "label": "excluded"}`
- **Note snippet:**

```
    [-P] 0. A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
```

- **Why it went wrong:** _to be written from this case_

## Case 7: missed_information (annotation 193, sigir-201411, NCT01526382)

- **Criterion (exclusion):** Patients were moribund.
- **Gold:** `not enough information`, evidence [0, 1, 2, 3, 4]
- **Predicted:** `not excluded`, evidence []
- **All categories:** missed_information
- **Raw output:** `{"evidence_ids": [], "label": "not excluded"}`
- **Note snippet:**

```
    [G-] 0. A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
    [G-] 1. She denies trauma.
    [G-] 2. On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
    [G-] 3. Her body temperature is normal and her blood pressure is 80/60.
    [G-] 4. Her right arm has no discoloration or movement limitation.
```

- **Why it went wrong:** _to be written from this case_

## Case 8: spurious_nei (annotation 185, sigir-201411, NCT01526382)

- **Criterion (inclusion):** Patients were included if they were diagnosed with Shock, Acute respiratory distress syndrome (ARDS), or both.
- **Gold:** `included`, evidence [2, 3]
- **Predicted:** `not enough information`, evidence [0, 2, 4]
- **All categories:** spurious_nei
- **Raw output:** `{"evidence_ids": [0, 2, 4], "label": "not enough information"}`
- **Note snippet:**

```
    [-P] 0. A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
    [GP] 2. On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
    [G-] 3. Her body temperature is normal and her blood pressure is 80/60.
    [-P] 4. Her right arm has no discoloration or movement limitation.
```

- **Why it went wrong:** _to be written from this case_

## Case 9: applicability (annotation 683, sigir-20158, NCT02562040)

- **Criterion (exclusion):** Previous tonsillectomy, including partial tonsillectomy
- **Gold:** `not applicable`, evidence []
- **Predicted:** `not excluded`, evidence []
- **All categories:** applicability
- **Raw output:** `{"evidence_ids": [], "label": "not excluded"}`
- **Note snippet:**

```
    (neither side cites a sentence; first sentences of the note shown)
    [--] 0. A 10 yo boy with nighttime snoring, pauses in breathing, and restlessness with nighttime awakenings.
    [--] 1. No history of headache or night terrors.
    [--] 2. The boy's teacher recently contacted his parents because she was concerned about his declining grades, lack of attention, and excessive sleepiness during class.
```

- **Why it went wrong:** _to be written from this case_

## Case 10: unsupported_evidence (annotation 162, sigir-201410, NCT02097186)

- **Criterion (inclusion):** Patients undergoing open abdominal aortic aneurysm repair
- **Gold:** `not included`, evidence []
- **Predicted:** `not included`, evidence [0]
- **All categories:** unsupported_evidence
- **Raw output:** `{"evidence_ids": [0], "label": "not included"}`
- **Note snippet:**

```
    [-P] 0. A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
```

- **Why it went wrong:** _to be written from this case_

## Case 11: polarity_flip (annotation 195, sigir-201411, NCT01723137)

- **Criterion (inclusion):** Reported pain greater than or equal to 3 out of 10
- **Gold:** `included`, evidence [0]
- **Predicted:** `not included`, evidence [0]
- **All categories:** polarity_flip
- **Raw output:** `{"evidence_ids": [0], "label": "not included"}`
- **Note snippet:**

```
    [GP] 0. A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
```

- **Why it went wrong:** _to be written from this case_

## Case 12: missed_information (annotation 323, sigir-201417, NCT02556359)

- **Criterion (inclusion):** Immune Deficiency and early BMF in childhood
- **Gold:** `not enough information`, evidence [0]
- **Predicted:** `not included`, evidence [0]
- **All categories:** missed_information
- **Raw output:** `{"evidence_ids": [0], "label": "not included"}`
- **Note snippet:**

```
    [GP] 0. A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.
```

- **Why it went wrong:** _to be written from this case_
