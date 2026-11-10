# Error analysis: run4_val.jsonl (val)

206 predictions, 53 wrong labels (25.7%). One record can fall into several categories.

| Category | Count | Meaning |
|---|---|---|
| polarity_flip | 3 | Contradictory label: the opposite decision (e.g. included vs not included) |
| missed_information | 23 | Gold says the note is insufficient, but the model made a definite decision |
| spurious_nei | 17 | The model claimed missing information, but the experts made a definite decision |
| applicability | 10 | Confusion involving "not applicable" |
| unsupported_evidence | 30 | Correct label, but the cited evidence does not support it (no overlap with the gold sentences, or citations where gold cites none) |
| hallucinated_ids | 0 | Cited sentence IDs that do not exist in the note |
| parse_failure | 0 | Output was not valid JSON with an allowed label |

Sentence tags: `G` = cited by the experts, `P` = cited by the model.

## Case 1: polarity_flip (annotation 21, sigir-20141, NCT00149227)

- **Criterion (inclusion):** Clinical diagnosis of hypertension
- **Gold:** `included`, evidence [3]
- **Predicted:** `not included`, evidence [3]
- **All categories:** polarity_flip
- **Raw output:** `{"evidence_ids": [3], "label": "not included"}`
- **Note snippet:**

```
    [GP] 3. She is known to have hypertension and obesity.
```

- **Why it went wrong:** _to be written from this case_

## Case 2: missed_information (annotation 8, sigir-20141, NCT01397994)

- **Criterion (exclusion):** Unstable angina
- **Gold:** `not enough information`, evidence [0, 1, 2]
- **Predicted:** `not excluded`, evidence [0, 1, 2]
- **All categories:** missed_information
- **Raw output:** `{"evidence_ids": [0, 1, 2], "label": "not excluded"}`
- **Note snippet:**

```
    [GP] 0. A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.
    [GP] 1. The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.
    [GP] 2. The latest episode of pain ended half an hour prior to her arrival.
```

- **Why it went wrong:** _to be written from this case_

## Case 3: spurious_nei (annotation 22, sigir-20141, NCT00149227)

- **Criterion (inclusion):** Clinical diagnosis of one or more risk factors, such as diabetes, smoking habit, lipid metabolism abnormality, history of ischemic heart disease (IHD) or cerebrovascular disease, obesity (BMI>25), chronic heart failure (NYHA II-III), and electrocardiogram (ECG) abnormality (LVH)
- **Gold:** `included`, evidence [3, 4]
- **Predicted:** `not enough information`, evidence [0, 3]
- **All categories:** spurious_nei
- **Raw output:** `{"evidence_ids": [0, 3], "label": "not enough information"}`
- **Note snippet:**

```
    [-P] 0. A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.
    [GP] 3. She is known to have hypertension and obesity.
    [G-] 4. She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease.
```

- **Why it went wrong:** _to be written from this case_

## Case 4: applicability (annotation 24, sigir-20141, NCT00149227)

- **Criterion (exclusion):** Patients with IHD within 6 months after percutaneous coronary intervention(PCI), and who are stable but are going to implement PCI or coronary artery bypass grafting(CABG)
- **Gold:** `not applicable`, evidence []
- **Predicted:** `not excluded`, evidence []
- **All categories:** applicability
- **Raw output:** `{"evidence_ids": [], "label": "not excluded"}`
- **Note snippet:**

```
    (neither side cites a sentence; first sentences of the note shown)
    [--] 0. A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.
    [--] 1. The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.
    [--] 2. The latest episode of pain ended half an hour prior to her arrival.
```

- **Why it went wrong:** _to be written from this case_

## Case 5: unsupported_evidence (annotation 0, sigir-20141, NCT01397994)

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

## Case 6: polarity_flip (annotation 515, sigir-201427, NCT01713881)

- **Criterion (inclusion):** Any adenoma found on screening/surveillance colonoscopy (ICD9-211.3, Procedure code 45385 or 45380)
- **Gold:** `included`, evidence [0, 3, 4]
- **Predicted:** `not included`, evidence [3, 4]
- **All categories:** polarity_flip
- **Raw output:** `{"evidence_ids": [3, 4], "label": "not included"}`
- **Note snippet:**

```
    [G-] 0. A 21-year-old college student undergoes colonoscopy due to family history of multiple polyps in his older siblings.
    [GP] 3. The patient undergoes sigmoidoscopy and is found to have dozens of small colonic polyps within rectosigmoid.
    [GP] 4. Several of these are biopsied and are all benign adenomas.
```

- **Why it went wrong:** _to be written from this case_

## Case 7: missed_information (annotation 15, sigir-20141, NCT01397994)

- **Criterion (exclusion):** Pregnant and nursing women
- **Gold:** `not enough information`, evidence []
- **Predicted:** `not applicable`, evidence []
- **All categories:** missed_information
- **Raw output:** `{"evidence_ids": [], "label": "not applicable"}`
- **Note snippet:**

```
    (neither side cites a sentence; first sentences of the note shown)
    [--] 0. A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.
    [--] 1. The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.
    [--] 2. The latest episode of pain ended half an hour prior to her arrival.
```

- **Why it went wrong:** _to be written from this case_

## Case 8: spurious_nei (annotation 31, sigir-20142, NCT02618655)

- **Criterion (inclusion):** patients have fever more than one week
- **Gold:** `not included`, evidence [0]
- **Predicted:** `not enough information`, evidence [0]
- **All categories:** spurious_nei
- **Raw output:** `{"evidence_ids": [0], "label": "not enough information"}`
- **Note snippet:**

```
    [GP] 0. An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.
```

- **Why it went wrong:** _to be written from this case_

## Case 9: applicability (annotation 270, sigir-201415, NCT00180739)

- **Criterion (exclusion):** 20. Individuals who are not able or willing to tolerate the required prolonged stationary prone position during treatment (approximately 3 hours.)
- **Gold:** `not excluded`, evidence [6]
- **Predicted:** `not applicable`, evidence []
- **All categories:** applicability
- **Raw output:** `{"evidence_ids": [], "label": "not applicable"}`
- **Note snippet:**

```
    [G-] 6. The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
```

- **Why it went wrong:** _to be written from this case_

## Case 10: unsupported_evidence (annotation 5, sigir-20141, NCT01397994)

- **Criterion (exclusion):** Hypertension of > 170/100 mm of Hg
- **Gold:** `not excluded`, evidence []
- **Predicted:** `not excluded`, evidence [3]
- **All categories:** unsupported_evidence
- **Raw output:** `{"evidence_ids": [3], "label": "not excluded"}`
- **Note snippet:**

```
    [-P] 3. She is known to have hypertension and obesity.
```

- **Why it went wrong:** _to be written from this case_

## Case 11: polarity_flip (annotation 809, sigir-201515, NCT01550588)

- **Criterion (inclusion):** Subjects who have had a cryptogenic stroke within the previous 3 months, radiologically verified
- **Gold:** `included`, evidence [0]
- **Predicted:** `not included`, evidence [0]
- **All categories:** polarity_flip
- **Raw output:** `{"evidence_ids": [0], "label": "not included"}`
- **Note snippet:**

```
    [GP] 0. Karen is a 72-year-old woman with hypertension and type 2 diabetes, who was hospitalized for cryptogenic stroke two weeks ago.
```

- **Why it went wrong:** _to be written from this case_

## Case 12: missed_information (annotation 18, sigir-20141, NCT01397994)

- **Criterion (exclusion):** Patients not eligible for Tc 99m SPECT
- **Gold:** `not enough information`, evidence []
- **Predicted:** `not excluded`, evidence []
- **All categories:** missed_information
- **Raw output:** `{"evidence_ids": [], "label": "not excluded"}`
- **Note snippet:**

```
    (neither side cites a sentence; first sentences of the note shown)
    [--] 0. A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.
    [--] 1. The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.
    [--] 2. The latest episode of pain ended half an hour prior to her arrival.
```

- **Why it went wrong:** _to be written from this case_
