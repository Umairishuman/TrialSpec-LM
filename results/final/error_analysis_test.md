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

## Failure patterns (hand-written analysis of the 12 cases below)

The 12 cases were sampled automatically, round-robin across categories (D-024). The "why it went wrong" text under each
case was written by hand after reading the full note and criterion. Recurring patterns:

1. **Silent-note convention (cases 2, 3, 9).** When the note says nothing, experts sometimes answer *not excluded*,
   sometimes *not enough information*, *not included* or *not applicable*. The model learned the majority choice
   (*not excluded*), which explains the low exclusion-NEI recall (0.107). This is partly label inconsistency, not only
   model error, and it is exactly where absent evidence must not be read as proof (the app flags these decisions).
2. **Clinical inference from raw findings (cases 7, 8).** The model cites the right vital signs but does not turn them into
   a clinical state (shock, moribund).
3. **Numeric thresholds and qualitative wording (case 11; app attempt 19).** Mapping "excruciating" to "≥ 3/10", or 8.1% to
   "7–10%", fails even when the right sentence is cited.
4. **Compound criteria (case 12).** One part met and one unknown should give *not enough information*; the model answers
   *not included*.
5. **Surface matching and vague criteria (cases 1, 6).** Topical overlap is taken as a match, and catch-all criteria get
   *excluded*. Oversampling the 8 *excluded* training examples (run 4) makes the second more likely.
6. **Ambiguous applicability (case 4).** Sex-mismatched criteria are labelled inconsistently in the data.
7. **Evidence-metric limits (cases 5, 10).** A correct label with a sensible citation counts as "unsupported" when the
   experts cited nothing, so the unsupported-evidence count (30) is an upper bound.

No case involved a parse failure or a hallucinated sentence ID (0 of 206 on test).

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

- **Why it went wrong:** **Surface matching instead of checking the event.** The note describes a femoral-artery complication after catheterisation (groin mass, cool foot, lost distal pulses). These are the words of limb ischaemia, so the model matched them to "lower limb revascularisation" and answered *included*. But the criterion requires the patient to be *undergoing* that surgery, and nothing in the note says she is. The cited sentences are about the right topic but do not support the label. A 1.5B model trained on 603 examples learns topical association more easily than the difference between a symptom and a planned procedure.

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

- **Why it went wrong:** **Expert convention for silent notes.** The note never mentions axillary surgery. The experts chose *not enough information*, but in training 75% of *not excluded* labels also cite no sentence (D-016): experts usually treat an unreported excluding condition as absent. The model learned that majority convention and applied it here. This is the largest error category on test (28 of 54 errors) and the main reason exclusion NEI recall is only 0.107. The label *not excluded* rests on the absence of information, which is why the app flags such answers as "no supporting evidence cited" instead of presenting them as confirmed.

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

- **Why it went wrong:** **The opposite side of the same ambiguity.** The note says nothing about kidney disease or dialysis. The experts inferred *not included* (an acute vascular presentation with no renal history) without citing a sentence, while the model abstained with *not enough information*. With a silent note, either answer can be defended. In training only 23% of *not included* labels have no evidence, so the model reserves *not included* for notes that state a contradiction. The model's answer is arguably the more cautious one; the gold label encodes an inference the note does not state.

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

- **Why it went wrong:** **Ambiguous label for sex-specific criteria.** The patient is a 48-year-old man, so a pregnancy/contraception exclusion cannot apply. The model said *not applicable* (the criterion concerns a different sex); the experts said *not excluded* and cited sentence 0, which states the sex. Both readings are reasonable, and the dataset is not consistent about which label sex-mismatched criteria get. The model read the patient correctly but did not cite the sentence that justified its answer.

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

- **Why it went wrong:** **Correct label, evidence counted as unsupported because gold is empty.** The model rejected "elective carotid endarterectomy" and cited sentence 0, which says what procedure the patient actually had (femoral catheterisation). That is a sensible justification, but the experts cited nothing, and the set-based evidence metric gives P = 0 whenever gold is empty and the prediction is not (D-011). Several "unsupported evidence" cases on test are of this kind (see case 10), so the 30 cases in that category overstate how often the cited sentences are actually irrelevant.

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

- **Why it went wrong:** **Vague catch-all criterion plus oversampled rare class.** "Other inappropriate situation" names no condition, so the note cannot decide it; the experts defaulted to *not excluded*. The model answered *excluded*, citing the acute complication in sentence 0. *Excluded* has only 8 training examples, and run 4 repeated rare (type, label) groups up to 4× (D-016). This raised exclusion recall on val (both *excluded* val cases correct) but also makes the model more willing to predict *excluded* on weak grounds. This is the price of the oversampling that made run 4 the best model.

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

- **Why it went wrong:** **Missing clinical inference from vital signs.** The experts marked "moribund" as *not enough information* and cited sentences 0–4: the patient is pale, tachycardic, tachypnoeic and hypotensive (BP 80/60), possibly in shock but not clearly dying. The model cited nothing and answered *not excluded*, the silent-note default. It did not recognise that these vital signs make the criterion uncertain rather than absent. Turning raw vital signs into a clinical state requires medical knowledge the 1.5B model has only weakly, and nothing in fine-tuning teaches it.

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

- **Why it went wrong:** **Same patient, same inference gap.** The experts read BP 80/60 with tachycardia and tachypnoea (sentences 2–3) as shock and labelled *included*. The model cited sentence 2 (partly right) but missed sentence 3, which has the blood pressure, and abstained. Together with case 7, this shows the model is not consistent across criteria for the same patient: it treats the patient as not moribund but cannot confirm shock. Each criterion is decided independently, so nothing makes the answers agree.

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

- **Why it went wrong:** **Not applicable vs not excluded for a silent note.** A 10-year-old with snoring and apnoea, and a criterion about a previous tonsillectomy that the note never mentions. The experts judged the criterion *not applicable* to this patient; the model gave the majority answer *not excluded* with no evidence. Exclusion *not applicable* is rare (23 training examples), and its boundary with *not excluded* depends on the experts' view of the trial rather than on anything in the note.

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

- **Why it went wrong:** **A systematic and reasonable citation pattern.** As in case 5, the model rejects a procedure criterion ("open abdominal aortic aneurysm repair") and cites the sentence that describes the procedure the patient really had. It does this consistently across the procedure criteria of trial NCT02097186. The label is right, and the citation is arguably useful to a reader, but the metric scores it as unsupported because the experts cited no sentence.

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

- **Why it went wrong:** **Qualitative-to-numeric reasoning failure.** The note says "excruciating pain", which is clearly at least 3/10, and the model cited exactly the right sentence, yet answered *not included*. The criterion is a numeric threshold, and the note gives a qualitative description with no score; the model seems to treat the missing number as a failure instead of mapping "excruciating" onto the scale. The same weakness showed up in the app test on an unseen note (HbA1c 8.1% judged *not enough information* for "HbA1c between 7% and 10%", runs.md attempt 19).

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

- **Why it went wrong:** **Compound criterion with one part unknown.** "Immune deficiency **and** early bone-marrow failure in childhood": the note confirms an immunodeficiency (CVID, sentence 0) but says nothing about bone-marrow failure or childhood onset. With one part met and the other unknown, the correct answer is *not enough information* (the experts' label). The model cited the right sentence but answered *not included*, treating the unknown part as failed (or reading "48-year-old" as ruling out "in childhood", although a childhood history is still possible). Conjunctions with partial information are a known weak point of single-label classification.
