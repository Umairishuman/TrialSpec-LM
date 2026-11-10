"""Evidence text lookup and deterministic, evidence-grounded explanations (D-009, D-024).

The model is trained to output only {"evidence_ids", "label"}: the dataset has no expert explanations and
GPT-4's are forbidden. The explanation shown to users is therefore composed from the label and the exact
sentences the model cited, so every statement traces back to the note.
"""

LABEL_PHRASES = {
    "inclusion": {
        "included": "The note indicates the patient meets this inclusion criterion",
        "not included": "The note indicates the patient does not meet this inclusion criterion",
        "not enough information": "The note does not give enough information to decide this inclusion criterion",
        "not applicable": "This inclusion criterion does not apply to this patient",
    },
    "exclusion": {
        "excluded": "The note indicates the patient has the excluding condition",
        "not excluded": "The note indicates the patient does not have the excluding condition",
        "not enough information": "The note does not give enough information to decide this exclusion criterion",
        "not applicable": "This exclusion criterion does not apply to this patient",
    },
}

NO_EVIDENCE = ("No sentence was cited, so this label rests on the absence of reported information, "
               "not on stated evidence.")
UNPARSEABLE = "The model output could not be parsed into an allowed label, so no decision was made."


def evidence_text(sentences: list[tuple[int, str]], ids: list[int]) -> list[dict]:
    """Original sentence IDs -> [{"id", "text"}], in the order given. Unknown IDs are skipped."""
    lookup = dict(sentences)
    return [{"id": i, "text": lookup[i]} for i in ids if i in lookup]


def _clip(text: str, n: int = 220) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def explanation(criterion_type: str, label: str | None, cited: list[dict]) -> str:
    if label is None:
        return UNPARSEABLE
    lead = LABEL_PHRASES[criterion_type][label]
    if not cited:
        return f"{lead}. {NO_EVIDENCE}"
    quotes = "; ".join(f'[{c["id"]}] "{_clip(c["text"])}"' for c in cited)
    noun = "sentence" if len(cited) == 1 else "sentences"
    return f"{lead}, based on {noun} {quotes}."
