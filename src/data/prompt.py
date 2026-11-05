import json
import re

from src.data.load import LABELS

SYSTEM_PROMPT = (
    "You are a clinical trial eligibility screener. Given a patient note split into numbered "
    "sentences and one eligibility criterion, decide how the criterion applies to the patient. "
    "Use only information stated in the note; if the note does not say enough, answer "
    "\"not enough information\". Cite the IDs of the sentences that support your decision "
    "(use an empty list if no sentence is relevant). Respond with JSON only."
)


def format_note(sentences: list[tuple[int, str]]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in sentences)


def build_messages(sentences, criterion_type: str, criterion_text: str) -> list[dict]:
    labels = " | ".join(f'"{l}"' for l in LABELS[criterion_type])
    user = (
        f"Patient note:\n{format_note(sentences)}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text}\n\n"
        f"Allowed labels: {labels}\n"
        'Answer format: {"evidence_ids": [<sentence ids>], "label": "<one allowed label>"}'
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_target(label: str, evidence_ids: list[int]) -> str:
    # Evidence first so the label is generated conditioned on the cited sentences.
    return json.dumps({"evidence_ids": sorted(evidence_ids), "label": label})


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_output(text: str, criterion_type: str, valid_ids: set[int]) -> dict:
    out = {"parse_ok": False, "label": None, "evidence_ids": [], "invalid_ids": [], "raw": text}
    m = _JSON_RE.search(text)
    if not m:
        return out
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    label = str(obj.get("label", "")).strip().lower()
    ids = obj.get("evidence_ids", [])
    if not isinstance(ids, list):
        ids = []
    ids = [int(i) for i in ids if isinstance(i, int) or (isinstance(i, str) and i.isdigit())]
    out["evidence_ids"] = sorted({i for i in ids if i in valid_ids})
    out["invalid_ids"] = sorted({i for i in ids if i not in valid_ids})
    if label in LABELS[criterion_type]:
        out["label"] = label
        out["parse_ok"] = True
    return out
