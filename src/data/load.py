import json
import re

import pandas as pd
from datasets import load_dataset

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"

# Previous-model outputs and judgments of them; must never reach model inputs or evaluation.
LEAKY_COLUMNS = [
    "gpt4_explanation",
    "gpt4_sentences",
    "gpt4_eligibility",
    "explanation_correctness",
    "training",
]

LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}

_SENT_RE = re.compile(r"^(\d+)\.\s+(.*)$")


def parse_note(note: str) -> list[tuple[int, str]]:
    sentences = []
    for line in note.split("\n"):
        m = _SENT_RE.match(line)
        if not m:
            raise ValueError(f"Unparseable note line: {line[:80]!r}")
        sentences.append((int(m.group(1)), m.group(2).strip()))
    return sentences


def load_annotations(revision: str | None = None) -> pd.DataFrame:
    df = load_dataset(DATASET_ID, revision=revision)["train"].to_pandas()
    df = df.drop(columns=LEAKY_COLUMNS)
    df["expert_sentences"] = df["expert_sentences"].map(json.loads)
    df["sentences"] = df["note"].map(parse_note)
    return df
