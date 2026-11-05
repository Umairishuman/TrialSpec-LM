"""Prescribed patient-grouped split: sorted IDs, default_rng(42) shuffle, 37/8/8."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load import DATASET_ID, load_annotations

SEED = 42
SIZES = {"train": 37, "val": 8, "test": 8}
MANIFEST_PATH = Path("data/splits/split_manifest.json")


def make_patient_split(patient_ids, seed: int = SEED) -> dict[str, list[str]]:
    ids = np.array(sorted(set(patient_ids)))
    expected = sum(SIZES.values())
    if len(ids) != expected:
        raise ValueError(f"Expected {expected} unique patients, found {len(ids)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    out, start = {}, 0
    for name, n in SIZES.items():
        out[name] = ids[start:start + n].tolist()
        start += n
    return out


def assign_splits(df: pd.DataFrame, split: dict[str, list[str]]) -> pd.DataFrame:
    lookup = {pid: name for name, pids in split.items() for pid in pids}
    return df.assign(split=df["patient_id"].map(lookup))


def build_manifest(df: pd.DataFrame, split: dict[str, list[str]]) -> dict:
    df = assign_splits(df, split)
    splits = {}
    for name, pids in split.items():
        part = df[df["split"] == name]
        splits[name] = {
            "patient_ids": pids,
            "n_patients": len(pids),
            "n_annotations": int(len(part)),
            "n_annotations_per_patient": {p: int(c) for p, c in part["patient_id"].value_counts().sort_index().items()},
            "label_counts": {
                ctype: {k: int(v) for k, v in g["expert_eligibility"].value_counts().sort_index().items()}
                for ctype, g in part.groupby("criterion_type")
            },
            "n_empty_gold_evidence": int((part["expert_sentences"].map(len) == 0).sum()),
        }
    return {
        "dataset": DATASET_ID,
        "procedure": "sorted unique patient_id -> numpy default_rng(42).shuffle -> first 37 train, next 8 val, last 8 test",
        "seed": SEED,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_patients": sum(s["n_patients"] for s in splits.values()),
        "total_annotations": int(len(df)),
        "splits": splits,
    }


def load_split(path: Path = MANIFEST_PATH) -> dict[str, list[str]]:
    manifest = json.loads(Path(path).read_text())
    return {name: s["patient_ids"] for name, s in manifest["splits"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=MANIFEST_PATH)
    args = ap.parse_args()

    df = load_annotations()
    split = make_patient_split(df["patient_id"])
    manifest = build_manifest(df, split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    for name, s in manifest["splits"].items():
        print(f"{name:5s} patients={s['n_patients']:2d} annotations={s['n_annotations']:4d} empty_evidence={s['n_empty_gold_evidence']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
