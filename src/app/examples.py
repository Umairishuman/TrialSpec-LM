"""Example trials for the app's "load example" menu, exported from the VAL and TRAIN splits only (D-028).

  python -m src.app.examples            # writes data/app_examples.json
  python -m src.app.examples --smoke    # prints the selection, writes nothing
An example = one (patient, trial) pair: the note in its original numbered format, its inclusion and exclusion
criteria, and the expert labels (shown in the app only for comparison). Test patients are never used.
"""
import argparse
import json
from pathlib import Path

from src.app.text import parse_criteria
from src.data.split import load_split

OUT = Path("data/app_examples.json")


def test_patients() -> set[str]:
    return set(load_split()["test"])


def check_no_test(examples: list[dict]) -> None:
    bad = sorted({e["patient_id"] for e in examples} & test_patients())
    if bad:
        raise ValueError(f"app examples contain TEST-split patients {bad}; examples must come from train/val only")


def _norm(text: str) -> str:
    items, _ = parse_criteria(text.replace("\n", " "))
    return items[0] if items else ""


def build(per_split: dict[str, int], min_crit: int = 4, max_crit: int = 12) -> list[dict]:
    from src.data.dataset import load_split_frames
    frames = load_split_frames()
    out = []
    for split, n in per_split.items():
        assert split in ("train", "val"), "examples must come from train or val"
        cands = []
        for (pid, tid), g in frames[split].groupby(["patient_id", "trial_id"], sort=True):
            types = set(g["criterion_type"])
            if not (min_crit <= len(g) <= max_crit and types == {"inclusion", "exclusion"}):
                continue
            has_nei = (g["expert_eligibility"] == "not enough information").any()
            n_labels = g["expert_eligibility"].nunique()
            cands.append((not has_nei, -n_labels, pid, tid, g))     # prefer NEI cases and label variety
        cands.sort(key=lambda c: c[:4])
        used_patients = set()
        for _, _, pid, tid, g in cands:
            if len([e for e in out if e["split"] == split]) >= n:
                break
            if pid in used_patients:                                  # one trial per patient for variety
                continue
            used_patients.add(pid)
            crit = [{"type": r["criterion_type"], "text": _norm(r["criterion_text"]), "label": r["expert_eligibility"],
                     "evidence": [int(i) for i in r["expert_sentences"]]} for _, r in g.iterrows()]
            crit = [c for c in crit if c["text"]]
            out.append({"name": f"{split}: patient {pid} × trial {tid} ({len(crit)} criteria)",
                        "split": split, "patient_id": pid, "trial_id": tid, "note": g["note"].iloc[0],
                        "inclusion": [c["text"] for c in crit if c["type"] == "inclusion"],
                        "exclusion": [c["text"] for c in crit if c["type"] == "exclusion"],
                        "expert": crit})
    check_no_test(out)
    return out


def load_examples(path: Path | str = OUT) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    examples = json.loads(p.read_text(encoding="utf-8"))
    check_no_test(examples)
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=2)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--smoke", action="store_true", help="1 val example, print only")
    args = ap.parse_args()
    ex = build({"val": 1} if args.smoke else {"val": args.n_val, "train": args.n_train})
    for e in ex:
        labels = [c["label"] for c in e["expert"]]
        print(f"{e['name']}: {len(e['inclusion'])} incl / {len(e['exclusion'])} excl; labels {sorted(set(labels))}")
    if not args.smoke:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(ex, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out} ({len(ex)} examples)")


if __name__ == "__main__":
    main()
