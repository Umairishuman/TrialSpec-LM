"""Categorise prediction errors and write a Markdown case file for the report (Task 2, C4; D-024).

Usage: python -m src.eval.error_analysis results/final/run1_test.jsonl --split test [--n 12]
The "why it went wrong" analysis is written by hand from the printed cases; this script only finds,
categorises and displays them deterministically.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

NEI = "not enough information"
NA = "not applicable"
POLARITY = {("included", "not included"), ("not included", "included"),
            ("excluded", "not excluded"), ("not excluded", "excluded")}

# Display order, which is also the round-robin order used for sampling.
CATEGORIES = {
    "polarity_flip": "Contradictory label: the opposite decision (e.g. included vs not included)",
    "missed_information": "Gold says the note is insufficient, but the model made a definite decision",
    "spurious_nei": "The model claimed missing information, but the experts made a definite decision",
    "applicability": "Confusion involving \"not applicable\"",
    "unsupported_evidence": "Correct label, but the cited evidence does not support it (no overlap with the gold sentences, or citations where gold cites none)",
    "hallucinated_ids": "Cited sentence IDs that do not exist in the note",
    "parse_failure": "Output was not valid JSON with an allowed label",
}


def categories(r: dict) -> list[str]:
    """All error categories a prediction record falls into (empty list = fully correct)."""
    out = []
    g, p = r["gold_label"], r["pred_label"]
    if not r.get("parse_ok", p is not None) or p is None:
        out.append("parse_failure")
    elif p != g:
        if (g, p) in POLARITY:
            out.append("polarity_flip")
        elif g == NEI:
            out.append("missed_information")
        elif p == NEI:
            out.append("spurious_nei")
        else:   # with 4 labels per type, every remaining confusion involves "not applicable"
            out.append("applicability")
    else:
        ge, pe = set(r["gold_evidence"]), set(r["pred_evidence"])
        if (ge and not ge & pe) or (not ge and pe):
            out.append("unsupported_evidence")
    if r.get("invalid_ids"):
        out.append("hallucinated_ids")
    return out


def sample_cases(records: list[dict], n: int) -> list[tuple[str, dict]]:
    """Round-robin over categories (ordered by annotation_id within each) until n distinct records are taken."""
    pools = {c: sorted((r for r in records if c in categories(r)), key=lambda r: r["annotation_id"])
             for c in CATEGORIES}
    taken, seen = [], set()
    while len(taken) < n and any(pools.values()):
        for c in CATEGORIES:
            while pools[c] and pools[c][0]["annotation_id"] in seen:
                pools[c].pop(0)
            if pools[c] and len(taken) < n:
                r = pools[c].pop(0)
                seen.add(r["annotation_id"])
                taken.append((c, r))
    return taken


def _sent_lines(sentences: list[tuple[int, str]], gold: set[int], pred: set[int]) -> list[str]:
    shown = sorted(gold | pred) or [i for i, _ in sentences[:3]]
    lookup = dict(sentences)
    lines = []
    for i in shown:
        tag = ("G" if i in gold else "-") + ("P" if i in pred else "-")
        lines.append(f"    [{tag}] {i}. {lookup.get(i, '<not in note>')}")
    if not gold | pred:
        lines.insert(0, "    (neither side cites a sentence; first sentences of the note shown)")
    return lines


def render(records: list[dict], sentences_by_id: dict[int, list], title: str, n: int) -> str:
    counts = Counter(c for r in records for c in categories(r))
    n_wrong = sum(r["pred_label"] != r["gold_label"] for r in records)
    md = [f"# Error analysis: {title}", "",
          f"{len(records)} predictions, {n_wrong} wrong labels "
          f"({n_wrong / max(len(records), 1):.1%}). One record can fall into several categories.", "",
          "| Category | Count | Meaning |", "|---|---|---|"]
    md += [f"| {c} | {counts.get(c, 0)} | {d} |" for c, d in CATEGORIES.items()]
    md += ["", "Sentence tags: `G` = cited by the experts, `P` = cited by the model.", ""]
    for k, (c, r) in enumerate(sample_cases(records, n), 1):
        sents = sentences_by_id.get(r["annotation_id"], [])
        md += [f"## Case {k}: {c} (annotation {r['annotation_id']}, {r['patient_id']}, {r['trial_id']})", "",
               f"- **Criterion ({r['criterion_type']}):** {r['criterion_text']}",
               f"- **Gold:** `{r['gold_label']}`, evidence {sorted(r['gold_evidence'])}",
               f"- **Predicted:** `{r['pred_label']}`, evidence {sorted(r['pred_evidence'])}"
               + (f", invalid IDs {r['invalid_ids']}" if r.get("invalid_ids") else ""),
               f"- **All categories:** {', '.join(categories(r))}",
               f"- **Raw output:** `{r.get('raw', '').strip()}`",
               "- **Note snippet:**", "", "```", *_sent_lines(sents, set(r["gold_evidence"]), set(r["pred_evidence"])),
               "```", "", "- **Why it went wrong:** _to be written from this case_", ""]
    if not counts:
        md.append("No errors found.")
    return "\n".join(md)


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_error_analysis(pred_file: Path, split: str, out: Path, n: int = 12) -> Path:
    from src.data.dataset import load_split_frames
    frame = load_split_frames()[split]
    sentences_by_id = {int(r["annotation_id"]): r["sentences"] for _, r in frame.iterrows()}
    records = load_records(pred_file)
    out.write_text(render(records, sentences_by_id, f"{Path(pred_file).name} ({split})", n), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path, help="*.jsonl written by src.eval.predict / final_eval")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--n", type=int, default=12, help="cases to print (the rubric asks for at least 8)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(str(args.predictions).replace(".jsonl", "_errors.md"))
    print(f"wrote {write_error_analysis(args.predictions, args.split, out, args.n)}")


if __name__ == "__main__":
    main()
