"""Final evaluation (Task 2, protocol D-024).

Stage A: zero-shot baselines and every fine-tuned candidate on VAL, pure greedy.
Selection: highest val macro-F1 among candidates (tie: val accuracy, then config order).
Stage B: the selected model and the zero-shot baseline with the same prompt on TEST, once.
Then: confusion matrices, prediction CSVs, error analysis and summary.md.

  python -m src.eval.final_eval --config configs/final_eval.yaml --smoke   # 4 val examples per model, no test
  python -m src.eval.final_eval --config configs/final_eval.yaml           # the real thing (test used once)
"""
import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from src.data.dataset import load_split_frames
from src.eval.compare import row as compare_row
from src.eval.error_analysis import write_error_analysis
from src.eval.plots import confusion_figure
from src.eval.predict import evaluate_and_write, load_model
from src.utils import git_commit, load_config, load_env

LOCK_NAME = "test_lock.json"


def select_best(results: dict[str, dict], order: list[str]) -> str:
    """results: name -> {"meta", "metrics"} on val. Max macro-F1, then accuracy, then earlier in `order`."""
    def key(name):
        o = results[name]["metrics"]["overall"]
        return (-o["macro_f1"], -o["accuracy"], order.index(name))
    return sorted(order, key=key)[0]


def check_test_lock(out_dir: Path, allow_rerun: bool):
    lock = out_dir / LOCK_NAME
    if lock.exists() and not allow_rerun:
        raise SystemExit(f"{lock} exists: the test split was already evaluated "
                         f"({json.loads(lock.read_text()).get('created_utc')}). Refusing to run test again. "
                         "Pass --allow-test-rerun only with a new decision-log entry explaining why.")


def write_csv(jsonl: Path, csv: Path):
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    df = pd.DataFrame([{
        "annotation_id": r["annotation_id"], "patient_id": r["patient_id"], "trial_id": r["trial_id"],
        "criterion_type": r["criterion_type"], "criterion_text": r["criterion_text"],
        "gold_label": r["gold_label"], "pred_label": r["pred_label"], "correct": r["gold_label"] == r["pred_label"],
        "gold_evidence": r["gold_evidence"], "pred_evidence": r["pred_evidence"], "invalid_ids": r["invalid_ids"],
        "pred_evidence_text": " | ".join(f"[{c['id']}] {c['text']}" for c in r.get("pred_evidence_text", [])),
        "explanation": r.get("explanation", ""), "parse_ok": r["parse_ok"],
    } for r in rows])
    df.to_csv(csv, index=False)


def per_class_table(rep: dict) -> list[str]:
    lines = ["| Type | Label | Precision | Recall | F1 | Support |", "|---|---|---|---|---|---|"]
    for part in ["inclusion", "exclusion"]:
        for label, s in rep.get(part, {}).get("per_class", {}).items():
            if s["support"]:
                lines.append(f"| {part} | {label} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['support']} |")
            else:
                lines.append(f"| {part} | {label} | N/A | N/A | N/A | 0 |")
    return lines


def evidence_table(results: dict[str, Path]) -> list[str]:
    lines = ["| Model | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | Gold empty | Pred empty | Both empty | Invalid IDs |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, f in results.items():
        e = json.loads(f.read_text())["metrics"]["evidence"]
        lines.append(f"| {name} | {e['micro']['precision']:.3f} | {e['micro']['recall']:.3f} | {e['micro']['f1']:.3f} | "
                     f"{e['macro']['precision']:.3f} | {e['macro']['recall']:.3f} | {e['macro']['f1']:.3f} | "
                     f"{e['n_gold_empty']} | {e['n_pred_empty']} | {e['n_both_empty']} | {e.get('n_invalid_ids', 0)} |")
    return lines


def write_summary(out_dir: Path, files: dict[str, dict[str, Path]], selected: str, baseline: str, smoke: bool):
    head = "| Model | Prompt | Acc | Macro-F1 | Incl. mF1 | Excl. mF1 | Evid. micro-F1 | Evid. macro-F1 | Parse OK | Commit |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    md = ["# Final evaluation summary" + (" (SMOKE: 4 val examples per model)" if smoke else ""), "",
          f"Protocol: docs/decisions.md D-024. Pure greedy (repetition_penalty 1.0). "
          f"Selected by val macro-F1: **{selected}**; zero-shot counterpart: **{baseline}**.", ""]
    for split in ["val", "test"]:
        if not files.get(split):
            continue
        md += [f"## {split.capitalize()}: labels", "", head, sep]
        md += [compare_row(n, f) for n, f in files[split].items()]
        md += ["", f"## {split.capitalize()}: evidence (set-based over sentence IDs; both-empty records score 1 in macro, nothing in micro)", ""]
        md += evidence_table(files[split])
        for n in [baseline, selected]:
            if n in files[split]:
                rep = json.loads(files[split][n].read_text())["metrics"]
                md += ["", f"## {split.capitalize()}: per-class, {n}", ""] + per_class_table(rep)
        md.append("")
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final_eval.yaml")
    ap.add_argument("--smoke", action="store_true", help="4 val examples per model; never touches test")
    ap.add_argument("--allow-test-rerun", action="store_true")
    args = ap.parse_args()
    load_env()
    cfg = load_config(args.config)
    dec = cfg["decoding"]
    out_dir = Path("results/smoke/final") if args.smoke else Path(cfg["out_dir"])
    if not args.smoke:
        check_test_lock(Path(cfg["out_dir"]), args.allow_test_rerun)   # fail before any GPU work
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    frames = load_split_frames()
    models = {m["name"]: m for m in cfg["baselines"] + cfg["candidates"]}
    cand_names = [m["name"] for m in cfg["candidates"]]

    def run(name: str, split: str) -> Path:
        m = models[name]
        frame = frames[split].head(4) if args.smoke else frames[split]
        print(f"\n===== {name} on {split} (adapter={m['adapter']}, prompt={m['prompt_version']}, n={len(frame)}) =====", flush=True)
        tok, model = load_model(cfg["model_id"], m["adapter"], dtype, device)
        out = out_dir / f"{name}_{split}"
        evaluate_and_write(tok, model, frame, out, model_id=cfg["model_id"], adapter=m["adapter"], split=split,
                           prompt_version=m["prompt_version"], batch_size=dec["batch_size"],
                           max_new_tokens=dec["max_new_tokens"], repetition_penalty=dec["repetition_penalty"],
                           dtype=dtype, extra_meta={"final_eval_name": name})
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        return Path(f"{out}.metrics.json")

    files: dict[str, dict[str, Path]] = {"val": {}, "test": {}}
    for name in models:                                                 # Stage A
        files["val"][name] = run(name, "val")

    val = {n: json.loads(files["val"][n].read_text()) for n in cand_names}
    selected = select_best(val, cand_names)
    pv = models[selected]["prompt_version"]
    baseline = next(b["name"] for b in cfg["baselines"] if b["prompt_version"] == pv)
    selection = {
        "criterion": "max val macro-F1 at pure greedy (tie: val accuracy, then config order); D-024",
        "selected": selected, "adapter": models[selected]["adapter"], "baseline": baseline,
        "val": {n: {k: v["metrics"]["overall"][k] for k in ("macro_f1", "accuracy")} for n, v in val.items()},
        "git_commit": git_commit(), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    print(f"\nselected {selected} ({models[selected]['adapter']}); baseline {baseline}")

    if not args.smoke:                                                  # Stage B
        print("\nWARNING: evaluating on the held-out TEST split. This happens once (D-024).")
        for name in [baseline, selected]:
            files["test"][name] = run(name, "test")
        (out_dir / LOCK_NAME).write_text(json.dumps({
            "selected": selected, "baseline": baseline, "git_commit": git_commit(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2))

    for split, fs in files.items():                                     # figures, CSVs, errors
        for name, f in fs.items():
            stem = str(f).replace(".metrics.json", "")
            confusion_figure(f, Path(f"{stem}_confusion.png"), f"{name}, {split}")
            write_csv(Path(f"{stem}.jsonl"), Path(f"{stem}.csv"))
        if selected in fs:
            write_error_analysis(Path(str(fs[selected]).replace(".metrics.json", ".jsonl")), split,
                                 out_dir / f"error_analysis_{split}.md", cfg.get("error_cases", 12))
    write_summary(out_dir, files, selected, baseline, args.smoke)
    print(f"\n===== FINAL EVALUATION {'SMOKE ' if args.smoke else ''}FINISHED: see {out_dir}/summary.md =====")


if __name__ == "__main__":
    main()
