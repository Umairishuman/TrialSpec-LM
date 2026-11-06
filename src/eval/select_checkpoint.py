"""Pick the checkpoint with the best validation macro-F1 (never uses test), record it, optionally push it."""
import argparse
import json
import os
import re
import shutil
from pathlib import Path

from src.utils import git_commit, load_config, load_env

ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    load_env()
    cfg = load_config(args.config)
    run = cfg["run_name"]

    cands = []
    for f in sorted(Path("results/finetuned").glob(f"{run}_checkpoint-*_val.metrics.json")):
        step = int(re.search(r"checkpoint-(\d+)", f.name).group(1))
        m = json.loads(f.read_text())["metrics"]
        cands.append({"checkpoint": f"checkpoint-{step}", "step": step, "metrics_file": str(f),
                      "macro_f1": m["overall"]["macro_f1"], "accuracy": m["overall"]["accuracy"],
                      "inclusion_macro_f1": m["inclusion"]["macro_f1"], "exclusion_macro_f1": m["exclusion"]["macro_f1"],
                      "evidence_micro_f1": m["evidence"]["micro"]["f1"]})
    if not cands:
        raise SystemExit(f"no per-checkpoint metrics found for {run}")
    # Highest val macro-F1; ties -> higher accuracy -> earlier step.
    best = sorted(cands, key=lambda c: (-c["macro_f1"], -c["accuracy"], c["step"]))[0]

    src = Path("outputs") / run / best["checkpoint"]
    dst = Path("outputs") / run / "selected"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_FILES:
        shutil.copy2(src / name, dst / name)
    for extra in Path("outputs", run, "final").glob("*"):   # tokenizer files + training_meta
        if extra.name not in ADAPTER_FILES and extra.is_file():
            shutil.copy2(extra, dst / extra.name)

    sel = {"run": run, "criterion": "max val macro-F1 (tie: accuracy, then earlier step)", "selected": best,
           "candidates": cands, "git_commit": git_commit()}
    (dst / "selection.json").write_text(json.dumps(sel, indent=2))
    Path(f"results/finetuned/{run}_selection.json").write_text(json.dumps(sel, indent=2))
    shutil.copy2(best["metrics_file"], f"results/finetuned/{run}_val.metrics.json")
    shutil.copy2(best["metrics_file"].replace(".metrics.json", ".jsonl"), f"results/finetuned/{run}_val.jsonl")
    for c in cands:
        print(f"{c['checkpoint']:>15s} macroF1={c['macro_f1']:.3f} acc={c['accuracy']:.3f} "
              f"excl_mF1={c['exclusion_macro_f1']:.3f} evid_microF1={c['evidence_micro_f1']:.3f}"
              + ("   <-- selected" if c is best else ""))

    if args.push and cfg.get("hub_repo") and os.environ.get("HF_TOKEN"):
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(cfg["hub_repo"], exist_ok=True)
        api.upload_folder(folder_path=str(dst), repo_id=cfg["hub_repo"],
                          commit_message=f"{best['checkpoint']} selected by val macro-F1 ({git_commit()[:8]})")
        print(f"pushed selected adapter -> https://huggingface.co/{cfg['hub_repo']}")


if __name__ == "__main__":
    main()
