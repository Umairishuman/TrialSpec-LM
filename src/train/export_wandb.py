"""Download training histories from W&B and draw the report's training curves (D-003, D-025).

Needs WANDB_API_KEY in the environment (never in code or files):
  python -m src.train.export_wandb                 # all non-smoke runs of the project
Writes results/training/wandb_export/<run>_history.csv, runs.json (ids, URLs, configs) and curve PNGs.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# W&B run name -> short label used in the report (run 1 predates per-run names).
SHORT = {"qwen2.5-1.5b-lora": "run1 (v1)", "run2-v2prompt-oversample": "run2 (v2 + oversample)",
         "run3-v2prompt-f1select": "run3 (v2)", "run4-v1prompt-oversample-f1select": "run4 (v1 + oversample)"}
PANELS = [("train/loss", "Training loss"), ("eval/loss", "Validation loss"),
          ("train/learning_rate", "Learning rate"), ("train/grad_norm", "Gradient norm")]


def export(project: str, out: Path) -> list[dict]:
    import wandb
    api = wandb.Api()
    entity = api.default_entity
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for r in api.runs(f"{entity}/{project}"):
        if r.group == "smoke" or r.name.startswith("smoke-"):
            continue
        keys = ["train/global_step", "train/epoch"] + [k for k, _ in PANELS]
        hist = pd.DataFrame(list(r.scan_history(keys=None)))
        hist = hist[[c for c in hist.columns if c in keys or c.startswith(("train/", "eval/"))]]
        hist.to_csv(out / f"{r.name}_history.csv", index=False)
        runs.append({"name": r.name, "id": r.id, "state": r.state, "created_at": r.created_at,
                     "url": f"https://wandb.ai/{entity}/{project}/runs/{r.id}", "rows": len(hist),
                     "config": {k: v for k, v in r.config.items() if not k.startswith("_")}})
        print(f"{r.name:40s} {r.state:9s} {len(hist):4d} rows  {runs[-1]['url']}")
    (out / "runs.json").write_text(json.dumps(runs, indent=2, default=str))
    return runs


def plot(out: Path, runs: list[dict]):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (key, title) in zip(axes.flat, PANELS):
        for r in runs:
            h = pd.read_csv(out / f"{r['name']}_history.csv")
            if key not in h or "train/global_step" not in h:
                continue
            d = h[["train/global_step", key]].dropna()
            style = dict(marker="o") if key == "eval/loss" else {}
            ax.plot(d["train/global_step"], d[key], label=SHORT.get(r["name"], r["name"]), **style)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out / 'training_curves.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="trialspec-lm")
    ap.add_argument("--out", type=Path, default=Path("results/training/wandb_export"))
    args = ap.parse_args()
    plot(args.out, export(args.project, args.out))


if __name__ == "__main__":
    main()
