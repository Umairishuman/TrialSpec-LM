"""Confusion-matrix figures (inclusion and exclusion side by side) from a *.metrics.json file."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SHORT = {"included": "incl.", "not included": "not incl.", "excluded": "excl.", "not excluded": "not excl.",
         "not enough information": "NEI", "not applicable": "N/A", "<invalid>": "invalid"}


def _panel(ax, block: dict, title: str):
    labels = block["confusion_matrix"]["labels"]
    m = np.array(block["confusion_matrix"]["matrix"])
    keep_cols = [j for j, l in enumerate(labels) if l != "<invalid>" or m[:, j].sum() > 0]
    rows = [i for i, l in enumerate(labels) if l != "<invalid>"]   # gold is never invalid
    m = m[np.ix_(rows, keep_cols)]
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(keep_cols)), [SHORT[labels[j]] for j in keep_cols], rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), [SHORT[labels[i]] for i in rows])
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(f"{title} (n={block['n']}, macro-F1 {block['macro_f1']:.2f})", fontsize=10)
    hi = m.max() if m.size else 0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if hi and m[i, j] > hi / 2 else "black")


def confusion_figure(metrics_file: Path, out: Path, title: str | None = None) -> Path:
    data = json.loads(Path(metrics_file).read_text())
    rep = data["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, part in zip(axes, ["inclusion", "exclusion"]):
        if part in rep:
            _panel(ax, rep[part], part.capitalize())
        else:
            ax.axis("off")
    fig.suptitle(title or Path(metrics_file).name.replace(".metrics.json", ""), fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", type=Path, nargs="+")
    args = ap.parse_args()
    for f in args.metrics:
        print(confusion_figure(f, Path(str(f).replace(".metrics.json", "_confusion.png"))))


if __name__ == "__main__":
    main()
