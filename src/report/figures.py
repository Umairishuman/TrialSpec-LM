"""Report figures drawn from the committed benchmark/evaluation results (no GPU, no network).

  python -m src.report.figures            # writes report/figures/*.png
  python -m src.report.figures --smoke    # draws into a temp folder and checks every input exists
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("report/figures")
KV = Path("results/kv_cache/benchmark_cuda_float16.json")
SPEC = Path("results/speculative/benchmark_cuda_float16.json")
ATT = Path("results/attention/benchmark_cuda_float16.json")
COPIES = {
    "fig_training_curves.png": Path("results/training/wandb_export/training_curves.png"),
    "fig_confusion_run4_test.png": Path("results/final/run4_test_confusion.png"),
    "fig_confusion_zeroshot_test.png": Path("results/final/zeroshot-v1_test_confusion.png"),
}
NAMES = {"no_cache": "uncached", "hf_cache": "HF cache loop", "manual_cache": "hand-written cache",
         "reference": "reference SDPA", "tiled": "tiled online softmax", "torch_sdpa": "PyTorch SDPA (fused)"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "figure.dpi": 200})


def kv_figure(out: Path):
    rows = json.loads(KV.read_text())["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for mode in ("no_cache", "hf_cache", "manual_cache"):
        r = sorted((x for x in rows if x["mode"] == mode and x["output_len"] == 256), key=lambda x: x["input_len"])
        ax[0].plot([x["input_len"] for x in r], [x["decode_tok_per_s"] for x in r], marker="o", label=NAMES[mode])
        ax[1].plot([x["input_len"] for x in r], [x["peak_extra_mib"] for x in r], marker="o", label=NAMES[mode])
    ax[0].set(title="Decode throughput (256 new tokens)", xlabel="prompt length (tokens)", ylabel="tokens / s")
    ax[1].set(title="Peak extra GPU memory", xlabel="prompt length (tokens)", ylabel="MiB")
    for a in ax:
        a.grid(alpha=.3); a.set_xticks([128, 512, 1024])
    ax[0].legend()
    fig.tight_layout(); fig.savefig(out / "fig_kv_cache.png"); plt.close(fig)


def spec_figure(out: Path):
    s = json.loads(SPEC.read_text())["summary"]
    ks = [2, 4, 6, 8]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for name, label in (("task", "task prompts (JSON answer)"), ("free", "free-text prompts")):
        ax[0].plot(ks, [s[f"{name}_k{k}"]["acceptance_rate"] for k in ks], marker="o", label=label)
        ax[1].plot(ks, [s[f"{name}_k{k}"]["mean_speedup"] for k in ks], marker="o", label=label)
    ax[1].axhline(1.0, color="grey", ls="--", lw=.8)
    ax[0].set(title="Draft acceptance rate", xlabel="draft length k", ylim=(0, 1))
    ax[1].set(title="Speed-up vs target greedy (T4, fp16)", xlabel="draft length k", ylim=(0, 1.1))
    for a in ax:
        a.grid(alpha=.3); a.set_xticks(ks)
    ax[0].legend()
    fig.tight_layout(); fig.savefig(out / "fig_speculative.png"); plt.close(fig)


def attention_figure(out: Path):
    rows = [r for r in json.loads(ATT.read_text())["rows"] if r["B"] == 1 and r["causal"]]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for impl in ("reference", "tiled", "torch_sdpa"):
        r = sorted((x for x in rows if x["impl"] == impl), key=lambda x: x["Lq"])
        ax[0].plot([x["Lq"] for x in r], [x["median_ms"] for x in r], marker="o", label=NAMES[impl])
        ax[1].plot([x["Lq"] for x in r], [max(x["peak_extra_mib"], 0.1) for x in r], marker="o", label=NAMES[impl])
    ax[0].set(title="Causal attention time (B=1, H=12, D=128, fp16)", xlabel="sequence length", ylabel="median ms")
    ax[1].set(title="Peak extra GPU memory", xlabel="sequence length", ylabel="MiB")
    for a in ax:
        a.set_xscale("log", base=2); a.set_yscale("log"); a.grid(alpha=.3, which="both")
    ax[0].legend()
    fig.tight_layout(); fig.savefig(out / "fig_attention.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--smoke", action="store_true", help="draw into a temporary folder only")
    args = ap.parse_args()
    for p in [KV, SPEC, ATT, *COPIES.values()]:
        assert p.exists(), f"missing input {p}"
    out = Path(tempfile.mkdtemp()) if args.smoke else args.out
    out.mkdir(parents=True, exist_ok=True)
    kv_figure(out); spec_figure(out); attention_figure(out)
    for name, src in COPIES.items():
        shutil.copyfile(src, out / name)
    print("wrote", sorted(p.name for p in out.glob("*.png")), "to", out)


if __name__ == "__main__":
    main()
