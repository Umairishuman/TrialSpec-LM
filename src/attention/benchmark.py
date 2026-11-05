"""Numerical error + runtime/memory benchmark: reference vs tiled online-softmax vs PyTorch SDPA.

Errors are measured against a float64 reference (ground truth). Timing uses warm-up, device
synchronisation and repeated runs (median and std). Peak memory is measured with the CUDA allocator;
on CPU, the analytic size of the largest score buffer is reported instead.
"""
import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from src.attention.attention import reference_attention, tiled_attention
from src.utils import git_commit, library_versions

FULL_SHAPES = [(1, 12, L, L, 128) for L in (128, 512, 1024, 2048, 4096)] + [(8, 12, 1, 2048, 128)]
SMOKE_SHAPES = [(1, 2, 64, 64, 32), (1, 2, 1, 64, 32)]


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def bench(fn, device, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1e3)
    res = {"median_ms": statistics.median(times), "std_ms": statistics.pstdev(times), "repeats": repeats}
    if device == "cuda":
        res["peak_extra_mib"] = (torch.cuda.max_memory_allocated() - base) / 2**20
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dtype", choices=["float32", "float16"], default=None)
    ap.add_argument("--block", type=int, nargs=2, default=[64, 64], metavar=("BQ", "BK"))
    ap.add_argument("--out", type=Path, default=Path("results/attention/benchmark"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype) if args.dtype else (torch.float16 if device == "cuda" else torch.float32)
    shapes = SMOKE_SHAPES if args.smoke else FULL_SHAPES
    warmup, repeats = (1, 3) if args.smoke else (3, 10)
    bq, bk = args.block
    rows = []

    for B, H, Lq, Lk, D in shapes:
        for causal in (False, True):
            g = torch.Generator(device="cpu").manual_seed(0)
            q64, k64, v64 = (torch.randn((B, H, L, D), generator=g, dtype=torch.float64) for L in (Lq, Lk, Lk))
            truth = reference_attention(q64, k64, v64, causal)
            q, k, v = (t.to(device=device, dtype=dtype) for t in (q64, k64, v64))
            mask = None
            if causal and Lq != Lk:
                from src.attention.attention import causal_mask
                mask = ~causal_mask(Lq, Lk, device)

            impls = {
                "reference": lambda: reference_attention(q, k, v, causal),
                "tiled": lambda: tiled_attention(q, k, v, causal, block_q=bq, block_k=bk),
                "torch_sdpa": lambda: F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, is_causal=causal and Lq == Lk),
            }
            for name, fn in impls.items():
                err = (fn().double().cpu() - truth).abs()
                r = {"impl": name, "B": B, "H": H, "Lq": Lq, "Lk": Lk, "D": D, "causal": causal,
                     "dtype": str(dtype).replace("torch.", ""), "max_abs_err": err.max().item(),
                     "mean_abs_err": err.mean().item(), **bench(fn, device, warmup, repeats)}
                score_elems = B * H * (Lq * Lk if name == "reference" else min(bq, Lq) * min(bk, Lk))
                if name != "torch_sdpa":
                    r["score_buffer_mib_fp32"] = score_elems * 4 / 2**20
                rows.append(r)
                print(f"{name:10s} B={B} H={H} Lq={Lq:4d} Lk={Lk:4d} causal={causal!s:5s} "
                      f"max_err={r['max_abs_err']:.2e} mean_err={r['mean_abs_err']:.2e} "
                      f"t={r['median_ms']:8.2f}±{r['std_ms']:.2f} ms"
                      + (f" peak={r['peak_extra_mib']:.1f} MiB" if "peak_extra_mib" in r else ""), flush=True)
            del q, k, v
            if device == "cuda":
                torch.cuda.empty_cache()

    meta = {"device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "dtype": str(dtype),
            "block_q": bq, "block_k": bk, "warmup": warmup, "smoke": args.smoke, "git_commit": git_commit(),
            "versions": library_versions(), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sdpa_backends": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                              "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                              "math": torch.backends.cuda.math_sdp_enabled()} if device == "cuda" else None}
    tag = "smoke" if args.smoke else f"{device}_{str(dtype).replace('torch.', '')}"
    out = Path(f"{args.out}_{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
