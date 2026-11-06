"""Task 3 benchmark: uncached vs KV-cached greedy decoding on the same model and inputs.

Prompts are real patient-note text (train split) cut to an exact token length, so every mode sees the
same ids. EOS is ignored so each run produces exactly n_out tokens. For every config: 1 warm-up run,
then `repeats` timed runs (device-synchronised); peak memory is the CUDA allocator peak above the
weights. Token agreement with the uncached reference is checked; on a mismatch, the reference
logit margin (top-1 minus top-2) at the first differing step is recorded to tell a numerical near-tie
from a logic bug.
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.decoding.generate import greedy_hf_cache, greedy_manual, greedy_no_cache
from src.decoding.manual_decoder import ManualQwen2, kv_bytes_per_token
from src.utils import git_commit, library_versions


def build_prompt(tok, n_tokens: int, device) -> torch.Tensor:
    from src.data.dataset import load_split_frames
    text = "\n\n".join(load_split_frames()["train"]["note"].drop_duplicates())
    ids = tok(text, add_special_tokens=False).input_ids
    while len(ids) < n_tokens:
        ids = ids + ids
    return torch.tensor([ids[:n_tokens]], device=device)


@torch.no_grad()
def mismatch_info(model, prompt, ref, other) -> dict | None:
    for i, (a, b) in enumerate(zip(ref, other)):
        if a != b:
            prefix = torch.cat([prompt, torch.tensor([ref[:i]], device=prompt.device, dtype=prompt.dtype)], dim=1) if i else prompt
            logits = model(input_ids=prefix, use_cache=False).logits[0, -1].float()
            top2 = logits.topk(2)
            return {"first_mismatch_step": i, "ref_token": a, "other_token": b,
                    "ref_top1_minus_top2": (top2.values[0] - top2.values[1]).item(),
                    "ref_logit_gap_to_other": (logits[a] - logits[b]).item()}
    return None if len(ref) == len(other) else {"length_mismatch": [len(ref), len(other)]}


def run_mode(fn, device, repeats: int) -> tuple[list[int], dict]:
    fn()  # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    runs = [fn() for _ in range(repeats)]
    r0 = runs[0]
    stats = {
        "prefill_ms": statistics.median(r.prefill_s for r in runs) * 1e3,
        "prefill_ms_std": statistics.pstdev(r.prefill_s for r in runs) * 1e3,
        "decode_ms": statistics.median(r.decode_s for r in runs) * 1e3,
        "decode_ms_std": statistics.pstdev(r.decode_s for r in runs) * 1e3,
        "decode_tok_per_s": statistics.median(r.decode_tok_per_s for r in runs),
        "end_to_end_tok_per_s": statistics.median(r.n_new / (r.prefill_s + r.decode_s) for r in runs),
        "n_new": r0.n_new, "repeats": repeats,
    }
    if r0.step_s:
        k = max(1, len(r0.step_s) // 10)
        stats["first_steps_ms"] = statistics.mean(r0.step_s[:k]) * 1e3
        stats["last_steps_ms"] = statistics.mean(r0.step_s[-k:]) * 1e3
    if device == "cuda":
        stats["peak_extra_mib"] = (torch.cuda.max_memory_allocated() - base) / 2**20
    return r0.tokens, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--inputs", type=int, nargs="+", default=[128, 512, 1024])
    ap.add_argument("--outputs", type=int, nargs="+", default=[32, 128, 256])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=["no_cache", "hf_cache", "manual_cache"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/kv_cache/benchmark"))
    args = ap.parse_args()
    if args.smoke:
        args.model, args.inputs, args.outputs, args.repeats = "Qwen/Qwen2.5-0.5B-Instruct", [16, 48], [4, 8], 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
    dec = ManualQwen2(model)
    weights_mib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20

    modes = {
        "no_cache": lambda p, n: greedy_no_cache(model, p, n),
        "hf_cache": lambda p, n: greedy_hf_cache(model, p, n),
        "manual_cache": lambda p, n: greedy_manual(dec, p, n),
        "manual_no_cache": lambda p, n: greedy_manual(dec, p, n, use_cache=False),
    }
    rows = []
    for n_in in args.inputs:
        prompt = build_prompt(tok, n_in, device)
        for n_out in args.outputs:
            ref = None
            for mode in args.modes:
                tokens, stats = run_mode(lambda: modes[mode](prompt, n_out), device, args.repeats)
                if ref is None:
                    ref = tokens
                row = {"mode": mode, "input_len": n_in, "output_len": n_out, **stats,
                       "tokens_equal_to_first_mode": tokens == ref,
                       "kv_cache_mib_at_end": kv_bytes_per_token(model.config, torch.finfo(dtype).bits // 8) * (n_in + n_out) / 2**20
                       if "cache" in mode and "no_cache" not in mode else 0.0}
                if tokens != ref:
                    row["mismatch"] = mismatch_info(model, prompt, ref, tokens)
                rows.append(row)
                print(f"{mode:15s} in={n_in:5d} out={n_out:4d} prefill={stats['prefill_ms']:8.1f}ms "
                      f"decode={stats['decode_ms']:9.1f}ms tok/s={stats['decode_tok_per_s']:7.1f} "
                      f"equal={row['tokens_equal_to_first_mode']}"
                      + (f" peak+={stats['peak_extra_mib']:.0f}MiB" if "peak_extra_mib" in stats else ""), flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()

    meta = {"model": args.model, "adapter": args.adapter, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "dtype": str(dtype), "decoding": "greedy, EOS ignored", "reference_mode": args.modes[0],
            "weights_mib": weights_mib, "kv_bytes_per_token": kv_bytes_per_token(model.config, torch.finfo(dtype).bits // 8),
            "smoke": args.smoke, "git_commit": git_commit(), "versions": library_versions(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    tag = "smoke" if args.smoke else f"{device}_{str(dtype).replace('torch.', '')}"
    out = Path(f"{args.out}_{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print(f"wrote {out}")
    if not all(r["tokens_equal_to_first_mode"] for r in rows):
        print("NOTE: token mismatches found; see 'mismatch' entries (logit margins) in the JSON.")


if __name__ == "__main__":
    main()
