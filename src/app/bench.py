"""Live decoding benchmark for the app (rubric G3): the Task 3 and Task 5 decoders on one prompt.

Modes (all greedy, same prompt, same token budget):
  uncached      greedy_no_cache: full forward over the whole sequence every step (reference)
  hf_cache      greedy_hf_cache: HF forward with an explicitly passed and grown DynamicCache
  manual_cache  greedy_manual: hand-written Qwen2 decoder with our own preallocated KV cache
  speculative   speculative_greedy: 0.5B draft proposes k tokens, target verifies in one pass
Each mode: optional short warm-up (2 tokens), then `repeats` timed runs; the median run is reported.
"""
import statistics
import time

import torch

from src.decoding.generate import greedy_hf_cache, greedy_manual, greedy_no_cache
from src.decoding.speculative import speculative_greedy

MODES = {
    "uncached": "Uncached reference",
    "hf_cache": "KV cache (HF DynamicCache loop)",
    "manual_cache": "KV cache (hand-written decoder)",
    "speculative": "Speculative decoding",
}

# Kaggle T4 reference numbers (docs/runs.md attempt 9), shown only as a labelled reference next to live runs.
T4_REFERENCE = [
    ("Manual KV cache vs uncached, 1024 in + 256 out", "2.8 → 25.9 tok/s (9.3×), tokens identical"),
    ("Manual KV cache vs uncached, 128 in + 32 out", "23.7 → 25.4 tok/s (1.07×)"),
    ("Speculative k=4 vs target greedy, task prompts", "24.2 → 17.8 tok/s (0.74×), acceptance 0.79, 8/8 identical"),
    ("Speculative k=6 vs target greedy, task prompts", "0.80×, acceptance 0.83, 8/8 identical"),
]


def chat_prompt(tok, text: str) -> str:
    """Wrap free text as a user turn (custom benchmark prompt)."""
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)


def _runner(backend, mode: str, ids, n: int, k: int, eos):
    if mode == "uncached":
        return lambda m=n: (greedy_no_cache(backend.model, ids, m, eos), None)
    if mode == "hf_cache":
        return lambda m=n: (greedy_hf_cache(backend.model, ids, m, eos), None)
    if mode == "manual_cache":
        return lambda m=n: (greedy_manual(backend.target_dec, ids, m, eos), None)
    if mode == "speculative":
        return lambda m=n: speculative_greedy(backend.target_dec, backend.draft_dec, ids, m, k=k, eos_ids=eos)
    raise ValueError(f"unknown mode {mode!r}")


def run_benchmark(backend, prompt: str, modes: list[str], max_new_tokens: int, k: int = 4, warmup: bool = True,
                  repeats: int = 1, stop_at_eos: bool = True, progress=None) -> dict:
    if not modes:
        raise ValueError("Select at least one decoding mode.")
    ids = backend.encode(prompt)
    T = ids.shape[1]
    eos = backend.eos_ids if stop_at_eos else None
    cuda = ids.device.type == "cuda"
    rows, texts, tokens = [], {}, {}
    for j, mode in enumerate(modes):
        if progress:
            progress(j, len(modes), mode)
        fn = _runner(backend, mode, ids, max_new_tokens, k, eos)
        if warmup:
            fn(2)
        if cuda:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        runs = [fn() for _ in range(max(1, repeats))]
        wall = (time.perf_counter() - t0) / len(runs)
        totals = [r.prefill_s + r.decode_s for r, _ in runs]
        res, stats = runs[totals.index(statistics.median_low(totals))]
        n = res.n_new
        total = res.prefill_s + res.decode_s
        kv = backend.kv_bytes_target * (T + n)
        if mode == "speculative":
            kv += backend.kv_bytes_draft * (T + n)
        row = {
            "mode": mode, "label": MODES[mode] + (f" (k={k})" if mode == "speculative" else ""),
            "input_tokens": T, "new_tokens": n, "prefill_s": res.prefill_s, "decode_s": res.decode_s,
            "total_s": total, "tok_per_s": n / total if total > 0 else float("nan"),
            "decode_tok_per_s": res.decode_tok_per_s, "wall_s": wall,
            "kv_cache_mib": None if mode == "uncached" else kv / 2**20,
            "target_forwards": stats.target_forwards if stats else n,
            "acceptance_rate": stats.acceptance_rate if stats else None,
            "mean_accepted_per_round": stats.as_dict()["mean_accepted_per_round"] if stats else None,
            "peak_extra_mib": (torch.cuda.max_memory_allocated() - base) / 2**20 if cuda else None,
            "repeats": len(runs),
        }
        rows.append(row)
        tokens[mode] = res.tokens
        texts[mode] = backend.tok.decode(res.tokens, skip_special_tokens=True)

    ref_mode = "uncached" if "uncached" in tokens else modes[0]
    ref = tokens[ref_mode]
    ref_total = next(r["total_s"] for r in rows if r["mode"] == ref_mode)
    for r in rows:
        t = tokens[r["mode"]]
        r["same_tokens_as_reference"] = t == ref
        r["first_diff"] = next((i for i, (a, b) in enumerate(zip(ref, t)) if a != b),
                               None if len(ref) == len(t) else min(len(ref), len(t)))
        r["speedup_vs_reference"] = ref_total / r["total_s"] if r["total_s"] > 0 else float("nan")
    return {"rows": rows, "texts": texts, "reference_mode": ref_mode, "input_tokens": T,
            "settings": {"max_new_tokens": max_new_tokens, "k": k, "warmup": warmup, "repeats": repeats,
                         "stop_at_eos": stop_at_eos, "dtype": backend.info["dtype"], "device": backend.info["device"]}}
