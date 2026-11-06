"""Task 5 benchmark: target greedy (KV-cached) vs greedy speculative decoding, same prompts and settings.

Two prompt sets:
  task  - real val criterion prompts (prompt v2); short structured JSON answers, EOS-terminated
  free  - free-text explanations of the same notes; longer outputs (up to --free-max-new, EOS-terminated)
Per config: warm-up, then timed runs (synchronised); token equality vs target greedy; acceptance stats;
peak CUDA memory. HF assisted generation (assistant_model=) is run once per prompt as an external
reference for token equality only.
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataset import load_split_frames, prompt_text
from src.decoding.generate import greedy_manual
from src.decoding.manual_decoder import ManualQwen2
from src.decoding.speculative import speculative_greedy
from src.eval.predict import greedy_config
from src.utils import git_commit, library_versions


def load(model_id, adapter, dtype, device):
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload().eval()
    return m


def prompts(tok, n: int, device):
    val = load_split_frames()["val"]
    rows = [r for _, r in val.drop_duplicates("patient_id").head(n).iterrows()]
    task = [tok(prompt_text(tok, r, "v2"), return_tensors="pt", add_special_tokens=False).input_ids.to(device) for r in rows]
    free = []
    for r in rows:
        msgs = [{"role": "user", "content": "Patient note:\n" + r["note"] + "\n\nExplain, in plain language, the patient's main "
                 "medical problems and what further information a clinician would need."}]
        free.append(tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
                        return_tensors="pt", add_special_tokens=False).input_ids.to(device))
    return {"task": task, "free": free}


def timed(fn, device, repeats):
    fn()
    if device == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    outs = [fn() for _ in range(repeats)]
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20 if device == "cuda" else None
    return outs, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter merged into the target")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--task-max-new", type=int, default=64)
    ap.add_argument("--free-max-new", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/speculative/benchmark"))
    args = ap.parse_args()
    if args.smoke:
        args.target, args.adapter, args.ks, args.n_prompts, args.task_max_new, args.free_max_new, args.repeats = \
            "Qwen/Qwen2.5-0.5B-Instruct", None, [2], 1, 8, 8, 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.target)
    dtok = AutoTokenizer.from_pretrained(args.draft)
    assert tok.get_vocab() == dtok.get_vocab() and tok.all_special_tokens == dtok.all_special_tokens, \
        "draft and target tokenizers differ"
    target, draft = load(args.target, args.adapter, dtype, device), load(args.draft, None, dtype, device)
    T, D = ManualQwen2(target), ManualQwen2(draft)
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    sets = prompts(tok, args.n_prompts, device)

    rows = []
    for set_name, plist in sets.items():
        max_new = args.task_max_new if set_name == "task" else args.free_max_new
        eos_ids = eos   # both sets stop at EOS, identically in every mode
        for pi, p in enumerate(plist):
            base_outs, base_peak = timed(lambda: greedy_manual(T, p, max_new, eos_ids), device, args.repeats)
            ref = base_outs[0].tokens
            base_lat = statistics.median(o.prefill_s + o.decode_s for o in base_outs)
            gcfg = greedy_config(tok, max_new)          # pure greedy (Qwen defaults add repetition_penalty=1.1)
            g = target.generate(p, attention_mask=torch.ones_like(p), assistant_model=draft, generation_config=gcfg)
            hf_equal = g[0, p.shape[1]:].tolist()[:len(ref)] == ref
            rows.append({"set": set_name, "prompt": pi, "mode": "target_greedy", "k": 0, "input_len": p.shape[1],
                         "n_new": len(ref), "latency_s": base_lat, "tok_per_s": len(ref) / base_lat,
                         "peak_extra_mib": base_peak, "hf_assisted_equal": hf_equal})
            for k in args.ks:
                outs, peak = timed(lambda: speculative_greedy(T, D, p, max_new, k=k, eos_ids=eos_ids), device, args.repeats)
                res, st = outs[0]
                lat = statistics.median(r.prefill_s + r.decode_s for r, _ in outs)
                first_diff = next((i for i, (a, b) in enumerate(zip(ref, res.tokens)) if a != b), None)
                rows.append({"set": set_name, "prompt": pi, "mode": "speculative", "k": k, "input_len": p.shape[1],
                             "n_new": len(res.tokens), "latency_s": lat, "tok_per_s": len(res.tokens) / lat,
                             "speedup": base_lat / lat, "tokens_equal": res.tokens == ref, "first_diff": first_diff,
                             "peak_extra_mib": peak, **st.as_dict()})
                print(f"{set_name:4s} p{pi} k={k} n={len(res.tokens):3d} acc={st.acceptance_rate:.2f} "
                      f"mean_acc/round={st.as_dict()['mean_accepted_per_round']:.2f} speedup={base_lat / lat:.2f}x "
                      f"equal={res.tokens == ref}" + (f" first_diff={first_diff}" if first_diff is not None else ""), flush=True)

    summary = {}
    for set_name in sets:
        for k in args.ks:
            rs = [r for r in rows if r["set"] == set_name and r["k"] == k]
            summary[f"{set_name}_k{k}"] = {
                "acceptance_rate": sum(r["accepted"] for r in rs) / sum(r["drafted"] for r in rs),
                "mean_speedup": statistics.mean(r["speedup"] for r in rs),
                "mean_tok_per_s": statistics.mean(r["tok_per_s"] for r in rs),
                "all_tokens_equal": all(r["tokens_equal"] for r in rs),
                "n_prompts": len(rs)}
        b = [r for r in rows if r["set"] == set_name and r["k"] == 0]
        summary[f"{set_name}_target_greedy"] = {"mean_tok_per_s": statistics.mean(r["tok_per_s"] for r in b),
                                                "hf_assisted_equal": [r["hf_assisted_equal"] for r in b]}
    for key, v in summary.items():
        print(key, {a: (round(x, 3) if isinstance(x, float) else x) for a, x in v.items()})

    meta = {"target": args.target, "adapter": args.adapter, "draft": args.draft, "dtype": str(dtype),
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "repeats": args.repeats,
            "task_max_new": args.task_max_new, "free_max_new": args.free_max_new, "decoding": "greedy",
            "smoke": args.smoke, "git_commit": git_commit(), "versions": library_versions(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    tag = "smoke" if args.smoke else f"{device}_{str(dtype).replace('torch.', '')}"
    out = Path(f"{args.out}_{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
