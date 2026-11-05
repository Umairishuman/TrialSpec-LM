"""Greedy batched prediction on a split with the base model or base + LoRA adapter."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataset import load_split_frames, prompt_text
from src.data.prompt import parse_output
from src.eval.metrics import full_report
from src.utils import git_commit, library_versions, load_env


def load_model(model_id: str, adapter: str | None, dtype: torch.dtype, device: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa").to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return tok, model


@torch.no_grad()
def predict_frame(tok, model, frame, batch_size: int, max_new_tokens: int, prompt_version: str = "v2") -> list[dict]:
    rows = [r for _, r in frame.iterrows()]
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        enc = tok([prompt_text(tok, r, prompt_version) for r in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(chunk, texts):
            valid = {i for i, _ in r["sentences"]}
            p = parse_output(t, r["criterion_type"], valid)
            out.append({
                "annotation_id": int(r["annotation_id"]), "patient_id": r["patient_id"], "trial_id": r["trial_id"],
                "criterion_type": r["criterion_type"], "criterion_text": r["criterion_text"],
                "gold_label": r["expert_eligibility"], "gold_evidence": list(r["expert_sentences"]),
                "pred_label": p["label"], "pred_evidence": p["evidence_ids"], "invalid_ids": p["invalid_ids"],
                "parse_ok": p["parse_ok"], "raw": t,
            })
        print(f"  {min(i + batch_size, len(rows))}/{len(rows)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="local path or HF repo of a LoRA adapter")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--prompt-version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="4 examples, checks the pipeline end-to-end")
    args = ap.parse_args()
    load_env()

    if args.split == "test":
        print("WARNING: evaluating on the held-out TEST split. Do this once, for final results only.")
    if args.smoke:
        args.limit = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok, model = load_model(args.model, args.adapter, dtype, device)

    frame = load_split_frames()[args.split]
    if args.limit:
        frame = frame.head(args.limit)

    t0 = time.perf_counter()
    records = predict_frame(tok, model, frame, args.batch_size, args.max_new_tokens, args.prompt_version)
    elapsed = time.perf_counter() - t0

    report = full_report(records)
    meta = {
        "model": args.model, "adapter": args.adapter, "prompt_version": args.prompt_version,
        "split": args.split, "n": len(records),
        "decoding": {"strategy": "greedy", "do_sample": False, "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size},
        "dtype": str(dtype), "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "git_commit": git_commit(), "versions": library_versions(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seconds": round(elapsed, 1),
        "parse_ok_rate": sum(r["parse_ok"] for r in records) / max(len(records), 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Append suffixes instead of with_suffix(): names like "qwen2.5-1.5b_val" contain dots.
    with open(f"{args.out}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    Path(f"{args.out}.metrics.json").write_text(json.dumps({"meta": meta, "metrics": report}, indent=2))

    o = report["overall"]
    print(f"accuracy={o['accuracy']:.3f} macro_f1={o['macro_f1']:.3f} parse_ok={meta['parse_ok_rate']:.3f} "
          f"evidence_micro_f1={report['evidence']['micro']['f1']:.3f} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
