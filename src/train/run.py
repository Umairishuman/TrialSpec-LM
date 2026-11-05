"""LoRA supervised fine-tuning on the prescribed train split; model selection on val loss.

Smoke mode (--smoke): 16 train / 8 val examples including the longest train example,
10 steps, eval+save at step 5/10, then reload the saved adapter and generate on 2 val examples.
"""
import argparse
import json
import os
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from src.data.dataset import SFTDataset, collate, load_split_frames
from src.utils import git_commit, load_config, load_env, wandb_enabled


def smoke_subset(frame, n: int):
    # Always include the longest example so the smoke run exercises peak memory.
    lengths = frame["note"].str.len() + frame["criterion_text"].str.len()
    longest = frame.loc[[lengths.idxmax()]]
    return pd.concat([longest, frame.drop(index=longest.index).head(n - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default=None, help="override config model_id (e.g. a smaller model for local tests)")
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    load_env()

    cfg = load_config(args.config)
    if args.model:
        cfg["model_id"] = args.model
    tc = cfg["train"]
    set_seed(cfg["seed"])
    on_gpu = torch.cuda.is_available()
    out_dir = Path(cfg["output_dir"] + ("-smoke" if args.smoke else ""))

    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    frames = load_split_frames()
    train_df, val_df = frames["train"], frames["val"]
    if args.smoke:
        train_df = smoke_subset(train_df, 16)
        val_df = val_df.head(8)
    train_ds = SFTDataset(tok, train_df, cfg["max_len"])
    val_ds = SFTDataset(tok, val_df, cfg["max_len"])
    print(f"train={len(train_ds)} val={len(val_ds)} longest_train_seq={max(len(x['input_ids']) for x in train_ds.items)}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=torch.float16 if (on_gpu and tc["fp16"]) else torch.float32, attn_implementation="sdpa"
    )
    if tc["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lc = cfg["lora"]
    # PEFT keeps adapter weights in fp32 (autocast_adapter_dtype) so fp16 AMP can unscale their grads.
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    report_to = "wandb" if wandb_enabled() else "none"
    if report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", cfg["wandb_project"])
        if args.smoke:
            os.environ["WANDB_RUN_GROUP"] = "smoke"

    smoke_steps = dict(max_steps=10, eval_strategy="steps", eval_steps=5, save_strategy="steps", save_steps=5, logging_steps=1)
    full_steps = dict(num_train_epochs=tc["epochs"], eval_strategy="epoch", save_strategy="epoch", logging_steps=tc["logging_steps"])
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=tc["per_device_batch_size"],
        per_device_eval_batch_size=tc["per_device_batch_size"],
        gradient_accumulation_steps=1 if args.smoke else tc["grad_accum"],
        learning_rate=tc["lr"], warmup_steps=tc["warmup_ratio"],  # float in [0,1) = ratio in transformers v5 weight_decay=tc["weight_decay"],
        lr_scheduler_type=tc["lr_scheduler"],
        fp16=on_gpu and tc["fp16"],
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, seed=cfg["seed"], report_to=report_to,
        run_name=("smoke-" if args.smoke else "") + Path(cfg["output_dir"]).name,
        remove_unused_columns=False, dataloader_num_workers=0,
        **(smoke_steps if args.smoke else full_steps),
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=partial(collate, pad_id=tok.pad_token_id))
    trainer.train(resume_from_checkpoint=args.resume)

    final = out_dir / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    losses = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    meta = {"config": cfg, "smoke": args.smoke, "git_commit": git_commit(), "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_eval_loss": trainer.state.best_metric, "log_history": losses,
            "precision": "fp16 AMP (fp32 adapters)" if targs.fp16 else "fp32"}
    (final / "training_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"saved adapter -> {final} (best eval_loss={trainer.state.best_metric})")

    train_losses = [h["loss"] for h in losses if "loss" in h]
    if not all(map(lambda x: x == x and abs(x) != float("inf"), train_losses)):
        raise RuntimeError("non-finite training loss detected; see D-002 fallback (fp32)")

    if args.smoke:
        from src.eval.predict import load_model, predict_frame
        del model, trainer
        if on_gpu:
            torch.cuda.empty_cache()
        tok2, m2 = load_model(cfg["model_id"], str(final), torch.float16 if on_gpu else torch.float32, "cuda" if on_gpu else "cpu")
        recs = predict_frame(tok2, m2, frames["val"].head(2), batch_size=2, max_new_tokens=64)
        for r in recs:
            print("  gold:", r["gold_label"], r["gold_evidence"], "| raw:", r["raw"][:120])
        if on_gpu:
            print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        print("SMOKE TEST PASSED")
    elif not args.no_push and cfg.get("hub_repo") and os.environ.get("HF_TOKEN"):
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(cfg["hub_repo"], exist_ok=True)
        api.upload_folder(folder_path=str(final), repo_id=cfg["hub_repo"], commit_message=f"adapter from {git_commit()[:8]}")
        print(f"pushed adapter -> https://huggingface.co/{cfg['hub_repo']}")


if __name__ == "__main__":
    main()
