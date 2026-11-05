import pandas as pd
import torch

from src.data.load import load_annotations
from src.data.prompt import build_messages, build_target
from src.data.split import assign_splits, load_split


def load_split_frames() -> dict[str, pd.DataFrame]:
    df = assign_splits(load_annotations(), load_split())
    return {name: g.reset_index(drop=True) for name, g in df.groupby("split")}


def oversample(frame: pd.DataFrame, target: int, max_factor: int, seed: int) -> pd.DataFrame:
    """Repeat rare (criterion_type, label) groups up to `target` rows, at most `max_factor` copies each."""
    parts = []
    for _, g in frame.groupby(["criterion_type", "expert_eligibility"]):
        factor = min(max_factor, max(1, -(-target // len(g))))
        parts.append(pd.concat([g] * factor))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prompt_text(tokenizer, row, prompt_version: str = "v2") -> str:
    msgs = build_messages(row["sentences"], row["criterion_type"], row["criterion_text"], prompt_version)
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def encode_example(tokenizer, row, max_len: int, prompt_version: str = "v2") -> dict:
    msgs = build_messages(row["sentences"], row["criterion_type"], row["criterion_text"], prompt_version)
    target = build_target(row["expert_eligibility"], row["expert_sentences"])
    prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    full = tokenizer.apply_chat_template(msgs + [{"role": "assistant", "content": target}], tokenize=False)
    assert full.startswith(prompt), "chat template prompt is not a prefix of the full conversation"

    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    assert f_ids[: len(p_ids)] == p_ids, "prompt tokenization differs inside full sequence"
    if len(f_ids) > max_len:
        raise ValueError(f"annotation {row['annotation_id']} has {len(f_ids)} tokens > max_len {max_len}")

    # Loss only on the assistant answer (target JSON + end-of-turn tokens).
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels}


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, frame: pd.DataFrame, max_len: int, prompt_version: str = "v2"):
        self.items = [encode_example(tokenizer, r, max_len, prompt_version) for _, r in frame.iterrows()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id: int):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        labels[i, :L] = torch.tensor(b["labels"])
        mask[i, :L] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}
