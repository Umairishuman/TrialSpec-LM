"""Greedy decoding loops: uncached reference, explicitly managed HF cache, and the hand-written decoder.

All loops take a [1, T] prompt and return the generated token ids plus timing. Each forward call is
written out so the cache handling is visible (no model.generate):
  - prefill: the whole prompt in one forward pass, filling the cache with positions [0, T)
  - decode:  one token per step at position T+i, attending to every cached key (no mask needed for a
             single query against the past: causal masking is implicit)
"""
import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from src.decoding.manual_decoder import ManualQwen2


@dataclass
class GenResult:
    tokens: list[int]
    prefill_s: float
    decode_s: float
    step_s: list[float] = field(default_factory=list)

    @property
    def n_new(self) -> int:
        return len(self.tokens)

    @property
    def decode_tok_per_s(self) -> float:
        # First token comes from prefill logits; decode steps produce the remaining n_new - 1.
        return (self.n_new - 1) / self.decode_s if self.decode_s > 0 and self.n_new > 1 else float("nan")


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _done(tok: int, eos_ids: set[int] | None) -> bool:
    return eos_ids is not None and tok in eos_ids


@torch.no_grad()
def greedy_no_cache(model, input_ids, max_new_tokens: int, eos_ids=None) -> GenResult:
    """Reference: every step recomputes attention over the full sequence (O(T^2) work per step)."""
    dev = input_ids.device
    ids = input_ids
    out, steps = [], []
    _sync(dev); t0 = time.perf_counter()
    logits = model(input_ids=ids, use_cache=False).logits
    tok = int(logits[0, -1].argmax())
    _sync(dev); prefill = time.perf_counter() - t0
    out.append(tok)
    t1 = time.perf_counter()
    while len(out) < max_new_tokens and not _done(tok, eos_ids):
        ts = time.perf_counter()
        ids = torch.cat([ids, torch.tensor([[tok]], device=dev)], dim=1)
        logits = model(input_ids=ids, use_cache=False).logits
        tok = int(logits[0, -1].argmax())
        _sync(dev); steps.append(time.perf_counter() - ts)
        out.append(tok)
    _sync(dev)
    return GenResult(out, prefill, time.perf_counter() - t1, steps)


@torch.no_grad()
def greedy_hf_cache(model, input_ids, max_new_tokens: int, eos_ids=None) -> GenResult:
    """HF model forward with an explicitly passed and grown DynamicCache, explicit positions and mask."""
    dev = input_ids.device
    T = input_ids.shape[1]
    cache = DynamicCache()
    out, steps = [], []
    _sync(dev); t0 = time.perf_counter()
    logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                   position_ids=torch.arange(T, device=dev)[None], past_key_values=cache, use_cache=True).logits
    tok = int(logits[0, -1].argmax())
    _sync(dev); prefill = time.perf_counter() - t0
    out.append(tok)
    assert cache.get_seq_length() == T
    t1 = time.perf_counter()
    while len(out) < max_new_tokens and not _done(tok, eos_ids):
        ts = time.perf_counter()
        pos = cache.get_seq_length()                      # absolute position of the new token
        logits = model(input_ids=torch.tensor([[tok]], device=dev),
                       attention_mask=torch.ones((1, pos + 1), dtype=torch.long, device=dev),  # mask covers past + new
                       position_ids=torch.tensor([[pos]], device=dev),
                       past_key_values=cache, use_cache=True).logits
        tok = int(logits[0, -1].argmax())
        _sync(dev); steps.append(time.perf_counter() - ts)
        out.append(tok)
    _sync(dev)
    return GenResult(out, prefill, time.perf_counter() - t1, steps)


@torch.no_grad()
def greedy_manual(dec: ManualQwen2, input_ids, max_new_tokens: int, eos_ids=None, use_cache: bool = True) -> GenResult:
    """Hand-written decoder; use_cache=False recomputes the full sequence each step (manual reference)."""
    dev = input_ids.device
    T = input_ids.shape[1]
    cache = dec.new_cache(1, T + max_new_tokens) if use_cache else None
    ids = input_ids
    out, steps = [], []
    _sync(dev); t0 = time.perf_counter()
    tok = int(dec.forward(ids, cache)[0, -1].argmax())
    _sync(dev); prefill = time.perf_counter() - t0
    out.append(tok)
    t1 = time.perf_counter()
    while len(out) < max_new_tokens and not _done(tok, eos_ids):
        ts = time.perf_counter()
        nxt = torch.tensor([[tok]], device=dev)
        if use_cache:
            logits = dec.forward(nxt, cache)              # one token, position = cache.length
        else:
            ids = torch.cat([ids, nxt], dim=1)
            logits = dec.forward(ids, None)
        tok = int(logits[0, -1].argmax())
        _sync(dev); steps.append(time.perf_counter() - ts)
        out.append(tok)
    _sync(dev)
    return GenResult(out, prefill, time.perf_counter() - t1, steps)
