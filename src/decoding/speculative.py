"""Greedy speculative decoding with explicit draft/target KV caches (both ManualQwen2 decoders).

Invariant at the start of each round: both caches hold every committed token except the newest one
(`last`), whose next-token distribution has not been computed yet.

Round:
  1. Draft: feed `last`, then its own guesses, one token at a time -> k proposals d1..dk.
     (Draft cache grows by k: last, d1..d_{k-1}.)
  2. Target: ONE forward over [last, d1..dk] -> greedy predictions p0..pk, where p_i is the target's
     choice after d_i (p0 after `last`).
  3. Accept the longest prefix with d_{i+1} == p_i (n tokens), then append the target's own token p_n
     (a correction if n < k, a free "bonus" token if n == k). Output = exactly what target greedy
     decoding would produce.
  4. Roll back: target cache keeps last, d1..dn (crop away rejected positions). Draft cache keeps the
     same; if all k were accepted, d_k was never fed to the draft, so it is fed now.
"""
import time
from dataclasses import dataclass, field

import torch

from src.decoding.generate import GenResult, _done, _sync
from src.decoding.manual_decoder import ManualQwen2


@dataclass
class SpecStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    accepted_per_round: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else float("nan")

    def as_dict(self) -> dict:
        return {"rounds": self.rounds, "drafted": self.drafted, "accepted": self.accepted,
                "acceptance_rate": self.acceptance_rate, "target_forwards": self.target_forwards,
                "draft_forwards": self.draft_forwards,
                "mean_accepted_per_round": self.accepted / self.rounds if self.rounds else float("nan"),
                "accepted_hist": {str(i): self.accepted_per_round.count(i) for i in sorted(set(self.accepted_per_round))}}


@torch.no_grad()
def speculative_greedy(target: ManualQwen2, draft: ManualQwen2, input_ids: torch.Tensor, max_new_tokens: int,
                       k: int = 4, eos_ids=None) -> tuple[GenResult, SpecStats]:
    assert k >= 1
    dev = input_ids.device
    T = input_ids.shape[1]
    cap = T + max_new_tokens + k + 1
    tc, dc = target.new_cache(1, cap), draft.new_cache(1, cap)
    st = SpecStats()

    _sync(dev); t0 = time.perf_counter()
    last = int(target.forward(input_ids, tc)[0, -1].argmax())   # first token from target prefill
    draft.forward(input_ids, dc)
    st.target_forwards += 1; st.draft_forwards += 1
    _sync(dev); prefill = time.perf_counter() - t0
    out = [last]
    steps = []

    t1 = time.perf_counter()
    while len(out) < max_new_tokens and not _done(last, eos_ids):
        ts = time.perf_counter()
        base = tc.length                                    # == dc.length == committed - 1
        assert dc.length == base
        # 1. draft k proposals
        props, nxt = [], last
        for _ in range(k):
            nxt = int(draft.forward(torch.tensor([[nxt]], device=dev), dc)[0, -1].argmax())
            props.append(nxt)
        st.draft_forwards += k
        # 2. verify all proposals with one target pass
        block = torch.tensor([[last] + props], device=dev)
        preds = target.forward(block, tc)[0].argmax(-1).tolist()     # p0..pk
        st.target_forwards += 1
        # 3. longest matching prefix + target's own next token
        n = 0
        while n < k and props[n] == preds[n]:
            n += 1
        new = props[:n] + [preds[n]]
        st.rounds += 1; st.drafted += k; st.accepted += n; st.accepted_per_round.append(n)
        # 4. synchronise caches: keep last, d1..dn
        tc.crop(base + 1 + n)
        if n == k:
            draft.forward(torch.tensor([[props[-1]]], device=dev), dc)   # draft never saw d_k
            st.draft_forwards += 1
        else:
            dc.crop(base + 1 + n)
        # commit, honouring EOS and the length budget
        for t in new:
            out.append(t)
            last = t
            if len(out) >= max_new_tokens or _done(t, eos_ids):
                break
        _sync(dev); steps.append(time.perf_counter() - ts)
    _sync(dev)
    return GenResult(out[:max_new_tokens], prefill, time.perf_counter() - t1, steps), st
