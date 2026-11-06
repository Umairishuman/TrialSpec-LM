import copy
import itertools

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.decoding.generate import greedy_manual, greedy_no_cache
from src.decoding.manual_decoder import ManualQwen2
from src.decoding.speculative import speculative_greedy


def tiny(seed, layers):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=61, hidden_size=48, intermediate_size=96, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      rope_parameters={"rope_type": "default", "rope_theta": 10000.0}, tie_word_embeddings=False)  # tied random models just copy the last token
    return Qwen2ForCausalLM(cfg).to(torch.float64).eval()


@pytest.fixture(scope="module")
def models():
    target = tiny(0, 3)
    # A draft that agrees with the target often but not always: target weights + noise of 20% of each
    # tensor's std (measured: 11-54% acceptance at k=4, so both accept and reject paths are exercised).
    draft = tiny(1, 3)
    with torch.no_grad():
        for p, src in zip(draft.parameters(), target.parameters()):
            p.copy_(src + 0.2 * src.std() * torch.randn_like(src))
    return target, draft


def prompt(n, seed):
    return torch.randint(0, 61, (1, n), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("k,n_in,n_out", list(itertools.product([1, 2, 4, 7], [1, 9, 30], [1, 6, 25])))
def test_speculative_equals_target_greedy(models, k, n_in, n_out):
    target, draft = models
    ids = prompt(n_in, seed=n_in + k)
    ref = greedy_no_cache(target, ids, n_out).tokens
    res, st = speculative_greedy(ManualQwen2(target), ManualQwen2(draft), ids, n_out, k=k)
    assert res.tokens == ref
    assert st.accepted <= st.drafted


def test_noisy_draft_partially_accepted(models):
    """Guards against a trivial setup where draft and target always agree (or never do)."""
    target, draft = models
    ids = prompt(12, seed=11)
    res, st = speculative_greedy(ManualQwen2(target), ManualQwen2(draft), ids, 40, k=4)
    assert res.tokens == greedy_no_cache(target, ids, 40).tokens
    assert 0.0 < st.acceptance_rate < 1.0
    assert len(set(res.tokens)) > 3


def test_identical_draft_accepts_everything(models):
    target, _ = models
    ids = prompt(8, seed=3)
    res, st = speculative_greedy(ManualQwen2(target), ManualQwen2(copy.deepcopy(target)), ids, 21, k=4)
    assert res.tokens == greedy_manual(ManualQwen2(target), ids, 21).tokens
    assert st.acceptance_rate == 1.0
    # 1 prefill + ceil(20 / (k+1)) verification passes instead of 20 single-token steps
    assert st.target_forwards == 1 + 4


def test_unrelated_draft_still_correct(models):
    target, _ = models
    ids = prompt(8, seed=4)
    res, st = speculative_greedy(ManualQwen2(target), ManualQwen2(tiny(7, 1)), ids, 15, k=3)
    assert res.tokens == greedy_no_cache(target, ids, 15).tokens
    assert st.acceptance_rate < 1.0


def test_eos_respected(models):
    target, draft = models
    ids = prompt(6, seed=5)
    ref = greedy_no_cache(target, ids, 20).tokens
    stop = ref[5]
    res, _ = speculative_greedy(ManualQwen2(target), ManualQwen2(draft), ids, 20, k=4, eos_ids={stop})
    assert res.tokens == ref[:ref.index(stop) + 1]
