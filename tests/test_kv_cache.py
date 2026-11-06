import itertools
import os

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.decoding.generate import greedy_hf_cache, greedy_manual, greedy_no_cache
from src.decoding.manual_decoder import KVCache, ManualQwen2, kv_bytes_per_token

INPUT_LENS = [1, 7, 40]
OUTPUT_LENS = [1, 5, 24]


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      rope_parameters={"rope_type": "default", "rope_theta": 10000.0}, tie_word_embeddings=True)
    m = Qwen2ForCausalLM(cfg).to(torch.float64).eval()
    return m


def prompt(n, seed=1):
    return torch.randint(0, 97, (1, n), generator=torch.Generator().manual_seed(seed))


def test_cache_write_crop_overflow():
    c = KVCache.allocate(2, 1, 2, 4, capacity=5, dtype=torch.float32, device="cpu")
    k = torch.ones(1, 2, 3, 4)
    kv, vv = c.write(0, k, k)
    assert kv.shape[-2] == 3
    c.advance(3)
    c.crop(1)
    assert c.length == 1
    with pytest.raises(ValueError):
        c.write(0, torch.ones(1, 2, 5, 4), torch.ones(1, 2, 5, 4))


def test_kv_bytes_formula(tiny):
    # 2 (K,V) x layers x kv_heads x head_dim x bytes
    assert kv_bytes_per_token(tiny.config, 2) == 2 * 3 * 2 * 16 * 2


@pytest.mark.parametrize("attention", ["reference", "tiled"])
def test_manual_forward_matches_hf(tiny, attention):
    ids = prompt(33)
    dec = ManualQwen2(tiny, attention=attention, block=8)
    ref = tiny(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(dec.forward(ids, None), ref, rtol=1e-9, atol=1e-9)
    # Incremental: prefill 20 tokens, then 13 single-token steps must reproduce the same logits.
    cache = dec.new_cache(1, 33)
    parts = [dec.forward(ids[:, :20], cache)] + [dec.forward(ids[:, t:t + 1], cache) for t in range(20, 33)]
    torch.testing.assert_close(torch.cat(parts, dim=1), ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("n_in,n_out", list(itertools.product(INPUT_LENS, OUTPUT_LENS)))
def test_greedy_tokens_identical(tiny, n_in, n_out):
    ids = prompt(n_in, seed=n_in)
    ref = greedy_no_cache(tiny, ids, n_out).tokens
    assert len(ref) == n_out
    assert greedy_hf_cache(tiny, ids, n_out).tokens == ref
    assert greedy_manual(ManualQwen2(tiny), ids, n_out).tokens == ref
    assert greedy_manual(ManualQwen2(tiny, attention="tiled", block=8), ids, n_out).tokens == ref
    assert greedy_manual(ManualQwen2(tiny), ids, n_out, use_cache=False).tokens == ref


def test_eos_stops(tiny):
    ids = prompt(5)
    ref = greedy_no_cache(tiny, ids, 10).tokens
    stop = ref[3]
    first = ref.index(stop)
    assert greedy_manual(ManualQwen2(tiny), ids, 10, eos_ids={stop}).tokens == ref[:first + 1]


@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 (downloads Qwen2.5-0.5B)")
def test_real_model_tokens_identical():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
    ids = tok("A 58-year-old woman presents with chest pain radiating to the back.", return_tensors="pt").input_ids
    ref = greedy_no_cache(m, ids, 16).tokens
    assert greedy_hf_cache(m, ids, 16).tokens == ref
    assert greedy_manual(ManualQwen2(m), ids, 16).tokens == ref
