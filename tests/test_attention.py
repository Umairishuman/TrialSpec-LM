import itertools

import pytest
import torch
import torch.nn.functional as F

from src.attention.attention import causal_mask, reference_attention, tiled_attention

SHAPES = [  # (B, H, Lq, Lk, D)
    (1, 1, 1, 1, 8),
    (2, 3, 17, 17, 32),     # not a multiple of the block size
    (1, 2, 64, 64, 64),
    (1, 2, 130, 130, 64),
    (2, 2, 1, 97, 64),      # single-token decode against a cache
    (1, 4, 16, 80, 128),    # chunked prefill: queries are the last 16 positions
]
BLOCKS = [(16, 16), (64, 32), (7, 13)]


def rand(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float64).to(dtype)


def qkv(B, H, Lq, Lk, D, dtype=torch.float64, seed=0):
    return rand((B, H, Lq, D), dtype, seed), rand((B, H, Lk, D), dtype, seed + 1), rand((B, H, Lk, D), dtype, seed + 2)


def test_causal_mask_offsets():
    assert causal_mask(3, 3, "cpu").tolist() == [[False, True, True], [False, False, True], [False, False, False]]
    assert not causal_mask(1, 5, "cpu").any()       # decode token sees the whole cache


@pytest.mark.parametrize("shape,causal", list(itertools.product(SHAPES, [False, True])))
def test_reference_matches_sdpa(shape, causal):
    B, H, Lq, Lk, D = shape
    q, k, v = qkv(B, H, Lq, Lk, D)
    mask = None if not causal else ~causal_mask(Lq, Lk, "cpu")   # SDPA bool mask: True = attend
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(reference_attention(q, k, v, causal), expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("shape,causal,blocks", list(itertools.product(SHAPES, [False, True], BLOCKS)))
def test_tiled_matches_reference_fp64(shape, causal, blocks):
    q, k, v = qkv(*shape)
    out = tiled_attention(q, k, v, causal, block_q=blocks[0], block_k=blocks[1])
    torch.testing.assert_close(out, reference_attention(q, k, v, causal), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.float16, 2e-3)])
def test_tiled_low_precision(dtype, tol):
    q, k, v = qkv(2, 2, 70, 70, 64, dtype=dtype)
    truth = reference_attention(*(t.double() for t in (q, k, v)), causal=True)
    out = tiled_attention(q, k, v, causal=True, block_q=32, block_k=16)
    assert (out.double() - truth).abs().max().item() < tol


def test_large_logits_stable():
    q, k, v = qkv(1, 1, 33, 33, 16, dtype=torch.float32)
    q = q * 1e3   # logits ~1e4: naive exp would overflow
    out = tiled_attention(q, k, v, causal=True, block_q=8, block_k=8)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, reference_attention(q, k, v, causal=True), rtol=1e-4, atol=1e-4)
