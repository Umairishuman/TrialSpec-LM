"""Reference scaled dot-product attention and a tiled online-softmax (FlashAttention-style) variant.

Shapes: q [B, H, Lq, D], k/v [B, H, Lk, D] with Lk >= Lq. Under causal masking the queries are the
*last* Lq positions of the key sequence (offset = Lk - Lq), which covers both prefill (Lq == Lk)
and cached decoding (Lq == 1): query i may attend to keys j <= i + offset.
"""
import math

import torch


def causal_mask(lq: int, lk: int, device) -> torch.Tensor:
    """Boolean [Lq, Lk], True where attention is NOT allowed."""
    offset = lk - lq
    qi = torch.arange(lq, device=device)[:, None]
    kj = torch.arange(lk, device=device)[None, :]
    return kj > qi + offset


def reference_attention(q, k, v, causal: bool = False, scale: float | None = None) -> torch.Tensor:
    """softmax(QK^T / sqrt(d) + mask) V, materialising the full [Lq, Lk] score matrix."""
    assert k.shape[-2] >= q.shape[-2], "Lk must be >= Lq"
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    s = (q.to(acc_dtype) @ k.to(acc_dtype).transpose(-1, -2)) * scale
    if causal:
        s = s.masked_fill(causal_mask(q.shape[-2], k.shape[-2], q.device), float("-inf"))
    s = s - s.amax(dim=-1, keepdim=True)          # numerically stable softmax
    p = torch.exp(s)
    p = p / p.sum(dim=-1, keepdim=True)
    return (p @ v.to(acc_dtype)).to(q.dtype)


def tiled_attention(q, k, v, causal: bool = False, scale: float | None = None,
                    block_q: int = 64, block_k: int = 64) -> torch.Tensor:
    """Blocked attention with online softmax: never materialises more than [block_q, block_k] scores.

    For each query block we keep a running row max m, running normaliser l and unnormalised output acc.
    When a new key block raises the max from m to m_new, previous contributions are rescaled by
    exp(m - m_new), so the final acc / l equals the exact softmax-weighted sum.
    """
    B, H, Lq, D = q.shape
    Lk = k.shape[-2]
    assert Lk >= Lq, "Lk must be >= Lq"
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    offset = Lk - Lq
    acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    out = torch.empty_like(q)

    for qs in range(0, Lq, block_q):
        qe = min(qs + block_q, Lq)
        qb = q[:, :, qs:qe].to(acc_dtype)
        m = torch.full((B, H, qe - qs), float("-inf"), dtype=acc_dtype, device=q.device)
        l = torch.zeros((B, H, qe - qs), dtype=acc_dtype, device=q.device)
        acc = torch.zeros((B, H, qe - qs, D), dtype=acc_dtype, device=q.device)

        # Causal: keys beyond the last query's position in this block are never visible.
        k_stop = min(Lk, qe + offset) if causal else Lk
        for ks in range(0, k_stop, block_k):
            ke = min(ks + block_k, k_stop)
            s = (qb @ k[:, :, ks:ke].to(acc_dtype).transpose(-1, -2)) * scale
            if causal and ke - 1 > qs + offset:   # tile crosses the diagonal: mask partially
                qi = torch.arange(qs, qe, device=q.device)[:, None]
                kj = torch.arange(ks, ke, device=q.device)[None, :]
                s = s.masked_fill(kj > qi + offset, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1))
            # Every row sees key 0 in the first tile, so m_new is finite from the first tile on.
            p = torch.exp(s - m_new[..., None])
            alpha = torch.exp(m - m_new)
            l = alpha * l + p.sum(dim=-1)
            acc = alpha[..., None] * acc + p @ v[:, :, ks:ke].to(acc_dtype)
            m = m_new

        out[:, :, qs:qe] = (acc / l[..., None]).to(q.dtype)
    return out
