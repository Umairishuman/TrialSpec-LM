"""Hand-written Qwen2 decoder forward pass with an explicit, growable KV cache.

Reuses the Hugging Face module *weights* only; embedding lookup, RMSNorm, RoPE, grouped-query attention,
cache update and MLP are computed here. Attention goes through src.attention (reference or tiled).

Cache layout per layer: K, V of shape [B, n_kv_heads, capacity, head_dim], preallocated; `length`
tokens are valid. New keys are rotated with RoPE at their absolute positions *before* being written,
so cached keys never need re-rotation. Batch size 1 is assumed for decoding (no padding).
"""
from dataclasses import dataclass, field

import torch

from src.attention.attention import reference_attention, tiled_attention


@dataclass
class KVCache:
    k: list[torch.Tensor] = field(default_factory=list)
    v: list[torch.Tensor] = field(default_factory=list)
    length: int = 0
    capacity: int = 0

    @classmethod
    def allocate(cls, n_layers, batch, n_kv, head_dim, capacity, dtype, device):
        shape = (batch, n_kv, capacity, head_dim)
        return cls([torch.empty(shape, dtype=dtype, device=device) for _ in range(n_layers)],
                   [torch.empty(shape, dtype=dtype, device=device) for _ in range(n_layers)], 0, capacity)

    def write(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """Append k/v for positions [length, length+T) of one layer; returns views of the valid prefix."""
        t = k_new.shape[-2]
        end = self.length + t
        if end > self.capacity:
            raise ValueError(f"KV cache overflow: need {end}, capacity {self.capacity}")
        self.k[layer][:, :, self.length:end] = k_new
        self.v[layer][:, :, self.length:end] = v_new
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, t: int):
        self.length += t

    def crop(self, length: int):
        """Discard positions >= length (used by speculative decoding on rejection)."""
        assert 0 <= length <= self.length
        self.length = length

    def nbytes(self) -> int:
        per_tok = sum(x[:, :, :1].numel() * x.element_size() for x in self.k + self.v)
        return per_tok * self.length


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class ManualQwen2:
    def __init__(self, hf_model, attention: str = "reference", block: int = 64):
        self.m = hf_model.model if hasattr(hf_model, "model") else hf_model
        self.lm_head = hf_model.lm_head
        c = hf_model.config
        self.n_layers = c.num_hidden_layers
        self.n_heads = c.num_attention_heads
        self.n_kv = c.num_key_value_heads
        self.head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        self.eps = c.rms_norm_eps
        rope = getattr(c, "rope_parameters", None) or getattr(c, "rope_scaling", None) or {}
        assert rope.get("rope_type", "default") == "default", "only unscaled RoPE is implemented"
        theta = rope.get("rope_theta", getattr(c, "rope_theta", 10000.0))
        dev = self.lm_head.weight.device
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, device=dev, dtype=torch.float32) / self.head_dim))
        self.attn_fn = {"reference": reference_attention,
                        "tiled": lambda q, k, v, causal: tiled_attention(q, k, v, causal, block_q=block, block_k=block)}[attention]
        self.dtype = self.lm_head.weight.dtype
        self.device = dev

    def new_cache(self, batch: int, capacity: int) -> KVCache:
        return KVCache.allocate(self.n_layers, batch, self.n_kv, self.head_dim, capacity, self.dtype, self.device)

    def rms_norm(self, x, weight):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return weight * xf.to(x.dtype)

    def rope(self, positions: torch.Tensor):
        freqs = positions.float()[:, None] * self.inv_freq[None, :]      # [T, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype)[None, None], emb.sin().to(self.dtype)[None, None]

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, cache: KVCache | None) -> torch.Tensor:
        """Logits for input_ids, which occupy positions [cache.length, cache.length + T).

        cache=None runs without caching (all positions recomputed from scratch, starting at 0).
        """
        B, T = input_ids.shape
        start = cache.length if cache is not None else 0
        positions = torch.arange(start, start + T, device=input_ids.device)
        cos, sin = self.rope(positions)
        h = self.m.embed_tokens(input_ids)

        for i, layer in enumerate(self.m.layers):
            a = layer.self_attn
            x = self.rms_norm(h, layer.input_layernorm.weight)
            q = a.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = a.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            v = a.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            if cache is not None:
                k, v = cache.write(i, k, v)          # keys/values for positions [0, start + T)
            # Grouped-query attention: each KV head serves n_heads / n_kv query heads.
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
            # Causal with offset: query t (absolute position start+t) sees keys 0..start+t.
            o = self.attn_fn(q, k, v, causal=True)
            h = h + a.o_proj(o.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim))
            x = self.rms_norm(h, layer.post_attention_layernorm.weight)
            mlp = layer.mlp
            h = h + mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))

        if cache is not None:
            cache.advance(T)
        return self.lm_head(self.rms_norm(h, self.m.norm.weight))


def kv_bytes_per_token(config, dtype_bytes: int) -> int:
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * dtype_bytes
