"""KVTok translator (v2): compact Stage-1 pooling + Receiver-aware output.

A deliberately naive baseline that keeps only LCF-X's Fig. 3 **Stage 1**
(within-span attention: one learned query per layer, ``N -> P``, no K/V
projection -- the mechanism that handles the Sharer's variable context
length ``N``) and drops **Stage 2** (the across-span attention that reduces
``L*P -> num_tokens`` in LCF-X/EIYW) entirely. Concretely:

1. **Pool the Sharer cache with Stage 1 only.** Per Sharer layer, one learned
   query collapses that layer's ``N_S`` real tokens into ``P`` span summaries.
   Flattening only heads and K/V gives ``[B, L_S*P, 2*H_S*D_S]``.
2. **Factorized KVTok down-projection.** A position-shared feature projection
   maps ``2*H_S*D_S -> d_model`` for every summary, then a token-axis linear
   maps ``L_S*P -> num_tokens`` independently for every feature. This is a
   plain linear down-projection, not Stage 2 attention. Crucially, it avoids
   the incorrect fully dense ``(L*P*2*H*D) -> (T*d)`` matrix, whose parameter
   count is the product of both axes and can exceed the base model itself.
3. **Receiver embedding processing (EIYW-v2 style).** The Receiver's own
    context is looked up as raw embeddings ``[B, N_r, hidden_r]``, projected to
    ``d_model``, then gets sinusoidal positional encoding + LayerNorm.
4. **Receiver-aware refinement + output head.** Slot queries cross-attend into
    the processed Receiver memory, then a pre-head LayerNorm and output linear
    ``d_model -> hidden_r`` emit soft tokens. As in EIYW-v2, output magnitude
    is calibrated by the Receiver embedding RMS with a learnable gain.

The Receiver contributes embeddings only. There is no initial Receiver model
forward, Receiver KV cache, Receiver Stage-1 pool, or Receiver cache sharing.

Those ``num_tokens`` tokens are **prepended to the Receiver's input
embeddings** (see ``model.KVTokPair.decode_receiver``), identical to EIYW's
mechanism -- never written into the KV cache.

A *gentle* (not exact-zero) small-scale init on the final projection starts
these tokens near, but not at, zero -- see EIYW's translator docstring for why
exact-zero is unsafe for an embedding-space token fed through a frozen model.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class _Stage1Pool(nn.Module):
    """LCF-X's Stage 1 (Fig. 3), unmodified, with Stage 2 removed.

    Within-span attention (``N -> P``, one query per layer, no K/V
    projection) only -- no across-span reduction. Returns the fixed-size
    ``(K, V)`` pair ``[B, H, L*P, D]``, in this side's own geometry.
    """

    def __init__(self, layers: int, kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.layers = layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim

        # One query per layer (q_base^(l) in the paper).
        self.q_base = nn.Parameter(torch.empty(layers, kv_heads, head_dim))
        nn.init.kaiming_normal_(self.q_base)
        with torch.no_grad():
            self.q_base.mul_(0.02)

    def _within_span(
        self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor, num_spans: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw ``K,V [1,H,n,D]`` -> span summaries ``[1,H,P,D]`` x2 (keeps geometry)."""

        n = key.shape[2]
        p = max(1, min(int(num_spans), n))
        scale = self.head_dim**0.5
        pooled_k, pooled_v = [], []
        for k_span, v_span in zip(torch.tensor_split(key, p, dim=2), torch.tensor_split(value, p, dim=2)):
            scores = torch.einsum("hd,bhnd->bhn", query, k_span) / scale
            weight = scores.softmax(dim=-1).to(k_span.dtype)
            pooled_k.append(torch.einsum("bhn,bhnd->bhd", weight, k_span))
            pooled_v.append(torch.einsum("bhn,bhnd->bhd", weight, v_span))
        spans_k = torch.stack(pooled_k, dim=2)  # [1, H, p, D]
        spans_v = torch.stack(pooled_v, dim=2)
        if p < num_spans:  # pad to a fixed P so every example lands on the same width
            pad_k = spans_k.new_zeros(1, spans_k.shape[1], num_spans - p, spans_k.shape[-1])
            pad_v = spans_v.new_zeros(1, spans_v.shape[1], num_spans - p, spans_v.shape[-1])
            spans_k = torch.cat([spans_k, pad_k], dim=2)
            spans_v = torch.cat([spans_v, pad_v], dim=2)
        return spans_k, spans_v

    def forward(
        self,
        summary: list[tuple[torch.Tensor, torch.Tensor]],
        lengths: "torch.Tensor | None" = None,
        num_spans: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``L`` x ``(K, V)`` each ``[B, H, N, D]`` (raw) -> ``(K, V)`` pair ``[B, H, L*P, D]``."""

        weight_dtype = self.q_base.dtype
        batch = summary[0][0].shape[0]
        full_n = summary[0][0].shape[2]

        token_k_sets, token_v_sets = [], []
        for b in range(batch):
            n = int(lengths[b]) if lengths is not None else full_n
            n = max(1, n)
            layer_k, layer_v = [], []
            for layer_idx, (k, v) in enumerate(summary):
                spans_k, spans_v = self._within_span(
                    k[b : b + 1, :, :n, :].to(weight_dtype),
                    v[b : b + 1, :, :n, :].to(weight_dtype),
                    self.q_base[layer_idx].to(weight_dtype),
                    num_spans,
                )
                layer_k.append(spans_k)
                layer_v.append(spans_v)
            token_k_sets.append(torch.cat(layer_k, dim=2))  # [1, H, L*P, D]
            token_v_sets.append(torch.cat(layer_v, dim=2))
        token_k = torch.cat(token_k_sets, dim=0)  # [B, H, L*P, D]
        token_v = torch.cat(token_v_sets, dim=0)
        return token_k, token_v


def _flatten_features(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """``[B,H,N,D]`` x2 -> ``[B,N,2*H*D]``, preserving summary positions."""

    batch, _, positions, _ = key.shape
    key_rows = key.permute(0, 2, 1, 3).reshape(batch, positions, -1)
    value_rows = value.permute(0, 2, 1, 3).reshape(batch, positions, -1)
    return torch.cat([key_rows, value_rows], dim=-1)


def _safe_num_heads(d_model: int, num_heads: int) -> int:
    """Largest head count <= ``num_heads`` that divides ``d_model`` (>=1)."""

    for h in range(min(num_heads, d_model), 0, -1):
        if d_model % h == 0:
            return h
    return 1


def _sinusoidal_pe(n: int, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Standard sin/cos positional encoding ``[n, d]`` (length-agnostic, no params)."""

    position = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    half = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    div = torch.exp(half * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)[:, : d // 2]
    return pe.to(dtype)


class KVTokTranslator(nn.Module):
    """Stage-1 pooled KVTok with Receiver-embedding-aware refinement and EIYW-v2 output head."""

    def __init__(
        self,
        *,
        sharer_layers: int,
        sharer_kv_heads: int,
        sharer_head_dim: int,
        receiver_hidden: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_spans: int = 5,
        num_tokens: int = 8,
        receiver_embed_rms: float = 1.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.sharer_layers = sharer_layers
        self.sharer_kv_heads = sharer_kv_heads
        self.sharer_head_dim = sharer_head_dim
        self.receiver_hidden = receiver_hidden
        self.d_model = int(d_model)
        self.num_heads = _safe_num_heads(self.d_model, num_heads)
        self.num_spans = num_spans
        self.num_tokens = num_tokens

        self.sharer_pool = _Stage1Pool(sharer_layers, sharer_kv_heads, sharer_head_dim)

        # Factorized KVTok projection. ``feature_proj`` is shared over all
        # layer/span positions; ``token_down`` mixes only that position axis.
        # Parameter count is O(2HD*d + LP*T), rather than O(2LPHD*T*d).
        self.feature_proj = nn.Linear(
            2 * sharer_kv_heads * sharer_head_dim, self.d_model
        )
        self.token_down = nn.Linear(sharer_layers * num_spans, num_tokens)
        self.norm_slots = nn.LayerNorm(self.d_model)

        # Receiver memory path (EIYW-v2 style): project raw receiver context
        # embeddings, inject sinusoidal order, then normalize once.
        self.receiver_proj = nn.Linear(receiver_hidden, self.d_model)
        self.norm_mem = nn.LayerNorm(self.d_model)

        # One lightweight receiver-aware fusion step: slot queries attend over
        # processed receiver memory.
        self.norm_cross = nn.LayerNorm(self.d_model)
        self.cross_attn = nn.MultiheadAttention(
            self.d_model, self.num_heads, batch_first=True
        )

        # EIYW-v2 style output head: normalize then map to receiver hidden,
        # scaled to the receiver's own embedding loudness.
        self.norm_out = nn.LayerNorm(self.d_model)
        self.out = nn.Linear(self.d_model, receiver_hidden)
        self.register_buffer("embed_scale", torch.tensor(float(receiver_embed_rms)))
        self.out_gain = nn.Parameter(torch.tensor(0.05))

        if zero_init:
            nn.init.normal_(self.out.weight, std=0.02)
            nn.init.zeros_(self.out.bias)

    def translate(
        self,
        sharer_summary: list[tuple[torch.Tensor, torch.Tensor]],
        receiver_embeds: torch.Tensor,
        *,
        sharer_lengths: "torch.Tensor | None" = None,
        receiver_lengths: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """Raw Sharer cache + Receiver embeddings -> soft tokens.

        Unlike EIYW's ``translate`` (where ``num_spans`` is a free per-call
        knob), KVTok's token-axis projection width is fixed by ``self.num_spans``
        at construction time (see :meth:`arch_config`), so it is always used
        here rather than accepted as an argument.
        """

        weight_dtype = self.feature_proj.weight.dtype
        aS_k, aS_v = self.sharer_pool(sharer_summary, lengths=sharer_lengths, num_spans=self.num_spans)

        summaries = _flatten_features(aS_k, aS_v).to(weight_dtype)
        features = self.feature_proj(summaries)  # [B, L_s*P, d]
        slots = self.token_down(features.transpose(1, 2)).transpose(1, 2)
        slots = self.norm_slots(slots)

        memory = self.receiver_proj(receiver_embeds.to(weight_dtype))
        n_r = memory.shape[1]
        memory = memory + _sinusoidal_pe(n_r, self.d_model, memory.device, memory.dtype)
        memory = self.norm_mem(memory)

        pad_mask = None
        if receiver_lengths is not None:
            positions = torch.arange(n_r, device=memory.device)
            pad_mask = positions.unsqueeze(0) >= receiver_lengths.to(memory.device).clamp(
                min=1
            ).unsqueeze(1)

        h = self.norm_cross(slots)
        attn, _ = self.cross_attn(
            h,
            memory,
            memory,
            key_padding_mask=pad_mask,
            need_weights=False,
        )
        x = slots + attn
        return self.out(self.norm_out(x)) * (self.embed_scale * self.out_gain)

    def forward(
        self,
        sharer_summary: list[tuple[torch.Tensor, torch.Tensor]],
        receiver_embeds: torch.Tensor,
        *,
        sharer_lengths: "torch.Tensor | None" = None,
        receiver_lengths: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        return self.translate(
            sharer_summary,
            receiver_embeds,
            sharer_lengths=sharer_lengths,
            receiver_lengths=receiver_lengths,
        )

    @classmethod
    def from_specs(
        cls,
        sharer_spec: "object",
        receiver_spec: "object",
        *,
        d_model: int = 512,
        num_heads: int = 8,
        num_spans: int = 5,
        num_tokens: int = 8,
        zero_init: bool = True,
    ) -> "KVTokTranslator":
        return cls(
            sharer_layers=sharer_spec.layers,
            sharer_kv_heads=sharer_spec.kv_heads,
            sharer_head_dim=sharer_spec.head_dim,
            receiver_hidden=receiver_spec.hidden,
            d_model=d_model,
            num_heads=num_heads,
            num_spans=num_spans,
            num_tokens=num_tokens,
            receiver_embed_rms=getattr(receiver_spec, "embed_rms", 1.0),
            zero_init=zero_init,
        )

    def arch_config(self) -> dict:
        """Self-describing architecture (the dims that determine every weight shape).

        The token-axis down-projection has shape ``[num_tokens, L_s*P]``, so
        ``num_spans`` and ``num_tokens`` remain checkpoint architecture.
        """

        return {
            "architecture_version": 2,
            "sharer_layers": self.sharer_layers,
            "sharer_kv_heads": self.sharer_kv_heads,
            "sharer_head_dim": self.sharer_head_dim,
            "receiver_hidden": self.receiver_hidden,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_spans": self.num_spans,
            "num_tokens": self.num_tokens,
        }

    @classmethod
    def from_arch_config(cls, cfg: dict, *, zero_init: bool = True) -> "KVTokTranslator":
        """Rebuild a translator from an :meth:`arch_config` dict."""

        if cfg.get("architecture_version") != 2:
            raise ValueError(
                "Legacy KVTok checkpoint uses the oversized dense/Receiver-cache "
                "architecture and cannot be loaded by compact KVTok v2; retrain it."
            )

        return cls(
            sharer_layers=cfg["sharer_layers"],
            sharer_kv_heads=cfg["sharer_kv_heads"],
            sharer_head_dim=cfg["sharer_head_dim"],
            receiver_hidden=cfg["receiver_hidden"],
            d_model=cfg.get("d_model", cfg["receiver_hidden"]),
            num_heads=cfg.get("num_heads", 8),
            num_spans=cfg["num_spans"],
            num_tokens=cfg["num_tokens"],
            zero_init=zero_init,
        )

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
