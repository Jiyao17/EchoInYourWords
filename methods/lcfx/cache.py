"""KV-cache extraction, pooling, and residual injection for LCF-X.

Targets the transformers >= 5 ``DynamicCache`` layout, where a cache exposes
``cache.layers[i].keys`` / ``.values`` as tensors shaped
``[batch, num_kv_heads, seq_len, head_dim]`` and ``len(cache)`` is the layer
count. Torch is imported lazily so the rest of the package stays importable
without a GPU stack.

Axis responsibilities (see fuser.py for the rest):

- ``pool_summary`` is a cheap, parameter-free **span pre-reduction** on the
  sequence axis (``N_S -> P`` span means), keeping the Sharer's own ``H_s, D_s``.
  It produces the token set the fuser's attention resampler consumes.
- The fuser's attention resampler then converts **all remaining axes the same
  way** (sequence ``P -> 1``, heads ``H_s -> H_r``, head_dim ``D_s -> D_r``, and
  layers ``L_s -> L_r``) via learnable Receiver-sized queries, so a residual that
  reaches ``add_residual_inplace`` is already in Receiver shape and is broadcast
  across the Receiver's ``N_R`` positions.
"""

from __future__ import annotations

from typing import Any


def num_layers(cache: Any) -> int:
    return len(cache.layers)


def extract_kv(cache: Any) -> list[tuple[Any, Any]]:
    """Return ``[(keys, values), ...]`` per layer, each ``[B, Hkv, S, D]``."""

    return [(layer.keys, layer.values) for layer in cache.layers]


def _span_pool(x: Any, num_spans: int) -> Any:
    """Mean-pool ``[B, H, S, D]`` into ``[B, H, P, D]`` span means (P = num_spans)."""

    import torch

    seq_len = x.shape[2]
    p = max(1, min(int(num_spans), int(seq_len)))
    spans = torch.tensor_split(x, p, dim=2)  # p chunks of [B, H, s_i, D]
    return torch.stack([span.mean(dim=2) for span in spans], dim=2)  # [B, H, P, D]


def pool_summary(cache: Any, num_spans: int = 1) -> list[tuple[Any, Any]]:
    """Span pre-reduction: per layer ``(K, V)`` each ``[B, Hkv, P, D]`` span means.

    ``num_spans == 1`` is a plain mean over the whole sequence (1 token/layer).
    The fuser's attention resampler turns these ``L_s * P`` tokens into a
    Receiver-shaped residual.
    """

    return [
        (_span_pool(layer.keys, num_spans), _span_pool(layer.values, num_spans))
        for layer in cache.layers
    ]


def add_residual_inplace(cache: Any, residuals: list[tuple[Any, Any]]) -> Any:
    """Add per-layer ``(dK, dV)`` residuals into a Receiver cache, in place.

    Each residual is broadcast over the sequence dimension, so a position-free
    ``[B, Hkv, 1, D]`` residual is added to every cached position (the LCF-X
    "broadcast across the Receiver sequence" behavior). Returns the same cache.
    """

    if len(residuals) != len(cache.layers):
        raise ValueError(
            f"residual layer count {len(residuals)} != cache layers {len(cache.layers)}"
        )
    for layer, (d_key, d_value) in zip(cache.layers, residuals):
        layer.keys = layer.keys + d_key.to(layer.keys.dtype)
        layer.values = layer.values + d_value.to(layer.values.dtype)
    return cache
