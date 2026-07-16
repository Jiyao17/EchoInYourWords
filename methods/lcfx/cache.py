"""KV-cache extraction and residual injection for LCF-X.

Targets the transformers >= 5 ``DynamicCache`` layout, where a cache exposes
``cache.layers[i].keys`` / ``.values`` as tensors shaped
``[batch, num_kv_heads, seq_len, head_dim]`` and ``len(cache)`` is the layer
count. Torch is imported lazily so the rest of the package stays importable
without a GPU stack.

This module only extracts raw KV and injects residuals. All pooling (the paper's
per-layer ``N_S -> P -> 1`` two-stage attention) lives in ``fuser.LCFXFuser.align``;
by the time a residual reaches ``add_residual_inplace`` it is already in Receiver
shape (broadcast over ``N_R`` inside ``fuser.project``).
"""

from __future__ import annotations

from typing import Any


def num_layers(cache: Any) -> int:
    return len(cache.layers)


def extract_kv(cache: Any) -> list[tuple[Any, Any]]:
    """Return ``[(keys, values), ...]`` per layer, each ``[B, Hkv, S, D]``."""

    return [(layer.keys, layer.values) for layer in cache.layers]


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
