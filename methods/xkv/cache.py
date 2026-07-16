"""KV-cache extraction and Receiver residual injection for XKV."""

from __future__ import annotations

from typing import Any


def extract_kv(cache: Any) -> list[tuple[Any, Any]]:
    """Return per-layer ``(K, V)`` tensors in ``[B,Hkv,N,D]`` layout."""

    return [(layer.keys, layer.values) for layer in cache.layers]


def add_residual_inplace(cache: Any, residuals: list[tuple[Any, Any]]) -> Any:
    """Add Receiver-shaped per-layer ``(dK,dV)`` tensors to ``cache``."""

    if len(residuals) != len(cache.layers):
        raise ValueError(
            f"residual layer count {len(residuals)} != cache layers {len(cache.layers)}"
        )
    for index, (layer, (d_key, d_value)) in enumerate(zip(cache.layers, residuals)):
        if tuple(d_key.shape) != tuple(layer.keys.shape):
            raise ValueError(
                f"layer {index} dK shape {tuple(d_key.shape)} != cache K "
                f"shape {tuple(layer.keys.shape)}"
            )
        if tuple(d_value.shape) != tuple(layer.values.shape):
            raise ValueError(
                f"layer {index} dV shape {tuple(d_value.shape)} != cache V "
                f"shape {tuple(layer.values.shape)}"
            )
        layer.keys = layer.keys + d_key.to(
            device=layer.keys.device, dtype=layer.keys.dtype
        )
        layer.values = layer.values + d_value.to(
            device=layer.values.device, dtype=layer.values.dtype
        )
    return cache


__all__ = ["add_residual_inplace", "extract_kv"]
