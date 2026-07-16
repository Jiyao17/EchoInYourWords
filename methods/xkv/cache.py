"""KV-cache extraction and Receiver residual injection for XKV."""

from __future__ import annotations

from typing import Any


def extract_kv(cache: Any) -> list[tuple[Any, Any]]:
    """Return per-layer ``(K, V)`` tensors in ``[B,Hkv,N,D]`` layout."""

    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    if isinstance(cache, (list, tuple)):
        return [(layer[0], layer[1]) for layer in cache]
    raise TypeError(f"unsupported KV cache type: {type(cache).__name__}")


def add_residual_inplace(cache: Any, residuals: list[tuple[Any, Any]]) -> Any:
    """Add Receiver-shaped per-layer ``(dK,dV)`` tensors to ``cache``."""

    layers = extract_kv(cache)
    if len(residuals) != len(layers):
        raise ValueError(
            f"residual layer count {len(residuals)} != cache layers {len(layers)}"
        )
    updated = []
    for index, ((key, value), (d_key, d_value)) in enumerate(zip(layers, residuals)):
        if tuple(d_key.shape) != tuple(key.shape):
            raise ValueError(
                f"layer {index} dK shape {tuple(d_key.shape)} != cache K "
                f"shape {tuple(key.shape)}"
            )
        if tuple(d_value.shape) != tuple(value.shape):
            raise ValueError(
                f"layer {index} dV shape {tuple(d_value.shape)} != cache V "
                f"shape {tuple(value.shape)}"
            )
        updated.append(
            (
                key + d_key.to(device=key.device, dtype=key.dtype),
                value + d_value.to(device=value.device, dtype=value.dtype),
            )
        )

    if hasattr(cache, "layers"):
        for layer, (key, value) in zip(cache.layers, updated):
            layer.keys, layer.values = key, value
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for index, (key, value) in enumerate(updated):
            cache.key_cache[index], cache.value_cache[index] = key, value
    else:
        raise TypeError(
            "legacy tuple/list caches cannot be updated in place; use DynamicCache"
        )
    return cache


__all__ = ["add_residual_inplace", "extract_kv"]
