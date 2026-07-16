"""KV-cache extraction for KVTok.

Targets the transformers >= 5 ``DynamicCache`` layout, where a cache exposes
``cache.layers[i].keys`` / ``.values`` as tensors shaped
``[batch, num_kv_heads, seq_len, head_dim]`` and ``len(cache)`` is the layer
count. Torch is imported lazily so the rest of the package stays importable
without a GPU stack.

KVTok reads only the Sharer's raw KV for Stage-1 pooling. The Receiver provides
an embedding-table lookup and does not create a cache until generation prefill;
the translated tokens are prepended to its *input embeddings* (see
``model.py``).
"""

from __future__ import annotations

from typing import Any


def num_layers(cache: Any) -> int:
    return len(cache.layers)


def extract_kv(cache: Any) -> list[tuple[Any, Any]]:
    """Return ``[(keys, values), ...]`` per layer, each ``[B, Hkv, S, D]``."""

    return [(layer.keys, layer.values) for layer in cache.layers]
