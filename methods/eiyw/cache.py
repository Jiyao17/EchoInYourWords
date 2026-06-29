"""KV-cache extraction for EIYW.

Targets the transformers >= 5 ``DynamicCache`` layout, where a cache exposes
``cache.layers[i].keys`` / ``.values`` as tensors shaped
``[batch, num_kv_heads, seq_len, head_dim]`` and ``len(cache)`` is the layer
count. Torch is imported lazily so the rest of the package stays importable
without a GPU stack.

Unlike LCF-X, EIYW does **not** inject a residual into the Receiver cache — the
Sharer's information is translated into soft tokens prepended to the Receiver
*input* (see ``translator.py`` / ``model.py``). We only need to read the Sharer's
raw KV here.
"""

from __future__ import annotations

from typing import Any


def num_layers(cache: Any) -> int:
    return len(cache.layers)


def extract_kv(cache: Any) -> list[tuple[Any, Any]]:
    """Return ``[(keys, values), ...]`` per layer, each ``[B, Hkv, S, D]``."""

    return [(layer.keys, layer.values) for layer in cache.layers]
