"""Prompt builders for LCF-X.

Two text surfaces:

- The **Sharer encode text** is what the Sharer reads to build the KV cache that
  gets pooled into a position-free latent summary. It is the Sharer's own context
  (plus the shared question), never the Receiver's evidence.
- The **Receiver prompt** is what the Receiver actually generates from. Crucially,
  it contains *only the Receiver's own context* — the Sharer's information arrives
  through the fused latent cache, **not** through text. So it is exactly the
  no-Sharer prompt, reused from the T2T package.
"""

from __future__ import annotations

from methods.t2t.prompts import build_receiver_only_prompt, format_choices
from methods.t2t.schema import T2TExample


def build_sharer_encode_text(example: T2TExample) -> str:
    """Text the Sharer encodes into its KV cache (its context + the question)."""

    context = example.sharer_context.strip() or "(no additional information)"
    text = f"Information:\n{context}\n\nQuestion:\n{example.question}"
    if example.choices:
        text += "\n\nChoices:\n" + format_choices(tuple(example.choices))
    return text


def build_receiver_prompt(example: T2TExample) -> str:
    """Receiver answers from its own context only; Sharer info comes via the cache."""

    return build_receiver_only_prompt(example, example.receiver_context)


__all__ = ["build_sharer_encode_text", "build_receiver_prompt", "format_choices"]
