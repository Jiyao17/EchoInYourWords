"""Span-boundary computation for the LCF-X pool.

The paper divides the sharer's context into ``P`` contiguous spans — "a
paragraph, passage, or fixed-size chunk" — and trains with HotpotQA's natural
five-paragraph structure (Appendix E.1: "Training partitions the sharer's
input using HotpotQA's five-paragraph structure, so P = 5 during training").
Our sharer encode text joins paragraphs (and the Information/Question/Choices
blocks) with blank lines, so natural spans are the ``"\\n\\n"``-separated
blocks; this module maps their character starts to token offsets. Torch-free.
"""

from __future__ import annotations

from typing import Any


def paragraph_span_starts(tokenizer: Any, text: str) -> list[int]:
    """Token start offsets of the ``"\\n\\n"``-separated blocks of ``text``.

    Tokenizes with the same arguments the sharer encode uses (defaults, so the
    offsets index the same id sequence) and returns the first token at-or-after
    each block's character start, deduplicated and anchored at 0. Consecutive
    blank lines and blocks that vanish into a single token collapse naturally.
    Returns ``[0]`` (one span) for unstructured text.
    """

    offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    if not offsets:
        return [0]

    char_starts, pos = [0], 0
    for block in text.split("\n\n")[:-1]:
        pos += len(block) + 2
        char_starts.append(pos)

    starts, tok_idx = [], 0
    for char_start in char_starts:
        # Skip tokens that end at/before the block's first character (special
        # tokens report (0, 0) and are skipped for every block but the first).
        while tok_idx < len(offsets) and offsets[tok_idx][1] <= char_start:
            tok_idx += 1
        starts.append(min(tok_idx, len(offsets) - 1))
    return sorted(set(starts))


__all__ = ["paragraph_span_starts"]
