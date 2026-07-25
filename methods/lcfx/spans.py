"""Span-boundary computation for the LCF-X pool.

The paper divides the sharer's context into ``P`` contiguous spans — "a
paragraph, passage, or fixed-size chunk" — and trains with HotpotQA's natural
five-paragraph structure (Appendix E.1: "Training partitions the sharer's
input using HotpotQA's five-paragraph structure, so P = 5 during training").
The formatted Sharer prompt also contains instruction/question blocks. Those
are not additional evidence spans: when the raw context is supplied, this
module derives starts only from its paragraphs, keeps the prompt prefix in the
first span, and leaves the prompt suffix in the last span. Thus HotpotQA's five
evidence paragraphs produce exactly ``P=5`` as in Appendix E.1. Torch-free.
"""

from __future__ import annotations

from typing import Any


def paragraph_span_starts_from_offsets(
    text: str,
    offsets: list[tuple[int, int]],
    context: str | None = None,
) -> list[int]:
    """Map natural paragraph starts to an already-tokenized text.

    This is the offset-level form of :func:`paragraph_span_starts`.  It lets
    callers use the exact same paragraph policy with token sequences produced
    by a chat template, whose rendered text must be tokenized with
    ``add_special_tokens=False``.
    """

    if not offsets:
        return [0]

    if context is not None:
        source = context.strip()
        if not source:
            return [0]
        source_start = text.find(source)
        if source_start < 0:
            # Do not silently turn formatting/question blocks into evidence
            # spans when a caller explicitly requested context boundaries.
            return [0]
    else:
        source = text
        source_start = 0

    # Only evidence paragraphs add starts. Formatting before the evidence stays
    # in the first span and the question/choices after it stay in the last.
    char_starts, pos = [0], 0
    for block in source.split("\n\n")[:-1]:
        pos += len(block) + 2
        char_starts.append(source_start + pos)

    starts, tok_idx = [], 0
    for char_start in char_starts:
        # Skip tokens that end at/before the block's first character (special
        # tokens report (0, 0) and are skipped for every block but the first).
        while tok_idx < len(offsets) and offsets[tok_idx][1] <= char_start:
            tok_idx += 1
        starts.append(min(tok_idx, len(offsets) - 1))
    return sorted(set(starts))


def paragraph_span_starts(
    tokenizer: Any,
    text: str,
    context: str | None = None,
) -> list[int]:
    """Token starts for natural evidence paragraphs in a formatted prompt.

    Tokenizes with the same arguments the sharer encode uses (defaults, so the
    offsets index the same id sequence) and returns the first token at-or-after
    each paragraph start. When ``context`` is provided, question/choice blocks
    in ``text`` do not create spans; the first span remains anchored at token 0
    so the prompt prefix is retained. Consecutive blank lines and blocks that
    vanish into a single token collapse naturally. Returns ``[0]`` for
    unstructured text.
    """

    offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    return paragraph_span_starts_from_offsets(text, offsets, context)


__all__ = ["paragraph_span_starts", "paragraph_span_starts_from_offsets"]
