"""Natural-span offsets for XKV's plain Sharer and chat Receiver inputs."""

from __future__ import annotations

from typing import Any

from methods.lcfx.spans import paragraph_span_starts


def _block_char_starts(text: str) -> list[int]:
    starts, pos = [0], 0
    for block in text.split("\n\n")[:-1]:
        pos += len(block) + 2
        starts.append(pos)
    return starts


def _render_chat(tokenizer: Any, text: str) -> str | None:
    if not getattr(tokenizer, "chat_template", None):
        return None
    messages = [{"role": "user", "content": text}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def chat_paragraph_span_starts(tokenizer: Any, text: str) -> list[int]:
    """Token offsets of ``text`` blocks inside the Receiver's rendered chat.

    If the tokenizer does not expose a chat template, this is identical to the
    plain-text helper. If a template transforms the content so it cannot be
    located verbatim, return ``[0]`` rather than inventing incorrect offsets.
    """

    rendered = _render_chat(tokenizer, text)
    if rendered is None:
        return paragraph_span_starts(tokenizer, text)
    content_start = rendered.find(text)
    if content_start < 0:
        return [0]

    offsets = tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True
    )["offset_mapping"]
    if not offsets:
        return [0]

    starts: list[int] = []
    tok_idx = 0
    for local_start in _block_char_starts(text):
        char_start = content_start + local_start
        while tok_idx < len(offsets) and offsets[tok_idx][1] <= char_start:
            tok_idx += 1
        starts.append(min(tok_idx, len(offsets) - 1))
    # Include chat-system/control tokens in the first span.
    return sorted(set([0, *starts]))


__all__ = ["chat_paragraph_span_starts", "paragraph_span_starts"]
