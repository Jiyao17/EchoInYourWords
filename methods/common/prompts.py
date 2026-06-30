"""Shared prompt primitives used across methods.

The pieces here are method-agnostic: choice formatting, and the **no-Sharer**
Receiver prompt (the Receiver answers from a given context with no message from
another model). T2T builds its own Sharer/Receiver message prompts on top of
these; LCF-X and EIYW reuse ``build_receiver_only_prompt`` directly (the Sharer's
information reaches the Receiver through the cache / soft tokens, not text).
"""

from __future__ import annotations

from .schema import Example


def format_choices(choices: list[str] | tuple[str, ...]) -> str:
    """Format choices as lettered multiple-choice options."""

    if not choices:
        return ""

    lines = []
    for idx, choice in enumerate(choices):
        label = chr(ord("A") + idx)
        lines.append(f"{label}. {choice}")
    return "\n".join(lines)


def _context_or_placeholder(context: str) -> str:
    return context.strip() if context.strip() else "(no additional information)"


def _choices_block(example: Example) -> str:
    choices = format_choices(tuple(example.choices))
    if not choices:
        return ""
    return f"\n\nChoices:\n{choices}"


def build_receiver_only_prompt(example: Example, context: str) -> str:
    """Build a no-Sharer prompt where the Receiver answers from ``context``.

    Used for the receiver reference baselines (question-only / half-context /
    full-context) and as the Receiver prompt for LCF-X / EIYW: there is no Sharer
    text message, only the Receiver's own available context (which may be empty
    for the question-only baseline).
    """

    if example.choices:
        final_instruction = "\n\nRespond with only the option letter."
    else:
        final_instruction = (
            "\n\nRespond with only the answer as a short phrase, "
            "with no explanation."
        )

    info_block = ""
    if context.strip():
        info_block = f"Your information:\n{context.strip()}\n\n"

    return (
        "Answer the question using your own information.\n\n"
        f"{info_block}"
        "Question:\n"
        f"{example.question}"
        f"{_choices_block(example)}"
        f"{final_instruction}\n\n"
        "Answer:"
    )


__all__ = ["format_choices", "build_receiver_only_prompt"]
