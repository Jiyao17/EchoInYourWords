"""Prompt builders specific to the text-to-text baseline.

The Sharer-message and Receiver-with-message prompts are T2T-specific (T2T is the
only method that passes a natural-language message). The method-agnostic pieces —
``format_choices`` and the no-Sharer ``build_receiver_only_prompt`` — live in
``methods.common.prompts`` and are re-exported here for convenience.
"""

from __future__ import annotations

from methods.common.prompts import (
    _choices_block,
    _context_or_placeholder,
    build_receiver_only_prompt,
    format_choices,
)

from .schema import T2TExample


def build_sharer_prompt(example: T2TExample) -> str:
    """Build the prompt used to ask the Sharer for a message.

    The Sharer may hold only part of the information needed (cross-context
    setting), so it is asked to relay relevant facts from its own information
    rather than guess a final answer it cannot support.
    """

    return (
        "You are helping another model answer a question.\n"
        "You may only have part of the information needed. Read your information "
        "below and write a concise message that relays the facts from it that are "
        "most relevant to the question.\n"
        "Only state facts supported by your information. Do not guess the final "
        "answer if your information is insufficient.\n\n"
        "Your information:\n"
        f"{_context_or_placeholder(example.sharer_context)}\n\n"
        "Question:\n"
        f"{example.question}"
        f"{_choices_block(example)}\n\n"
        "Message:"
    )


def build_receiver_prompt(example: T2TExample, sharer_message: str) -> str:
    """Build the prompt used to ask the Receiver for the final answer."""

    if example.choices:
        final_instruction = "\n\nRespond with only the option letter."
    else:
        final_instruction = (
            "\n\nRespond with only the answer as a short phrase, "
            "with no explanation."
        )

    return (
        "Answer the task using your own information and the message from another model.\n\n"
        "Your information:\n"
        f"{_context_or_placeholder(example.receiver_context)}\n\n"
        "Message from the other model:\n"
        f"{sharer_message.strip()}\n\n"
        "Question:\n"
        f"{example.question}"
        f"{_choices_block(example)}"
        f"{final_instruction}\n\n"
        "Answer:"
    )


__all__ = [
    "format_choices",
    "build_receiver_only_prompt",
    "build_sharer_prompt",
    "build_receiver_prompt",
]
