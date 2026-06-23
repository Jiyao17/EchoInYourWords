"""Prompt builders for the text-to-text baseline."""

from __future__ import annotations

from .schema import T2TExample


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


def _choices_block(example: T2TExample) -> str:
    choices = format_choices(tuple(example.choices))
    if not choices:
        return ""
    return f"\n\nChoices:\n{choices}"


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


def build_receiver_only_prompt(
    example: T2TExample, context: str
) -> str:
    """Build a no-Sharer prompt where the Receiver answers from ``context``.

    Used for the LCF-X reference baselines (question-only / half-context /
    full-context): there is no Sharer message, only the Receiver's own
    available context (which may be empty for the question-only baseline).
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

