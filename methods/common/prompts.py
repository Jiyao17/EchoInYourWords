"""Single home for every prompt builder used across methods.

Three families live here so methods don't duplicate or accidentally share them:

- **Communication prompts** used by the methods (T2T, LCF-X, EIYW):
  - ``build_sharer_prompt`` / ``build_receiver_message_prompt`` — T2T's
    natural-language Sharer message and the Receiver-with-message prompt.
  - ``build_sharer_encode_text`` / ``build_receiver_prompt`` — the latent
    methods' Sharer encode text and the no-message Receiver prompt (the Sharer's
    information reaches the Receiver through the cache / soft tokens, not text).
- **Historical single-model inference**: ``build_single_model_prompt`` — the
  dedicated prompt used by older full-context runs. Current half/full Receiver
  baselines use ``build_receiver_prompt`` for matched answer instructions.

All builders are duck-typed on the example fields ``question``, ``choices``,
``sharer_context``, ``receiver_context``; every ``*Example`` provides them.
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


def _final_instruction(example: Example) -> str:
    if example.choices:
        return "\n\nRespond with only the option letter."
    return (
        "\n\nRespond with only the answer as a short phrase, "
        "with no explanation."
    )


# --------------------------------------------------------------------------- #
# T2T communication prompts (natural-language message between two models)
# --------------------------------------------------------------------------- #
def build_sharer_prompt(example: Example) -> str:
    """Build the prompt used to ask the Sharer for a message.

    The Sharer may hold only part of the information needed (cross-context
    setting), so it is asked to relay relevant facts from its own information
    rather than guess a final answer it cannot support.
    """

    return (
        "You and another model are answering a question together. You hold only "
        "PART of the information; the other model holds the rest and will write "
        "the final answer using your message plus its own information.\n\n"
        "Write a concise message that passes on the facts from your information "
        "that could help answer the question. You cannot see the other model's "
        "information, so include the specific names, entities, dates, numbers, "
        "and relationships from yours — even ones whose relevance is not obvious "
        "— since one of them may be the link the other model needs.\n\n"
        "State only facts contained in your information. Do not guess or give a "
        "final answer; your information alone is not enough to answer.\n\n"
        "Your information:\n"
        f"{_context_or_placeholder(example.sharer_context)}\n\n"
        "Question:\n"
        f"{example.question}"
        f"{_choices_block(example)}\n\n"
        "Message:"
    )


def build_receiver_message_prompt(example: Example, sharer_message: str) -> str:
    """Build the T2T prompt asking the Receiver for the final answer.

    The Receiver answers from its own information plus the Sharer's text message.
    """

    return (
        "Answer the task using your own information and the message from another model.\n\n"
        "Your information:\n"
        f"{_context_or_placeholder(example.receiver_context)}\n\n"
        "Message from the other model:\n"
        f"{sharer_message.strip()}\n\n"
        "Question:\n"
        f"{example.question}"
        f"{_choices_block(example)}"
        f"{_final_instruction(example)}\n\n"
        "Answer:"
    )


# --------------------------------------------------------------------------- #
# Latent-method communication prompts (LCF-X, EIYW)
# --------------------------------------------------------------------------- #
def build_sharer_encode_text(example: Example) -> str:
    """Text the Sharer encodes into its KV cache (its context + the question).

    Shared by LCF-X and EIYW: the Sharer's own context, never the Receiver's
    evidence, with the shared question (and choices, if any).
    """

    context = example.sharer_context.strip() or "(no additional information)"
    text = f"Information:\n{context}\n\nQuestion:\n{example.question}"
    if example.choices:
        text += "\n\nChoices:\n" + format_choices(tuple(example.choices))
    return text


def build_receiver_encode_text(example: Example) -> str:
    """Text the Receiver encodes into its own KV cache (its context + the question).

    EIYW-only: mirrors ``build_sharer_encode_text`` but for the Receiver's own
    context, so the translator can pool the Receiver's cache the same way it
    pools the Sharer's (symmetric, position-free) before concatenating the two.
    Plain text, not chat-templated -- this cache is pooled, never generated from.
    """

    context = example.receiver_context.strip() or "(no additional information)"
    text = f"Information:\n{context}\n\nQuestion:\n{example.question}"
    if example.choices:
        text += "\n\nChoices:\n" + format_choices(tuple(example.choices))
    return text


def build_receiver_prompt(example: Example) -> str:
    """Latent-method Receiver prompt: the Receiver answers from its own context.

    Shared by LCF-X and EIYW. The Sharer's information arrives through the fused
    cache / soft tokens, **not** through text, so this is exactly the Receiver's
    own-context prompt with no Sharer message.
    """

    context = example.receiver_context
    info_block = ""
    if context.strip():
        info_block = f"Your information:\n{context.strip()}\n\n"

    return (
        "Answer the question using your own information.\n\n"
        f"{info_block}"
        "Question:\n"
        f"{example.question}"
        f"{_choices_block(example)}"
        f"{_final_instruction(example)}\n\n"
        "Answer:"
    )


# --------------------------------------------------------------------------- #
# Single-model inference (one model answers from a given context)
# --------------------------------------------------------------------------- #
def _single_model_final_instruction(example: Example) -> str:
    """Answer-format instruction for the single-model baseline.

    Deliberately separate from the shared ``_final_instruction`` (used by the
    method receiver prompts) so tuning the baseline never disturbs the methods.

    The wording follows the datasets' evaluation conventions, where the
    answer-format instruction is the dominant lever for the scoring metric:

    - Multiple choice (QASC): the OpenAI simple-evals / OLMES "MCF" style —
      answer with only the option letter, and state the valid range (A-H) so
      the model doesn't emit an out-of-range or free-text answer.
    - Free-form (HotpotQA-bridge, MuSiQue): gold answers are short extractive
      spans scored by EM/F1, so ask for the shortest span copied from the
      evidence with no explanation (verbose answers are the main EM killer).
    """

    if example.choices:
        last_letter = chr(ord("A") + min(len(example.choices), 26) - 1)
        return (
            f"\n\nRespond with only the letter (A-{last_letter}) of the correct "
            "option."
        )
    return (
        "\n\nAnswer with the shortest possible span copied from the information "
        "(usually a single name, entity, date, or number). "
        "Give only the answer, with no explanation."
    )


def build_single_model_prompt(example: Example, context: str) -> str:
    """Dedicated prompt for a single model answering from ``context``.

    Intentionally independent of the method receiver prompts: this is the
    single-model baseline (one model, no communication), so changes to the
    method prompts must never affect it (and vice versa).
    """

    info_block = ""
    if context.strip():
        info_block = f"Information:\n{context.strip()}\n\n"

    return (
        "Answer the question using the information provided.\n\n"
        f"{info_block}"
        "Question:\n"
        f"{example.question}"
        f"{_choices_block(example)}"
        f"{_single_model_final_instruction(example)}\n\n"
        "Answer:"
    )


__all__ = [
    "format_choices",
    "build_sharer_prompt",
    "build_receiver_message_prompt",
    "build_sharer_encode_text",
    "build_receiver_encode_text",
    "build_receiver_prompt",
    "build_single_model_prompt",
]
