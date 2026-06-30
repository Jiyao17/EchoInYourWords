"""Pure-Python runner for one text-to-text communication example."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from methods.common.parsing import parse_choice_answer

from .prompts import (
    build_receiver_only_prompt,
    build_receiver_prompt,
    build_sharer_prompt,
)
from .schema import T2TExample


@dataclass(frozen=True)
class GenerationOutput:
    """Result of a single text generation call.

    A ``TextGenerator`` may return either a plain string (token counts unknown)
    or this richer object when the backend can report token usage.
    """

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


# A generator maps a prompt to either raw text or a GenerationOutput.
TextGenerator = Callable[[str], "str | GenerationOutput"]


@dataclass(frozen=True)
class T2TResult:
    """Structured result for one text-to-text example."""

    id: str
    method: str
    sharer_prompt: str
    sharer_message: str
    receiver_prompt: str
    receiver_output: str
    prediction: str | None
    answer: str | None
    is_correct: bool | None
    scores: dict[str, Any]
    latency: dict[str, float]
    token_counts: dict[str, int | None]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "id": self.id,
            "method": self.method,
            "sharer_prompt": self.sharer_prompt,
            "sharer_message": self.sharer_message,
            "receiver_prompt": self.receiver_prompt,
            "receiver_output": self.receiver_output,
            "prediction": self.prediction,
            "answer": self.answer,
            "is_correct": self.is_correct,
            "scores": dict(self.scores),
            "latency": dict(self.latency),
            "token_counts": dict(self.token_counts),
            "metadata": dict(self.metadata),
        }


def _timed_generate(
    generator: TextGenerator, prompt: str
) -> tuple[GenerationOutput, float]:
    start = perf_counter()
    output = generator(prompt)
    latency_ms = (perf_counter() - start) * 1000.0
    if isinstance(output, GenerationOutput):
        normalized = GenerationOutput(
            text=str(output.text).strip(),
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
        )
    else:
        normalized = GenerationOutput(text=str(output).strip())
    return normalized, latency_ms


ScoreFn = Callable[[str, str], dict[str, Any]]


def _score_prediction(
    prediction: str | None, answer: str | None, score_fn: ScoreFn | None
) -> tuple[bool | None, dict[str, Any]]:
    """Score a prediction, returning ``(is_correct, scores)``."""

    is_correct: bool | None = None
    scores: dict[str, Any] = {}
    if answer is not None and prediction is not None:
        if score_fn is not None:
            scores = dict(score_fn(prediction, answer))
            if "is_correct" in scores:
                is_correct = bool(scores["is_correct"])
        else:
            is_correct = prediction.strip().upper() == answer.strip().upper()
    return is_correct, scores


def run_t2t_example(
    example: T2TExample,
    sharer_generate: TextGenerator,
    receiver_generate: TextGenerator,
    score_fn: ScoreFn | None = None,
) -> T2TResult:
    """Run the text-to-text baseline for a single example.

    Args:
        example: Normalized task example.
        sharer_generate: Generator for the Sharer message.
        receiver_generate: Generator for the Receiver answer.
        score_fn: Optional free-form scorer ``(prediction, answer) -> dict``.
            The dict should include ``is_correct`` (bool) plus any extra metrics
            (e.g. ``em``, ``f1``) which are stored on the result. When omitted,
            multiple-choice exact-letter matching is used.
    """

    sharer_prompt = build_sharer_prompt(example)
    sharer_gen, sharer_ms = _timed_generate(sharer_generate, sharer_prompt)
    sharer_message = sharer_gen.text

    receiver_prompt = build_receiver_prompt(example, sharer_message)
    receiver_gen, receiver_ms = _timed_generate(receiver_generate, receiver_prompt)
    receiver_output = receiver_gen.text

    prediction = parse_choice_answer(receiver_output, example.choices)
    is_correct, scores = _score_prediction(prediction, example.answer, score_fn)

    return T2TResult(
        id=example.id,
        method="text_to_text",
        sharer_prompt=sharer_prompt,
        sharer_message=sharer_message,
        receiver_prompt=receiver_prompt,
        receiver_output=receiver_output,
        prediction=prediction,
        answer=example.answer,
        is_correct=is_correct,
        scores=scores,
        latency={
            "sharer_ms": sharer_ms,
            # No translation/fusion stage in text-to-text; kept for a uniform
            # cross-method latency contract (cache/latent methods report >0).
            "fuser_ms": 0.0,
            "receiver_ms": receiver_ms,
            "total_ms": sharer_ms + receiver_ms,
        },
        token_counts={
            "sharer_input": sharer_gen.input_tokens,
            "sharer_output": sharer_gen.output_tokens,
            "receiver_input": receiver_gen.input_tokens,
            "receiver_output": receiver_gen.output_tokens,
        },
        metadata=dict(example.metadata),
    )


# Receiver-only context for each LCF-X reference baseline.
RECEIVER_CONTEXT_MODES = ("question_only", "half", "full")


def _receiver_context(example: T2TExample, context_mode: str) -> str:
    """Resolve the Receiver's available context for a no-Sharer baseline."""

    if context_mode == "question_only":
        return ""
    if context_mode == "half":
        return example.receiver_context
    if context_mode == "full":
        parts = [example.sharer_context, example.receiver_context]
        return "\n\n".join(p for p in parts if p.strip())
    raise ValueError(
        f"Unknown context_mode {context_mode!r}; expected one of "
        f"{RECEIVER_CONTEXT_MODES}"
    )


def run_receiver_only_example(
    example: T2TExample,
    receiver_generate: TextGenerator,
    context_mode: str = "half",
    score_fn: ScoreFn | None = None,
) -> T2TResult:
    """Run a no-Sharer reference baseline (LCF-X).

    The Receiver answers from a subset of the evidence and no Sharer message:

    - ``question_only``: no context at all;
    - ``half``: only the Receiver's own partition (5 of 10 paragraphs);
    - ``full``: both partitions combined (all 10 paragraphs).
    """

    context = _receiver_context(example, context_mode)
    receiver_prompt = build_receiver_only_prompt(example, context)
    receiver_gen, receiver_ms = _timed_generate(receiver_generate, receiver_prompt)
    receiver_output = receiver_gen.text

    prediction = parse_choice_answer(receiver_output, example.choices)
    is_correct, scores = _score_prediction(prediction, example.answer, score_fn)

    metadata = dict(example.metadata)
    metadata["context_mode"] = context_mode

    return T2TResult(
        id=example.id,
        method=f"receiver_{context_mode}",
        sharer_prompt="",
        sharer_message="",
        receiver_prompt=receiver_prompt,
        receiver_output=receiver_output,
        prediction=prediction,
        answer=example.answer,
        is_correct=is_correct,
        scores=scores,
        latency={
            "sharer_ms": 0.0,
            "fuser_ms": 0.0,
            "receiver_ms": receiver_ms,
            "total_ms": receiver_ms,
        },
        token_counts={
            "sharer_input": None,
            "sharer_output": None,
            "receiver_input": receiver_gen.input_tokens,
            "receiver_output": receiver_gen.output_tokens,
        },
        metadata=metadata,
    )

