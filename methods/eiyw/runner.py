"""Per-example runner for EIYW (the receiver-anchored method).

Orchestrates the soft-token pipeline for one example: encode the Sharer context
-> compute the Receiver anchor -> translate to K soft tokens -> decode the
Receiver with those tokens prepended to its input.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from methods.common.parsing import parse_choice_answer
from methods.common.prompts import build_receiver_prompt, build_sharer_encode_text

from .model import EIYWPair
from .schema import EIYWExample
from .translator import EIYWTranslator

ScoreFn = Callable[[str, str], dict[str, Any]]


@dataclass(frozen=True)
class EIYWResult:
    """Structured result for one EIYW example (uniform with the T2T contract)."""

    id: str
    method: str
    sharer_prompt: str
    sharer_message: str  # short descriptor of the soft tokens (no NL message)
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


def _score_prediction(
    prediction: str | None, answer: str | None, score_fn: ScoreFn | None
) -> tuple[bool | None, dict[str, Any]]:
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


def run_eiyw_example(
    example: EIYWExample,
    pair: EIYWPair,
    translator: EIYWTranslator,
    *,
    num_spans: int = 4,
    max_new_tokens: int = 64,
    score_fn: ScoreFn | None = None,
) -> EIYWResult:
    """Run the EIYW receiver-anchored soft-token pipeline for a single example."""

    import torch

    sharer_text = build_sharer_encode_text(example)
    receiver_prompt = build_receiver_prompt(example)

    start = perf_counter()
    summary, sharer_tokens = pair.sharer_summary(sharer_text)
    sharer_ms = (perf_counter() - start) * 1000.0

    # Translate the Sharer cache to receiver-anchored soft tokens.
    start = perf_counter()
    with torch.no_grad():
        anchor = pair.receiver_anchor(receiver_prompt)
        soft_tokens = translator.translate(summary, anchor, num_spans=num_spans)
    fuser_ms = (perf_counter() - start) * 1000.0

    start = perf_counter()
    output, recv_in, recv_out = pair.decode_receiver(
        receiver_prompt, soft_tokens=soft_tokens, max_new_tokens=max_new_tokens
    )
    receiver_ms = (perf_counter() - start) * 1000.0

    prediction = parse_choice_answer(output, example.choices)
    is_correct, scores = _score_prediction(prediction, example.answer, score_fn)

    n_layers = len(summary)
    k = translator.num_soft_tokens
    return EIYWResult(
        id=example.id,
        method="eiyw",
        sharer_prompt=sharer_text,
        sharer_message=f"[soft tokens: {k} (pooled from {n_layers} layers x {num_spans} spans)]",
        receiver_prompt=receiver_prompt,
        receiver_output=output,
        prediction=prediction,
        answer=example.answer,
        is_correct=is_correct,
        scores=scores,
        latency={
            "sharer_ms": sharer_ms,
            "fuser_ms": fuser_ms,
            "receiver_ms": receiver_ms,
            "total_ms": sharer_ms + fuser_ms + receiver_ms,
        },
        token_counts={
            "sharer_input": sharer_tokens,
            "sharer_output": k,  # soft-token slots, not text tokens
            "receiver_input": recv_in,
            "receiver_output": recv_out,
        },
        metadata=dict(example.metadata),
    )


