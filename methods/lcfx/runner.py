"""Per-example runner for LCF-X.

Orchestrates the cross-context latent pipeline for one example:
encode the Sharer context -> pool to a position-free summary -> fuse to a
Receiver cache residual -> decode the Receiver with that residual injected.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from methods.common.parsing import parse_choice_answer
from methods.common.prompts import build_receiver_prompt, build_sharer_encode_text

from .cache import extract_kv
from .fuser import LCFXFuser
from .model import LCFXPair
from .schema import LCFXExample

ScoreFn = Callable[[str, str], dict[str, Any]]


@dataclass(frozen=True)
class LCFXResult:
    """Structured result for one LCF-X example (uniform with the T2T contract)."""

    id: str
    method: str
    sharer_prompt: str
    sharer_message: str  # short descriptor of the latent summary (no NL message)
    receiver_prompt: str
    receiver_output: str
    prediction: str | None
    answer: str | None
    is_correct: bool | None
    scores: dict[str, Any]
    latency: dict[str, float | None]
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


def run_lcfx_example(
    example: LCFXExample,
    pair: LCFXPair,
    fuser: LCFXFuser,
    *,
    num_spans: int = 1,
    max_new_tokens: int = 64,
    score_fn: ScoreFn | None = None,
) -> LCFXResult:
    """Run the LCF-X latent-communication pipeline for a single example."""

    sharer_text = build_sharer_encode_text(example)
    receiver_prompt = build_receiver_prompt(example)

    start = perf_counter()
    summary, sharer_tokens = pair.sharer_summary(sharer_text)
    sharer_ms = (perf_counter() - start) * 1000.0

    # Align the Sharer summary to Receiver shape up front; the joint projection
    # (which needs the Receiver cache) runs inside decode via this callback.
    start = perf_counter()
    aligned = fuser.align(summary, num_spans=num_spans)
    align_ms = (perf_counter() - start) * 1000.0
    project_ms: list[float] = []

    def residual_fn(receiver_cache: Any) -> list[tuple[Any, Any]]:
        tic = perf_counter()
        residual = fuser.project(aligned, extract_kv(receiver_cache))
        project_ms.append((perf_counter() - tic) * 1000.0)
        return residual

    start = perf_counter()
    output, recv_in, recv_out, recv_first_ms, recv_last_ms = pair.decode_receiver(
        receiver_prompt, residual_fn=residual_fn, max_new_tokens=max_new_tokens
    )
    decode_ms = (perf_counter() - start) * 1000.0

    # Attribute the projection time to the fuser, not the Receiver decode.
    fuser_ms = align_ms + sum(project_ms)
    receiver_ms = decode_ms - sum(project_ms)

    # decode_receiver's own timestamps are relative to its call; shift them to be
    # relative to the start of this example (sharer + align already elapsed).
    pre_decode_ms = sharer_ms + align_ms
    first_token_ms = pre_decode_ms + recv_first_ms if recv_first_ms is not None else None
    last_token_ms = pre_decode_ms + recv_last_ms if recv_last_ms is not None else None

    prediction = parse_choice_answer(output, example.choices)
    is_correct, scores = _score_prediction(prediction, example.answer, score_fn)

    n_layers = len(summary)
    return LCFXResult(
        id=example.id,
        method="lcf_x",
        sharer_prompt=sharer_text,
        sharer_message=f"[latent cache summary: {n_layers} layers x {num_spans} spans]",
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
            "first_token_ms": first_token_ms,
            "last_token_ms": last_token_ms,
        },
        token_counts={
            "sharer_input": sharer_tokens,
            "sharer_output": n_layers * num_spans,  # latent summary slots, not text tokens
            "receiver_input": recv_in,
            "receiver_output": recv_out,
        },
        metadata=dict(example.metadata),
    )
