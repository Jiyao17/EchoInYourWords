"""Per-example XKV inference orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from methods.common.parsing import parse_choice_answer
from methods.common.prompts import build_receiver_prompt, build_sharer_encode_text

from .cache import extract_kv
from .model import XKVPair
from .schema import XKVExample
from .spans import chat_paragraph_span_starts, paragraph_span_starts
from .translator import XKVTranslator

ScoreFn = Callable[[str, str], dict[str, Any]]


@dataclass(frozen=True)
class XKVResult:
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


def _score(
    prediction: str | None, answer: str | None, score_fn: ScoreFn | None
) -> tuple[bool | None, dict[str, Any]]:
    if prediction is None or answer is None:
        return None, {}
    if score_fn is None:
        return prediction.strip().upper() == answer.strip().upper(), {}
    scores = dict(score_fn(prediction, answer))
    correct = bool(scores["is_correct"]) if "is_correct" in scores else None
    return correct, scores


def run_xkv_example(
    example: XKVExample,
    pair: XKVPair,
    translator: XKVTranslator,
    *,
    num_spans: int = 5,
    span_scheme: str = "paragraphs",
    max_new_tokens: int = 64,
    score_fn: ScoreFn | None = None,
) -> XKVResult:
    sharer_text = build_sharer_encode_text(example)
    receiver_prompt = build_receiver_prompt(example)

    started = perf_counter()
    sharer_cache, sharer_tokens = pair.sharer_summary(sharer_text)
    sharer_ms = (perf_counter() - started) * 1000.0

    sharer_boundaries = (
        [paragraph_span_starts(pair.sharer_tok, sharer_text)]
        if span_scheme == "paragraphs"
        else None
    )
    receiver_boundaries = (
        [chat_paragraph_span_starts(pair.receiver_tok, receiver_prompt)]
        if span_scheme == "paragraphs"
        else None
    )
    translator_times: list[float] = []

    def residual_fn(receiver_dynamic_cache: Any) -> list[tuple[Any, Any]]:
        tic = perf_counter()
        residual = translator.translate(
            sharer_cache,
            extract_kv(receiver_dynamic_cache),
            num_spans=num_spans,
            sharer_span_boundaries=sharer_boundaries,
            receiver_span_boundaries=receiver_boundaries,
            sharer_windowed_cache=pair.sharer_spec.is_gemma3,
            receiver_windowed_cache=pair.receiver_spec.is_gemma3,
        )
        translator_times.append((perf_counter() - tic) * 1000.0)
        return residual

    started = perf_counter()
    output, receiver_in, receiver_out, first_ms, last_ms = pair.decode_receiver(
        receiver_prompt,
        residual_fn=residual_fn,
        max_new_tokens=max_new_tokens,
    )
    decode_total_ms = (perf_counter() - started) * 1000.0
    translator_ms = sum(translator_times)
    receiver_ms = decode_total_ms - translator_ms
    first_token_ms = sharer_ms + first_ms if first_ms is not None else None
    last_token_ms = sharer_ms + last_ms if last_ms is not None else None

    prediction = parse_choice_answer(output, example.choices)
    is_correct, scores = _score(prediction, example.answer, score_fn)
    spans = len(sharer_boundaries[0]) if sharer_boundaries else num_spans
    return XKVResult(
        id=example.id,
        method="xkv",
        sharer_prompt=sharer_text,
        sharer_message=(
            f"[XKV dual-cache update: {len(sharer_cache)} Sharer layers, "
            f"{translator.receiver_layers} Receiver layers, {spans} spans, "
            f"{translator.pool_slots} pooled slots/layer]"
        ),
        receiver_prompt=receiver_prompt,
        receiver_output=output,
        prediction=prediction,
        answer=example.answer,
        is_correct=is_correct,
        scores=scores,
        latency={
            "sharer_ms": sharer_ms,
            "fuser_ms": translator_ms,
            "receiver_ms": receiver_ms,
            "total_ms": sharer_ms + translator_ms + receiver_ms,
            "first_token_ms": first_token_ms,
            "last_token_ms": last_token_ms,
        },
        token_counts={
            "sharer_input": sharer_tokens,
            "sharer_output": len(sharer_cache) * translator.pool_slots,
            "receiver_input": receiver_in,
            "receiver_output": receiver_out,
        },
        metadata=dict(example.metadata),
    )


__all__ = ["XKVResult", "run_xkv_example"]
