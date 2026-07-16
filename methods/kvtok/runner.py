"""Per-example runner for KVTok.

Orchestrates the pipeline for one example: prefill and Stage-1-pool the Sharer,
look up Receiver context embeddings without a Receiver forward, factorized-
project the Sharer summaries into ``num_tokens`` slots, condition them on the
Receiver embeddings, then decode with the soft tokens prepended.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from methods.common.parsing import parse_choice_answer
from methods.common.prompts import (
    build_receiver_encode_text,
    build_receiver_prompt,
    build_sharer_encode_text,
)

from .model import KVTokPair
from .schema import KVTokExample
from .translator import KVTokTranslator

ScoreFn = Callable[[str, str], dict[str, Any]]


@dataclass(frozen=True)
class KVTokResult:
    """Structured result for one KVTok example (uniform with the T2T contract)."""

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


def run_kvtok_example(
    example: KVTokExample,
    pair: KVTokPair,
    translator: KVTokTranslator,
    *,
    max_new_tokens: int = 64,
    score_fn: ScoreFn | None = None,
) -> KVTokResult:
    """Run the KVTok pipeline for a single example."""

    import torch

    sharer_text = build_sharer_encode_text(example)
    receiver_text = build_receiver_encode_text(example)
    receiver_prompt = build_receiver_prompt(example)

    start = perf_counter()
    sharer_kv, sharer_tokens = pair.sharer_summary(sharer_text)
    receiver_embeds, receiver_tokens = pair.receiver_context_embeds(receiver_text)
    sharer_ms = (perf_counter() - start) * 1000.0

    start = perf_counter()
    with torch.no_grad():
        soft_token = translator.translate(
            sharer_kv,
            receiver_embeds,
            receiver_lengths=torch.tensor([receiver_tokens], device=pair.device),
        )
    translator_ms = (perf_counter() - start) * 1000.0

    start = perf_counter()
    output, recv_in, recv_out, recv_first_ms, recv_last_ms = pair.decode_receiver(
        receiver_prompt, soft_token=soft_token, max_new_tokens=max_new_tokens
    )
    receiver_ms = (perf_counter() - start) * 1000.0

    # decode_receiver's own timestamps are relative to its call; shift them to be
    # relative to the start of this example (encode + translate already elapsed).
    pre_decode_ms = sharer_ms + translator_ms
    first_token_ms = pre_decode_ms + recv_first_ms if recv_first_ms is not None else None
    last_token_ms = pre_decode_ms + recv_last_ms if recv_last_ms is not None else None

    prediction = parse_choice_answer(output, example.choices)
    is_correct, scores = _score_prediction(prediction, example.answer, score_fn)
    num_soft_tokens = int(soft_token.shape[1])

    return KVTokResult(
        id=example.id,
        method="kvtok",
        sharer_prompt=sharer_text,
        sharer_message=(
            f"[{num_soft_tokens} soft tokens, Stage-1-only pool of {len(sharer_kv)} Sharer "
            f"layers x {translator.num_spans} spans, factorized down-projection + "
            f"receiver-memory cross-attn + EIYW-v2 output head]"
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
            "sharer_output": num_soft_tokens,  # soft-token slots, not text tokens
            "receiver_input": recv_in,
            "receiver_output": recv_out,
        },
        metadata=dict(example.metadata),
    )
