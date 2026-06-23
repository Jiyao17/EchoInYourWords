"""Output helpers for text-to-text baseline runs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runner import T2TResult


def summarize_metrics(results: Sequence[T2TResult]) -> dict[str, Any]:
    """Compute simple aggregate accuracy metrics."""

    evaluated = [result for result in results if result.is_correct is not None]
    correct = [result for result in evaluated if result.is_correct]

    accuracy = None
    if evaluated:
        accuracy = len(correct) / len(evaluated)

    metrics = {
        "total_examples": len(results),
        "evaluated_examples": len(evaluated),
        "correct_examples": len(correct),
        "accuracy": accuracy,
    }

    # Average any numeric free-form scores (e.g. em, f1) reported by a score_fn.
    score_totals: dict[str, float] = {}
    score_counts: dict[str, int] = {}
    for result in results:
        for key, value in result.scores.items():
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)):
                score_totals[key] = score_totals.get(key, 0.0) + float(value)
                score_counts[key] = score_counts.get(key, 0) + 1
    if score_totals:
        metrics["scores"] = {
            key: score_totals[key] / score_counts[key] for key in score_totals
        }

    return metrics


def summarize_latency(results: Sequence[T2TResult]) -> dict[str, Any]:
    """Compute aggregate latency statistics from result records."""

    totals = {
        "sharer_ms": 0.0,
        "receiver_ms": 0.0,
        "total_ms": 0.0,
    }

    for result in results:
        for key in totals:
            totals[key] += float(result.latency.get(key, 0.0))

    count = len(results)
    averages = {
        key: (value / count if count else None)
        for key, value in totals.items()
    }

    return {
        "total": totals,
        "average": averages,
    }


def summarize_tokens(results: Sequence[T2TResult]) -> dict[str, Any]:
    """Aggregate token counts across result records.

    Counts that are ``None`` (backend did not report them) are skipped, and the
    number of results that contributed to each field is recorded.
    """

    keys = ("sharer_input", "sharer_output", "receiver_input", "receiver_output")
    totals: dict[str, int] = {key: 0 for key in keys}
    counted: dict[str, int] = {key: 0 for key in keys}

    for result in results:
        for key in keys:
            value = result.token_counts.get(key)
            if value is not None:
                totals[key] += int(value)
                counted[key] += 1

    averages = {
        key: (totals[key] / counted[key] if counted[key] else None)
        for key in keys
    }

    return {
        "total": totals,
        "average": averages,
        "counted_examples": counted,
    }


def write_run_outputs(
    results: Sequence[T2TResult],
    output_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write standardized output files for a T2T run."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions_path = output_path / "predictions.jsonl"
    metrics_path = output_path / "metrics.json"
    latency_path = output_path / "latency.json"
    metadata_path = output_path / "run_metadata.json"

    with predictions_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    metrics = summarize_metrics(results)
    metrics["token_counts"] = summarize_tokens(results)
    latency = summarize_latency(results)
    run_metadata = {
        "method": "text_to_text",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **dict(metadata or {}),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    latency_path.write_text(json.dumps(latency, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")

    return {
        "predictions": predictions_path,
        "metrics": metrics_path,
        "latency": latency_path,
        "metadata": metadata_path,
    }

