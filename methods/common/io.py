"""Shared result aggregation + the standardized output writer.

Every method writes the same files via :func:`write_run_outputs`. The aggregators
and the writer are duck-typed over the per-method result dataclasses (T2TResult /
LCFXResult / EIYWResult): they only read ``is_correct`` / ``scores`` /
``latency`` / ``token_counts`` / ``prediction`` / ``answer`` and call
``result.to_dict()``, so one implementation serves every method.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def summarize_metrics(results: Sequence[Any]) -> dict[str, Any]:
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


def summarize_latency(results: Sequence[Any]) -> dict[str, Any]:
    """Compute aggregate latency statistics from result records."""

    totals = {
        "sharer_ms": 0.0,
        "fuser_ms": 0.0,
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


def summarize_tokens(results: Sequence[Any]) -> dict[str, Any]:
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


def _clean_cell(text: Any) -> str:
    """Collapse whitespace so a value fits one TSV cell."""

    return " ".join(str("" if text is None else text).split())


def write_predictions_brief(results: Sequence[Any], path: str | Path) -> Path:
    """Write a compact, eyeball-friendly TSV: one row per example with just the
    answer, the model prediction, and the EM/F1 scores.

    A quick-scan companion to ``predictions.jsonl`` (which also carries the full
    prompts and is too verbose to skim). ``em``/``f1`` come from the free-form
    ``scores`` when present; otherwise ``em`` falls back to ``is_correct`` (e.g.
    multiple-choice) and ``f1`` is left blank.
    """

    path = Path(path)
    columns = ("id", "em", "f1", "prediction", "answer")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for result in results:
            scores = getattr(result, "scores", {}) or {}
            if "em" in scores:
                em = "1" if scores["em"] else "0"
            elif result.is_correct is not None:
                em = "1" if result.is_correct else "0"
            else:
                em = ""
            f1 = f"{float(scores['f1']):.3f}" if "f1" in scores else ""
            row = (
                _clean_cell(result.id),
                em,
                f1,
                _clean_cell(result.prediction),
                _clean_cell(result.answer),
            )
            handle.write("\t".join(row) + "\n")
    return path


def write_run_outputs(
    results: Sequence[Any],
    output_dir: str | Path,
    *,
    method: str = "unknown",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the standardized output files for any method's run.

    Files: ``predictions.jsonl``, ``predictions_brief.tsv``, ``metrics.json``,
    ``latency.json``, ``run_metadata.json``. ``method`` names the run; a
    ``"method"`` key in ``metadata`` overrides it (e.g. the no-fusion path of a
    latent method records ``"receiver_only"``).
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions_path = output_path / "predictions.jsonl"
    brief_path = output_path / "predictions_brief.tsv"
    metrics_path = output_path / "metrics.json"
    latency_path = output_path / "latency.json"
    metadata_path = output_path / "run_metadata.json"

    with predictions_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    write_predictions_brief(results, brief_path)

    metrics = summarize_metrics(results)
    metrics["token_counts"] = summarize_tokens(results)
    latency = summarize_latency(results)
    run_metadata = {
        "method": method,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **dict(metadata or {}),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    latency_path.write_text(json.dumps(latency, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")

    return {
        "predictions": predictions_path,
        "predictions_brief": brief_path,
        "metrics": metrics_path,
        "latency": latency_path,
        "metadata": metadata_path,
    }


__all__ = [
    "summarize_metrics",
    "summarize_latency",
    "summarize_tokens",
    "write_predictions_brief",
    "write_run_outputs",
]
