"""Output helpers for EIYW runs.

Reuses the T2T aggregators (they read only ``is_correct`` / ``scores`` /
``latency`` / ``token_counts`` off each result, so ``EIYWResult`` works
unchanged) and writes the same four standardized files, keeping every method
comparable over result files alone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from methods.t2t.io import summarize_latency, summarize_metrics, summarize_tokens

from .runner import EIYWResult


def write_run_outputs(
    results: Sequence[EIYWResult],
    output_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write ``predictions.jsonl`` / ``metrics.json`` / ``latency.json`` / ``run_metadata.json``."""

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
        "method": "eiyw",
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
