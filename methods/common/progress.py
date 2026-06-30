"""Persistent JSONL progress logger shared by training loops and eval runs.

Stdout prints are lost when the session ends; this appends one JSON object per
event to a log file so a long run — training or evaluation — survives and can be
plotted, diffed, or checked mid-flight (each line is flushed immediately).

Training writes ``training_log.jsonl`` beside the checkpoint:
    {"event": "start", "created_at": ..., <config>}
    {"event": "train", "wall_s": ..., "opt_step": ..., "train_loss": ..., "lr": ...}
    {"event": "eval",  "wall_s": ..., "opt_step": ..., "eval_loss": ..., "em": ..., "f1": ..., "best": ...}
    {"event": "end",   "wall_s": ..., "opt_step": ..., "best_eval": ...}

Eval runs write ``eval_log.jsonl`` in the output dir (see ``eval_progress_fields``):
    {"event": "start", "created_at": ..., <config>}
    {"event": "progress", "wall_s": ..., "processed": ..., "total": ..., "eta_s": ..., "accuracy": ..., "scores": ...}
    {"event": "done", ...}
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .io import summarize_metrics


class ProgressLogger:
    """Append-only JSONL event logger. A no-op when ``path`` is ``None``."""

    def __init__(self, path: str | Path | None, *, config: dict[str, Any] | None = None) -> None:
        self.path = Path(path) if path else None
        self._start = time.perf_counter()
        self._fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Fresh file per run (mirrors the checkpoint/output overwrite behaviour).
            self._fh = self.path.open("w", encoding="utf-8")
            self._write(
                {
                    "event": "start",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **(config or {}),
                }
            )

    def _write(self, record: dict[str, Any]) -> None:
        if self._fh is not None:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()  # durable mid-run: survives a crash / kill

    def log(self, event: str, **fields: Any) -> None:
        """Append one ``event`` record with the elapsed wall time and ``fields``."""

        self._write(
            {
                "event": event,
                "wall_s": round(time.perf_counter() - self._start, 2),
                **fields,
            }
        )

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def eval_progress_fields(
    results: Sequence[Any], total: int, start_perf: float
) -> dict[str, Any]:
    """Compute progress fields for an eval run: counts, throughput, ETA, running metrics.

    ``results`` is the list filled so far; ``total`` the planned example count;
    ``start_perf`` a ``time.perf_counter()`` captured before the loop. The running
    accuracy / EM / F1 come from :func:`summarize_metrics` over the results so far.
    """

    processed = len(results)
    elapsed = time.perf_counter() - start_perf
    rate = processed / elapsed if elapsed > 0 else 0.0
    eta = (total - processed) / rate if rate > 0 and total else None
    metrics = summarize_metrics(results)
    return {
        "processed": processed,
        "total": total,
        "elapsed_s": round(elapsed, 1),
        "rate_per_s": round(rate, 2),
        "eta_s": round(eta, 1) if eta is not None else None,
        "accuracy": metrics.get("accuracy"),
        "scores": metrics.get("scores"),
    }


__all__ = ["ProgressLogger", "eval_progress_fields"]
