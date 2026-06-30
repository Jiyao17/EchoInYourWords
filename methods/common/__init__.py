"""Thin shared experiment layer reused by every method (T2T, LCF-X, EIYW).

Holds the cross-method data contract, scoring, answer parsing, shared prompt
primitives, and the standardized result aggregation / output writer. Methods own
their communication mechanism (cache/fuser/translator, model wrapper, training,
runner); they consume these shared pieces rather than importing one another.
"""

from .io import (
    summarize_latency,
    summarize_metrics,
    summarize_tokens,
    write_predictions_brief,
    write_run_outputs,
)
from .parsing import parse_choice_answer
from .prompts import build_receiver_only_prompt, format_choices
from .progress import ProgressLogger, eval_progress_fields
from .schema import Example
from .scoring import exact_match, f1_score, normalize_answer, score_free_form

__all__ = [
    "Example",
    "exact_match",
    "f1_score",
    "normalize_answer",
    "score_free_form",
    "parse_choice_answer",
    "format_choices",
    "build_receiver_only_prompt",
    "summarize_latency",
    "summarize_metrics",
    "summarize_tokens",
    "write_predictions_brief",
    "write_run_outputs",
    "ProgressLogger",
    "eval_progress_fields",
]
