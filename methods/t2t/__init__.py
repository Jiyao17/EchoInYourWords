"""Text-to-text communication baseline."""

from .io import (
    summarize_latency,
    summarize_metrics,
    summarize_tokens,
    write_run_outputs,
)
from .model import load_generator
from .parsing import parse_choice_answer
from .prompts import (
    build_receiver_only_prompt,
    build_receiver_prompt,
    build_sharer_prompt,
    format_choices,
)
from .runner import (
    GenerationOutput,
    T2TResult,
    run_receiver_only_example,
    run_t2t_example,
)
from .schema import T2TExample
from .scoring import exact_match, f1_score, normalize_answer, score_free_form

__all__ = [
    "T2TExample",
    "T2TResult",
    "GenerationOutput",
    "build_sharer_prompt",
    "build_receiver_prompt",
    "build_receiver_only_prompt",
    "format_choices",
    "load_generator",
    "parse_choice_answer",
    "run_t2t_example",
    "run_receiver_only_example",
    "exact_match",
    "f1_score",
    "normalize_answer",
    "score_free_form",
    "summarize_latency",
    "summarize_metrics",
    "summarize_tokens",
    "write_run_outputs",
]
