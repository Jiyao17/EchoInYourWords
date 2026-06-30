"""Text-to-text communication baseline.

Method-specific code lives here (prompts, model wrapper, runner); the data
contract, scoring, parsing, and result aggregation come from
``methods.common`` and are re-exported for convenience.
"""

from functools import partial

from methods.common.io import (
    summarize_latency,
    summarize_metrics,
    summarize_tokens,
)
from methods.common.io import write_run_outputs as _write_run_outputs
from methods.common.parsing import parse_choice_answer
from methods.common.prompts import (
    build_receiver_message_prompt,
    build_sharer_prompt,
    format_choices,
)
from methods.common.scoring import (
    exact_match,
    f1_score,
    normalize_answer,
    score_free_form,
)

from .model import load_generator
from .runner import (
    GenerationOutput,
    T2TResult,
    run_t2t_example,
)
from .schema import T2TExample

# Standard writer with this method's default name (a "method" in metadata still
# overrides it).
write_run_outputs = partial(_write_run_outputs, method="text_to_text")

__all__ = [
    "T2TExample",
    "T2TResult",
    "GenerationOutput",
    "build_sharer_prompt",
    "build_receiver_message_prompt",
    "format_choices",
    "load_generator",
    "parse_choice_answer",
    "run_t2t_example",
    "exact_match",
    "f1_score",
    "normalize_answer",
    "score_free_form",
    "summarize_latency",
    "summarize_metrics",
    "summarize_tokens",
    "write_run_outputs",
]
