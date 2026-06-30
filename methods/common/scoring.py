"""Free-form answer scoring (HotpotQA / SQuAD style EM and F1).

Shared by every method. These are the standard normalization and metric
functions used by HotpotQA and SQuAD: lowercase, strip punctuation and articles,
collapse whitespace, then compute exact match and token-level F1.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Exact match after normalization."""

    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 after normalization (HotpotQA convention)."""

    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    # Special handling for yes/no/noanswer: F1 is 1.0 only on exact match.
    if normalize_answer(prediction) in {"yes", "no", "noanswer"} or normalize_answer(
        ground_truth
    ) in {"yes", "no", "noanswer"}:
        return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_free_form(prediction: str, answer: str) -> dict[str, Any]:
    """Score a free-form prediction against a reference answer.

    Returns a dict with ``em`` (bool), ``f1`` (float), and ``is_correct``
    (alias of ``em``) so it can be plugged into the runner's ``score_fn`` hook.
    """

    em = exact_match(prediction, answer)
    return {
        "is_correct": em,
        "em": em,
        "f1": f1_score(prediction, answer),
    }


__all__ = ["normalize_answer", "exact_match", "f1_score", "score_free_form"]
