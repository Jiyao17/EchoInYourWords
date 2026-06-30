"""Answer parsing helpers shared by every method's runner."""

from __future__ import annotations

import re
from collections.abc import Sequence


def parse_choice_answer(text: str, choices: Sequence[str]) -> str | None:
    """Extract a multiple-choice letter from model output.

    For free-form tasks (no ``choices``) the stripped text is returned as-is.
    """

    if not choices:
        return text.strip() or None

    max_idx = min(len(choices), 26) - 1
    valid = "".join(chr(ord("A") + idx) for idx in range(max_idx + 1))
    normalized = text.strip().upper()

    patterns = [
        rf"\b([{valid}])\b",
        rf"ANSWER\s*[:\-]?\s*([{valid}])\b",
        rf"OPTION\s*[:\-]?\s*([{valid}])\b",
        rf"CHOICE\s*[:\-]?\s*([{valid}])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1)

    return None


__all__ = ["parse_choice_answer"]
