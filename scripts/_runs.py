"""Parse replicate selections used by repeated runs and visual aggregation."""

from __future__ import annotations

import argparse
import re


def parse_runs(spec: str) -> list[str]:
    """Expand ``1:3``, ``1-3``, ``1,3,5``, or ``[1, 3, 5]``."""

    text = spec.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if not text:
        raise ValueError("run selection is empty")

    selected: list[int] = []
    for token in re.split(r"[\s,]+", text):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\s*([:-])\s*(\d+)", token)
        if match:
            start, end = int(match.group(1)), int(match.group(3))
            if end < start:
                raise ValueError(f"descending run range is not supported: {token!r}")
            selected.extend(range(start, end + 1))
        elif token.isdigit():
            selected.append(int(token))
        else:
            raise ValueError(
                f"invalid run token {token!r}; use 1:5, 1-5, or 1,3,5"
            )

    if not selected:
        raise ValueError("run selection is empty")
    # Preserve the user's order while removing duplicates.
    return [str(value) for value in dict.fromkeys(selected)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection")
    args = parser.parse_args()
    try:
        print(" ".join(parse_runs(args.selection)))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
