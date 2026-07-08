"""Prepare cross-context ROPES examples (LCF-X partitioning).

ROPES (Reasoning Over Paragraph Effects in Situations) is already a natural
two-part dataset -- no invented partitioning needed. Each example has a
``background`` passage (a general causal-relationship rule, e.g. from a
science textbook) and a ``situation`` passage (a specific scenario the rule
applies to); the answer is a span drawn from the situation (occasionally the
question) and requires combining the rule with the scenario. One side gets
the background, the other the situation -- the coin flip is seeded per
example so the split is deterministic but not systematically biased toward
either side. The question is shared by both sides.

Output is normalized `Example` rows (one JSON object per line), ready for
every method/baseline.

    python scripts/prepare_ropes.py --split train      --num-examples -1 --output data/tasks/ropes_train.jsonl
    python scripts/prepare_ropes.py --split validation --num-examples -1 --output data/tasks/ropes_val.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Route HF caches under data/hf_cache (before importing datasets).
from scripts._hf_cache import configure_hf_cache  # noqa: E402

configure_hf_cache()

DEFAULT_OUTPUT = REPO_ROOT / "data" / "tasks" / "ropes_val.jsonl"
DEFAULT_HF_DATASET = "allenai/ropes"


def _seed_from_id(example_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode()).hexdigest()
    return int(digest[:16], 16)


def build_example(raw: dict[str, Any], base_seed: int) -> dict[str, Any] | None:
    """Convert one ROPES row into a normalized cross-context row.

    Returns ``None`` if the row is missing the background/situation text or
    has no answer span.
    """

    background = (raw.get("background") or "").strip()
    situation = (raw.get("situation") or "").strip()
    answers = list((raw.get("answers") or {}).get("text") or [])
    if not background or not situation or not answers:
        return None

    rng = random.Random(_seed_from_id(str(raw["id"]), base_seed))

    # One passage per side; deterministically decide which side gets which.
    sharer_part, receiver_part = background, situation
    sharer_label, receiver_label = "background", "situation"
    if rng.random() < 0.5:
        sharer_part, receiver_part = receiver_part, sharer_part
        sharer_label, receiver_label = receiver_label, sharer_label

    return {
        "id": str(raw["id"]),
        "question": raw["question"],
        "choices": [],
        "answer": answers[0],
        "sharer_context": sharer_part,
        "receiver_context": receiver_part,
        "metadata": {
            "dataset": "ropes",
            "setting": "cross_context",
            "answer_aliases": answers,
            "sharer_part": sharer_label,
            "receiver_part": receiver_label,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", default="validation", help="ROPES split (train/validation)"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=200,
        help="Number of examples to write (default: 200; use -1 for all)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base seed for deterministic partitioning"
    )
    parser.add_argument(
        "--hf-dataset", default=DEFAULT_HF_DATASET, help="HF dataset id (ROPES)"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path"
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading ROPES {args.hf_dataset} split={args.split} ...", flush=True)
    ds = load_dataset(args.hf_dataset, split=args.split)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for raw in ds:
            row = build_example(raw, args.seed)
            if row is None:
                skipped += 1
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if args.num_examples != -1 and written >= args.num_examples:
                break

    print(f"Wrote {written} examples to {args.output} (skipped {skipped} without background/situation/answer)")


if __name__ == "__main__":
    main()
