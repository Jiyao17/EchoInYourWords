"""Prepare cross-context QASC examples (LCF-X partitioning).

QASC is an 8-way multiple-choice science QA dataset where every question is
composed from exactly **two** facts (``fact1`` + ``fact2``); answering requires
combining them. That two-fact structure is a natural bridge split: one fact goes
to the Sharer and the other to the Receiver, so neither side can answer alone
(mirroring the HotpotQA-bridge / MuSiQue setup, but multiple-choice instead of
free-form). The question and the 8 answer options are shared by both sides; only
the two supporting facts are split.

Unlike HotpotQA / MuSiQue there are no distractor paragraphs — each side holds
exactly its one gold fact. The stored ``answer`` is the option **letter** (A-H)
at the gold choice's position in ``choices`` (auto-lettered by index downstream),
which is what the multiple-choice scoring path compares against.

Only rows with a non-empty ``answerKey`` and both facts are kept (the ``test``
split has no answer key, so use ``train`` / ``validation``). Output is normalized
`Example` rows (one JSON object per line), ready for every method/baseline.

    python scripts/prepare_qasc.py --split train      --num-examples -1 --output data/tasks/qasc_train.jsonl
    python scripts/prepare_qasc.py --split validation --num-examples -1 --output data/tasks/qasc_val.jsonl
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

DEFAULT_OUTPUT = REPO_ROOT / "data" / "tasks" / "qasc_val.jsonl"
DEFAULT_HF_DATASET = "allenai/qasc"


def _seed_from_id(example_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode()).hexdigest()
    return int(digest[:16], 16)


def build_example(raw: dict[str, Any], base_seed: int) -> dict[str, Any] | None:
    """Convert one QASC row into a normalized cross-context row.

    Returns ``None`` if the row lacks an answer key or either supporting fact
    (the structure the two-fact one-per-side split requires).
    """

    choices = raw.get("choices") or {}
    texts = list(choices.get("text") or [])
    labels = list(choices.get("label") or [])
    answer_key = (raw.get("answerKey") or "").strip()
    fact1 = (raw.get("fact1") or "").strip()
    fact2 = (raw.get("fact2") or "").strip()

    if not texts or not answer_key or not fact1 or not fact2:
        return None
    if answer_key not in labels:
        return None

    # Re-letter the gold by its position in the stored choices (format_choices /
    # parse_choice_answer letter options by index), so the answer letter stays
    # correct regardless of the source label scheme.
    gold_idx = labels.index(answer_key)
    answer_letter = chr(ord("A") + gold_idx)

    rng = random.Random(_seed_from_id(str(raw["id"]), base_seed))

    # One fact per side; deterministically decide which fact goes to the Sharer.
    sharer_fact, receiver_fact = fact1, fact2
    if rng.random() < 0.5:
        sharer_fact, receiver_fact = receiver_fact, sharer_fact

    return {
        "id": str(raw["id"]),
        "question": raw["question"],
        "choices": [str(t).strip() for t in texts],
        "answer": answer_letter,
        "sharer_context": sharer_fact,
        "receiver_context": receiver_fact,
        "metadata": {
            "dataset": "qasc",
            "setting": "cross_context",
            "hops": 2,
            "answer_key": answer_key,
            "combined_fact": (raw.get("combinedfact") or "").strip(),
            "sharer_fact": "fact1" if sharer_fact == fact1 else "fact2",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", default="validation", help="QASC split (train/validation)"
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
        "--hf-dataset", default=DEFAULT_HF_DATASET, help="HF dataset id (QASC)"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path"
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading QASC {args.hf_dataset} split={args.split} ...", flush=True)
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

    print(f"Wrote {written} examples to {args.output} (skipped {skipped} without key/facts)")


if __name__ == "__main__":
    main()
