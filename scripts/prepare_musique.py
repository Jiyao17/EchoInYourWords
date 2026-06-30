"""Prepare cross-context MuSiQue examples (LCF-X partitioning).

MuSiQue-Ans 2-hop questions are bridge-style: each has exactly **2 supporting
paragraphs** (one per hop) among ~20 distractors. Mirroring the HotpotQA-bridge
setup, we split the paragraphs into two disjoint partitions, each holding exactly
one supporting paragraph plus some distractors. One partition goes to the Sharer
and the other to the Receiver, so neither side can answer alone. The question is
shared by both sides (the Sharer encodes it too); only the evidence is split.

Only 2-hop questions are used (3-/4-hop have 3-4 supporting paragraphs, which do
not split one-per-side). Output is normalized `Example` rows (one JSON object
per line), ready for every method/baseline.

    python scripts/prepare_musique.py --split train  --num-examples -1 --output data/tasks/musique_train.jsonl
    python scripts/prepare_musique.py --split validation --num-examples -1 --output data/tasks/musique_val.jsonl
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

DEFAULT_OUTPUT = REPO_ROOT / "data" / "tasks" / "musique_val.jsonl"
DEFAULT_HF_DATASET = "dgslibisey/MuSiQue"


def _seed_from_id(example_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode()).hexdigest()
    return int(digest[:16], 16)


def _paragraph_text(title: str, body: str) -> str:
    return f"{title}\n{body.strip()}"


def _format_partition(paragraphs: list[tuple[str, str]]) -> str:
    """Join (title, body) paragraphs into a single context string."""

    return "\n\n".join(_paragraph_text(title, body) for title, body in paragraphs)


def build_example(
    raw: dict[str, Any], base_seed: int, distractors_per_side: int
) -> dict[str, Any] | None:
    """Convert one MuSiQue 2-hop row into a normalized cross-context row.

    Returns ``None`` if the row is not a 2-hop question with exactly two
    supporting paragraphs (the structure the one-per-side split requires).
    """

    paragraphs = list(raw["paragraphs"])
    gold = [p for p in paragraphs if p.get("is_supporting")]
    distractors = [p for p in paragraphs if not p.get("is_supporting")]
    if len(gold) != 2:
        return None

    rng = random.Random(_seed_from_id(str(raw["id"]), base_seed))
    rng.shuffle(distractors)

    # One supporting paragraph per side; deterministically pick which side.
    gold_a, gold_b = gold
    if rng.random() < 0.5:
        gold_a, gold_b = gold_b, gold_a

    if distractors_per_side >= 0:
        # Cap to a fixed count per side (drops the rest).
        k = min(distractors_per_side, len(distractors) // 2)
        sharer_dist = distractors[:k]
        receiver_dist = distractors[k : 2 * k]
    else:
        # Default: split ALL distractors as evenly as possible (extra -> receiver).
        k = len(distractors) // 2
        sharer_dist = distractors[:k]
        receiver_dist = distractors[k:]
    sharer = [gold_a] + sharer_dist
    receiver = [gold_b] + receiver_dist
    rng.shuffle(sharer)
    rng.shuffle(receiver)

    def _pairs(paras: list[dict[str, Any]]) -> list[tuple[str, str]]:
        return [(p["title"], p["paragraph_text"]) for p in paras]

    return {
        "id": str(raw["id"]),
        "question": raw["question"],
        "choices": [],
        "answer": raw["answer"],
        "sharer_context": _format_partition(_pairs(sharer)),
        "receiver_context": _format_partition(_pairs(receiver)),
        "metadata": {
            "dataset": "musique",
            "setting": "cross_context",
            "hops": 2,
            "answer_aliases": list(raw.get("answer_aliases") or []),
            "sharer_gold_idx": gold_a["idx"],
            "receiver_gold_idx": gold_b["idx"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", default="validation", help="MuSiQue split (train/validation)"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=200,
        help="Number of 2-hop examples to write (default: 200; use -1 for all)",
    )
    parser.add_argument(
        "--distractors-per-side",
        type=int,
        default=-1,
        help="Distractor paragraphs per partition (default: -1 = split all ~18 evenly; "
        "set a positive N to cap and shorten contexts)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base seed for deterministic partitioning"
    )
    parser.add_argument(
        "--hf-dataset", default=DEFAULT_HF_DATASET, help="HF dataset id (MuSiQue-Ans)"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path"
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading MuSiQue {args.hf_dataset} split={args.split} ...", flush=True)
    ds = load_dataset(args.hf_dataset, split=args.split)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for raw in ds:
            row = build_example(raw, args.seed, args.distractors_per_side)
            if row is None:
                skipped += 1
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if args.num_examples != -1 and written >= args.num_examples:
                break

    print(f"Wrote {written} examples to {args.output} (skipped {skipped} non-2hop)")


if __name__ == "__main__":
    main()
