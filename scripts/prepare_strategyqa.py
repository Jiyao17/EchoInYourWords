"""Prepare cross-context StrategyQA examples (LCF-X partitioning).

StrategyQA is implicit multi-hop reasoning: each boolean (yes/no) question is
backed by a short ``facts`` list (usually 2-5 sentences) that must be chained
together to reach the answer. Mirroring the MuSiQue split, the facts are
shuffled and divided into two disjoint halves, one per side, so neither side
alone holds every fact needed. The question is shared by both sides; the
boolean answer is encoded as a 2-way multiple choice ("Yes" / "No") so it
reuses the project's existing MC scoring path. Only rows with at least 2
facts are kept (the split needs at least one fact per side).

The source dataset only publishes labeled examples in its ``train`` split
(``test`` is the hidden official leaderboard split -- no answer/facts, so it
is unusable here). We therefore carve our own train/validation split out of
that single labeled split via a deterministic per-id hash bucket, keyed by
``--val-frac``, rather than passing ``--split`` straight through to
``load_dataset`` like the other prepare scripts.

Output is normalized `Example` rows (one JSON object per line), ready for
every method/baseline.

    python scripts/prepare_strategyqa.py --split train      --num-examples -1 --output data/tasks/strategyqa_train.jsonl
    python scripts/prepare_strategyqa.py --split validation --num-examples -1 --output data/tasks/strategyqa_val.jsonl
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

DEFAULT_OUTPUT = REPO_ROOT / "data" / "tasks" / "strategyqa_val.jsonl"
DEFAULT_HF_DATASET = "voidful/StrategyQA"
CHOICES = ["Yes", "No"]


def _seed_from_id(example_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode()).hexdigest()
    return int(digest[:16], 16)


def _assign_split(example_id: str, base_seed: int, val_frac: float) -> str:
    """Deterministically bucket an id into 'train' or 'validation'."""

    digest = hashlib.sha256(f"strategyqa-split:{base_seed}:{example_id}".encode()).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    return "validation" if frac < val_frac else "train"


def build_example(
    raw: dict[str, Any], base_seed: int
) -> dict[str, Any] | None:
    """Convert one StrategyQA row into a normalized cross-context row.

    Returns ``None`` if the row lacks a boolean answer or fewer than 2 facts
    (the structure the one-fact-per-side-minimum split requires).
    """

    qid = str(raw.get("qid") or "")
    answer = raw.get("answer")
    facts = [str(f).strip() for f in (raw.get("facts") or []) if str(f).strip()]
    if not qid or answer is None or len(facts) < 2:
        return None

    rng = random.Random(_seed_from_id(qid, base_seed))
    shuffled = facts[:]
    rng.shuffle(shuffled)

    half = len(shuffled) // 2
    sharer_facts = shuffled[:half]
    receiver_facts = shuffled[half:]

    return {
        "id": qid,
        "question": raw["question"],
        "choices": list(CHOICES),
        "answer": "A" if bool(answer) else "B",
        "sharer_context": " ".join(sharer_facts),
        "receiver_context": " ".join(receiver_facts),
        "metadata": {
            "dataset": "strategyqa",
            "setting": "cross_context",
            "term": raw.get("term"),
            "description": raw.get("description"),
            "decomposition": list(raw.get("decomposition") or []),
            "num_facts": len(facts),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"],
        help="Which carved-out half to write (see module docstring)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of the labeled data deterministically held out as 'validation'",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=200,
        help="Number of examples to write (default: 200; use -1 for all)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base seed for deterministic partitioning/split"
    )
    parser.add_argument(
        "--hf-dataset", default=DEFAULT_HF_DATASET, help="HF dataset id (StrategyQA)"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path"
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading StrategyQA {args.hf_dataset} split=train ...", flush=True)
    # Streaming avoids a batched-JSON-parsing bug in some `datasets` versions
    # that trips on this repo's non-streaming Arrow build past ~5K rows.
    ds = load_dataset(args.hf_dataset, split="train", streaming=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for raw in ds:
            qid = str(raw.get("qid") or "")
            if not qid or _assign_split(qid, args.seed, args.val_frac) != args.split:
                continue
            row = build_example(raw, args.seed)
            if row is None:
                skipped += 1
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if args.num_examples != -1 and written >= args.num_examples:
                break

    print(f"Wrote {written} examples to {args.output} (skipped {skipped} without answer/2+ facts)")


if __name__ == "__main__":
    main()
