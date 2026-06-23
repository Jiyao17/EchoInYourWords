"""Prepare cross-context HotpotQA-bridge examples (LCF-X partitioning).

Reproduces the LCF-X cross-context setup described in the project baselines
notes: each HotpotQA distractor example has 10 paragraphs (2 supporting /
"gold" + 8 distractors). We keep only ``bridge`` questions, then split the
paragraphs into two disjoint 5-paragraph partitions, each containing exactly
one gold paragraph and four distractors. One partition goes to the Sharer and
the other to the Receiver, so neither side can answer alone.

Output is written as normalized ``T2TExample`` rows (one JSON object per line)
to ``data/tasks/hotpotqa_bridge_cross.jsonl`` by default, ready for
``scripts/run_text_to_text.py``.
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

DEFAULT_OUTPUT = REPO_ROOT / "data" / "tasks" / "hotpotqa_bridge_cross.jsonl"


def _seed_from_id(example_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode()).hexdigest()
    return int(digest[:16], 16)


def _paragraph_text(title: str, sentences: list[str]) -> str:
    body = " ".join(s.strip() for s in sentences).strip()
    return f"{title}\n{body}"


def _format_partition(paragraphs: list[tuple[str, str]]) -> str:
    """Join (title, body) paragraphs into a single context string."""

    return "\n\n".join(
        _paragraph_text(title, [body]) for title, body in paragraphs
    )


def build_example(raw: dict[str, Any], base_seed: int) -> dict[str, Any] | None:
    """Convert one HotpotQA distractor row into a normalized cross-context row.

    Returns ``None`` if the row does not fit the bridge / 10-paragraph /
    2-gold-paragraph structure required by the partitioning scheme.
    """

    if raw.get("type") != "bridge":
        return None

    context = raw["context"]
    titles = list(context["title"])
    sentences = list(context["sentences"])
    if len(titles) != 10:
        return None

    # Map title -> joined paragraph body.
    para_body: dict[str, str] = {}
    for title, sents in zip(titles, sentences):
        para_body[title] = " ".join(s.strip() for s in sents).strip()

    # Gold (supporting) paragraph titles, de-duplicated, order preserved.
    gold_titles: list[str] = []
    for title in raw["supporting_facts"]["title"]:
        if title not in gold_titles:
            gold_titles.append(title)
    if len(gold_titles) != 2 or any(t not in para_body for t in gold_titles):
        return None

    distractor_titles = [t for t in titles if t not in gold_titles]
    if len(distractor_titles) != 8:
        return None

    rng = random.Random(_seed_from_id(str(raw["id"]), base_seed))
    shuffled_distractors = distractor_titles[:]
    rng.shuffle(shuffled_distractors)

    # One gold paragraph per side; deterministically decide which side.
    gold_a, gold_b = gold_titles
    if rng.random() < 0.5:
        gold_a, gold_b = gold_b, gold_a

    sharer_titles = [gold_a] + shuffled_distractors[:4]
    receiver_titles = [gold_b] + shuffled_distractors[4:]
    rng.shuffle(sharer_titles)
    rng.shuffle(receiver_titles)

    sharer_paras = [(t, para_body[t]) for t in sharer_titles]
    receiver_paras = [(t, para_body[t]) for t in receiver_titles]

    return {
        "id": str(raw["id"]),
        "question": raw["question"],
        "choices": [],
        "answer": raw["answer"],
        "sharer_context": _format_partition(sharer_paras),
        "receiver_context": _format_partition(receiver_paras),
        "metadata": {
            "dataset": "hotpotqa_bridge",
            "setting": "cross_context",
            "level": raw.get("level"),
            "type": raw.get("type"),
            "gold_titles": gold_titles,
            "sharer_gold": gold_a,
            "receiver_gold": gold_b,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", default="validation", help="HotpotQA split (default: validation)"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=200,
        help="Number of bridge examples to write (default: 200; use -1 for all)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base seed for deterministic partitioning"
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        default=None,
        help="Optional difficulty filter, e.g. --levels hard medium",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path"
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading HotpotQA distractor split={args.split} ...", flush=True)
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)

    levels = set(args.levels) if args.levels else None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for raw in ds:
            if levels is not None and raw.get("level") not in levels:
                continue
            row = build_example(raw, args.seed)
            if row is None:
                skipped += 1
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if args.num_examples != -1 and written >= args.num_examples:
                break

    print(f"Wrote {written} examples to {args.output} (skipped {skipped})")


if __name__ == "__main__":
    main()
