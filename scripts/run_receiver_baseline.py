"""Run a no-Sharer Receiver baseline (LCF-X reference points).

These baselines have no Sharer message; the Receiver answers from a subset of
the evidence:

- ``question_only``: no context at all;
- ``half``: only the Receiver's own partition (5 of 10 paragraphs);
- ``full``: both partitions combined (all 10 paragraphs).

Example (half-context, the "no sharer / 5 of 10 paragraphs" setting):

    python scripts/run_receiver_baseline.py \
        --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
        --context half \
        --receiver-model Qwen/Qwen2.5-0.5B-Instruct \
        --output-dir results/receiver_half_context

Writes the same standardized output files as the T2T runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Route HF caches under data/hf_cache (before transformers loads).
from scripts._defaults import DEFAULT_RECEIVER  # noqa: E402
from scripts._hf_cache import configure_hf_cache  # noqa: E402

configure_hf_cache()

from methods.t2t import (  # noqa: E402
    T2TExample,
    load_generator,
    run_receiver_only_example,
    score_free_form,
    write_run_outputs,
)


def load_examples(path: Path, limit: int | None) -> list[T2TExample]:
    examples: list[T2TExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(T2TExample.from_mapping(json.loads(line)))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Normalized JSONL")
    parser.add_argument(
        "--context",
        choices=["question_only", "half", "full"],
        default="half",
        help="How much evidence the Receiver sees (default: half)",
    )
    parser.add_argument("--receiver-model", default=DEFAULT_RECEIVER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--receiver-max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--scoring",
        choices=["auto", "choice", "free_form"],
        default="auto",
    )
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    examples = load_examples(args.dataset, args.num_examples)
    if not examples:
        raise SystemExit(f"No examples loaded from {args.dataset}")
    print(f"Loaded {len(examples)} examples from {args.dataset}")
    print(f"Receiver context mode: {args.context}")

    has_choices = any(ex.choices for ex in examples)
    if args.scoring == "auto":
        free_form = not has_choices
    else:
        free_form = args.scoring == "free_form"
    score_fn = score_free_form if free_form else None
    print(f"Scoring mode: {'free_form (EM/F1)' if free_form else 'multiple_choice'}")

    print(f"Loading receiver model: {args.receiver_model}")
    receiver_gen = load_generator(
        args.receiver_model,
        max_new_tokens=args.receiver_max_new_tokens,
        deterministic=True,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    results = []
    for idx, example in enumerate(examples, start=1):
        results.append(
            run_receiver_only_example(
                example,
                receiver_gen,
                context_mode=args.context,
                score_fn=score_fn,
            )
        )
        if args.print_every and idx % args.print_every == 0:
            print(f"  processed {idx}/{len(examples)}", flush=True)

    run_metadata = {
        "method": f"receiver_{args.context}",
        "dataset": str(args.dataset),
        "receiver_model": args.receiver_model,
        "context_mode": args.context,
        "num_examples": len(examples),
        "receiver_max_new_tokens": args.receiver_max_new_tokens,
        "scoring": "free_form" if free_form else "multiple_choice",
        "dtype": args.dtype,
    }
    paths = write_run_outputs(results, args.output_dir, metadata=run_metadata)

    metrics = json.loads(paths["metrics"].read_text())
    print("\n=== metrics ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
