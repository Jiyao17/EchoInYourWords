"""Run the text-to-text communication baseline over a normalized JSONL dataset.

Example (cross-context HotpotQA-bridge):

    python scripts/prepare_hotpotqa_bridge.py --num-examples 50
    python scripts/run_text_to_text.py \
        --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
        --sharer-model unsloth/Llama-3.2-3B-Instruct \
        --receiver-model Qwen/Qwen3-0.6B \
        --output-dir results/t2t_hotpotqa_bridge

Writes ``predictions.jsonl``, ``metrics.json``, ``latency.json``, and
``run_metadata.json`` into the output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Route HF caches under data/hf_cache (before transformers loads).
from scripts._defaults import DEFAULT_RECEIVER, DEFAULT_SHARER  # noqa: E402
from scripts._hf_cache import configure_hf_cache  # noqa: E402

configure_hf_cache()

from methods.common import (  # noqa: E402
    ProgressLogger,
    eval_progress_fields,
    write_predictions_brief,
)
from methods.t2t import (  # noqa: E402
    T2TExample,
    load_generator,
    run_t2t_example,
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
    # Sharer and Receiver are always independent models; default to a symmetric
    # Qwen pair when unspecified (the project assumes heterogeneous settings).
    parser.add_argument("--sharer-model", default=DEFAULT_SHARER)
    parser.add_argument("--receiver-model", default=DEFAULT_RECEIVER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--sharer-max-new-tokens", type=int, default=128)
    parser.add_argument("--receiver-max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--scoring",
        choices=["auto", "choice", "free_form"],
        default="auto",
        help="auto: free-form when examples have no choices",
    )
    parser.add_argument(
        "--print-every", type=int, default=10, help="Progress print interval"
    )
    args = parser.parse_args()

    receiver_model = args.receiver_model

    examples = load_examples(args.dataset, args.num_examples)
    if not examples:
        raise SystemExit(f"No examples loaded from {args.dataset}")
    print(f"Loaded {len(examples)} examples from {args.dataset}")

    # Decide scoring mode.
    has_choices = any(ex.choices for ex in examples)
    if args.scoring == "auto":
        free_form = not has_choices
    else:
        free_form = args.scoring == "free_form"
    score_fn = score_free_form if free_form else None
    print(f"Scoring mode: {'free_form (EM/F1)' if free_form else 'multiple_choice'}")

    print(f"Loading sharer model: {args.sharer_model}")
    sharer_gen = load_generator(
        args.sharer_model,
        max_new_tokens=args.sharer_max_new_tokens,
        deterministic=True,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if receiver_model == args.sharer_model:
        print("Receiver shares the sharer model instance")
        receiver_gen = sharer_gen
    else:
        print(f"Loading receiver model: {receiver_model}")
        receiver_gen = load_generator(
            receiver_model,
            max_new_tokens=args.receiver_max_new_tokens,
            deterministic=True,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )

    # Persistent eval progress (long runs survive the session); flushed per record.
    progress = ProgressLogger(
        args.output_dir / "eval_log.jsonl",
        config={
            "method": "text_to_text",
            "dataset": str(args.dataset),
            "sharer_model": args.sharer_model,
            "receiver_model": receiver_model,
            "num_examples": len(examples),
        },
    )
    t0 = perf_counter()

    # Written incrementally so the brief can be tailed mid-run (the full set of
    # standardized outputs is rewritten once at the end via write_run_outputs).
    args.output_dir.mkdir(parents=True, exist_ok=True)
    brief_path = args.output_dir / "predictions_brief.tsv"

    results = []
    for idx, example in enumerate(examples, start=1):
        results.append(
            run_t2t_example(example, sharer_gen, receiver_gen, score_fn=score_fn)
        )
        if args.print_every and idx % args.print_every == 0:
            print(f"  processed {idx}/{len(examples)}", flush=True)
            progress.log("progress", **eval_progress_fields(results, len(examples), t0))
            write_predictions_brief(results, brief_path)

    progress.log("done", **eval_progress_fields(results, len(examples), t0))
    progress.close()

    run_metadata = {
        "dataset": str(args.dataset),
        "sharer_model": args.sharer_model,
        "receiver_model": receiver_model,
        "num_examples": len(examples),
        "sharer_max_new_tokens": args.sharer_max_new_tokens,
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
