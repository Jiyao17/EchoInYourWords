"""Run the single-model inference baseline.

One model answers each question from the **full** context (both partitions
combined) — no Sharer, no communication. This is the upper-bound reference the
communication methods (T2T, LCF-X, EIYW) are measured against.

It uses a dedicated single-model prompt (``build_single_model_prompt``) that is
deliberately separate from the method receiver prompts, so changes to one never
disturb the other.

Example (full-context single model on cross-context HotpotQA-bridge):

    python scripts/run_single_model.py \
        --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --output-dir results/single_model

Writes the same standardized output files as the method runners.
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
from scripts._defaults import DEFAULT_RECEIVER  # noqa: E402
from scripts._hf_cache import configure_hf_cache  # noqa: E402

configure_hf_cache()

from methods.common import (  # noqa: E402
    ProgressLogger,
    eval_progress_fields,
    write_predictions_brief,
)
from methods.common.prompts import build_single_model_prompt  # noqa: E402
from methods.t2t import (  # noqa: E402
    GenerationOutput,
    T2TExample,
    T2TResult,
    load_generator,
    parse_choice_answer,
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


def _full_context(example: T2TExample) -> str:
    """Both partitions combined — the single model sees all the evidence."""

    parts = [example.sharer_context, example.receiver_context]
    return "\n\n".join(p for p in parts if p.strip())


def run_single_model_example(example, generate, score_fn=None) -> T2TResult:
    """Generate one single-model answer from the full context."""

    prompt = build_single_model_prompt(example, _full_context(example))

    start = perf_counter()
    output = generate(prompt)
    receiver_ms = (perf_counter() - start) * 1000.0
    if not isinstance(output, GenerationOutput):
        output = GenerationOutput(text=str(output))
    receiver_output = output.text.strip()

    prediction = parse_choice_answer(receiver_output, example.choices)
    is_correct: bool | None = None
    scores: dict = {}
    if example.answer is not None and prediction is not None:
        if score_fn is not None:
            scores = dict(score_fn(prediction, example.answer))
            if "is_correct" in scores:
                is_correct = bool(scores["is_correct"])
        else:
            is_correct = prediction.strip().upper() == example.answer.strip().upper()

    return T2TResult(
        id=example.id,
        method="single_model",
        sharer_prompt="",
        sharer_message="",
        receiver_prompt=prompt,
        receiver_output=receiver_output,
        prediction=prediction,
        answer=example.answer,
        is_correct=is_correct,
        scores=scores,
        latency={
            "sharer_ms": 0.0,
            "fuser_ms": 0.0,
            "receiver_ms": receiver_ms,
            "total_ms": receiver_ms,
        },
        token_counts={
            "sharer_input": None,
            "sharer_output": None,
            "receiver_input": output.input_tokens,
            "receiver_output": output.output_tokens,
        },
        metadata=dict(example.metadata),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Normalized JSONL")
    parser.add_argument("--model", default=DEFAULT_RECEIVER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
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

    has_choices = any(ex.choices for ex in examples)
    if args.scoring == "auto":
        free_form = not has_choices
    else:
        free_form = args.scoring == "free_form"
    score_fn = score_free_form if free_form else None
    print(f"Scoring mode: {'free_form (EM/F1)' if free_form else 'multiple_choice'}")

    print(f"Loading model: {args.model}")
    generate = load_generator(
        args.model,
        max_new_tokens=args.max_new_tokens,
        deterministic=True,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    # Persistent eval progress (long runs survive the session); flushed per record.
    progress = ProgressLogger(
        args.output_dir / "eval_log.jsonl",
        config={
            "method": "single_model",
            "dataset": str(args.dataset),
            "model": args.model,
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
        results.append(run_single_model_example(example, generate, score_fn=score_fn))
        if args.print_every and idx % args.print_every == 0:
            print(f"  processed {idx}/{len(examples)}", flush=True)
            progress.log("progress", **eval_progress_fields(results, len(examples), t0))
            write_predictions_brief(results, brief_path)

    progress.log("done", **eval_progress_fields(results, len(examples), t0))
    progress.close()

    run_metadata = {
        "method": "single_model",
        "dataset": str(args.dataset),
        "model": args.model,
        "num_examples": len(examples),
        "max_new_tokens": args.max_new_tokens,
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
