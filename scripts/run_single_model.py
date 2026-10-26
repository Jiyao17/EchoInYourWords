"""Evaluate a Receiver with private (half) or combined (full) context.

Both conditions use the latent methods' Receiver prompt and greedy decoder,
with no Sharer forward pass, learned translator, or cache residual. ``both``
loads one model and evaluates half then full context. Each condition writes
its own subdirectory under ``--output-dir``; historical flat outputs remain
separate. Full context is a reference, not a guaranteed upper bound.

Example (both conditions on ROPES):

    python scripts/run_single_model.py \
        --dataset data/tasks/ropes_val.jsonl \
        --model Qwen/Qwen3-0.6B \
        --context-mode both --output-dir results/single/ropes/Qwen--Qwen3-0.6B

Writes the same standardized output files as the method runners.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Route HF caches under data/hf_cache (before transformers loads).
from scripts._defaults import DEFAULT_RECEIVER, resolve_seed  # noqa: E402
from scripts._hf_cache import configure_hf_cache  # noqa: E402

configure_hf_cache()

from methods.common import (  # noqa: E402
    ProgressLogger,
    eval_progress_fields,
    write_predictions_brief,
)
from methods.common.prompts import build_receiver_prompt  # noqa: E402
from methods.t2t import (  # noqa: E402
    GenerationOutput,
    T2TExample,
    T2TResult,
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


def load_receiver_generator(
    model_name: str,
    *,
    max_new_tokens: int = 64,
    device: str | None = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
):
    """Load one frozen model and reuse the method Receiver's exact decode path."""

    from methods.xkv.model import load_xkv_pair

    # The symmetric loader aliases Sharer/Receiver to the SAME model/tokenizer;
    # it loads only one set of weights. Only decode_receiver is invoked here.
    receiver = load_xkv_pair(
        model_name,
        model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    def generate(prompt: str) -> GenerationOutput:
        text, input_tokens, output_tokens, first_ms, last_ms = receiver.decode_receiver(
            prompt, residual_fn=None, max_new_tokens=max_new_tokens
        )
        return GenerationOutput(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            first_token_ms=first_ms,
            last_token_ms=last_ms,
        )

    return generate


def run_single_model_example(
    example, generate, score_fn=None, *, context_mode: str = "full"
) -> T2TResult:
    """Answer with the original Receiver context or both evidence portions."""

    if context_mode not in {"full", "half"}:
        raise ValueError(f"unknown context mode: {context_mode!r}")
    context = _full_context(example) if context_mode == "full" else example.receiver_context
    receiver_example = replace(example, receiver_context=context, sharer_context="")
    prompt = build_receiver_prompt(receiver_example)

    start = perf_counter()
    output = generate(prompt)
    receiver_ms = (perf_counter() - start) * 1000.0
    if not isinstance(output, GenerationOutput):
        output = GenerationOutput(text=str(output))
    receiver_output = output.text.strip()

    prediction = parse_choice_answer(receiver_output, example.choices)
    is_correct: bool | None = None
    scores: dict = {}
    if example.answer is not None:
        if score_fn is not None:
            # Empty/invalid predictions are errors, not excluded denominators.
            scores = dict(score_fn(prediction or "", example.answer))
            if "is_correct" in scores:
                is_correct = bool(scores["is_correct"])
        else:
            is_correct = prediction is not None and (
                prediction.strip().upper() == example.answer.strip().upper()
            )

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
            "first_token_ms": output.first_token_ms,
            "last_token_ms": output.last_token_ms,
        },
        token_counts={
            "sharer_input": 0,
            "sharer_output": 0,
            "receiver_input": output.input_tokens,
            "receiver_output": output.output_tokens,
        },
        metadata={
            **example.metadata,
            "context_mode": context_mode,
            "prompt_template": "build_receiver_prompt",
            "communication": "none",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Normalized JSONL")
    parser.add_argument("--model", default=DEFAULT_RECEIVER)
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output root; conditions are written under half/ and full/",
    )
    parser.add_argument(
        "--context-mode", choices=["half", "full", "both"], default="both",
        help="half=original Receiver evidence; full=both partitions; both=run each",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
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
    args.seed = resolve_seed(args.seed)
    print(f"Run id: {args.run_id or '(unscoped)'} | seed: {args.seed}")

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
    generate = load_receiver_generator(
        args.model,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    common_metadata = {
        "method": "single_model",
        "run_id": args.run_id,
        "seed": args.seed,
        "dataset": str(args.dataset),
        "model": args.model,
        "num_examples": len(examples),
        "max_new_tokens": args.max_new_tokens,
        "scoring": "free_form" if free_form else "multiple_choice",
        "dtype": args.dtype,
        "prompt_template": "build_receiver_prompt",
        "decoding": "receiver_greedy_no_residual",
        "communication": "none",
        "invalid_predictions": "count_as_incorrect",
    }
    modes = ("half", "full") if args.context_mode == "both" else (args.context_mode,)
    for mode in modes:
        output_dir = args.output_dir / mode
        run_metadata = {**common_metadata, "context_mode": mode}
        print(f"\nContext: {mode} | output: {output_dir}")
        progress = ProgressLogger(output_dir / "eval_log.jsonl", config=run_metadata)
        t0 = perf_counter()
        brief_path = output_dir / "predictions_brief.tsv"
        results = []
        try:
            for idx, example in enumerate(examples, start=1):
                results.append(run_single_model_example(
                    example, generate, score_fn=score_fn, context_mode=mode
                ))
                if args.print_every and idx % args.print_every == 0:
                    print(f"  {mode}: processed {idx}/{len(examples)}", flush=True)
                    progress.log("progress", **eval_progress_fields(results, len(examples), t0))
                    write_predictions_brief(results, brief_path)
            progress.log("done", **eval_progress_fields(results, len(examples), t0))
        finally:
            progress.close()

        paths = write_run_outputs(results, output_dir, metadata=run_metadata)
        metrics = json.loads(paths["metrics"].read_text())
        print(f"\n=== {mode} metrics ===")
        print(json.dumps(metrics, indent=2))
        print(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    main()
