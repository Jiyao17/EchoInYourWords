"""Run KVTok, a Stage-1-only-pooling baseline against LCF-X/EIYW.

The Sharer's cache is pooled with LCF-X's Stage 1 *only* (within-span
attention), then compact feature-axis and token-axis linear projections produce
``--num-tokens`` slots. The Receiver contributes processed context embeddings
only -- no initial Receiver forward/cache. Cross-attention and a calibrated
output head emit soft tokens prepended to the Receiver generation prompt.

Example (cross-context HotpotQA-bridge, trained translator):

    python scripts/run_kvtok.py \
        --dataset data/tasks/hotpotqa_bridge_val.jsonl \
        --sharer-model unsloth/Llama-3.2-3B-Instruct \
        --receiver-model Qwen/Qwen3-0.6B \
        --num-spans 5 --num-tokens 8 \
        --translator-checkpoint data/checkpoints/kvtok/translator.pt \
        --output-dir results/kvtok_hotpotqa

With no ``--translator-checkpoint`` the output head is gently initialized near
zero for a plumbing run. Writes the four standardized output files.
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

from methods.kvtok import (  # noqa: E402
    KVTokExample,
    KVTokTranslator,
    check_pair_compatible,
    load_kvtok_pair,
    load_translator,
    run_kvtok_example,
    write_run_outputs,
)
from methods.common import (  # noqa: E402
    ProgressLogger,
    eval_progress_fields,
    score_free_form,
    write_predictions_brief,
)


def load_examples(path: Path, limit: int | None) -> list[KVTokExample]:
    examples: list[KVTokExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(KVTokExample.from_mapping(json.loads(line)))
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
    parser.add_argument("--translator-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--num-spans",
        type=int,
        default=5,
        help="Sharer Stage-1 pooling spans; must match the trained --translator-checkpoint",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=8,
        help="Soft tokens the down-projection outputs; must match the trained --translator-checkpoint",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Translator width; must match a loaded --translator-checkpoint",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Cross-attention heads; must match a loaded --translator-checkpoint after divisor-safe resolution",
    )
    parser.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help="Load the translator even if its stamped Sharer/Receiver/arch differ",
    )
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--scoring", choices=["auto", "choice", "free_form"], default="auto"
    )
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    receiver_model = args.receiver_model

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

    print(f"Loading sharer={args.sharer_model} receiver={receiver_model}")
    pair = load_kvtok_pair(
        args.sharer_model,
        receiver_model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    print("pair compatibility:", json.dumps(check_pair_compatible(pair)))

    if args.translator_checkpoint is not None:
        translator = load_translator(
            str(args.translator_checkpoint),
            pair,
            num_spans=args.num_spans,
            num_tokens=args.num_tokens,
            d_model=args.d_model,
            num_heads=args.num_heads,
            device=pair.device,
            allow_model_mismatch=args.allow_model_mismatch,
        ).to(pair.device)
        print(f"Loaded translator checkpoint: {args.translator_checkpoint}")
    else:
        translator = KVTokTranslator.from_specs(
            pair.sharer_spec,
            pair.receiver_spec,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_spans=args.num_spans,
            num_tokens=args.num_tokens,
            zero_init=True,
        ).to(pair.device)
        print("No checkpoint: using ZERO-INIT translator (== single-model floor)")
    translator.eval()
    print(f"Translator trainable params: {translator.trainable_parameters()}")

    # Persistent eval progress (long runs survive the session); flushed per record.
    progress = ProgressLogger(
        args.output_dir / "eval_log.jsonl",
        config={
            "method": "kvtok",
            "dataset": str(args.dataset),
            "sharer_model": args.sharer_model,
            "receiver_model": receiver_model,
            "num_examples": len(examples),
            "d_model": translator.d_model,
            "num_heads": translator.num_heads,
        },
    )
    t0 = perf_counter()

    # Written incrementally so the brief can be tailed mid-run (the full set of
    # standardized outputs is rewritten once at the end via write_run_outputs).
    args.output_dir.mkdir(parents=True, exist_ok=True)
    brief_path = args.output_dir / "predictions_brief.tsv"

    results = []
    for idx, example in enumerate(examples, start=1):
        result = run_kvtok_example(
            example,
            pair,
            translator,
            max_new_tokens=args.max_new_tokens,
            score_fn=score_fn,
        )
        results.append(result)
        if args.print_every and idx % args.print_every == 0:
            print(f"  processed {idx}/{len(examples)}", flush=True)
            progress.log("progress", **eval_progress_fields(results, len(examples), t0))
            write_predictions_brief(results, brief_path)

    progress.log("done", **eval_progress_fields(results, len(examples), t0))
    progress.close()

    run_metadata = {
        "method": "kvtok",
        "dataset": str(args.dataset),
        "sharer_model": args.sharer_model,
        "receiver_model": receiver_model,
        "num_examples": len(examples),
        "max_new_tokens": args.max_new_tokens,
        "num_spans": args.num_spans,
        "num_tokens": args.num_tokens,
        "d_model": translator.d_model,
        "num_heads": translator.num_heads,
        "translator_checkpoint": (
            str(args.translator_checkpoint) if args.translator_checkpoint else None
        ),
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
