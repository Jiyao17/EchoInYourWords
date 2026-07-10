"""Run EIYW, the receiver-anchored cross-context communication method.

The Sharer encodes its own context; its KV cache is translated into K soft
pseudo-tokens — conditioned on the Receiver's question — that are prepended to
the Receiver's input embeddings. The Receiver then answers from its own context
plus those soft tokens.

Example (cross-context HotpotQA-bridge, trained translator):

    python scripts/run_eiyw.py \
        --dataset data/tasks/hotpotqa_bridge_val.jsonl \
        --sharer-model Qwen/Qwen2.5-0.5B-Instruct \
        --receiver-model Qwen/Qwen3-0.6B \
        --translator-checkpoint data/checkpoints/eiyw/translator.pt \
        --output-dir results/eiyw_hotpotqa

With no ``--translator-checkpoint`` the translator is zero-initialized (soft
tokens start near zero — a plumbing run, not the method). Writes the four
standardized output files.
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

from methods.eiyw import (  # noqa: E402
    EIYWExample,
    EIYWTranslator,
    check_pair_compatible,
    load_eiyw_pair,
    load_translator,
    run_eiyw_example,
    write_run_outputs,
)
from methods.common import (  # noqa: E402
    ProgressLogger,
    eval_progress_fields,
    score_free_form,
    write_predictions_brief,
)


def load_examples(path: Path, limit: int | None) -> list[EIYWExample]:
    examples: list[EIYWExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(EIYWExample.from_mapping(json.loads(line)))
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
        "--num-soft-tokens",
        type=int,
        default=64,
        help="K soft tokens; must match the trained --translator-checkpoint",
    )
    parser.add_argument(
        "--bottleneck-dim",
        type=int,
        default=256,
        help="Translator latent dim; must match the trained checkpoint",
    )
    parser.add_argument(
        "--stage2-num-heads",
        type=int,
        default=8,
        help="Multi-head count for the stage-2 cross-attention; must match the trained checkpoint",
    )
    parser.add_argument(
        "--stage2-head-dim",
        type=int,
        default=256,
        help="Per-head width for the stage-2 cross-attention; must match the trained checkpoint",
    )
    parser.add_argument("--num-spans", type=int, default=4, help="Sharer pooling spans")
    parser.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help="Load the translator even if its stamped Sharer/Receiver differ",
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
    pair = load_eiyw_pair(
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
            device=pair.device,
            allow_model_mismatch=args.allow_model_mismatch,
            num_soft_tokens=args.num_soft_tokens,
            bottleneck_dim=args.bottleneck_dim,
            stage2_num_heads=args.stage2_num_heads,
            stage2_head_dim=args.stage2_head_dim,
        ).to(pair.device)
        print(f"Loaded translator checkpoint: {args.translator_checkpoint}")
    else:
        translator = EIYWTranslator.from_specs(
            pair.sharer_spec,
            pair.receiver_spec,
            num_soft_tokens=args.num_soft_tokens,
            bottleneck_dim=args.bottleneck_dim,
            stage2_num_heads=args.stage2_num_heads,
            stage2_head_dim=args.stage2_head_dim,
            zero_init=True,
        ).to(pair.device)
        print("No checkpoint: using ZERO-INIT translator (soft tokens ~ 0; plumbing only)")
    translator.eval()
    print(f"Translator trainable params: {translator.trainable_parameters()}")

    # Persistent eval progress (long runs survive the session); flushed per record.
    progress = ProgressLogger(
        args.output_dir / "eval_log.jsonl",
        config={
            "method": "eiyw",
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
        result = run_eiyw_example(
            example,
            pair,
            translator,
            num_spans=args.num_spans,
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
        "method": "eiyw",
        "dataset": str(args.dataset),
        "sharer_model": args.sharer_model,
        "receiver_model": receiver_model,
        "num_examples": len(examples),
        "max_new_tokens": args.max_new_tokens,
        "num_spans": args.num_spans,
        "num_soft_tokens": args.num_soft_tokens,
        "bottleneck_dim": args.bottleneck_dim,
        "stage2_num_heads": args.stage2_num_heads,
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
