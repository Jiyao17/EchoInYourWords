"""Run EIYW, the cross-context latent-communication method.

The Sharer's own context is pooled per layer with the paper-exact LCF-X pool
(up to k position-free slots per layer); the projected, layer/query-tagged
slots cross-attend into the Receiver's own context *embeddings* (a table
lookup + sinusoidal PE — no Receiver forward pass) and become k*L_s soft
tokens in the Receiver's embedding width, prepended to the Receiver's input
-- and the Receiver answers with those tokens leading its own prompt.
(Zero-init shape knobs are ignored when --translator-checkpoint is given; the
checkpoint carries its own stamped architecture, including the adaptive-k
setting.)

Example (cross-context HotpotQA-bridge, trained translator):

    python scripts/run_eiyw.py \
        --dataset data/tasks/hotpotqa_bridge_val.jsonl \
        --sharer-model unsloth/Llama-3.2-3B-Instruct \
        --receiver-model Qwen/Qwen3-0.6B \
        --translator-checkpoint data/checkpoints/eiyw/translator.pt \
        --output-dir results/eiyw_hotpotqa

With no ``--translator-checkpoint`` the translator uses a gentle
near-zero-output init, so the run sits close to (not exactly at) the
single-model floor (see the translator docstring for why exact zero is unsafe
for embedding-space tokens). Writes the four standardized output files.
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
from scripts._defaults import DEFAULT_RECEIVER, DEFAULT_SHARER, resolve_seed  # noqa: E402
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
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--translator-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--span-scheme",
        choices=["paragraphs", "fixed"],
        default="paragraphs",
        help=(
            "Sharer pooling span partition: 'paragraphs' uses the encode text's "
            "blank-line blocks, 'fixed' uses --num-spans token chunks. The pool "
            "has no P-dependent parameters, so any checkpoint accepts either."
        ),
    )
    parser.add_argument(
        "--num-spans",
        type=int,
        default=5,
        help="Span count for --span-scheme fixed",
    )
    parser.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help="Load the translator even if its stamped Sharer/Receiver differ",
    )
    parser.add_argument(
        "--queries-per-layer",
        type=int,
        default=1,
        help="Zero-init only (ignored with --translator-checkpoint): max Sharer pool queries per layer k (tokens = k*L_s)",
    )
    parser.add_argument(
        "--adaptive-k-tokens",
        type=int,
        default=0,
        help="Zero-init only (ignored with --translator-checkpoint): w>0 scales active k with Sharer length (k=clamp(ceil(N_s/w),1,k)); 0 = fixed",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=1,
        help="Zero-init only (ignored with --translator-checkpoint): fusion block count (self-attn + cross-attn + MLP)",
    )
    parser.add_argument(
        "--d-model",
        default=512,
        help="Zero-init only (ignored with --translator-checkpoint): translator width (int, default 512; 'half'|'hidden' as sweep options)",
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
    args.seed = resolve_seed(args.seed)

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
        ).to(pair.device)
        print(f"Loaded translator checkpoint: {args.translator_checkpoint}")
    else:
        translator = EIYWTranslator.from_specs(
            pair.sharer_spec,
            pair.receiver_spec,
            d_model=args.d_model,
            queries_per_layer=args.queries_per_layer,
            adaptive_k_tokens=args.adaptive_k_tokens,
            num_blocks=args.num_blocks,
            zero_init=True,
        ).to(pair.device)
        print("No checkpoint: using ZERO-INIT translator (== single-model floor)")
    translator.eval()
    run_seed = int(getattr(translator, "training_seed", args.seed))
    print(f"Run id: {args.run_id or '(unscoped)'} | seed: {run_seed}")
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
            span_scheme=args.span_scheme,
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
        "run_id": args.run_id,
        "seed": run_seed,
        "dataset": str(args.dataset),
        "sharer_model": args.sharer_model,
        "receiver_model": receiver_model,
        "num_examples": len(examples),
        "max_new_tokens": args.max_new_tokens,
        "num_spans": args.num_spans,
        "span_scheme": args.span_scheme,
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
