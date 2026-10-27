"""Evaluate XKV dual-cache translation and write standardized outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._defaults import DEFAULT_RECEIVER, DEFAULT_SHARER, resolve_seed  # noqa: E402
from scripts._hf_cache import configure_hf_cache  # noqa: E402
from scripts._transfer import (  # noqa: E402
    add_transfer_arguments, transfer_metadata, validate_checkpoint_history,
)

configure_hf_cache()

from methods.common import (  # noqa: E402
    ProgressLogger,
    eval_progress_fields,
    score_free_form,
    write_predictions_brief,
)
from methods.xkv import (  # noqa: E402
    XKVExample,
    XKVTranslator,
    XKV_VARIANTS,
    check_pair_compatible,
    load_translator,
    load_xkv_pair,
    run_xkv_example,
    write_run_outputs,
)


def _load_examples(path: Path, limit: int | None) -> list[XKVExample]:
    examples: list[XKVExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(XKVExample.from_mapping(json.loads(line)))
                if limit is not None and len(examples) >= limit:
                    break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sharer-model", default=DEFAULT_SHARER)
    parser.add_argument("--receiver-model", default=DEFAULT_RECEIVER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--translator-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--d-model",
        default="eighth",
        help=(
            "Translator width: 'eighth' (default, receiver_hidden//8) or an "
            "explicit integer such as 128 or 256"
        ),
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=1,
        help="Translator depth (default: 1 for every Receiver).",
    )
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--variant", choices=XKV_VARIANTS, default="xkv")
    parser.add_argument(
        "--pool-k",
        "--pool-slots",
        dest="pool_slots",
        type=int,
        default=None,
        help=(
            "Learned summaries per layer/head/side. Default follows Receiver "
            "size: 0.6B=1, 1B=2, 3B=3, 8B=4."
        ),
    )
    parser.add_argument(
        "--pool-spans",
        type=int,
        default=5,
        help=(
            "Fallback fixed spans P for exact xkv-lcfx-pooling N->P->1; "
            "natural evidence paragraphs are used when available."
        ),
    )
    parser.add_argument("--allow-model-mismatch", action="store_true")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--scoring", choices=["auto", "choice", "free_form"], default="auto"
    )
    parser.add_argument("--print-every", type=int, default=10)
    add_transfer_arguments(parser)
    args = parser.parse_args()
    try:
        transfer_info = transfer_metadata(args, args.translator_checkpoint)
    except ValueError as exc:
        parser.error(str(exc))
    args.seed = resolve_seed(args.seed)

    examples = _load_examples(args.dataset, args.num_examples)
    if not examples:
        raise SystemExit(f"No examples loaded from {args.dataset}")
    free_form = (
        not any(example.choices for example in examples)
        if args.scoring == "auto"
        else args.scoring == "free_form"
    )
    score_fn = score_free_form if free_form else None
    print(f"Loaded {len(examples)} examples from {args.dataset}")
    print(f"Loading sharer={args.sharer_model} receiver={args.receiver_model}")
    pair = load_xkv_pair(
        args.sharer_model,
        args.receiver_model,
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
            d_model=args.d_model,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout=args.dropout,
            pool_slots=args.pool_slots,
            variant=args.variant,
            pool_spans=args.pool_spans,
        ).to(pair.device)
        print(f"Loaded XKV checkpoint: {args.translator_checkpoint}")
    else:
        translator = XKVTranslator.from_specs(
            pair.sharer_spec,
            pair.receiver_spec,
            d_model=args.d_model,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout=args.dropout,
            pool_slots=args.pool_slots,
            variant=args.variant,
            pool_spans=args.pool_spans,
            zero_init=True,
        ).to(pair.device)
        print("No checkpoint: zero-init XKV (Receiver-only equivalence)")
    validate_checkpoint_history(translator, transfer_info)
    translator.eval()
    run_seed = int(getattr(translator, "training_seed", args.seed))
    print(f"Run id: {args.run_id or '(unscoped)'} | seed: {run_seed}")
    print(f"XKV trainable params: {translator.trainable_parameters()}")

    progress = ProgressLogger(
        args.output_dir / "eval_log.jsonl",
        config={
            "method": args.variant,
            "run_id": args.run_id,
            "seed": run_seed,
            "dataset": str(args.dataset),
            "sharer_model": args.sharer_model,
            "receiver_model": args.receiver_model,
            "num_examples": len(examples),
            **transfer_info,
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    brief_path = args.output_dir / "predictions_brief.tsv"
    started = perf_counter()
    results = []
    for index, example in enumerate(examples, start=1):
        results.append(
            run_xkv_example(
                example,
                pair,
                translator,
                max_new_tokens=args.max_new_tokens,
                score_fn=score_fn,
            )
        )
        if args.print_every and index % args.print_every == 0:
            print(f"  processed {index}/{len(examples)}", flush=True)
            progress.log(
                "progress", **eval_progress_fields(results, len(examples), started)
            )
            write_predictions_brief(results, brief_path)
    progress.log("done", **eval_progress_fields(results, len(examples), started))
    progress.close()

    paths = write_run_outputs(
        results,
        args.output_dir,
        method=args.variant,
        metadata={
            "method": args.variant,
            "run_id": args.run_id,
            "seed": run_seed,
            "dataset": str(args.dataset),
            "sharer_model": args.sharer_model,
            "receiver_model": args.receiver_model,
            "num_examples": len(examples),
            "max_new_tokens": args.max_new_tokens,
            "d_model": translator.d_model,
            "d_model_spec": args.d_model,
            "num_blocks": translator.num_blocks,
            "num_heads": translator.num_heads,
            "dropout": args.dropout,
            "pool_slots": translator.pool_slots,
            "pool_k": translator.pool_slots,
            "pooling_mode": translator.pooling_mode,
            "output_mode": translator.output_mode,
            "receiver_pooling": translator.receiver_pooling,
            "pool_spans": translator.pool_spans,
            **({"training_seed": run_seed} if args.translator_checkpoint else {}),
            "translator_checkpoint": str(args.translator_checkpoint)
            if args.translator_checkpoint
            else None,
            "scoring": "free_form" if free_form else "multiple_choice",
            "dtype": args.dtype,
            **transfer_info,
        },
    )
    print("\n=== metrics ===")
    print(paths["metrics"].read_text())
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
