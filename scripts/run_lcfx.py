"""Run the LCF-X cross-context latent cache communication method.

The Sharer encodes its own context; its KV cache is pooled into a fixed-size,
position-free summary, fused (via a trained — or zero-init — bottleneck) into the
Receiver's cache, and the Receiver answers from its own context plus that summary.

Example (cross-context HotpotQA-bridge, zero-init fuser == receiver baseline):

    python scripts/run_lcfx.py \
        --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
        --sharer-model Qwen/Qwen2.5-0.5B-Instruct \
        --receiver-model Qwen/Qwen2.5-0.5B-Instruct \
        --output-dir results/lcfx_hotpotqa_zeroinit

With no ``--fuser-checkpoint`` the fuser is zero-initialized, so LCF-X reproduces
the Receiver-only run (a plumbing-correctness check). ``--no-fusion`` runs the
Receiver alone explicitly. Writes the four standardized output files.
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
from scripts._defaults import DEFAULT_RECEIVER, DEFAULT_SHARER  # noqa: E402
from scripts._hf_cache import configure_hf_cache  # noqa: E402

configure_hf_cache()

from methods.lcfx import (  # noqa: E402
    LCFXExample,
    LCFXFuser,
    check_pair_compatible,
    load_fuser,
    load_lcfx_pair,
    run_lcfx_example,
    run_receiver_only_example,
    write_run_outputs,
)
from methods.t2t import score_free_form  # noqa: E402


def load_examples(path: Path, limit: int | None) -> list[LCFXExample]:
    examples: list[LCFXExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(LCFXExample.from_mapping(json.loads(line)))
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
    parser.add_argument("--fuser-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--bottleneck-dim",
        type=int,
        default=128,
        help="Latent dim; must match the trained --fuser-checkpoint (paper: 128)",
    )
    parser.add_argument("--num-spans", type=int, default=1, help="Sharer pooling spans")
    parser.add_argument(
        "--no-fusion", action="store_true", help="Run the Receiver alone (no fusion)"
    )
    parser.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help="Load the fuser even if its stamped Sharer/Receiver differ from the pair",
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
    pair = load_lcfx_pair(
        args.sharer_model,
        receiver_model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    print("pair compatibility:", json.dumps(check_pair_compatible(pair)))

    fuser = None
    if not args.no_fusion:
        if args.fuser_checkpoint is not None:
            fuser = load_fuser(
                str(args.fuser_checkpoint),
                pair,
                device=pair.device,
                allow_model_mismatch=args.allow_model_mismatch,
                bottleneck_dim=args.bottleneck_dim,
            ).to(pair.device)
            print(f"Loaded fuser checkpoint: {args.fuser_checkpoint}")
        else:
            fuser = LCFXFuser.from_specs(
                pair.sharer_spec,
                pair.receiver_spec,
                bottleneck_dim=args.bottleneck_dim,
                zero_init=True,
            ).to(pair.device)
            print("No checkpoint: using ZERO-INIT fuser (== receiver-only baseline)")
        fuser.eval()  # deterministic gating (plain sigmoid, no Gumbel noise)
        print(f"Fuser trainable params: {fuser.trainable_parameters()}")

    results = []
    for idx, example in enumerate(examples, start=1):
        if args.no_fusion:
            result = run_receiver_only_example(
                example, pair, max_new_tokens=args.max_new_tokens, score_fn=score_fn
            )
        else:
            result = run_lcfx_example(
                example,
                pair,
                fuser,
                num_spans=args.num_spans,
                max_new_tokens=args.max_new_tokens,
                score_fn=score_fn,
            )
        results.append(result)
        if args.print_every and idx % args.print_every == 0:
            print(f"  processed {idx}/{len(examples)}", flush=True)

    run_metadata = {
        "method": "receiver_only" if args.no_fusion else "lcf_x",
        "dataset": str(args.dataset),
        "sharer_model": args.sharer_model,
        "receiver_model": receiver_model,
        "num_examples": len(examples),
        "max_new_tokens": args.max_new_tokens,
        "num_spans": args.num_spans,
        "bottleneck_dim": args.bottleneck_dim,
        "fuser_checkpoint": (
            str(args.fuser_checkpoint) if args.fuser_checkpoint else None
        ),
        "fusion": not args.no_fusion,
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
