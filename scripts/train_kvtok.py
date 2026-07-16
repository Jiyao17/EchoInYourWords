"""Train the KVTok translator (frozen Sharer/Receiver; only the translator learns).

Each example: prefill the Sharer and pool its cache with LCF-X's Stage 1 only;
factorized feature/token-axis linears produce ``--num-tokens`` slots; processed
Receiver context embeddings (lookup only, no Receiver prefill/cache) condition
them through cross-attention; the output head emits soft tokens prepended to the
Receiver prompt for next-token cross-entropy on the gold answer. The loop lives
in ``methods.kvtok.training.train_translator``.

Unlike EIYW, ``--num-spans``/``--num-tokens`` are baked into the translator's
weight shapes at construction time here, so they must be re-supplied
identically (defaults match) to ``scripts/run_kvtok.py`` when loading this
checkpoint.

Example (smoke on the cross-context task file):

    python scripts/train_kvtok.py \
        --train-data data/tasks/hotpotqa_bridge_train.jsonl \
        --num-spans 5 --num-tokens 8 --max-steps 200 \
        --checkpoint-out data/checkpoints/kvtok/translator.pt

Then evaluate the trained translator:

    python scripts/run_kvtok.py --dataset data/tasks/hotpotqa_bridge_val.jsonl \
        --num-spans 5 --num-tokens 8 \
        --translator-checkpoint data/checkpoints/kvtok/translator.pt \
        --output-dir results/kvtok_trained
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._defaults import DEFAULT_RECEIVER, DEFAULT_SHARER  # noqa: E402
from scripts._hf_cache import configure_hf_cache  # noqa: E402

configure_hf_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True, help="Instruction JSONL")
    # Sharer and Receiver are always independent models; default to a symmetric
    # Qwen pair when unspecified (the project assumes heterogeneous settings).
    parser.add_argument("--sharer-model", default=DEFAULT_SHARER)
    parser.add_argument("--receiver-model", default=DEFAULT_RECEIVER)
    parser.add_argument("--num-spans", type=int, default=5, help="Sharer Stage-1 pooling spans")
    parser.add_argument(
        "--num-tokens", type=int, default=8, help="Soft tokens the down-projection outputs"
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Translator width used before the EIYW-v2-style output head",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Cross-attention heads for slot->receiver-memory fusion",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    # KVTok's Receiver prompt forward is grad-enabled end to end, same as EIYW:
    # default to EIYW's half-batch / double-accumulation schedule.
    parser.add_argument("--batch-size", type=int, default=8, help="Per-step micro-batch (same default as EIYW)")
    parser.add_argument("--grad-accum", type=int, default=32, help="Micro-batches per opt step (eff. batch 256)")
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Linear LR warmup steps (default: auto, round(10%% of total optimizer steps))",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Cap on optimizer steps")
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Use only the first N training examples (partial run)",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-max-examples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument(
        "--eval-em-examples",
        type=int,
        default=0,
        help="Generation-based EM/F1 on N eval examples each eval (0=off; slower)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Training-progress JSONL (default: training_log.jsonl beside the checkpoint)",
    )
    parser.add_argument(
        "--eval-data", type=Path, default=None, help="Held-out JSONL (else use --val-frac)"
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.028,
        help="Dev holdout fraction of --train-data when --eval-data is unset",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("data/checkpoints/kvtok/translator.pt"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    import json
    import random

    from methods.kvtok import KVTokTranslator, load_kvtok_pair, train_translator

    def _read_jsonl(path: Path) -> list:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    rows = _read_jsonl(args.train_data)
    if args.eval_data is not None:
        train_rows, eval_rows = rows, _read_jsonl(args.eval_data)
    elif args.val_frac and args.val_frac > 0:
        random.Random(args.seed).shuffle(rows)
        n_val = max(1, int(len(rows) * args.val_frac))
        eval_rows, train_rows = rows[:n_val], rows[n_val:]
    else:
        train_rows, eval_rows = rows, None
    if args.max_train_examples is not None:
        train_rows = train_rows[: args.max_train_examples]
    if eval_rows and args.eval_max_examples is not None:
        eval_rows = eval_rows[: args.eval_max_examples]
    if args.warmup is None:
        effective_batch = max(1, int(args.batch_size)) * max(1, int(args.grad_accum))
        steps_per_epoch = max(1, -(-len(train_rows) // effective_batch))
        total_steps = (
            max(1, int(args.max_steps))
            if args.max_steps is not None
            else max(1, int(args.epochs)) * steps_per_epoch
        )
        warmup_steps = max(1, int(0.10 * total_steps + 0.5))
    else:
        warmup_steps = max(0, int(args.warmup))
    eval_msg = f" / {len(eval_rows)} eval" if eval_rows else " (no eval split)"
    print(f"Loaded {len(train_rows)} train" + eval_msg + f" | warmup={warmup_steps}")

    pair = load_kvtok_pair(
        args.sharer_model, args.receiver_model, device=args.device, dtype=args.dtype
    )
    translator = KVTokTranslator.from_specs(
        pair.sharer_spec,
        pair.receiver_spec,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_spans=args.num_spans,
        num_tokens=args.num_tokens,
        zero_init=True,
    ).to(pair.device)
    print(f"Translator trainable params: {translator.trainable_parameters()}")

    train_translator(
        pair,
        translator,
        train_rows,
        eval_data=eval_rows,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        warmup_steps=warmup_steps,
        max_steps=args.max_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_max_examples=args.eval_max_examples,
        save_every=args.save_every,
        eval_em_examples=args.eval_em_examples,
        seed=args.seed,
        checkpoint_path=str(args.checkpoint_out),
        log_path=str(args.log_file) if args.log_file else None,
    )


if __name__ == "__main__":
    main()
