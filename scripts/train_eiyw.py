"""Train the EIYW translator (frozen Sharer/Receiver; only the translator learns).

Each example: encode the Sharer context -> translate to K receiver-anchored soft
tokens -> prepend to the Receiver input -> next-token cross-entropy on the gold
answer. Unlike LCF-X the whole prompt forward is grad-enabled (the soft tokens
sit at the front), so this is heavier on memory. The loop lives in
``methods.eiyw.training.train_translator``.

Example (smoke on the cross-context task file):

    python scripts/train_eiyw.py \
        --train-data data/tasks/hotpotqa_bridge_train.jsonl \
        --num-soft-tokens 16 --num-spans 4 --max-steps 200 \
        --checkpoint-out data/checkpoints/eiyw/translator.pt

Then evaluate the trained translator:

    python scripts/run_eiyw.py --dataset data/tasks/hotpotqa_bridge_val.jsonl \
        --translator-checkpoint data/checkpoints/eiyw/translator.pt \
        --output-dir results/eiyw_trained
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
    # On a smaller GPU, lower --batch-size and raise --grad-accum to keep the
    # effective batch (B*A) at 256 (see README).
    parser.add_argument("--num-soft-tokens", type=int, default=64, help="K soft tokens")
    parser.add_argument("--bottleneck-dim", type=int, default=256, help="Translator latent dim")
    parser.add_argument("--stage2-num-heads", type=int, default=8, help="Multi-head count for the stage-2 cross-attention")
    parser.add_argument("--stage2-head-dim", type=int, default=256, help="Per-head width for the stage-2 cross-attention")
    parser.add_argument("--num-spans", type=int, default=4, help="Sharer pooling spans")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16, help="Per-step micro-batch")
    parser.add_argument("--grad-accum", type=int, default=16, help="Micro-batches per opt step (eff. batch 256)")
    parser.add_argument("--warmup", type=int, default=200, help="Linear LR warmup steps")
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
        default=Path("data/checkpoints/eiyw/translator.pt"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    import json
    import random

    from methods.eiyw import EIYWTranslator, load_eiyw_pair, train_translator

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
    eval_msg = f" / {len(eval_rows)} eval" if eval_rows else " (no eval split)"
    print(f"Loaded {len(train_rows)} train" + eval_msg)

    pair = load_eiyw_pair(
        args.sharer_model, args.receiver_model, device=args.device, dtype=args.dtype
    )
    translator = EIYWTranslator.from_specs(
        pair.sharer_spec,
        pair.receiver_spec,
        num_soft_tokens=args.num_soft_tokens,
        bottleneck_dim=args.bottleneck_dim,
        stage2_num_heads=args.stage2_num_heads,
        stage2_head_dim=args.stage2_head_dim,
        zero_init=True,
    ).to(pair.device)

    train_translator(
        pair,
        translator,
        train_rows,
        eval_data=eval_rows,
        num_spans=args.num_spans,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        warmup_steps=args.warmup,
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
