"""Train the LCF-X fuser (frozen Sharer/Receiver; only the fuser learns).

Each example: encode the Sharer context -> pooled summary -> align -> project into
the Receiver's prompt cache (grad-enabled) -> next-token cross-entropy on the gold
answer. The loop lives in ``methods.lcfx.training.train_fuser``.

Example (smoke on the cross-context task file):

    python scripts/train_lcfx_fuser.py \
        --train-data data/tasks/hotpotqa_bridge_cross.jsonl \
        --num-spans 4 --max-steps 200 \
        --checkpoint-out data/checkpoints/lcfx/fuser.pt

Then evaluate the trained fuser:

    python scripts/run_lcfx.py --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
        --fuser-checkpoint data/checkpoints/lcfx/fuser.pt --output-dir results/lcfx_trained
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._hf_cache import configure_hf_cache  # noqa: E402

configure_hf_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True, help="Instruction JSONL")
    parser.add_argument("--sharer-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--receiver-model", default=None)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--num-spans", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument(
        "--eval-data", type=Path, default=None, help="Held-out JSONL (else use --val-frac)"
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of --train-data held out for eval when --eval-data is unset",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("data/checkpoints/lcfx/fuser.pt"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    import json
    import random

    from methods.lcfx import LCFXFuser, load_lcfx_pair, train_fuser

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
    print(
        f"Loaded {len(train_rows)} train"
        + (f" / {len(eval_rows)} eval" if eval_rows else " (no eval split)")
    )

    pair = load_lcfx_pair(
        args.sharer_model, args.receiver_model, device=args.device, dtype=args.dtype
    )
    fuser = LCFXFuser.from_specs(
        pair.sharer_spec,
        pair.receiver_spec,
        bottleneck_dim=args.bottleneck_dim,
        zero_init=True,
    ).to(pair.device)

    train_fuser(
        pair,
        fuser,
        train_rows,
        eval_data=eval_rows,
        num_spans=args.num_spans,
        lr=args.lr,
        epochs=args.epochs,
        max_steps=args.max_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        seed=args.seed,
        checkpoint_path=str(args.checkpoint_out),
    )


if __name__ == "__main__":
    main()
