"""Train XKV with frozen Sharer and Receiver language models."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._defaults import DEFAULT_RECEIVER, DEFAULT_SHARER  # noqa: E402
from scripts._hf_cache import configure_hf_cache  # noqa: E402

configure_hf_cache()


def _read_jsonl(path: Path) -> list:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, default=None)
    parser.add_argument("--sharer-model", default=DEFAULT_SHARER)
    parser.add_argument("--receiver-model", default=DEFAULT_RECEIVER)
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
    parser.add_argument(
        "--pool-k",
        "--pool-slots",
        dest="pool_slots",
        type=int,
        default=None,
        help=(
            "Learned summaries per layer/head/side. Default follows Receiver "
            "size: 0.6B=1, 1B=2, 4B=3, 8B=4."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-max-examples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--eval-em-examples", type=int, default=0)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--val-frac", type=float, default=0.028)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("data/checkpoints/xkv/translator.pt"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    from methods.xkv import XKVTranslator, load_xkv_pair, train_translator

    rows = _read_jsonl(args.train_data)
    if args.eval_data is not None:
        training, evaluation = rows, _read_jsonl(args.eval_data)
    elif args.val_frac > 0:
        random.Random(args.seed).shuffle(rows)
        count = max(1, int(len(rows) * args.val_frac))
        evaluation, training = rows[:count], rows[count:]
    else:
        training, evaluation = rows, None
    if args.max_train_examples is not None:
        training = training[: args.max_train_examples]
    if evaluation and args.eval_max_examples is not None:
        evaluation = evaluation[: args.eval_max_examples]
    print(
        f"Loaded {len(training)} train"
        + (f" / {len(evaluation)} eval" if evaluation else " (no eval split)")
    )

    pair = load_xkv_pair(
        args.sharer_model,
        args.receiver_model,
        device=args.device,
        dtype=args.dtype,
    )
    translator = XKVTranslator.from_specs(
        pair.sharer_spec,
        pair.receiver_spec,
        d_model=args.d_model,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        pool_slots=args.pool_slots,
        zero_init=True,
    ).to(pair.device)
    train_translator(
        pair,
        translator,
        training,
        eval_data=evaluation,
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
