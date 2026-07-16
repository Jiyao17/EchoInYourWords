"""Train the EIYW translator (frozen Sharer/Receiver; only the translator learns).

Each example: encode the Sharer's own context (paper-exact LCF-X pool, up to
``--queries-per-layer`` k slots per layer -> k*L_s soft-token queries at
``--d-model`` width, default receiver_hidden//2, with layer/query tags); look
up the Receiver's own context *embeddings* (no forward pass) as the
cross-attention memory (+ sinusoidal PE); ``--num-blocks`` blocks of
self-attn + cross-attn + MLP refine the slots into the soft tokens ->
prepended to the Receiver's prompt (grad-enabled) -> next-token cross-entropy
on the gold answer. ``--adaptive-k-tokens w`` scales the active k with the
Sharer's context length (one query per w tokens, capped at k). The loop lives
in ``methods.eiyw.training.train_translator``.

Example (smoke on the cross-context task file):

    python scripts/train_eiyw.py \
        --train-data data/tasks/hotpotqa_bridge_train.jsonl \
        --max-steps 200 \
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
    # Defaults follow LCF-X (this iteration reuses its pooling design on both
    # sides). On a smaller GPU, lower --batch-size and raise --grad-accum to
    # keep the effective batch (B*A) at 256 (see README).
    parser.add_argument(
        "--span-scheme",
        choices=["paragraphs", "fixed"],
        default="paragraphs",
        help=(
            "Sharer pooling span partition: 'paragraphs' uses the encode "
            "text's blank-line blocks (LCF-X's scheme), 'fixed' uses --num-spans "
            "near-equal token chunks"
        ),
    )
    parser.add_argument("--num-spans", type=int, default=5, help="Span count for --span-scheme fixed")
    parser.add_argument(
        "--queries-per-layer",
        type=int,
        default=1,
        help=(
            "Max cross-span queries per layer in the Sharer pool (k). Output "
            "soft-token count = k*L_s; k>1 keeps k distinct summaries per layer "
            "(k=1 = paper-exact pool)"
        ),
    )
    parser.add_argument(
        "--adaptive-k-tokens",
        type=int,
        default=0,
        help=(
            "w>0: scale the ACTIVE queries per layer with the Sharer's context "
            "length — k(N_s)=clamp(ceil(N_s/w),1,--queries-per-layer), resolved "
            "per batch from its longest example (soft tokens = k*L_s, so short "
            "contexts emit fewer, long ones cap at the max; e.g. "
            "--queries-per-layer 8 --adaptive-k-tokens 256). 0 = fixed budget"
        ),
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=1,
        help=(
            "Self-attn + cross-attn + MLP blocks (the fusion depth); each "
            "block is ~16*d_model^2 params, so the default stays lean"
        ),
    )
    parser.add_argument(
        "--d-model",
        default=512,
        help=(
            "Translator width: an integer (default 512, fixed and "
            "receiver-independent — the old 'half' rule scaled params "
            "quadratically with the Receiver's width); 'half' "
            "(receiver_hidden//2) and 'hidden' remain as sweep options"
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    # EIYW's generation forward is grad-enabled end to end, so it holds ~2x
    # LCF-X's activations at the same batch: default to half LCF-X's batch and
    # double the accumulation (same effective batch 256).
    parser.add_argument("--batch-size", type=int, default=8, help="Per-step micro-batch (half LCF-X's; EIYW's forward is grad-enabled end to end)")
    parser.add_argument("--grad-accum", type=int, default=32, help="Micro-batches per opt step (eff. batch 256)")
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
        d_model=args.d_model,
        queries_per_layer=args.queries_per_layer,
        adaptive_k_tokens=args.adaptive_k_tokens,
        num_blocks=args.num_blocks,
        zero_init=True,
    ).to(pair.device)

    train_translator(
        pair,
        translator,
        train_rows,
        eval_data=eval_rows,
        num_spans=args.num_spans,
        span_scheme=args.span_scheme,
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
