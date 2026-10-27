"""Optional transfer provenance shared by the XKV and LCF-X evaluators."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_transfer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--training-dataset", default=None)
    parser.add_argument("--inference-dataset", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--training-datasets", nargs="+", default=None)


def transfer_metadata(args: argparse.Namespace, checkpoint: Path | None) -> dict:
    """Require trained weights for transfer and preserve source/target identity."""

    if not any((args.training_dataset, args.inference_dataset, args.source_run_id,
                args.training_datasets)):
        return {}
    if not args.training_dataset or not args.inference_dataset:
        raise ValueError("transfer requires both --training-dataset and --inference-dataset")
    if checkpoint is None or not checkpoint.is_file():
        raise ValueError(f"transfer requires an existing trained checkpoint: {checkpoint}")
    history = args.training_datasets or [args.training_dataset]
    if len(history) != len(set(history)):
        raise ValueError("training datasets must not repeat")
    if args.inference_dataset in history:
        raise ValueError("transfer target has already been used for training")
    if args.training_datasets and args.training_dataset != "+".join(history):
        raise ValueError("--training-dataset must be the '+'-joined training history")
    return {
        "experiment_type": "cross_dataset_transfer",
        "training_dataset": args.training_dataset,
        "inference_dataset": args.inference_dataset,
        "source_run_id": args.source_run_id,
        "target_finetuning": False,
        "training_datasets": history,
        "transfer_step": len(history),
    }


def validate_checkpoint_history(module, metadata: dict) -> None:
    """Check declared transfer provenance against stamps in newer checkpoints."""
    history = getattr(module, "training_history", {}).get("training_datasets")
    if metadata and history and history != metadata["training_datasets"]:
        raise ValueError("declared training history does not match the checkpoint")


def add_warm_start_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--initial-checkpoint", type=Path, default=None,
                        help="Continue learned weights; reset optimizer and warmup")
    parser.add_argument("--training-datasets", nargs="+", default=None,
                        help="Full ordered dataset history, including this stage")


def validate_warm_start_paths(args) -> None:
    if args.initial_checkpoint is not None:
        if not args.initial_checkpoint.is_file():
            raise ValueError(f"missing initial checkpoint: {args.initial_checkpoint}")
        if args.initial_checkpoint.resolve() == args.checkpoint_out.resolve():
            raise ValueError("initial checkpoint and output checkpoint must differ")
        if not args.training_datasets or len(args.training_datasets) < 2:
            raise ValueError("warm start requires the full --training-datasets history")


def stamp_training_history(module, args) -> None:
    history = args.training_datasets
    if history is None:
        return
    if len(history) != len(set(history)):
        raise ValueError("training datasets must not repeat")
    previous = getattr(module, "training_history", {}).get("training_datasets")
    if previous and history[:-1] != previous:
        raise ValueError("new training history must extend the checkpoint by one dataset")
    module.training_history = {
        "training_datasets": history,
        "transfer_step": len(history),
        "initial_checkpoint": str(args.initial_checkpoint.resolve())
        if args.initial_checkpoint else None,
        "optimizer_reset": args.initial_checkpoint is not None,
    }
