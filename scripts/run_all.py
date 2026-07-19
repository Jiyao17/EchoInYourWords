"""Top-level orchestrator: distribute the full experiment sweep across every
dataset over the GPUs listed in ``CUDA_VISIBLE_DEVICES`` (see
``prompt/EXPERIMENT_SWEEP_PLAN.md``). One worker per GPU id; each worker
processes its assigned datasets sequentially via ``scripts/run_sweep.py
--dataset <ds> --gpus <that one gpu>``; workers for different GPUs run in
parallel with each other. Datasets are assigned round-robin, so e.g. with 2
GPUs and the project's 6 datasets, each GPU runs 3 datasets one after
another while both GPUs run in parallel.

Data/model prep is **not** done here -- `make sweep` handles that first via
its `prepare` dependency. Calling this script directly skips prepare (useful
once the repo is already fully prepared).

    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_all.py
    CUDA_VISIBLE_DEVICES=1,2 python scripts/run_all.py --stage smoke --dry-run
    python scripts/run_all.py                      # no GPUs pinned -> single sequential worker
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_DATASETS = ["hotpotqa_bridge", "musique", "qasc", "2wikimultihop", "ropes", "strategyqa"]


def build_sweep_args(args: argparse.Namespace) -> list[str]:
    """Flags forwarded verbatim to every `run_sweep.py --dataset ...` call."""

    forwarded = [
        "--stage", args.stage,
        "--concurrency-per-gpu", str(args.concurrency_per_gpu),
    ]
    for method in args.method or []:
        forwarded += ["--method", method]
    for sharer in args.sharer or []:
        forwarded += ["--sharer", sharer]
    for receiver in args.receiver or []:
        forwarded += ["--receiver", receiver]
    if args.extra_models:
        forwarded.append("--extra-models")
    else:
        forwarded += ["--pair-scope", args.pair_scope]
    if args.resume:
        forwarded.append("--resume")
    if args.dry_run:
        forwarded.append("--dry-run")
    return forwarded


def worker(
    gpu: str | None,
    datasets: list[str],
    sweep_args: list[str],
    results: dict[tuple[str | None, str], int],
) -> None:
    for dataset in datasets:
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_sweep.py"), "--dataset", dataset] + sweep_args
        if gpu is not None:
            cmd += ["--gpus", gpu]
        label = f"gpu={gpu or '-'} dataset={dataset}"
        print(f"=== worker start: {label} ===", flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        print(f"=== worker done:  {label} (exit {proc.returncode}) ===", flush=True)
        results[(gpu, dataset)] = proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None, help="Override the dataset list (default: all 6)"
    )
    parser.add_argument("--stage", default="smoke", choices=["smoke", "full"])
    parser.add_argument(
        "--method", action="append", choices=["lcfx", "xkv", "t2t"]
    )
    parser.add_argument("--sharer", action="append")
    parser.add_argument("--receiver", action="append")
    parser.add_argument("--extra-models", action="store_true")
    parser.add_argument(
        "--pair-scope",
        choices=["base", "extend", "all"],
        default="base",
        help="Model-pair grid forwarded to run_sweep.py (default: base)",
    )
    parser.add_argument("--concurrency-per-gpu", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells already ok in the manifest (default: rerun everything fresh)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = args.datasets or ALL_DATASETS

    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    gpu_ids = [g.strip() for g in raw.split(",") if g.strip()] if raw else []
    workers_gpus: list[str | None] = gpu_ids if gpu_ids else [None]

    buckets: dict[str | None, list[str]] = {gpu: [] for gpu in workers_gpus}
    for i, dataset in enumerate(datasets):
        gpu = workers_gpus[i % len(workers_gpus)]
        buckets[gpu].append(dataset)

    print(f"CUDA_VISIBLE_DEVICES={raw or '(unset)'} -> {len(workers_gpus)} worker(s)")
    for gpu in workers_gpus:
        print(f"  gpu={gpu or '-'}: {buckets[gpu]}")

    sweep_args = build_sweep_args(args)
    results: dict[tuple[str | None, str], int] = {}
    threads = [
        threading.Thread(target=worker, args=(gpu, buckets[gpu], sweep_args, results), daemon=True)
        for gpu in workers_gpus
        if buckets[gpu]
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = {k: v for k, v in results.items() if v != 0}
    if failed:
        print(f"FAILED dataset sweeps: {failed}", file=sys.stderr)
        sys.exit(1)
    print("All dataset sweeps completed successfully.")


if __name__ == "__main__":
    main()
