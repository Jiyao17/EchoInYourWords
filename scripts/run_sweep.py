"""Drive the model-pair x dataset x method experiment grid (see
``prompt/EXPERIMENT_SWEEP_PLAN.md``) via subprocess ``make`` calls, one
dataset at a time (the Makefile's ``exp`` target loops over methods and
datasets the same way it already does for ``data``/``model``).

Implements the **smoke** stage in full: reduced training/eval budget, a
resumable JSONL manifest (skip cells already ``ok`` unless ``--force`` /
``--rerun-failed``), and a fixed-size GPU worker pool via
``CUDA_VISIBLE_DEVICES``. The **full** stage uses the Makefile's paper-scale
defaults (full train split, full val EM/F1) except for BATCH_SIZE/GRAD_ACCUM,
which are picked per Sharer/Receiver pair (see ``pick_batch_grad_accum``) so
larger pairs don't OOM, always keeping the effective batch
(BATCH_SIZE x GRAD_ACCUM) at 256. Stage 2's priority-tiering (symmetric pairs
first, etc.) is not implemented yet -- run it manually via
``--sharer``/``--receiver`` filters for now.

    python scripts/run_sweep.py --dataset hotpotqa_bridge --gpus 1,2,3
    python scripts/run_sweep.py --dataset hotpotqa_bridge --dry-run
    python scripts/run_sweep.py --dataset hotpotqa_bridge --rerun-failed --gpus 1,2,3
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Primary experiment model roster (one flat tier, no sub-tiers): the two
# small Qwen models LCF-X was developed against, plus one model per size at
# 1B/4B/8B, different families.
PRIMARY_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/gemma-3-4b-it",
    "Qwen/Qwen3-8B",
]
METHODS = ["t2t", "lcfx", "eiyw"]

# Approx parameter count (billions) per roster model, used only to size
# BATCH_SIZE/GRAD_ACCUM for the full stage (see pick_batch_grad_accum below).
MODEL_PARAMS_B = {
    "Qwen/Qwen2.5-0.5B-Instruct": 0.5,
    "Qwen/Qwen3-0.6B": 0.6,
    "unsloth/Llama-3.2-1B-Instruct": 1.0,
    "unsloth/gemma-3-4b-it": 4.0,
    "Qwen/Qwen3-8B": 8.0,
}

# Budget overrides passed to `make <method> ...` as command-line variables.
# Empty dict ("full") means no overrides beyond the pair-size-aware
# BATCH_SIZE/GRAD_ACCUM added by resolve_overrides() -- the Makefile's other
# defaults (full train split, full val EM/F1, paper LR/warmup/etc.) apply.
STAGE_OVERRIDES = {
    "smoke": {
        "MAX_TRAIN": "800",
        "MAX_STEPS": "40",
        "EVAL_N": "150",
        "BATCH_SIZE": "4",
        "GRAD_ACCUM": "8",
        "EVAL_EVERY": "20",
        "SAVE_EVERY": "40",
        "LOG_EVERY": "5",
        "EVAL_MAX": "150",
    },
    "full": {},
}

# Sized for an ~80GB GPU (the paper's own hardware -- see README.md's
# "Smaller GPUs" table): the paper's reference pair (Qwen2.5-0.5B Sharer +
# Qwen3-0.6B Receiver, combined 1.1B) fits the paper's batch16/accum16
# default there. Larger Sharer/Receiver combos hold more frozen-model
# activations for the same projector, so batch halves (and grad-accum
# doubles) as combined pair size grows, keeping BATCH_SIZE x GRAD_ACCUM =
# 256 (same effective batch, same optimization, less parallelism) -- only
# slower, not different, matching README's guidance on this tradeoff.
# (threshold_billions, batch_size, grad_accum), first match wins:
BATCH_ACCUM_TIERS = [
    (2.5, 16, 16),   # up to 1B+1B
    (5.5, 8, 32),    # up to 1B+4B
    (9.5, 4, 64),    # up to 4B+4B / 1B+8B
    (13.0, 2, 128),  # up to 4B+8B
    (float("inf"), 1, 256),  # 8B+8B
]


def pick_batch_grad_accum(sharer: str, receiver: str) -> tuple[int, int]:
    """BATCH_SIZE/GRAD_ACCUM for a pair, keeping effective batch at 256."""

    combined = MODEL_PARAMS_B.get(sharer, 0.0) + MODEL_PARAMS_B.get(receiver, 0.0)
    for threshold, batch, accum in BATCH_ACCUM_TIERS:
        if combined <= threshold:
            return batch, accum
    return BATCH_ACCUM_TIERS[-1][1], BATCH_ACCUM_TIERS[-1][2]


def resolve_overrides(stage: str, method: str, sharer: str, receiver: str) -> dict[str, str]:
    """Per-cell `make <method> ...` overrides.

    smoke keeps its fixed reduced-budget knobs regardless of pair. full
    leaves data/step budgets at the Makefile's paper defaults but still
    needs pair-size-aware BATCH_SIZE/GRAD_ACCUM (t2t has no batch knob, so
    it's skipped there) so larger pairs don't OOM.
    """

    overrides = dict(STAGE_OVERRIDES[stage])
    if stage == "full" and method != "t2t":
        batch, accum = pick_batch_grad_accum(sharer, receiver)
        overrides["BATCH_SIZE"] = str(batch)
        overrides["GRAD_ACCUM"] = str(accum)
    return overrides

# Substrings (checked case-insensitively against the tail of a failed run's
# log) that indicate an out-of-memory failure rather than a generic bug.
OOM_MARKERS = ("out of memory", "outofmemoryerror", "cuda error: out of memory")


def _slug(model_id: str) -> str:
    return model_id.replace("/", "--")


def resolve_pairs() -> list[tuple[str, str]]:
    """Return every Sharer/Receiver pair in the primary model roster."""

    return [(s, r) for s in PRIMARY_MODELS for r in PRIMARY_MODELS]


def load_manifest(path: Path) -> dict[tuple[str, str, str, str, str], dict]:
    """Read the manifest and keep only the latest record per cell key."""

    latest: dict[tuple[str, str, str, str, str], dict] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (
                record["stage"],
                record["method"],
                record["dataset"],
                record["sharer"],
                record["receiver"],
            )
            latest[key] = record
    return latest


def append_manifest(path: Path, record: dict, lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def should_run(prev_status: str | None, force: bool, rerun_failed: bool) -> bool:
    if force:
        return True
    if prev_status is None:
        return True
    if prev_status == "ok":
        return False
    return rerun_failed  # covers "failed" / "oom" / a stale "running"


def classify_failure(log_tail: str) -> str:
    lowered = log_tail.lower()
    if any(marker in lowered for marker in OOM_MARKERS):
        return "oom"
    return "failed"


def run_cell(
    *,
    stage: str,
    method: str,
    dataset: str,
    sharer: str,
    receiver: str,
    gpu: str | None,
    manifest_path: Path,
    manifest_lock: threading.Lock,
    log_dir: Path,
) -> None:
    overrides = resolve_overrides(stage, method, sharer, receiver)
    cmd = ["make", method, f"DATASET={dataset}", f"SHARER={sharer}", f"RECEIVER={receiver}"]
    cmd += [f"{key}={value}" for key, value in overrides.items()]

    env = None
    if gpu is not None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu

    log_path = log_dir / stage / method / dataset / f"{_slug(sharer)}__{_slug(receiver)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    label = f"[{stage}/{method}] {dataset} {sharer} -> {receiver}"
    if "BATCH_SIZE" in overrides:
        label += f" (batch={overrides['BATCH_SIZE']} accum={overrides['GRAD_ACCUM']})"
    if gpu is not None:
        label += f" (gpu {gpu})"
    print(f"START {label}", flush=True)

    started = datetime.now(timezone.utc)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT
        )
    wall_s = time.time() - t0
    finished = datetime.now(timezone.utc)

    log_tail = ""
    try:
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    except OSError:
        pass

    status = "ok" if proc.returncode == 0 else classify_failure(log_tail)
    print(f"{status.upper():6s} {label} ({wall_s:.0f}s)", flush=True)

    record = {
        "stage": stage,
        "method": method,
        "dataset": dataset,
        "sharer": sharer,
        "receiver": receiver,
        "status": status,
        "batch_size": overrides.get("BATCH_SIZE"),
        "grad_accum": overrides.get("GRAD_ACCUM"),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "wall_s": round(wall_s, 1),
        "returncode": proc.returncode,
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "error_tail": log_tail[-500:] if status != "ok" else "",
    }
    append_manifest(manifest_path, record, manifest_lock)


def worker_loop(gpu: str | None, cell_queue: "queue.Queue", run_kwargs: dict) -> None:
    while True:
        try:
            method, dataset, sharer, receiver = cell_queue.get_nowait()
        except queue.Empty:
            return
        try:
            run_cell(method=method, dataset=dataset, sharer=sharer, receiver=receiver, gpu=gpu, **run_kwargs)
        finally:
            cell_queue.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Single dataset name, e.g. hotpotqa_bridge")
    parser.add_argument("--stage", default="smoke", choices=sorted(STAGE_OVERRIDES))
    parser.add_argument(
        "--method", action="append", choices=METHODS, help="Repeatable; default all 3 (t2t, lcfx, eiyw)"
    )
    parser.add_argument(
        "--sharer", action="append", help="Repeatable filter; default all models in PRIMARY_MODELS"
    )
    parser.add_argument(
        "--receiver", action="append", help="Repeatable filter; default all models in PRIMARY_MODELS"
    )
    parser.add_argument(
        "--gpus", default=None, help="Comma-separated GPU ids, e.g. 1,2,3 (default: no pinning, one worker)"
    )
    parser.add_argument("--concurrency-per-gpu", type=int, default=1)
    parser.add_argument(
        "--rerun-failed", action="store_true", help="Also retry cells whose latest status is failed/oom"
    )
    parser.add_argument(
        "--force", action="store_true", help="Rerun matching cells regardless of manifest status"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved cell list and exit")
    parser.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "results" / "sweep_manifest.jsonl"
    )
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "results" / "sweep_logs")
    args = parser.parse_args()

    pairs = resolve_pairs()
    if args.sharer:
        pairs = [(s, r) for s, r in pairs if s in args.sharer]
    if args.receiver:
        pairs = [(s, r) for s, r in pairs if r in args.receiver]
    if not pairs:
        print("No pairs matched --sharer/--receiver.", file=sys.stderr)
        sys.exit(1)

    methods = args.method or METHODS
    manifest_latest = load_manifest(args.manifest)

    all_cells = [(m, args.dataset, s, r) for m in methods for s, r in pairs]
    to_run: list[tuple[str, str, str, str]] = []
    skipped = 0
    for method, dataset, sharer, receiver in all_cells:
        key = (args.stage, method, dataset, sharer, receiver)
        prev = manifest_latest.get(key)
        prev_status = prev["status"] if prev else None
        if should_run(prev_status, args.force, args.rerun_failed):
            to_run.append((method, dataset, sharer, receiver))
        else:
            skipped += 1

    print(
        f"[{args.stage}] dataset={args.dataset}: "
        f"{len(all_cells)} cells total, {len(to_run)} to run, {skipped} skipped (already ok)"
    )

    if args.dry_run:
        for method, dataset, sharer, receiver in to_run:
            print(f"  RUN  {method:5s} {dataset} {sharer} -> {receiver}")
        return

    if not to_run:
        print("Nothing to run.")
        return

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_lock = threading.Lock()

    gpu_ids = args.gpus.split(",") if args.gpus else [None]
    slots = [gpu for gpu in gpu_ids for _ in range(args.concurrency_per_gpu)]

    cell_queue: "queue.Queue" = queue.Queue()
    for cell in to_run:
        cell_queue.put(cell)

    run_kwargs = dict(
        stage=args.stage,
        manifest_path=args.manifest,
        manifest_lock=manifest_lock,
        log_dir=args.log_dir,
    )

    threads = [
        threading.Thread(target=worker_loop, args=(gpu, cell_queue, run_kwargs), daemon=True)
        for gpu in slots
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    manifest_latest = load_manifest(args.manifest)
    counts: dict[str, int] = {}
    for method, dataset, sharer, receiver in to_run:
        key = (args.stage, method, dataset, sharer, receiver)
        status = manifest_latest.get(key, {}).get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    print(f"Done. {counts}")


if __name__ == "__main__":
    main()
