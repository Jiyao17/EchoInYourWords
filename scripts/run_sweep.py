"""Drive the model-pair x dataset x method experiment grid (see
``prompt/EXPERIMENT_SWEEP_PLAN.md``) via subprocess ``make`` calls, one
dataset at a time (the Makefile's ``exp`` target loops over methods and
datasets the same way it already does for ``data``/``model``).

Implements the **smoke** stage in full: reduced training/eval budget, a JSONL
manifest recording every run (every launch reruns every selected cell by
default — pass ``--resume`` to skip cells already ``ok``, e.g. to continue a
crashed sweep), and a fixed-size GPU worker pool via
``CUDA_VISIBLE_DEVICES``. The **full** stage uses the Makefile's paper-scale
defaults (full train split, full val EM/F1) except for BATCH_SIZE/GRAD_ACCUM,
which are picked per (method, Sharer/Receiver pair, dataset) (see
``pick_batch_grad_accum``; trained methods default to batch16/accum16 and
halve the batch as pair size grows;
qasc's short contexts get an 8x batch multiplier on top — 128/2 for the LCF-X
reference pair) so larger pairs don't OOM, always
keeping the effective batch (BATCH_SIZE x GRAD_ACCUM) at 256. Stage 2's priority-tiering (symmetric pairs
first, etc.) is not implemented yet -- run it manually via
``--sharer``/``--receiver`` filters for now.

    python scripts/run_sweep.py --dataset hotpotqa_bridge --gpus 1,2,3
    python scripts/run_sweep.py --dataset hotpotqa_bridge --dry-run
    python scripts/run_sweep.py --dataset hotpotqa_bridge --resume --gpus 1,2,3
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._defaults import resolve_seed  # noqa: E402

# Primary experiment model roster (one flat tier, no sub-tiers): the small
# Qwen model LCF-X was developed against, plus Gemma 1B and Llama 3B --
# one family per size and a 3x3 grid by default.
PRIMARY_MODELS = [
    "Qwen/Qwen3-0.6B",
    "unsloth/gemma-3-1b-it",
    "unsloth/Llama-3.2-3B-Instruct",
]
# Opt-in large tier: pair scope "all" (or legacy --extra-models) appends these
# to both axes. Scope "extend" selects only the seven cells touching this tier.
EXTRA_MODELS = [
    "mistralai/Ministral-8B-Instruct-2410",
]
METHODS = ["lcfx", "xkv", "t2t"]
METHOD_CHOICES = METHODS + [
    "xkv-no-receiver-pooling",
    "xkv-no-receiver-xattn",
    "xkv-lcfx-pooling",
]

# Approx parameter count (billions) per roster model, used only to size
# BATCH_SIZE/GRAD_ACCUM for the full stage (see pick_batch_grad_accum below).
MODEL_PARAMS_B = {
    "Qwen/Qwen3-0.6B": 0.6,
    "unsloth/gemma-3-1b-it": 1.0,
    "unsloth/Llama-3.2-3B-Instruct": 3.2,
    "mistralai/Ministral-8B-Instruct-2410": 8.02,
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
# "Smaller GPUs" table): the paper's reference pair (Qwen3-0.6B Sharer +
# Qwen3-0.6B Receiver, combined 1.2B) fits the paper's batch16/accum16
# default there. Larger Sharer/Receiver combos hold more frozen-model
# activations for the same projector, so batch halves (and grad-accum
# doubles) as combined pair size grows, keeping BATCH_SIZE x GRAD_ACCUM =
# 256 (same effective batch, same optimization, less parallelism) -- only
# slower, not different, matching README's guidance on this tradeoff.
# (threshold_billions, batch_size, grad_accum), first match wins:
BATCH_ACCUM_TIERS = [
    (2.5, 16, 16),  # up to 1B+1B
    (5.5, 8, 32),  # up to 1B+3B
    (9.5, 4, 64),  # up to 3B+3B / 1B+8B
    (13.0, 2, 128),  # up to 3B+8B
    (float("inf"), 1, 256),  # 8B+8B
]

# Per-dataset batch multiplier on top of the pair-size tiers. Memory scales
# with batch x sequence length, and the tiers above are sized for
# hotpotqa-length contexts (~750 prompt tokens); qasc's contexts are a few
# times shorter, so its per-device batch starts 8x higher (128 for the
# reference pair, verified against real memory usage) and still halves/
# doubles through the same tiers. accum shrinks by the same factor, keeping
# the effective batch at 256.
DATASET_BATCH_MULT = {"qasc": 8}


def pick_batch_grad_accum(
    method: str, sharer: str, receiver: str, dataset: str = ""
) -> tuple[int, int]:
    """BATCH_SIZE/GRAD_ACCUM for a (method, pair, dataset), keeping effective batch at 256."""

    tiers = BATCH_ACCUM_TIERS
    combined = MODEL_PARAMS_B.get(sharer, 0.0) + MODEL_PARAMS_B.get(receiver, 0.0)
    batch, accum = tiers[-1][1], tiers[-1][2]
    for threshold, tier_batch, tier_accum in tiers:
        if combined <= threshold:
            batch, accum = tier_batch, tier_accum
            break
    # XKV's no-grad dual-prefix path fits twice the reference micro-batch on
    # ROPES. Keep the same effective batch and the same pair-size progression:
    # 32/8 -> 16/16 -> 8/32 -> 4/64 -> 2/128.
    if method.startswith("xkv") and dataset == "ropes":
        batch *= 2
        accum = max(1, accum // 2)
    mult = DATASET_BATCH_MULT.get(dataset, 1)
    return batch * mult, max(1, accum // mult)


def resolve_overrides(
    stage: str, method: str, sharer: str, receiver: str, dataset: str = ""
) -> dict[str, str]:
    """Per-cell `make <method> ...` overrides.

    smoke keeps its fixed reduced-budget knobs regardless of pair. full leaves
    data/step budgets at the Makefile's paper defaults but still needs pair-size-aware
    BATCH_SIZE/GRAD_ACCUM (t2t has no batch knob, so it's skipped there) so
    larger pairs don't OOM.
    """

    overrides = dict(STAGE_OVERRIDES[stage])
    if stage == "full" and method != "t2t":
        batch, accum = pick_batch_grad_accum(method, sharer, receiver, dataset)
        overrides["BATCH_SIZE"] = str(batch)
        overrides["GRAD_ACCUM"] = str(accum)
    return overrides


# Substrings (checked case-insensitively against the tail of a failed run's
# log) that indicate an out-of-memory failure rather than a generic bug.
OOM_MARKERS = (
    "out of memory",
    "outofmemoryerror",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
)


def _slug(model_id: str) -> str:
    return model_id.replace("/", "--")


def resolve_pairs(scope: str = "base") -> list[tuple[str, str]]:
    """Return the base 3x3, 8B-extension 7, or complete 4x4 pair grid."""

    if scope == "base":
        roster = PRIMARY_MODELS
    elif scope in {"extend", "all"}:
        roster = PRIMARY_MODELS + EXTRA_MODELS
    else:
        raise ValueError(f"unknown pair scope: {scope!r}")

    pairs = [(sharer, receiver) for sharer in roster for receiver in roster]
    if scope == "extend":
        extras = set(EXTRA_MODELS)
        pairs = [
            (sharer, receiver)
            for sharer, receiver in pairs
            if sharer in extras or receiver in extras
        ]
    return pairs


def load_manifest(path: Path) -> dict[tuple[str | None, str, str, str, str, str], dict]:
    """Read the manifest and keep only the latest record per cell key."""

    latest: dict[tuple[str | None, str, str, str, str, str], dict] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (
                record.get("run_id"),
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


def should_run(prev_status: str | None, resume: bool) -> bool:
    """Every launch reruns every selected cell by default (a launch means "give
    me a fresh set of results"). ``--resume`` opts into skipping cells whose
    latest manifest status is ``ok`` — for continuing a crashed/interrupted
    sweep — while still rerunning failed/oom/stale-running cells."""

    if not resume:
        return True
    return prev_status != "ok"


def classify_failure(log_tail: str) -> str:
    lowered = log_tail.lower()
    if any(marker in lowered for marker in OOM_MARKERS):
        return "oom"
    return "failed"


def reduce_batch_after_oom(overrides: dict[str, str]) -> bool:
    """Halve per-device batch while preserving the effective batch size."""

    batch = int(overrides.get("BATCH_SIZE", 0) or 0)
    if batch <= 1:
        return False
    reduced = max(1, batch // 2)
    accum = int(overrides.get("GRAD_ACCUM", 1) or 1)
    overrides["BATCH_SIZE"] = str(reduced)
    overrides["GRAD_ACCUM"] = str(accum * batch // reduced)
    return True


def run_cell(
    *,
    stage: str,
    method: str,
    dataset: str,
    sharer: str,
    receiver: str,
    run_id: str | None,
    seed: int,
    results_root: Path,
    checkpoints_root: Path,
    gpu: str | None,
    manifest_path: Path,
    manifest_lock: threading.Lock,
    log_dir: Path,
) -> None:
    overrides = resolve_overrides(stage, method, sharer, receiver, dataset)

    env = None
    if gpu is not None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu

    base_log = (
        log_dir / stage / method / dataset / f"{_slug(sharer)}__{_slug(receiver)}.log"
    )
    base_log.parent.mkdir(parents=True, exist_ok=True)

    # Unattended-run resilience: repeatedly halve batch and increase gradient
    # accumulation on OOM, preserving the effective batch, until the cell
    # succeeds or batch 1 also fails. Each attempt gets its own log + manifest
    # record; --resume reads the latest record per cell, so a successful retry
    # marks the cell ok.
    attempt = 0
    while True:
        cmd = [
            "make",
            method,
            f"DATASET={dataset}",
            f"SHARER={sharer}",
            f"RECEIVER={receiver}",
            f"SEED={seed}",
            f"RESULTS_ROOT={results_root}",
            f"CHECKPOINTS_ROOT={checkpoints_root}",
        ]
        if run_id is not None:
            cmd.append(f"RUN={run_id}")
        cmd += [f"{key}={value}" for key, value in overrides.items()]
        log_path = (
            base_log if attempt == 0 else base_log.with_suffix(f".retry{attempt}.log")
        )

        run_label = f"run-{run_id}/" if run_id is not None else ""
        label = f"[{run_label}{stage}/{method}] {dataset} {sharer} -> {receiver}"
        if "BATCH_SIZE" in overrides:
            label += (
                f" (batch={overrides['BATCH_SIZE']} accum={overrides['GRAD_ACCUM']})"
            )
        if gpu is not None:
            label += f" (gpu {gpu})"
        if attempt:
            label += f" (oom retry {attempt})"
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
            "run_id": run_id,
            "seed": seed,
            "results_root": str(results_root),
            "checkpoints_root": str(checkpoints_root),
            "stage": stage,
            "method": method,
            "dataset": dataset,
            "sharer": sharer,
            "receiver": receiver,
            "status": status,
            "batch_size": overrides.get("BATCH_SIZE"),
            "grad_accum": overrides.get("GRAD_ACCUM"),
            "attempt": attempt,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "wall_s": round(wall_s, 1),
            "returncode": proc.returncode,
            "log_path": str(log_path.relative_to(REPO_ROOT)),
            "error_tail": log_tail[-500:] if status != "ok" else "",
        }
        append_manifest(manifest_path, record, manifest_lock)

        if status == "oom" and reduce_batch_after_oom(overrides):
            attempt += 1
            continue
        return


def worker_loop(gpu: str | None, cell_queue: "queue.Queue", run_kwargs: dict) -> None:
    while True:
        try:
            method, dataset, sharer, receiver = cell_queue.get_nowait()
        except queue.Empty:
            return
        try:
            run_cell(
                method=method,
                dataset=dataset,
                sharer=sharer,
                receiver=receiver,
                gpu=gpu,
                **run_kwargs,
            )
        finally:
            cell_queue.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, help="Single dataset name, e.g. hotpotqa_bridge"
    )
    parser.add_argument("--run-id", default=None, help="Replicate identifier")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Experiment seed (default: a fresh random seed per sweep invocation)",
    )
    parser.add_argument("--stage", default="smoke", choices=sorted(STAGE_OVERRIDES))
    parser.add_argument(
        "--method",
        action="append",
        choices=METHOD_CHOICES,
        help="Repeatable; default lcfx, xkv, t2t (XKV ablations are opt-in)",
    )
    parser.add_argument(
        "--sharer",
        action="append",
        help="Repeatable filter; default all models in PRIMARY_MODELS",
    )
    parser.add_argument(
        "--receiver",
        action="append",
        help="Repeatable filter; default all models in PRIMARY_MODELS",
    )
    parser.add_argument(
        "--extra-models",
        action="store_true",
        help="Legacy alias for --pair-scope all",
    )
    parser.add_argument(
        "--pair-scope",
        choices=["base", "extend", "all"],
        default="base",
        help="Pair grid: base=0.6/1/3B 3x3 (9), extend=only pairs touching 8B (7), all=0.6/1/3/8B 4x4 (16)",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids, e.g. 1,2,3 (default: no pinning, one worker)",
    )
    parser.add_argument("--concurrency-per-gpu", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip cells whose latest manifest status is ok (continue a crashed/"
            "interrupted sweep). Default: rerun every selected cell fresh."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the resolved cell list and exit"
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Explicit result root forwarded to every nested make cell",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=None,
        help="Explicit checkpoint root forwarded to every nested make cell",
    )
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)
    run_root = REPO_ROOT / "results"
    checkpoint_root = REPO_ROOT / "data" / "checkpoints"
    if args.run_id not in (None, "0"):
        run_root = run_root / "runs" / f"run-{args.run_id}"
        checkpoint_root = checkpoint_root / "runs" / f"run-{args.run_id}"
    args.results_root = args.results_root or run_root
    args.checkpoints_root = args.checkpoints_root or checkpoint_root
    args.manifest = args.manifest or run_root / "sweep_manifest.jsonl"
    args.log_dir = args.log_dir or run_root / "sweep_logs"

    pair_scope = "all" if args.extra_models else args.pair_scope
    pairs = resolve_pairs(pair_scope)
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
        key = (args.run_id, args.stage, method, dataset, sharer, receiver)
        prev = manifest_latest.get(key)
        prev_status = prev["status"] if prev else None
        if should_run(prev_status, args.resume):
            to_run.append((method, dataset, sharer, receiver))
        else:
            skipped += 1

    skip_note = f", {skipped} skipped (already ok; --resume)" if skipped else ""
    print(
        f"[{args.run_id or 'unscoped'}/{args.stage}] seed={args.seed} "
        f"dataset={args.dataset}: "
        f"{len(all_cells)} cells total, {len(to_run)} to run{skip_note}"
    )
    print(f"Results: {args.results_root}")
    print(f"Checkpoints: {args.checkpoints_root}")

    if args.concurrency_per_gpu < 1:
        parser.error("--concurrency-per-gpu must be at least 1")
    gpu_ids = (
        [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
        if args.gpus
        else [None]
    )
    if not gpu_ids:
        parser.error("--gpus must contain at least one GPU id")
    slots = [gpu for gpu in gpu_ids for _ in range(args.concurrency_per_gpu)]
    worker_names = [f"gpu:{gpu}" if gpu is not None else "unpinned" for gpu in slots]
    print(f"Workers ({len(slots)}): {', '.join(worker_names)}")

    if args.dry_run:
        for method, dataset, sharer, receiver in to_run:
            print(f"  RUN  {method:5s} {dataset} {sharer} -> {receiver}")
        return

    if not to_run:
        print("Nothing to run.")
        return

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_lock = threading.Lock()

    cell_queue: "queue.Queue" = queue.Queue()
    for cell in to_run:
        cell_queue.put(cell)

    run_kwargs = dict(
        stage=args.stage,
        run_id=args.run_id,
        seed=args.seed,
        results_root=args.results_root,
        checkpoints_root=args.checkpoints_root,
        manifest_path=args.manifest,
        manifest_lock=manifest_lock,
        log_dir=args.log_dir,
    )

    threads = [
        threading.Thread(
            target=worker_loop, args=(gpu, cell_queue, run_kwargs), daemon=True
        )
        for gpu in slots
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    manifest_latest = load_manifest(args.manifest)
    counts: dict[str, int] = {}
    for method, dataset, sharer, receiver in to_run:
        key = (args.run_id, args.stage, method, dataset, sharer, receiver)
        status = manifest_latest.get(key, {}).get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    print(f"Done. {counts}")


if __name__ == "__main__":
    main()
