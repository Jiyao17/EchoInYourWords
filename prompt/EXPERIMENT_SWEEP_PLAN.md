# Experiment Sweep Plan: 5-Model Roster x 6 Datasets x 3 Methods

## Goal

Systematically evaluate all three methods (T2T, LCF-X, EIYW) across every
Sharer/Receiver pair drawn from the project's primary model roster, on every
one of the six prepared datasets, so we have one comparable results grid to
analyze.

## Axes

### Models (primary roster, 5 models, one flat tier)

One model per size from 0.5B to 8B, different families, all ungated:

- `Qwen/Qwen2.5-0.5B-Instruct` (0.5B) — used by LCF-X
- `Qwen/Qwen3-0.6B` (0.6B) — used by LCF-X
- `unsloth/Llama-3.2-1B-Instruct` (1B)
- `unsloth/gemma-3-4b-it` (4B)
- `Qwen/Qwen3-8B` (8B tier)

`Qwen/Qwen3-4B` and `unsloth/gemma-3-1b-it` were considered but are **not**
part of the roster — excluded as redundant with the models above (extra
Qwen/Gemma sizes outside the 0.5/0.6/1/4/8B ladder this sweep targets).

- **Datasets** (6): `hotpotqa_bridge`, `musique`, `qasc`, `2wikimultihop`,
  `ropes`, `strategyqa`
- **Sharer x Receiver pairs** (5 x 5 = 25, includes symmetric self-pairs)
- **Methods** (3): T2T (eval only, no training), LCF-X (train fuser + eval),
  EIYW (train translator + eval)

**Grid size:** 25 pairs x 6 datasets = **150 configs**; x 3 methods = **450
experiment cells**. 300 of those cells (LCF-X + EIYW, 150 each) need a
training run first; the other 150 (T2T) are eval-only.

**Complementary, not part of the 150/450:** the single-model oracle baseline
(`make single`) is not pair-dependent — 5 models x 6 datasets = 30 cheap
eval-only runs. Worth collecting alongside the grid as an upper-bound
reference column, but tracked separately.

One cell already has a head start: `data/checkpoints/lcfx/hotpotqa_bridge/
Qwen--Qwen2.5-0.5B-Instruct/Qwen--Qwen3-0.6B/fuser.pt` exists from earlier
development (trained, not yet evaluated at sweep scale). The sweep driver
doesn't currently detect/reuse pre-existing checkpoints (see "Still needed"
below) — revisit this once the `full` stage is in play, where reuse matters
more than at smoke scale.

## Stage 0 — Prep (one-time, cheap)

- `make data` — builds all 6 dataset train/val files. `hotpotqa_bridge`,
  `musique`, `qasc` are already built; `2wikimultihop` / `ropes` /
  `strategyqa` still need a real (`-1`) run (only smoke-tested so far).
- `make model` — fetches all 5 primary-roster models into `data/hf_cache`;
  already downloaded (confirmed via `data/hf_cache/hub/`); idempotent to
  re-run. `Qwen/Qwen3-4B`'s cache (7.6GB, downloaded while it was still part
  of the roster) has been deleted now that it's excluded.

## Stage 1 — Smoke sweep (all 450 cells, reduced budget)

**Purpose:** catch structural failures per pair/dataset cheaply before
committing to full compute — tokenizer quirks, Gemma's unusual
`head_dim=256`, OOM at the 8B<->8B cell, dataset-specific edge cases (e.g.
StrategyQA's small size) — and get a rough directional signal on which
pairs/datasets look interesting.

Proposed budget overrides (vs. Makefile defaults):

| Knob | Stage 1 value | Why |
|---|---|---|
| `MAX_TRAIN` | ~800 | Enough steps to see the loss move, not enough to actually converge |
| `MAX_STEPS` | ~40 | Caps wall time per cell regardless of dataset size |
| `EVAL_N` | ~150 | Directional EM/F1, not the final number |
| `BATCH_SIZE` / `GRAD_ACCUM` | scaled down for 8B pairs | Keep the 8B<->8B cell inside ~40GB free per GPU (see Risk notes) |

**Execution:** `scripts/run_sweep.py` (see "Sweep driver script" below)
iterates `make lcfx` / `make eiyw` / `make t2t` over all 25 pairs for one
dataset+method at a time, with the Stage 1 overrides, logs one manifest line
per cell (ok / failed / oom + wall time), and is resumable (skips cells
already marked done) so a crash doesn't lose progress.

**Deliverable:** `results/sweep_manifest.jsonl` (stage=`smoke`) — used to
triage before spending Stage 2 compute.

## Stage 2 — Full-scale runs, staged by priority

Priority tiers (review results after each tier before starting the next —
don't blindly launch all 300 full trainings at once):

1. **Priority 1 — symmetric pairs**: the 5 same-model pairs (0.5B-0.5B,
   0.6B-0.6B, 1B-1B, 4B-4B, 8B-8B) x 6 datasets x 3 methods = 90 cells.
   Establishes same-model baselines across every dataset first.
2. **Priority 2 — heterogeneous pairs that passed Stage 1 cleanly** x 6
   datasets x 3 methods.
3. **Priority 3 — remaining/edge-case cells** that errored or looked
   degenerate in Stage 1, revisited individually once the cause is
   understood.

Budget: paper defaults (full train split, effective batch 256, full val
EM/F1) — `make lcfx` / `make eiyw` / `make t2t` with no manual overrides.
`run_sweep.py --stage full` now auto-picks `BATCH_SIZE`/`GRAD_ACCUM` per
Sharer/Receiver pair (see `pick_batch_grad_accum` — sized for an ~80GB GPU,
always keeping the effective batch at 256), so no per-tier manual tuning is
needed for pair size alone. Re-check `nvidia-smi` before launching each tier
and cap concurrency to what actually fits alongside whatever else is running
on the box.

## Stage 3 — Aggregate & analyze

New script `scripts/aggregate_results.py` (not yet written): walks
`results/<method>/<dataset>/<sharer>/<receiver>/metrics.json` for every
completed cell and emits one CSV/table with method, dataset, sharer,
receiver, sharer/receiver size, EM, F1, token counts, latency. Optionally
folds in the single-model oracle runs as an upper-bound reference column per
dataset/model.

## Sweep driver script (`scripts/run_sweep.py` — Stage 1 implemented)

**Status: implemented and verified for the smoke stage** (real end-to-end
run: T2T, 1B<->1B, `hotpotqa_bridge`, 150 eval examples, wrote a correct
`metrics.json` and manifest entry; resume-skip and `--force` both verified).
The `full` stage uses the Makefile's paper-scale defaults (full train split,
full val EM/F1) plus a pair-size-aware `BATCH_SIZE`/`GRAD_ACCUM`
(`pick_batch_grad_accum`, sized for an ~80GB GPU, effective batch always
256) so larger pairs don't OOM without hand-tuning per cell. Stage 2's
priority-tiering (`--priority-tier`) is **not yet implemented** — for now,
scope a full-stage run manually with `--sharer`/`--receiver` filters.

The script enumerates cells (method x sharer x receiver, for the one
`--dataset` it's given) from the fixed 5-model `PRIMARY_MODELS` roster,
maintains a resumable manifest, and shells out to the existing `make lcfx` /
`make eiyw` / `make t2t` targets — so all of the project's existing budget
knobs, checkpoint stamping, and results auto-separation keep working
unmodified. The **Makefile `exp` target** splits tests **method-major,
dataset-minor**: `METHOD=` selects one method (omit it to loop `t2t` ->
`lcfx` -> `eiyw` in turn), and within that, `DATASET=` selects one dataset
(omit it to loop all six) — mirroring how `data`/`model` resolve
`DATASET=`/`MODEL=`, just nested one level deeper. Internally this is two
targets: `exp` resolves the method axis (recursing into itself once per
method if `METHOD=` is unset), then hands off to the private `_exp-dataset`
target, which resolves the dataset axis the same way `data`/`model` already
do (recursing into itself once per dataset if `DATASET=` is unset) before
finally calling `run_sweep.py`.

**Cell identity & manifest** — `results/sweep_manifest.jsonl`, one JSON line
per attempt: `{stage, method, dataset, sharer, receiver, status,
started_at, finished_at, wall_s, returncode, log_path, error_tail}`.
`status` in `{ok, failed, oom}`. Default behavior is idempotent: a cell
already `ok` for the requested stage is skipped, so the same command can be
re-run safely to pick up where a crash left off. Per-cell stdout/stderr is
captured to `results/sweep_logs/<stage>/<method>/<dataset>/<sharer>__<
receiver>.log` (slugged like the Makefile's `/` -> `--`); a failed cell's
log tail is scanned for CUDA OOM markers to distinguish `oom` from a
generic `failed`.

**The "simple command" — via `make exp` (mirrors `make data`/`make
model`, now nested method -> dataset):**

```bash
make exp                                                 # every method, every dataset (450 cells)
make exp METHOD=t2t                                      # t2t only, every dataset (150 cells)
make exp METHOD=t2t DATASET=hotpotqa_bridge              # t2t only, one dataset (25 cells)
make exp DATASET=hotpotqa_bridge                          # every method, one dataset (75 cells)
make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_GPUS=1,2,3
```

**Picking specific cells to (re)run — via `EXP_ARGS=` passthrough:**

```bash
make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_ARGS="--dry-run"          # preview the cell list only
make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_ARGS="--rerun-failed" EXP_GPUS=1,2,3
make exp METHOD=lcfx DATASET=hotpotqa_bridge EXP_ARGS="--force"           # force-rerun one method
make exp DATASET=hotpotqa_bridge \
  EXP_ARGS="--sharer unsloth/gemma-3-4b-it --receiver Qwen/Qwen3-8B"      # narrow to one pair
```

Or call `scripts/run_sweep.py` directly for the same effect (what
`_exp-dataset` shells out to) — useful when scripting outside `make`; note
it always takes a single `--dataset` (the method-loop / dataset-loop live in
the Makefile, not the script):

```bash
python scripts/run_sweep.py --dataset hotpotqa_bridge --method t2t --gpus 1,2,3 --rerun-failed
```

## Full pipeline (`make sweep` + `scripts/run_all.py` — implemented)

One command for the entire workflow, prep through experiments:
`make sweep` depends on `prepare` (data + model, idempotent), then hands off
to **`scripts/run_all.py`**, which distributes the six datasets across the
GPU ids in `CUDA_VISIBLE_DEVICES` — **one worker per GPU id, running in
parallel; each worker processes its assigned datasets sequentially**
(round-robin assignment, so e.g. 2 GPUs x 6 datasets = 3 datasets per GPU).
Every per-dataset run goes through the same `run_sweep.py` used by `exp`, so
the same manifest/resume/GPU-pinning/rerun semantics apply. `METHOD=` also
works here (forwarded straight to `run_sweep.py --method`) — but note `sweep`
distributes by *dataset* across GPUs regardless of `METHOD=`; it does not
adopt `exp`'s method-major recursion (each GPU's worker still runs all
methods -- or just the one given -- for each of its assigned datasets).

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 make sweep                          # prepare everything, then sweep at full scale, 4-way parallel
CUDA_VISIBLE_DEVICES=1,2     make sweep SWEEP_ARGS="--dry-run"   # preview the whole plan, no GPUs touched
CUDA_VISIBLE_DEVICES=1,2     make sweep SWEEP_ARGS="--rerun-failed"
make sweep                                                        # no GPUs pinned -> single sequential worker
CUDA_VISIBLE_DEVICES=0,1,2,3 make sweep EXP_STAGE=smoke           # cheap reduced-budget sanity pass instead
```

Verified for real (at the prior 3-model/9-pair roster, before the 0.5B/0.6B
merge below -- mechanics unchanged, only the pair count grew):
`CUDA_VISIBLE_DEVICES=1,2,3 make sweep SWEEP_ARGS="--dry-run"` ran `prepare`
(fast, everything already cached) then correctly split the 6 datasets 2/2/2
across GPUs 1/2/3 and previewed every cell with the right parallel worker
interleaving; a 2-GPU split and the no-GPU fallback (single sequential
worker over all 6 datasets) were also verified directly against
`run_all.py`. Re-verified after the roster change that `make exp`/`make
sweep` produce the new 25-pair cell counts (25/75/150/450) correctly.

Along the way, fixed a pre-existing bug this surfaced: the Makefile's
`SHARER` default was `Qwen/Qwen3-0.5B-Instruct`, which doesn't exist on the
Hub (Qwen3's smallest is 0.6B) — every other reference in the repo uses
`Qwen/Qwen2.5-0.5B-Instruct` for the 0.5B LCF-X default, so `make
model`/`make prepare`/`make sweep` would have failed trying to download it.
Fixed to match.

**Full flag list:** `--dataset` (required), `--stage {smoke,full}` (script's
own default is `smoke`, but `make exp`/`make sweep` pass `--stage full` by
default via `EXP_STAGE ?= full` in the Makefile -- pass `EXP_STAGE=smoke` to
get the script's cheap default instead), `--method {t2t,lcfx,eiyw}`
(repeatable, default all 3),
`--sharer`/`--receiver` (repeatable filters into the 5-model roster),
`--gpus` (comma-separated ids; default: no `CUDA_VISIBLE_DEVICES` pinning,
one worker), `--concurrency-per-gpu` (default 1), `--rerun-failed`,
`--force`, `--dry-run`, `--manifest`/`--log-dir` (override the default
paths).

**Still needed:** `scripts/aggregate_results.py` (Stage 3) — flattens all
`metrics.json` files into one comparison table. Stage 2's priority-tier
filter (`--priority-tier`) and pre-existing-checkpoint reuse (skip retraining
a cell whose checkpoint already exists on disk from a manual run) are also
not yet built.

No changes needed to `methods/`, `scripts/prepare_*.py`, or the Makefile's
existing targets — the sweep is entirely driven through the existing
per-config targets and their automatic `results/`/`data/checkpoints/`
separation by dataset+pair.

## Risk notes

- **Gemma pairs** (`head_dim=256`, vs. 64/128 for Llama/Qwen) exercise the
  LCF-X fuser's shape-bridging hardest — worth checking first in Stage 1.
- **Qwen-8B <-> Qwen-8B** is the heaviest memory cell (two 8B models loaded
  frozen simultaneously, plus the trainable fuser/translator) — automatically
  gets the smallest `BATCH_SIZE`/largest `GRAD_ACCUM` tier (1/256) at "full
  scale" (see Stage 2 above); still worth watching as the first real test of
  the tiering.
- **StrategyQA is small** (~2.3K examples total, ~337 val) — its full-scale
  "epoch" is naturally much shorter than HotpotQA's (~70.7K train); that's
  expected, not a bug.
- **The box is shared**: at plan time, GPU0 had ~77/80GB used by another
  job, GPUs 1-3 had ~38-40GB free each. Pin sweep runs to GPUs 1-3 and
  re-check `nvidia-smi` before each stage — availability will change. The
  `--gpus` flag on `run_sweep.py` is exactly how this gets respected per
  invocation.
- **The 0.5B/0.6B pairs are much cheaper** than the 1B/4B/8B ones, so a
  `--gpus` worker can likely run several of them concurrently per GPU
  (`--concurrency-per-gpu`) even while an 8B cell is running one-at-a-time
  elsewhere — worth accounting for when scheduling Stage 2 priority tiers.

## Open items before executing the full sweep

- **Stage 0 still not done:** `2wikimultihop` / `ropes` / `strategyqa` still
  need a real (`-1`) `make data` run — only smoke-tested with a handful of
  examples so far. `make exp` will fail on those datasets until then.
- The Stage 1 budget knobs (`MAX_TRAIN=800` etc.) are unvalidated at scale —
  the one real cell run so far was T2T (eval-only, no training knobs
  exercised). Worth watching the first LCF-X/EIYW cell, and especially the
  first Qwen-8B<->Qwen-8B cell, for OOM before launching the full 25-pair
  sweep unattended.
- Stage 2 (`--priority-tier`), Stage 3 (`aggregate_results.py`), and
  pre-existing-checkpoint reuse are still just planned, not built.
