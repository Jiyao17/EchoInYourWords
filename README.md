# Echo In Your Words

Research codebase for **flexible, cross-context communication between language
models** for multi-agent collaboration: a *Sharer* and *Receiver* hold different
evidence (and may differ in size/architecture), and we study how the Sharer can
help the Receiver answer. It compares three methods under one standardized
output contract: **T2T** (text message), **LCF-X** (position-free latent cache
communication), and **EIYW** — our **receiver-anchored** method (the Sharer
cache is translated into soft pseudo-tokens, conditioned on the Receiver's
question, prepended to the Receiver's input).

- Architecture & contracts: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- Full run reference (every flag): [scripts/README.md](scripts/README.md)
- Design notes: [prompt/](prompt/)

## Setup

- All commands here (Makefile and scripts) invoke plain `python` — they don't
  reach into any `.venv/bin/python`. **Activate your own Python environment
  first** (e.g. `source .venv/bin/activate`, or your conda/pyenv equivalent) so
  that `python` on your `PATH` resolves to the right interpreter.
- The Hugging Face cache is auto-routed to `data/hf_cache` by
  `scripts/_hf_cache.py`. Prefix a command with `HF_HUB_OFFLINE=1` once assets are
  downloaded to skip Hub lookups.

## Quickstart (via the Makefile)

**The one command that does everything** — prepares every dataset + model,
then sweeps every method × pair × dataset, parallelized across your GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 make sweep
```

**You must set `CUDA_VISIBLE_DEVICES`** to however many GPUs you actually
want to use (comma-separated ids, e.g. `0,1,2,3`) — `make sweep` reads it to
decide how many parallel workers to launch and which physical GPUs to pin
them to. Omit it and everything still runs, just as a single sequential
worker (much slower). See [Run experiments](#2-run-experiments) below for
scoping this down (one method, one dataset, one exact model pair, etc.)
instead of running the full grid.

**Always drive experiments through `make`, and name every axis explicitly** —
this keeps runs reproducible and self-documenting. Run `make help` for the
full target + knob list. There are two phases, kept separate: **prepare**
(download data + models, once, idempotent) and **run** (train/eval, as many
times as you like against whatever's already prepared).

### 1. Prepare (data + models)

One-time, idempotent (skipped if already present) — do it all with a single
command, no `DATASET=`/`MODEL=` means "all of them":

```bash
make prepare     # = make data + make model: every dataset, primary roster + pair-testing extras
```

Or pick just the axis you need:

```bash
make data                            # every dataset: hotpotqa_bridge, musique, qasc, 2wikimultihop, ropes, strategyqa
make data  DATASET=hotpotqa_bridge   # ...or just one: train (~70.7K) + val (5,899)

make model                                    # fetch every known-good model into data/hf_cache (primary roster + pair-testing extras)
make model MODEL=Qwen/Qwen2.5-0.5B-Instruct   # ...or just one
```

Full lists: [Available datasets](#available-datasets-dataset) /
[Available models](#available-models-sharer--receiver) below.

### 2. Run experiments

**Sweep the whole grid** — every method (T2T, LCF-X, EIYW) × every model
pair from the primary roster × every dataset — with `make exp`, split
**method-major, then dataset-minor**. A single command runs everything (450
runs: 3 methods × 25 pairs × 6 datasets):

```bash
make exp     # every method, every dataset, every pair
```

Or `make sweep` to also run Step 1 first and spread the grid across GPUs
(reads `CUDA_VISIBLE_DEVICES` — one worker per GPU id, each running its
share of datasets one after another, workers in parallel):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 make sweep     # prepare, then sweep, 4-way parallel
```

Pick an axis on `exp` (`METHOD=` and `DATASET=` each: give one to run just
that, omit either to run all):

```bash
make exp METHOD=t2t                                             # t2t only, every dataset
make exp DATASET=hotpotqa_bridge                                # every method, one dataset
make exp METHOD=t2t DATASET=hotpotqa_bridge                     # t2t only, one dataset
make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_GPUS=1,2,3      # pin to specific GPUs
make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_ARGS="--dry-run"       # preview the cell list only
make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_ARGS="--rerun-failed"  # retry failed/oom cells
make exp METHOD=t2t DATASET=hotpotqa_bridge \
  EXP_ARGS="--sharer Qwen/Qwen3-8B --receiver Qwen/Qwen3-8B"    # narrow to one pair
```

Or run **one exact model pair** directly (skip the sweep grid entirely),
naming `DATASET=` plus a `SHARER=`/`RECEIVER=` pair explicitly:

```bash
make lcfx   DATASET=hotpotqa_bridge SHARER=Qwen/Qwen2.5-0.5B-Instruct RECEIVER=Qwen/Qwen3-0.6B  # train -> eval
make eiyw   DATASET=hotpotqa_bridge SHARER=Qwen/Qwen2.5-0.5B-Instruct RECEIVER=Qwen/Qwen3-0.6B  # ours: train -> eval
make t2t    DATASET=hotpotqa_bridge SHARER=Qwen/Qwen2.5-0.5B-Instruct RECEIVER=Qwen/Qwen3-0.6B  # baseline (no training)
make single DATASET=hotpotqa_bridge MODEL=Qwen/Qwen3-0.6B                                        # upper bound: one model, all evidence
```

`make single` follows the same "given=one, unset=all" pattern on both axes
(`MODEL=`/`DATASET=`): omit `MODEL=` to test every known-good model, omit
`DATASET=` to test every dataset, omit both to test the full model × dataset
grid.

Every such run is isolated automatically by **dataset + pair + pooling-span
count** — results go to `results/<method>/<dataset>/<sharer>/<receiver>/sp<SPANS>/`
and the checkpoint to `data/checkpoints/lcfx/<dataset>/<sharer>/<receiver>/
sp<SPANS>/fuser.pt` (model id `/` → `--`), so runs never overwrite each other,
including across a `SPANS=` sweep. `SHARER`/`RECEIVER` move every method at
once; the per-method `LCFX_*`/`EIYW_*`/`T2T_*` vars override a single method.
EIYW adds `SOFT_TOKENS` (K, the number of soft pseudo-tokens; default 64),
`STAGE1_NUM_HEADS` (independent pooling heads over each span — default 1,
the original single-query pooling; see [Run EIYW](#run-eiyw-our-receiver-anchored-method)),
and `STAGE2_NUM_HEADS`/`STAGE2_HEAD_DIM` (the soft-token cross-attention
width), and otherwise reuses the LCF-X training knobs. Its translator
checkpoint lands at `data/checkpoints/eiyw/<dataset>/<sharer>/<receiver>/
k<SOFT_TOKENS>/b<BOTTLENECK>/s1h<STAGE1_NUM_HEADS>/h<STAGE2_NUM_HEADS>/
d<STAGE2_HEAD_DIM>/sp<SPANS>/translator.pt`. `lcfx-train`/`lcfx-eval` and
`eiyw-train`/`eiyw-eval` also exist standalone if you want to separate
training from evaluation.

### Available datasets (`DATASET=`)

- `hotpotqa_bridge` — cross-context HotpotQA-bridge: Sharer and Receiver hold
  disjoint evidence, so neither can answer alone.
- `musique` — cross-context MuSiQue-Ans 2-hop: each question's two supporting
  paragraphs are split one per side (harder, less shortcut-able than HotpotQA).
- `qasc` — 8-way multiple-choice science QA; two-fact bridge split (fact1 |
  fact2) between Sharer and Receiver.
- `2wikimultihop` — cross-context 2WikiMultihopQA (compositional/bridge-chain
  questions): same 10-paragraph/2-gold-paragraph shape as `hotpotqa_bridge`,
  split the same way.
- `ropes` — cross-context ROPES: natively two-part (a general causal-rule
  `background` passage + a specific `situation` passage that applies it), one
  per side — no invented partitioning needed.
- `strategyqa` — cross-context StrategyQA: implicit multi-hop boolean (yes/no)
  reasoning; the question's supporting `facts` list is split in half between
  Sharer and Receiver.

**Size and typical scores:**

| Dataset | Train | Val | 1B oracle EM/F1 | 4B oracle EM/F1 | 8B oracle EM/F1 |
|---|---|---|---|---|---|
| `hotpotqa_bridge` | 72,763 | 5,899 | — | — | — |
| `musique` | 14,376 | 1,252 | 7.7 / 13.4 | — | — |
| `qasc` | 8,134 | 926 | 14.1 | 92.5 | — |
| `2wikimultihop` | 76,481 | 5,236 | 14.6 / 20.8 | — | — |
| `ropes` | 10,924 | 1,688 | 33.5 / 42.4 | — | — |
| `strategyqa` | 1,953 | 337 | 61.4 | — | — |

Train/Val are example counts. "Oracle" = **single-model upper bound** (`make
single`): one model answers from the Sharer's and Receiver's evidence
*combined*, no split — a rough per-size ceiling for the split-input methods
(T2T/LCF-X/EIYW), which must clear this bar to prove the communication itself
is adding value rather than just the model's own capability. Sizes: 1B =
`unsloth/Llama-3.2-1B-Instruct`, 4B = `unsloth/gemma-3-4b-it`, 8B =
`Qwen/Qwen3-8B`. `qasc`/`strategyqa` are classification-style and report a
single accuracy number instead of EM/F1. `—` = not yet run. This is a
point-in-time snapshot from the results on disk, not a fixed spec — regenerate
it as coverage grows:

```bash
make visual ITEM=datasets   # writes results/comparison/datasets.md
```

### Available models (`SHARER=` / `RECEIVER=`)

Any id below works as **either** the Sharer or the Receiver — mix families and
sizes freely (the LCF-X fuser bridges differing shapes; specs in
[Models](#models-sharer--receiver)). This is the **primary model roster** —
one model per size from 0.5B to 8B, different families — used by `make exp`/
`make sweep`/`make single` when no `MODEL=`/`SHARER=`/`RECEIVER=` is given.
All ids are **ungated** (no `HF_TOKEN` needed). Fetch one with
`make model MODEL=<id>` (or let the first run download it):

- `Qwen/Qwen2.5-0.5B-Instruct` — 0.5B; used by LCF-X
- `Qwen/Qwen3-0.6B` — 0.6B; used by LCF-X
- `unsloth/Llama-3.2-1B-Instruct` — 1B; ungated mirror of Llama-3.2-1B
- `unsloth/gemma-3-4b-it` — 4B; ungated mirror of Gemma-3-4B
- `Qwen/Qwen3-8B` — 8B tier

Pair-testing extras, available for manual `SHARER=`/`RECEIVER=` runs and
downloaded by `make model`, but **not** included in `make exp`/`make sweep`:

- `unsloth/gemma-3-1b-it` — 1B; ungated mirror of Gemma-3-1B
- `Qwen/Qwen3-4B` — 4B tier

Defaults follow the paper (per-device batch 16 × grad-accum 16 = effective batch
256), which needs a large GPU. On a smaller GPU, override to keep the effective
batch at 256, e.g. `make lcfx-train BATCH_SIZE=4 GRAD_ACCUM=64` (see
[Smaller GPUs](#smaller-gpus)).

## Prepare datasets (manual)

Cross-context **HotpotQA-bridge**: each bridge question's 10 paragraphs (2
supporting + 8 distractors) are split into two disjoint 5-paragraph partitions
(1 supporting + 4 distractors each) — Sharer gets one, Receiver the other, so
neither can answer alone. Downloads HotpotQA on first run (omit `HF_HUB_OFFLINE`).

```bash
# train split (all bridge examples ~70.7K) and full validation (~5,899)
python scripts/prepare_hotpotqa_bridge.py --split train --num-examples -1 \
  --output data/tasks/hotpotqa_bridge_train.jsonl
python scripts/prepare_hotpotqa_bridge.py --split validation --num-examples -1 \
  --output data/tasks/hotpotqa_bridge_val.jsonl
```

## Models (Sharer / Receiver)

The Sharer and Receiver are two independent models, set per run with
`--sharer-model` / `--receiver-model` (or `SHARER` / `RECEIVER` for the Makefile).
Use a **heterogeneous** pair: they may differ in family, size, hidden dim, head
dim, and depth — the LCF-X fuser's align stage bridges the differing shapes. Any
id works as either role.

Primary roster (mix freely as Sharer/Receiver) — all **ungated**, no `HF_TOKEN`
needed:

| Model | hidden | head_dim | layers | notes |
|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | 1024 | 128 | 28 | used by LCF-X |
| `Qwen/Qwen2.5-0.5B-Instruct` | 896 | 64 | 24 | used by LCF-X |
| `unsloth/Llama-3.2-1B-Instruct` | 2048 | 64 | 16 | **1B tier**; ungated mirror of Llama-3.2-1B |
| `unsloth/gemma-3-4b-it` | 2560 | 256 | 34 | **4B tier**; ungated mirror of Gemma-3-4B |
| `Qwen/Qwen3-8B` | 4096 | 128 | 36 | **8B tier** |

Pair-testing extras (manual `SHARER=` / `RECEIVER=` use; not part of the primary roster): `unsloth/gemma-3-1b-it`, `Qwen/Qwen3-4B`.

Trained fuser checkpoints are **stamped** with the Sharer/Receiver they were
trained for, and `run_lcfx.py` hard-errors if you load one against a different
pair (override with `--allow-model-mismatch`). Keep a distinct checkpoint per
pair — e.g. `make lcfx-hetero` runs Llama→Qwen and writes `fuser_hetero.pt`.

```bash
# heterogeneous example: Llama Sharer -> Qwen Receiver (ungated, no HF_TOKEN)
make lcfx-hetero
```

## Run T2T (baseline)

The simplest baseline: the Sharer writes a natural-language message describing
its own context (a capped `generate()` call), and the Receiver reads that
message plus its own context and answers (another `generate()` call). No
fusion, no training — just two independent forward generations per example.

```bash
make t2t DATASET=hotpotqa_bridge SHARER=Qwen/Qwen2.5-0.5B-Instruct RECEIVER=Qwen/Qwen3-0.6B

# or the script directly
python scripts/run_text_to_text.py --dataset data/tasks/hotpotqa_bridge_val.jsonl \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --sharer-max-new-tokens 100 --receiver-max-new-tokens 64 \
  --output-dir results/t2t
```

### All T2T options

| Makefile var | Flag | Default | Meaning |
|---|---|---|---|
| `DATASET` | `--dataset` (via `$(VAL)`) | `hotpotqa_bridge` | Dataset name (see [Available datasets](#available-datasets-dataset)). |
| `T2T_SHARER` | `--sharer-model` | `$(SHARER)` = `Qwen/Qwen3-0.6B` | Sharer model id. |
| `T2T_RECEIVER` | `--receiver-model` | `$(RECEIVER)` = `Qwen/Qwen3-0.6B` | Receiver model id. |
| `SHARER_MAX` | `--sharer-max-new-tokens` | `100` | Cap on the Sharer's generated message (the paper's own T2T default was 200). |
| `RECV_MAX` | `--receiver-max-new-tokens` | `64` | Cap on the Receiver's generated answer (matches the LCF-X/EIYW eval budget). |
| `EVAL_N` | `--num-examples` | (empty = full val split) | Cap eval examples (a quick partial run). |
| `T2T_OUT` | `--output-dir` | `results/t2t/<dataset>/<sharer>/<receiver>` | Output directory (auto dataset+pair-scoped). |

T2T has no training, so it has no `SEED`/`LR`/`EPOCHS`/checkpoint knobs — only
the eval-time flags above.

## Run LCF-X (the method)

The scenario: Sharer and Receiver hold **different** context (disjoint evidence),
the Sharer's KV cache is pooled → projected → **fused into the Receiver's cache**,
and the Receiver answers. The two models are set independently (`--sharer-model` /
`--receiver-model`) and may differ; the fuser's align stage bridges different dims.

```bash
DATA=data/tasks/hotpotqa_bridge_val.jsonl
python scripts/run_lcfx.py --dataset $DATA --num-spans 4 \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --fuser-checkpoint data/checkpoints/lcfx/fuser.pt --output-dir results/lcfx
```

> Needs a trained `--fuser-checkpoint` (see below). With **no** checkpoint the
> fuser is zero-initialized and adds nothing — output then equals the Receiver
> alone, which is only a plumbing sanity check, not the method.

**Optional baselines for context** (not part of the LCF-X pipeline): T2T
(`scripts/run_text_to_text.py`) is the text-message comparison; `--no-fusion`
(or `scripts/run_receiver_baseline.py`) is the Receiver-only floor.

Every run writes the same four files to its `--output-dir`
(`predictions.jsonl`, `metrics.json`, `latency.json`, `run_metadata.json`), so
methods are directly comparable. Headline numbers are in `metrics.json`.

## Train the LCF-X fuser

Frozen Sharer/Receiver; only the fuser learns (teacher-forced next-token CE on
the gold answer, with the Sharer summary fused into the Receiver's prompt cache).
Optimization follows the paper: AdamW, lr 1e-4, 200-step warmup, latent 128, 2.8%
dev holdout, per-device batch 16 × grad-accum 16 = **effective batch 256**, 1
epoch (~276 steps), EM/F1. Always name the Sharer/Receiver pair explicitly.

```bash
# Needs a large GPU (~80GB); on smaller GPUs see "Smaller GPUs".
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_lcfx_fuser.py \
  --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --eval-data  data/tasks/hotpotqa_bridge_val.jsonl \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --checkpoint-out data/checkpoints/lcfx/fuser.pt
# (dim/batch/warmup/etc. default to the paper values above)
```

### Smaller GPUs

The memory bottleneck is the receiver-aware projector: it is **position-wise over
~750 prompt tokens × 28 layers**, so the *per-device* `--batch-size` dominates
memory. The fix is to **lower `--batch-size` and raise `--grad-accum`
proportionally**, keeping the effective batch at 256 — the optimization is then
**identical to the paper**, only slower (less parallelism):

| GPU memory (free) | `--batch-size` | `--grad-accum` | effective batch |
|---|---|---|---|
| ~80 GB (paper) | 16 | 16 | 256 |
| ~24 GB | 4 | 64 | 256 |
| ~16 GB | 2 | 128 | 256 |
| ~12 GB | 1 | 256 | 256 |

```bash
make lcfx-train BATCH_SIZE=4 GRAD_ACCUM=64    # 24GB GPU, still effective batch 256
```

Other memory/time knobs:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduces fragmentation (the
  Makefile sets this automatically).
- `--eval-max-examples N` (Makefile `EVAL_MAX`) — cap the in-training dev eval
  (monitoring only; doesn't change the trained model).
- `--bottleneck-dim D` (Makefile `BOTTLENECK`) — smaller latent dim uses less
  memory, but **changes capacity** (deviates from the paper's 128).
- `--num-spans P` (Makefile `SPANS`) — fewer Sharer pooling spans = fewer tokens
  for the resampler.
- `--stage1-num-heads H1` (Makefile `STAGE1_NUM_HEADS`, EIYW only) — more
  independent stage-1 pooling heads = a wider span-token feature dim feeding
  stage 2 (more trainable params, more memory).

### Quick partial runs (limit data for verification)

Limit how much data each stage uses — for a fast end-to-end sanity check
(independent of the memory knobs above):

| Stage | Flag (Makefile var) | Effect |
|---|---|---|
| Train | `--max-train-examples N` (`MAX_TRAIN`) | Use only the first `N` training examples. |
| Train | `--max-steps N` (`MAX_STEPS`) | Stop after `N` optimizer steps (any dataset size). |
| Dev (in-training eval) | `--eval-max-examples N` (`EVAL_MAX`) | Cap the dev eval to `N` examples (monitoring only). |
| Test (final eval) | `--num-examples N` (`EVAL_N`) | Eval on only `N` examples in `lcfx-eval` / `run_lcfx.py`. |

One-shot verification (tiny train + dev + test, end to end in minutes):

```bash
P="SHARER=Qwen/Qwen2.5-0.5B-Instruct RECEIVER=Qwen/Qwen3-0.6B"
# train 5 steps on 200 examples, dev-eval on 20
make lcfx-train $P MAX_TRAIN=200 MAX_STEPS=5 EVAL_MAX=20 BATCH_SIZE=4 GRAD_ACCUM=2
# evaluate on 50 test examples (+ the no-fusion floor)
make lcfx-eval  $P EVAL_N=50
```

Same with the scripts directly:

```bash
python scripts/train_lcfx_fuser.py \
  --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --max-train-examples 200 --val-frac 0.1 --eval-max-examples 20 \
  --batch-size 4 --grad-accum 2 --max-steps 5 \
  --checkpoint-out data/checkpoints/lcfx/fuser_smoke.pt
python scripts/run_lcfx.py \
  --dataset data/tasks/hotpotqa_bridge_val.jsonl --num-examples 50 \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --fuser-checkpoint data/checkpoints/lcfx/fuser_smoke.pt --output-dir results/lcfx_smoke
```

> Note: a full epoch on a single 24 GB GPU is compute-bound (~hours); the paper's
> ~1 h used a much faster GPU. Lowering `--batch-size` while raising `--grad-accum`
> keeps the *effective* batch (256) and optimization identical to the paper.

### All LCF-X options

| Makefile var | Flag | Default | Meaning |
|---|---|---|---|
| `DATASET` | `--dataset` (via `$(VAL)`/`$(TRAIN)`) | `hotpotqa_bridge` | Dataset name (see [Available datasets](#available-datasets-dataset)). |
| `LCFX_SHARER` | `--sharer-model` | `$(SHARER)` = `Qwen/Qwen3-0.6B` | Sharer model id. |
| `LCFX_RECEIVER` | `--receiver-model` | `$(RECEIVER)` = `Qwen/Qwen3-0.6B` | Receiver model id. |
| `SPANS` | `--num-spans` | `5` | Sharer pooling spans (position-free summary granularity). |
| `BOTTLENECK` | `--bottleneck-dim` | `128` | Fuser latent width. |
| `LR` | `--lr` | `1e-4` | Learning rate. |
| `EPOCHS` | `--epochs` | `1` | Training epochs. |
| `BATCH_SIZE` | `--batch-size` | `16` | Per-device micro-batch. |
| `GRAD_ACCUM` | `--grad-accum` | `16` | Micro-batches per optimizer step (effective batch = `BATCH_SIZE`×`GRAD_ACCUM` = 256). |
| `WARMUP` | `--warmup` | `200` | Linear LR warmup steps. |
| `VAL_FRAC` | `--val-frac` | `0.028` | Dev holdout fraction of train data (when no `--eval-data`). |
| `LOG_EVERY` | `--log-every` | `10` | Steps between training log lines. |
| `EVAL_EVERY` | `--eval-every` | `50` | Steps between in-training dev evals. |
| `EVAL_MAX` | `--eval-max-examples` | `500` | Cap on in-training dev eval examples (monitoring only). |
| `SAVE_EVERY` | `--save-every` | `50` | Steps between checkpoint saves. |
| `EVAL_EM` | `--eval-em-examples` | `0` (off) | Generation-based EM/F1 on N dev examples per eval (slower; the teacher-forced loss alone hides stop-token issues). |
| `MAX_TRAIN` | `--max-train-examples` | (empty = all) | Cap training examples (partial run). |
| `MAX_STEPS` | `--max-steps` | (empty = full epoch(s)) | Cap optimizer steps. |
| `EVAL_N` | `--num-examples` | (empty = full val split) | Cap final-eval examples in `lcfx-eval`. |
| `SEED` | `--seed` | `0` | Random seed. |
| `CKPT` | `--checkpoint-out` / `--fuser-checkpoint` | `data/checkpoints/lcfx/<dataset>/<sharer>/<receiver>/sp<SPANS>/fuser.pt` | Checkpoint path (train writes it, eval reads it). |

## Run EIYW (our receiver-anchored method)

The Sharer encodes its own context; its KV cache is translated into **K soft
pseudo-tokens, conditioned on the Receiver's question**, which are **prepended to
the Receiver's input embeddings**. The Receiver answers from its own context plus
those soft tokens — the Sharer's information is added at the Receiver's *input*
(not broadcast into its KV cache as in LCF-X). Only the translator is trained;
both base models stay frozen.

```bash
# train -> eval end to end (model-stamped checkpoint per dataset+pair)
make eiyw DATASET=hotpotqa_bridge SHARER=Qwen/Qwen2.5-0.5B-Instruct RECEIVER=Qwen/Qwen3-0.6B

# or the scripts directly
python scripts/train_eiyw.py --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --num-soft-tokens 64 --num-spans 5 --checkpoint-out data/checkpoints/eiyw/translator.pt
python scripts/run_eiyw.py --dataset data/tasks/hotpotqa_bridge_val.jsonl \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --num-soft-tokens 64 --translator-checkpoint data/checkpoints/eiyw/translator.pt \
  --output-dir results/eiyw
```

> `--no-fusion` runs the Receiver alone (the comparison floor / plumbing check).
> EIYW's prompt forward is grad-enabled (soft tokens lead the sequence), so it
> uses more memory than LCF-X — lower `--batch-size` and raise `--grad-accum` to
> keep the effective batch at 256. See
> [prompt/EIYW_IMPLEMENTATION_PLAN.md](prompt/EIYW_IMPLEMENTATION_PLAN.md).

The translator pools the Sharer cache in two stages (see
[methods/eiyw/translator.py](methods/eiyw/translator.py)): stage 1 compresses
each Sharer layer's tokens down to `--num-spans` (`SPANS`) span-tokens, then
stage 2 runs `--stage2-num-heads`/`--stage2-head-dim` cross-attention from the
`K` soft-token queries over those span-tokens. `--stage1-num-heads`
(`STAGE1_NUM_HEADS`, default 1) widens stage 1 itself: each head learns an
independent pooling query over a span's raw tokens (no value projection, so no
extra information is thrown away — only *how many independent views* of a
span survive into stage 2), which is often the tighter bottleneck since a
single shared query has to pool every Sharer layer identically. Sweep it the
same way as the stage-2 knobs:

```bash
make eiyw DATASET=hotpotqa_bridge STAGE1_NUM_HEADS=4 SPANS=10
```

### All EIYW options

| Makefile var | Flag | Default | Meaning |
|---|---|---|---|
| `DATASET` | `--dataset` (via `$(VAL)`/`$(TRAIN)`) | `hotpotqa_bridge` | Dataset name (see [Available datasets](#available-datasets-dataset)). |
| `EIYW_SHARER` | `--sharer-model` | `$(SHARER)` = `Qwen/Qwen3-0.6B` | Sharer model id. |
| `EIYW_RECEIVER` | `--receiver-model` | `$(RECEIVER)` = `Qwen/Qwen3-0.6B` | Receiver model id. |
| `SOFT_TOKENS` | `--num-soft-tokens` | `64` | K, soft pseudo-tokens prepended to the Receiver input. |
| `BOTTLENECK` | `--bottleneck-dim` | `128` | Translator latent width. |
| `STAGE1_NUM_HEADS` | `--stage1-num-heads` | `1` | Independent stage-1 span-pooling heads (`1` = original single-query pooling). |
| `STAGE2_NUM_HEADS` | `--stage2-num-heads` | `8` | Multi-head count for the stage-2 soft-token cross-attention. |
| `STAGE2_HEAD_DIM` | `--stage2-head-dim` | `256` | Per-head width for the stage-2 cross-attention. |
| `SPANS` | `--num-spans` | `5` | Sharer pooling spans. |
| `LR` | `--lr` | `1e-4` | Learning rate. |
| `EPOCHS` | `--epochs` | `1` | Training epochs. |
| `BATCH_SIZE` | `--batch-size` | `16` | Per-device micro-batch (EIYW's forward is grad-enabled end to end, so it needs more memory than LCF-X at the same batch size — lower this before raising `GRAD_ACCUM`). |
| `GRAD_ACCUM` | `--grad-accum` | `16` | Micro-batches per optimizer step (effective batch = `BATCH_SIZE`×`GRAD_ACCUM` = 256). |
| `WARMUP` | `--warmup` | `200` | Linear LR warmup steps. |
| `VAL_FRAC` | `--val-frac` | `0.028` | Dev holdout fraction of train data (when no `--eval-data`). |
| `LOG_EVERY` | `--log-every` | `10` | Steps between training log lines. |
| `EVAL_EVERY` | `--eval-every` | `50` | Steps between in-training dev evals. |
| `EVAL_MAX` | `--eval-max-examples` | `500` | Cap on in-training dev eval examples (monitoring only). |
| `SAVE_EVERY` | `--save-every` | `50` | Steps between checkpoint saves. |
| `EVAL_EM` | `--eval-em-examples` | `0` (off) | Generation-based EM/F1 on N dev examples per eval (slower; the teacher-forced loss alone hides stop-token issues). |
| `MAX_TRAIN` | `--max-train-examples` | (empty = all) | Cap training examples (partial run). |
| `MAX_STEPS` | `--max-steps` | (empty = full epoch(s)) | Cap optimizer steps. |
| `EVAL_N` | `--num-examples` | (empty = full val split) | Cap final-eval examples in `eiyw-eval`. |
| `SEED` | `--seed` | `0` | Random seed. |
| `EIYW_CKPT` | `--checkpoint-out` / `--translator-checkpoint` | `data/checkpoints/eiyw/<dataset>/<sharer>/<receiver>/k<SOFT_TOKENS>/b<BOTTLENECK>/s1h<STAGE1_NUM_HEADS>/h<STAGE2_NUM_HEADS>/d<STAGE2_HEAD_DIM>/sp<SPANS>/translator.pt` | Checkpoint path (train writes it, eval reads it). |

## Visual comparison table

Once you've got some results on disk, render a Markdown comparison table with
`make visual`, scoped with `DATASET=` and `ITEM=`:

```bash
make visual DATASET=hotpotqa_bridge                  # EM+F1 grid (default ITEM=scores)
make visual DATASET=hotpotqa_bridge ITEM=scores       # same, explicit
make visual DATASET=hotpotqa_bridge ITEM=params       # trainable-parameter grid (lcfx/eiyw only)
make visual DATASET=hotpotqa_bridge ITEM=latency      # average per-example latency grid
make visual ITEM=datasets                              # dataset overview (DATASET= ignored)
```

`ITEM=` picks what gets rendered:

- `scores` (default) — Sharer × Receiver grid (primary roster, 5×5) for one
  `DATASET=`, with LCF-X / EIYW / T2T EM+F1 per cell (best EM and best F1 in
  each cell bolded). Read from `results/<method>/<dataset>/<sharer>/<receiver>/metrics.json`.
- `params` — the same Sharer × Receiver grid, but trainable-parameter counts
  for LCF-X / EIYW (T2T trains nothing, so it's omitted). Read from the
  `training_log.jsonl` under each checkpoint dir.
- `latency` — the same Sharer × Receiver grid, but LCF-X / EIYW / T2T average
  per-example total latency (fastest of each cell bolded). Read from the
  `average.total_ms` field of `results/<method>/<dataset>/<sharer>/<receiver>/latency.json`.
- `datasets` — one overview table across *every* dataset (`DATASET=` is
  ignored): train/val split sizes plus the single-model-oracle EM+F1 for a
  representative 1B/4B/8B model (`make single`'s results).

Any method/checkpoint/result that hasn't been run yet is left blank rather
than erroring, so you can call `make visual` at any point mid-sweep. Tables
are written to `results/comparison/<dataset>[_params|_latency].md` (or
`results/comparison/datasets.md` for `ITEM=datasets`) — override the path
with `OUT=`:

```bash
make visual DATASET=ropes OUT=results/comparison/ropes_custom.md
```

## Layout

```text
methods/      # one package per method (t2t, lcfx, eiyw — all implemented)
scripts/      # runnable entrypoints (prepare / run / train) + cache config
data/tasks/   # prepared task JSONL (reused across runs)
results/      # run outputs (one dir per run)
Makefile      # data / lcfx{-train,-eval} / eiyw{-train,-eval} / t2t / visual one-liners
```
