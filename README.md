# Echo In Your Words

Research codebase for **flexible, cross-context communication between language
models** for multi-agent collaboration: a *Sharer* and *Receiver* hold different
evidence (and may differ in size/architecture), and we study how the Sharer can
help the Receiver answer. The main experiment sweep compares three methods
under one standardized output contract: **LCF-X** (position-free latent cache
communication), **XKV** (pooled Sharer and Receiver caches translated into
Receiver KV updates), and **T2T** (text message). EIYW and KVTok remain in the
repository for legacy/manual runs but are excluded from experiment sweeps.

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
make prepare     # = make data + make model: every dataset + 0.6/1/3/8B experiment roster
```

Or pick just the axis you need:

```bash
make data                            # every dataset: hotpotqa_bridge, musique, qasc, 2wikimultihop, ropes, strategyqa
make data  DATASET=hotpotqa_bridge   # ...or just one: train (~70.7K) + val (5,899)

make model                                    # fetch the 0.6/1/3/8B experiment roster into data/hf_cache
make model MODEL=unsloth/Llama-3.2-3B-Instruct   # ...or just one
make model MODEL=mistralai/Ministral-8B-Instruct-2410 # ...or fetch only the 8B model
```

Model downloads retry transient Hugging Face timeouts, rate limits, and 5xx
responses three times with exponential backoff. If Hugging Face remains
unavailable, Ministral automatically falls back to the Transformers-format
`AI-ModelScope/Ministral-8B-Instruct-2410` mirror and installs it into the same
project Hugging Face cache. Override retry behavior with
`HF_MODEL_DOWNLOAD_ATTEMPTS` and `HF_MODEL_DOWNLOAD_RETRY_DELAY` if needed.

Full lists: [Available datasets](#available-datasets-dataset) /
[Available models](#available-models-sharer--receiver) below.

### 2. Run experiments

**Sweep the whole grid** — every method (LCF-X, XKV, T2T) × every model
pair from the primary roster × every dataset — with `make exp`, split
**method-major, then dataset-minor**. A single command runs everything (162
runs: 3 methods × 9 pairs × 6 datasets). `make exp-extend` runs only the seven
new pairs touching the 8B model; `make exp-all` runs the complete 16-pair grid:

```bash
make exp     # every method, every dataset, every pair
make exp-extend  # only 0.6/1/3B <-> 8B plus 8B -> 8B (7 pairs)
make exp-all     # full 0.6/1/3/8B x 0.6/1/3/8B grid (16 pairs)
```

Or `make sweep` to also run Step 1 first, select the full 4×4 model grid, and spread it across GPUs
(reads `CUDA_VISIBLE_DEVICES` — one worker per GPU id, each running its
share of datasets one after another, workers in parallel):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 make sweep     # prepare, then all 288 cells, 4-way parallel
```

Pick axes on `exp`. `METHOD=` and `DATASET=` select one method/dataset or all
when omitted. An explicit `SHARER=` fixes a row and an explicit `RECEIVER=`
fixes a column; setting both selects their single cell. The fixed model must
belong to the selected 3×3 (`exp`) or 4×4 (`exp-all`) roster:

```bash
make exp METHOD=t2t                                             # t2t only, every dataset
make exp DATASET=hotpotqa_bridge                                # every method, one dataset
make exp METHOD=t2t DATASET=hotpotqa_bridge                     # t2t only, one dataset
make exp METHOD=xkv DATASET=hotpotqa_bridge                     # XKV only, one dataset
make exp METHOD=xkv DATASET=hotpotqa_bridge SHARER=unsloth/gemma-3-1b-it # Gemma row: 3 receivers
make exp METHOD=xkv DATASET=hotpotqa_bridge RECEIVER=Qwen/Qwen3-0.6B     # Qwen column: 3 sharers
make exp-extend METHOD=xkv DATASET=ropes                        # only the 7 new XKV cells
make exp-all METHOD=xkv DATASET=ropes                           # all 16 XKV cells
make exp-all METHOD=xkv DATASET=ropes SHARER=unsloth/gemma-3-1b-it      # Gemma row: 4 receivers
make exp-all METHOD=xkv DATASET=ropes RECEIVER=mistralai/Ministral-8B-Instruct-2410 # Mistral column: 4 sharers
CUDA_VISIBLE_DEVICES=0,1,2,3 make exp METHOD=xkv DATASET=ropes   # one cell worker per visible GPU
make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_GPUS=1,2,3      # pin to specific GPUs
make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_ARGS="--dry-run"       # preview the cell list only
make exp METHOD=t2t DATASET=hotpotqa_bridge EXP_ARGS="--resume"  # continue a crashed sweep (skip cells already ok)
make exp-all METHOD=t2t DATASET=hotpotqa_bridge \
  SHARER=mistralai/Ministral-8B-Instruct-2410 RECEIVER=mistralai/Ministral-8B-Instruct-2410 # one pair
```

Repeat exactly the same selected grid with `make repeat`. `RUNS` accepts an
inclusive range or a list; `RANGE` is an equivalent alias:

```bash
make repeat RUNS=1:5 DATASET=ropes
make repeat RUNS=1,3,5 METHOD=xkv DATASET=ropes
make repeat RANGE='[1,2,3]' REPEAT_TARGET=exp-all DATASET=ropes
```

Each replicate gets a fresh random seed unless `SEED=` is explicitly supplied.
The resolved seed is propagated to every cell in that replicate, stored in its
manifest/checkpoints and `run_metadata.json`, and printed at launch. Dataset
construction uses the separate fixed `DATA_SEED=0` default, so changing the
training seed never changes the examples. Replicate artifacts are isolated at
`results/runs/run-<id>/...` and
`data/checkpoints/runs/run-<id>/...`. Run one manually with `RUN=`, for example
`make exp RUN=7 DATASET=ropes`; `N=7` is an alias for this originally proposed
notation. Omitting both preserves the legacy paths.

The historical unscoped tree is treated as virtual **run 0**. No migration or
copy is required: `RUN=0`/`N=0` reads and writes the existing `results/<method>`
and `data/checkpoints/<method>` locations. Use `RUNS=0:5` in visualization to
include the current result together with runs 1–5. If run 0 appears in
`make repeat RUNS=...`, it is reused and skipped; invoke `make exp RUN=0`
explicitly only when you intend to overwrite the legacy result.

For trained methods, the sweep forwards both roots explicitly to every nested
cell. Checkpoints therefore remain isolated even across recursive/parallel
sweeps, for example
`data/checkpoints/runs/run-2/xkv/<dataset>/<sharer>/<receiver>/<hp>/translator.pt`.

Sweep cells that run out of GPU memory automatically retry with successively
smaller per-device batches down to batch 1, increasing gradient accumulation
to preserve the effective batch size. CUDA/CUBLAS allocation failures are
treated as OOMs. Non-memory failures remain recorded for an explicit
`EXP_ARGS="--resume"` run.

Or run **one exact model pair** directly (skip the sweep grid entirely),
naming `DATASET=` plus a `SHARER=`/`RECEIVER=` pair explicitly:

```bash
make lcfx   DATASET=hotpotqa_bridge SHARER=unsloth/Llama-3.2-3B-Instruct RECEIVER=Qwen/Qwen3-0.6B  # train -> eval
make xkv    DATASET=hotpotqa_bridge SHARER=unsloth/Llama-3.2-3B-Instruct RECEIVER=Qwen/Qwen3-0.6B  # dual-cache KV update
make t2t    DATASET=hotpotqa_bridge SHARER=unsloth/Llama-3.2-3B-Instruct RECEIVER=Qwen/Qwen3-0.6B  # baseline (no training)
make single DATASET=hotpotqa_bridge MODEL=Qwen/Qwen3-0.6B                                        # one Receiver, half + full context
```

`make single` follows the same "given=one, unset=all" pattern on both axes
(`MODEL=`/`DATASET=`): omit `MODEL=` to test every primary-roster model, omit
`DATASET=` to test every dataset, omit both to test the full model × dataset
grid.

`make single` defaults to `SINGLE_CONTEXT=both`: it evaluates the original
Receiver evidence (`half`) and both evidence portions combined (`full`), with
the **same Receiver prompt, chat template and greedy cache decoder** used by
XKV/LCF-X. It performs no Sharer forward pass or translation. Each model is
loaded once per dataset and reused across the two conditions. Full context is
a reference, not a guaranteed upper bound.

For the six ROPES baselines (Qwen3-0.6B, Gemma-3-1B and Llama-3.2-3B × two
contexts), on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 make single DATASET=ropes SINGLE_CONTEXT=both RUN=7 SEED=2131195937
# Select one model/condition instead:
CUDA_VISIBLE_DEVICES=0 make single DATASET=ropes MODEL=unsloth/gemma-3-1b-it SINGLE_CONTEXT=half RUN=7
```

Results are separate under
`results/runs/run-7/single/ropes/<model>/half/` and `.../full/`, including
per-example predictions, progress logs, metrics, timing and context-mode
metadata. Without `RUN=`, the root is `results/single/`. `SINGLE_OUT=` overrides
the model/dataset root; the runner always appends the selected condition.
`EVAL_N=` limits each condition for a smoke check. Invalid/empty answers count
as incorrect. Old flat full-context results retain their historical dedicated
prompt; they are separate from these new matched-prompt evaluations.

Every run is isolated automatically by **dataset + pair + method-specific
architecture settings**. For example, LCF-X results go to
`results/lcfx/<dataset>/<sharer>/<receiver>/sp<SPANS>/` and its checkpoint to
`data/checkpoints/lcfx/<dataset>/<sharer>/<receiver>/sp<SPANS>/fuser.pt`
(model id `/` → `--`), so architecture sweeps never overwrite each other.
`SHARER`/`RECEIVER` move every method at
once; the per-method `LCFX_*`/`EIYW_*`/`KVTOK_*`/`XKV_*`/`T2T_*` vars override a single method.
EIYW reuses LCF-X's pooling design (and hence its `SPANS` knob) on the Sharer —
there's no separate `EIYW_SPANS`. Its own shape knobs are `EIYW_DMODEL`
(translator width, default `512`, receiver-independent), `EIYW_QPL` (max pool
queries per layer k; soft tokens = k·L_s), `EIYW_ADAPT` (w>0 scales the
active k with the Sharer's context length), and `EIYW_BLOCKS` (fusion depth;
see [Run EIYW](#run-eiyw-our-method)). Its translator checkpoint lands at
`data/checkpoints/eiyw/<dataset>/<sharer>/<receiver>/sp<SPANS>_dm<EIYW_DMODEL>_k<EIYW_QPL>/translator.pt`.
`lcfx-train`/`lcfx-eval`, `eiyw-train`/`eiyw-eval`, and
`kvtok-train`/`kvtok-eval` and `xkv-train`/`xkv-eval` also exist standalone if
you want to separate training from evaluation.

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
(T2T/LCF-X/XKV), which must clear this bar to prove the communication itself
is adding value rather than just the model's own capability. Sizes: 1B =
`unsloth/gemma-3-1b-it`, 3B = `unsloth/Llama-3.2-3B-Instruct`, 8B =
`mistralai/Ministral-8B-Instruct-2410`. `qasc`/`strategyqa` are classification-style and report a
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
one model per size from 0.6B to 3B, different families — used by `make exp`
and `make single` when no `MODEL=`/`SHARER=`/`RECEIVER=` is given. `make sweep`
adds the 8B large tier for the complete grid.
All ids are **ungated** (no `HF_TOKEN` needed). Fetch one with
`make model MODEL=<id>` (or let the first run download it):

- `Qwen/Qwen3-0.6B` — 0.6B; used by LCF-X
- `unsloth/gemma-3-1b-it` — 1B; ungated mirror of Gemma-3-1B
- `unsloth/Llama-3.2-3B-Instruct` — 3B; ungated mirror of Llama-3.2-3B

Large tier — `make exp-extend` runs only its seven new pairings, `make exp-all`
runs the complete 4×4 experiment grid, and `make visual-all` renders the 4×4
reports. It remains opt-in for `make single` and the unchanged `make visual`
command through `EXTRA_MODELS=TRUE`; bare `make sweep` includes it because
that target means the complete experiment matrix:

- `mistralai/Ministral-8B-Instruct-2410` — 8.02B; interleaved full/32K sliding attention; Mistral Research License

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
needed. Note both Qwen3 and Gemma-3 decouple `head_dim` from
`hidden/attn heads`:

| Model | hidden | head_dim | layers | attn heads | KV heads | notes |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | 1024 | 128 | 28 | 16 | 8 | used by LCF-X |
| `unsloth/gemma-3-1b-it` | 1152 | 256 | 26 | 4 | 1 | **1B tier**; ungated mirror of Gemma-3-1B; MQA |
| `unsloth/Llama-3.2-3B-Instruct` | 3072 | 128 | 28 | 24 | 8 | **3B tier**; ungated mirror of Llama-3.2-3B |
| `mistralai/Ministral-8B-Instruct-2410` | 4096 | 128 | 36 | 32 | 8 | **8B tier**; interleaved full/32K sliding attention; MRL; used by `exp-extend`/`exp-all` |

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
make t2t DATASET=hotpotqa_bridge SHARER=unsloth/Llama-3.2-3B-Instruct RECEIVER=Qwen/Qwen3-0.6B

# or the script directly
python scripts/run_text_to_text.py --dataset data/tasks/hotpotqa_bridge_val.jsonl \
  --sharer-model unsloth/Llama-3.2-3B-Instruct --receiver-model Qwen/Qwen3-0.6B \
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
  --sharer-model unsloth/Llama-3.2-3B-Instruct --receiver-model Qwen/Qwen3-0.6B \
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
Optimization follows the paper's LCF-X recipe: AdamW, lr 1e-4, latent 128,
dropout 0.1, 2.8% dev holdout, per-device batch 16 × grad-accum 16 =
**effective batch 256**, one epoch, Kaiming-scaled pool/up-projector
initialization, +1 K/V layer-gate logits, and gate-temperature annealing from
1.0 to 0.5. The intended warmup policy here is 10% of the exact resolved
optimizer-step budget (after the split, caps, batching, accumulation, and
epochs), rather than the paper's fixed 200 steps. Always name the
Sharer/Receiver pair explicitly.

The projector follows the paper's explicit latent equation
`z -> (z_K, z_V)`, so each K/V up-projection consumes `d/2`. The paper's
693K-per-layer / 25.87M parameter tables instead imply full-`d` inputs and are
internally inconsistent with that equation; this implementation follows the
equation. For heterogeneous pairs, the only architecture extension is a
linear Sharer-layer-to-Receiver-layer map applied after Sharer pooling.

```bash
# Needs a large GPU (~80GB); on smaller GPUs see "Smaller GPUs".
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_lcfx_fuser.py \
  --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --eval-data  data/tasks/hotpotqa_bridge_val.jsonl \
  --sharer-model unsloth/Llama-3.2-3B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --checkpoint-out data/checkpoints/lcfx/fuser.pt
# (warmup defaults to 10% of the resolved optimizer steps)
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
- `--num-spans P` (Makefile `SPANS`) — span count for `--span-scheme fixed`
  (LCF-X defaults to `--span-scheme paragraphs`, the paper's natural-block
  scheme, where `P` follows the text). EIYW/KVTok still use `SPANS` chunks.

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
P="SHARER=unsloth/Llama-3.2-3B-Instruct RECEIVER=Qwen/Qwen3-0.6B"
# train 5 steps on 200 examples, dev-eval on 20
make lcfx-train $P MAX_TRAIN=200 MAX_STEPS=5 EVAL_MAX=20 BATCH_SIZE=4 GRAD_ACCUM=2
# evaluate on 50 test examples (+ the no-fusion floor)
make lcfx-eval  $P EVAL_N=50
```

Same with the scripts directly:

```bash
python scripts/train_lcfx_fuser.py \
  --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --sharer-model unsloth/Llama-3.2-3B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --max-train-examples 200 --val-frac 0.1 --eval-max-examples 20 \
  --batch-size 4 --grad-accum 2 --max-steps 5 \
  --checkpoint-out data/checkpoints/lcfx/fuser_smoke.pt
python scripts/run_lcfx.py \
  --dataset data/tasks/hotpotqa_bridge_val.jsonl --num-examples 50 \
  --sharer-model unsloth/Llama-3.2-3B-Instruct --receiver-model Qwen/Qwen3-0.6B \
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
| `SPAN_SCHEME` | `--span-scheme` | `paragraphs` | Span partition: `paragraphs` = raw evidence paragraphs (prompt/question blocks do not add spans), `fixed` = `SPANS` near-equal token chunks. The pool has no `P`-dependent parameters, so a trained checkpoint accepts either (paper E.4). |
| `SPANS` | `--num-spans` | `5` | Span count for `--span-scheme fixed`. |
| `BOTTLENECK` | `--bottleneck-dim` | `128` | Fuser latent width. |
| `LCFX_DROPOUT` | `--dropout` | `0.1` | Dropout in the two-layer bottleneck MLP; stamped into the checkpoint architecture. |
| `LR` | `--lr` | `1e-4` | Learning rate. |
| `EPOCHS` | `--epochs` | `1` | Training epochs. |
| `BATCH_SIZE` | `--batch-size` | `16` (`128` on qasc) | Per-device micro-batch (qasc's short contexts fit 8x more per device; `GRAD_ACCUM` shrinks 8x to match). |
| `GRAD_ACCUM` | `--grad-accum` | `16` | Micro-batches per optimizer step (effective batch = `BATCH_SIZE`×`GRAD_ACCUM` = 256). |
| `WARMUP` | `--warmup` | 10% of resolved optimizer steps | Computed after the train/dev split, `MAX_TRAIN`, batch/accumulation, epochs, and `MAX_STEPS`. Set `WARMUP=...` only to override it explicitly. |
| `VAL_FRAC` | `--val-frac` | `0.028` | Dev holdout fraction of train data (when no `--eval-data`). |
| `LOG_EVERY` | `--log-every` | `10` | Steps between training log lines. |
| `EVAL_EVERY` | `--eval-every` | `50` | Steps between in-training dev evals. |
| `EVAL_MAX` | `--eval-max-examples` | `500` | Cap on in-training dev eval examples (monitoring only). |
| `SAVE_EVERY` | `--save-every` | `50` | Steps between checkpoint saves. |
| `EVAL_EM` | `--eval-em-examples` | `0` (off) | Generation-based EM/F1 on N dev examples per eval (slower; the teacher-forced loss alone hides stop-token issues). |
| `MAX_TRAIN` | `--max-train-examples` | (empty = all) | Cap training examples (partial run). |
| `MAX_STEPS` | `--max-steps` | (empty = full epoch(s)) | Cap optimizer steps. |
| `EVAL_N` | `--num-examples` | (empty = full val split) | Cap final-eval examples in `lcfx-eval`. |
| `SEED` | `--seed` | random | Resolved once per run and recorded; set explicitly for reproducibility. |
| `CKPT` | `--checkpoint-out` / `--fuser-checkpoint` | `data/checkpoints/lcfx/<dataset>/<sharer>/<receiver>/sp<SPANS>/fuser.pt` | Checkpoint path (train writes it, eval reads it). |

## Run EIYW (our method)

**Current iteration (v2): EIYW keeps LCF-X's paper-exact pooling on the
Sharer and replaces everything after it with a cross-attention translator
whose output is input tokens rather than a KV-cache residual.** The Sharer's
context is pooled **per layer** with the paper-exact LCF-X pool (up to
`--queries-per-layer` k position-free `[B,H,1,D]` slots per layer, in Sharer
geometry) — so any gain over LCF-X is attributable to what happens *after*
pooling, not to a richer pool. Flattening `K*‖V*` per slot gives the slot
matrix `[B, k·L_s, 2·H_s·D_s]`; a `Linear` projects it to `--d-model` width
(default 512 — a fixed, receiver-independent hyperparameter, so the adapter
does not grow quadratically with the Receiver) with learned layer /
query-index tags.
The Receiver's own context enters as raw input **embeddings** — an
embedding-table lookup, *no Receiver forward pass* (one whole frozen-model
call cheaper than v1) — projected to the same width plus **sinusoidal
positional encoding** (raw embeddings carry no order, unlike the KV caches v1
pooled; the pooled slots are position-free and need none; the soft tokens get
positions 0..S-1 from the Receiver's own RoPE at prefill). `--num-blocks`
decoder-shaped pre-LN blocks — self-attention over the slots,
**cross-attention from the slots into the Receiver memory** (cost
`k·L_s × N_r`, no `N_r²` term), then an MLP — refine the slots, and an output
head calibrated to the Receiver's measured embedding RMS emits `k·L_s` soft
tokens **prepended to the Receiver's input embeddings** — the Receiver
attends to them like any other input position; unlike LCF-X, nothing is
reshaped into a KV delta (see
[methods/eiyw/translator.py](methods/eiyw/translator.py)). The token count is
independent of both contexts' lengths; `--adaptive-k-tokens w` makes the
*active* k scale with the Sharer's length (`clamp(ceil(N_s/w), 1, k)`, e.g.
`--queries-per-layer 8 --adaptive-k-tokens 256`), so short contexts emit
fewer tokens and long ones cap at the fixed worst-case budget. Only the
translator is trained; both base models stay frozen. (See
[prompt/EIYW_IMPLEMENTATION_PLAN.md](prompt/EIYW_IMPLEMENTATION_PLAN.md) for
the original receiver-anchored design.)

```bash
# train -> eval end to end (model-stamped checkpoint per dataset+pair+bottleneck)
make eiyw DATASET=hotpotqa_bridge SHARER=unsloth/Llama-3.2-3B-Instruct RECEIVER=Qwen/Qwen3-0.6B

# or the scripts directly
python scripts/train_eiyw.py --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --sharer-model unsloth/Llama-3.2-3B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --d-model 512 --queries-per-layer 1 --num-blocks 1 \
  --checkpoint-out data/checkpoints/eiyw/translator.pt
python scripts/run_eiyw.py --dataset data/tasks/hotpotqa_bridge_val.jsonl \
  --sharer-model unsloth/Llama-3.2-3B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --translator-checkpoint data/checkpoints/eiyw/translator.pt \
  --output-dir results/eiyw
```

> No `--translator-checkpoint` gentle-inits the translator (soft tokens near,
> but not exactly, zero — exact-zero embeddings are a numerical degeneracy
> that makes the frozen Receiver's backward pass non-finite), so the run is a
> plumbing pass close to, but not guaranteed identical to, the Receiver-only
> floor (unlike LCF-X's exact-zero KV residual, which *is* an exact
> equality). Because the soft tokens lead the sequence, the whole Receiver
> generation forward is grad-enabled end to end (heavier than LCF-X's
> no_grad-prefill design) — gradient checkpointing on the frozen Receiver is
> enabled automatically; lower `--batch-size` / raise `--grad-accum` as needed.

### All EIYW options

| Makefile var | Flag | Default | Meaning |
|---|---|---|---|
| `DATASET` | `--dataset` (via `$(VAL)`/`$(TRAIN)`) | `hotpotqa_bridge` | Dataset name (see [Available datasets](#available-datasets-dataset)). |
| `EIYW_SHARER` | `--sharer-model` | `$(SHARER)` = `Qwen/Qwen3-0.6B` | Sharer model id. |
| `EIYW_RECEIVER` | `--receiver-model` | `$(RECEIVER)` = `Qwen/Qwen3-0.6B` | Receiver model id. |
| `SPAN_SCHEME` | `--span-scheme` | `paragraphs` | Sharer pooling span partition (shared with LCF-X): natural blank-line blocks or `fixed` token chunks. |
| `SPANS` | `--num-spans` | `5` | Span count for `--span-scheme fixed`. |
| `EIYW_QPL` | `--queries-per-layer` | `1` | Max cross-span pool queries per layer k (soft tokens = k·L_s; k=1 = paper-exact pool). |
| `EIYW_ADAPT` | `--adaptive-k-tokens` | `0` (off) | w>0: active k = `clamp(ceil(N_s/w), 1, EIYW_QPL)` — one query per w Sharer tokens, capped (e.g. `EIYW_QPL=8 EIYW_ADAPT=256`). |
| `EIYW_BLOCKS` | `--num-blocks` | `1` | Fusion depth (self-attn + cross-attn + MLP blocks; ~16·d_model² params each, ~4.2M at d_model 512). |
| `EIYW_DMODEL` | `--d-model` | `512` (receiver-independent) | Translator width (slot/memory projections and blocks); `half`/`hidden` remain as sweep options. |
| `LR` | `--lr` | `1e-4` | Learning rate. |
| `EPOCHS` | `--epochs` | `1` | Training epochs. |
| `BATCH_SIZE` | `--batch-size` | `8` (EIYW-specific; global default is 16) | Per-device micro-batch. EIYW's forward is grad-enabled end to end, so it holds ~2× LCF-X's activations at the same batch — `make eiyw-train` defaults to half LCF-X's batch, and `make exp` uses halved per-pair tiers for eiyw automatically. |
| `GRAD_ACCUM` | `--grad-accum` | `32` (EIYW-specific) | Micro-batches per optimizer step (effective batch = `BATCH_SIZE`×`GRAD_ACCUM` = 256, unchanged). |
| `WARMUP` | `--warmup` | ~10% of steps/epoch, per dataset | Linear LR warmup steps (`WARMUP_<dataset>` in the Makefile: hotpotqa 28, 2wiki 29, musique 5, ropes 4, qasc 3, strategyqa 1). |
| `VAL_FRAC` | `--val-frac` | `0.028` | Dev holdout fraction of train data (when no `--eval-data`). |
| `LOG_EVERY` | `--log-every` | `10` | Steps between training log lines. |
| `EVAL_EVERY` | `--eval-every` | `50` | Steps between in-training dev evals. |
| `EVAL_MAX` | `--eval-max-examples` | `500` | Cap on in-training dev eval examples (monitoring only). |
| `SAVE_EVERY` | `--save-every` | `50` | Steps between checkpoint saves. |
| `EVAL_EM` | `--eval-em-examples` | `0` (off) | Generation-based EM/F1 on N dev examples per eval (slower; the teacher-forced loss alone hides stop-token issues). |
| `MAX_TRAIN` | `--max-train-examples` | (empty = all) | Cap training examples (partial run). |
| `MAX_STEPS` | `--max-steps` | (empty = full epoch(s)) | Cap optimizer steps. |
| `EVAL_N` | `--num-examples` | (empty = full val split) | Cap final-eval examples in `eiyw-eval`. |
| `SEED` | `--seed` | random | Resolved once per run and recorded; set explicitly for reproducibility. |
| `EIYW_CKPT` | `--checkpoint-out` / `--translator-checkpoint` | `data/checkpoints/eiyw/<dataset>/<sharer>/<receiver>/sp<SPANS>_dm<EIYW_DMODEL>_k<EIYW_QPL>/translator.pt` | Checkpoint path (train writes it, eval reads it). |

## Run KVTok (baseline)

**KVTok reuses LCF-X's Fig. 3 Stage 1 (within-span attention) on the Sharer
only and drops Stage 2 (across-span attention).** Its `[B,L_s·P,2·H_s·D_s]`
summaries use a factorized linear down-projection: a position-shared
`2·H_s·D_s → d_model` feature projection followed by a feature-shared
`L_s·P → num_tokens` token-axis projection. This keeps KVTok linear (there is
no Stage-2 attention) without the erroneous billion-parameter fully dense
`(L_s·P·2·H_s·D_s) → (num_tokens·d_model)` map. The Receiver contributes only
raw context embeddings — no initial Receiver forward or KV cache — processed
with `receiver_proj + sinusoidal PE + LayerNorm`. Cross-attention and the
EIYW-v2 calibrated output head emit Receiver-compatible soft tokens, which are
prepended to the generation prompt embeddings. For Qwen3-0.6B → Qwen3-0.6B at
`SPANS=5`, `KVTOK_TOKENS=8`, and `d_model=512`, the translator has 3,183,721
trainable parameters (about 12.1 MiB in fp32), not 2.35B.

```bash
# train -> eval end to end (model-stamped checkpoint per dataset+pair+shape)
make kvtok DATASET=hotpotqa_bridge SHARER=unsloth/Llama-3.2-3B-Instruct RECEIVER=Qwen/Qwen3-0.6B

# or the scripts directly
python scripts/train_kvtok.py --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --sharer-model unsloth/Llama-3.2-3B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --num-spans 5 --num-tokens 8 --d-model 512 --num-heads 8 \
  --checkpoint-out data/checkpoints/kvtok/translator.pt
python scripts/run_kvtok.py --dataset data/tasks/hotpotqa_bridge_val.jsonl \
  --sharer-model unsloth/Llama-3.2-3B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --num-spans 5 --num-tokens 8 --d-model 512 --num-heads 8 \
  --translator-checkpoint data/checkpoints/kvtok/translator.pt \
  --output-dir results/kvtok
```

> No `--translator-checkpoint` gentle-inits the translator (same rationale as
> EIYW: exact-zero soft-token embeddings make the frozen Receiver's backward
> pass non-finite), so the run is a plumbing pass close to the Receiver-only
> floor. `--num-spans`, `--num-tokens`, `--d-model`, and resolved `--num-heads`
> are baked into the translator architecture, so they must be re-supplied identically
> between `train_kvtok.py` and `run_kvtok.py` — `load_translator` hard-errors
> on a mismatch.

### All KVTok options

| Makefile var | Flag | Default | Meaning |
|---|---|---|---|
| `DATASET` | `--dataset` (via `$(VAL)`/`$(TRAIN)`) | `hotpotqa_bridge` | Dataset name (see [Available datasets](#available-datasets-dataset)). |
| `KVTOK_SHARER` | `--sharer-model` | `$(SHARER)` = `Qwen/Qwen3-0.6B` | Sharer model id. |
| `KVTOK_RECEIVER` | `--receiver-model` | `$(RECEIVER)` = `Qwen/Qwen3-0.6B` | Receiver model id. |
| `SPANS` | `--num-spans` | `5` | Sharer Stage-1 pooling spans (shared with LCF-X/EIYW). |
| `KVTOK_TOKENS` | `--num-tokens` | `8` | Soft tokens the down-projection outputs (free knob; baked into the checkpoint shape). |
| `KVTOK_DMODEL` | `--d-model` | `512` | Translator width. |
| `KVTOK_HEADS` | `--num-heads` | `8` | Cross-attention heads; resolved down to the largest value that divides `KVTOK_DMODEL`. |
| `LR` | `--lr` | `1e-4` | Learning rate. |
| `EPOCHS` | `--epochs` | `1` | Training epochs. |
| `BATCH_SIZE` | `--batch-size` | `8` (KVTok-specific; global default is 16) | Per-device micro-batch. KVTok's Receiver prompt forward is grad-enabled like EIYW, so `make kvtok-train` uses EIYW's half-batch default. |
| `GRAD_ACCUM` | `--grad-accum` | `32` (KVTok-specific) | Micro-batches per optimizer step (effective batch = `BATCH_SIZE`×`GRAD_ACCUM` = 256). |
| `WARMUP` | `--warmup` | ~10% of total optimizer steps | Linear LR warmup steps. `make kvtok-train` and direct `scripts/train_kvtok.py` auto-compute 10% after train/eval split and `MAX_TRAIN`/`MAX_STEPS` are applied; pass `WARMUP=...` or `--warmup ...` to override. |
| `VAL_FRAC` | `--val-frac` | `0.028` | Dev holdout fraction of train data (when no `--eval-data`). |
| `LOG_EVERY` | `--log-every` | `10` | Steps between training log lines. |
| `EVAL_EVERY` | `--eval-every` | `50` | Steps between in-training dev evals. |
| `EVAL_MAX` | `--eval-max-examples` | `500` | Cap on in-training dev eval examples (monitoring only). |
| `SAVE_EVERY` | `--save-every` | `50` | Steps between checkpoint saves. |
| `EVAL_EM` | `--eval-em-examples` | `0` (off) | Generation-based EM/F1 on N dev examples per eval. |
| `MAX_TRAIN` | `--max-train-examples` | (empty = all) | Cap training examples (partial run). |
| `MAX_STEPS` | `--max-steps` | (empty = full epoch(s)) | Cap optimizer steps. |
| `EVAL_N` | `--num-examples` | (empty = full val split) | Cap final-eval examples in `kvtok-eval`. |
| `SEED` | `--seed` | random | Resolved once per run and recorded; set explicitly for reproducibility. |
| `KVTOK_CKPT` | `--checkpoint-out` / `--translator-checkpoint` | `data/checkpoints/kvtok/<dataset>/<sharer>/<receiver>/sp<SPANS>_tok<KVTOK_TOKENS>_dm<KVTOK_DMODEL>_h<KVTOK_HEADS>/translator.pt` | Checkpoint path (train writes it, eval reads it). |

## Run XKV

XKV prefills both frozen models and applies the same vectorized learned-query
attention pool directly to each cache. For every layer and KV head, `k`
independent queries reduce raw positions `N → k`, producing separate pooled K
and V tensors `[B,H,k,D]`. The default follows Receiver size: `k=1` for 0.6B,
`k=2` for 1B, and `k=3` for 3B. There are no intermediate spans. Pool queries
use the v9 initialization, Kaiming-normal followed by `x0.02`; there is no
learned attention temperature.
After the optional learned `L_s → L_r` layer-axis map, concatenation has width
`2*k*(H_s*D_s + H_r*D_r)`. A learned bottleneck returns this to
`[B,L_r,d]`, with `d=receiver_hidden//8`, before `n=1` attention block by
default for every Receiver. The width resolves to
128/144/384 for the Qwen-0.6B, Gemma-1B, and
Llama-3B receivers. After attention, each
raw Receiver K/V position is projected into a query and cross-attends to the
translated layer memory. The resulting position-aware latent is scaled by a
learned sigmoid per Receiver KV head and passed through one shared K
up-projection and one shared V up-projection. Output weights remain shared
across Receiver layers; the former stochastic layer/type gates have been
removed.

Hybrid caches are handled explicitly for Gemma-3, Ministral, and any model
whose config declares both full- and sliding-attention layers. When layer token
widths differ, XKV uses the shortest width as a recent common suffix for
pooling/translation. Updates are scattered back into each Receiver layer's
original shape; older full-attention positions receive a zero update and remain
available to the frozen Receiver. Uniform-cache models retain strict
equal-layer-width validation.

```bash
# train -> eval
make xkv DATASET=hotpotqa_bridge \
  SHARER=unsloth/Llama-3.2-3B-Instruct RECEIVER=Qwen/Qwen3-0.6B

# include XKV in the model-pair experiment sweep
make exp METHOD=xkv DATASET=hotpotqa_bridge
```

### XKV ablations

The three XKV ablations use the same training/evaluation implementation.
The first two retain Full XKV's Receiver-dependent `k`; the LCF-X pooling
ablation uses the original pool's single output slot. Full XKV, `−RP`, and
`−RX` remain frozen at their finalized `v11` implementations and result paths;
only corrected LCFP uses a separate `v12` path. Each method name has isolated
checkpoint and result roots under `data/checkpoints/<method>/...` and
`results/<method>/...`:

```bash
make exp METHOD=xkv-no-receiver-pooling DATASET=ropes
make exp METHOD=xkv-no-receiver-xattn DATASET=ropes
make exp METHOD=xkv-lcfx-pooling DATASET=ropes

make exp METHOD=xkv-no-receiver-pooling DATASET=qasc
make exp METHOD=xkv-no-receiver-xattn DATASET=qasc
make exp METHOD=xkv-lcfx-pooling DATASET=qasc
```

Launch all three ablations together in one 27-cell sweep. Finalized Full XKV
v11 is deliberately not rerun; the visual command reads its existing results:

```bash
make exp-ablation DATASET=ropes
make exp-ablation DATASET=qasc
```

It accepts the same sweep controls as `make exp`, including `EXP_GPUS=`,
`EXP_STAGE=smoke`, `SHARER=`, `RECEIVER=`, and
`EXP_ARGS="--resume"`/`EXP_ARGS="--dry-run"`.

Render ablation-only Markdown grids with:

```bash
make visual-ablation DATASET=ropes
make visual-ablation DATASET=qasc

# Or render one view only:
make visual-ablation DATASET=ropes ITEM=scores
make visual-ablation DATASET=ropes ITEM=params
make visual-ablation DATASET=ropes ITEM=latency
```

With no `ITEM=`, the command writes all three views under
`results/comparison/ablation/<dataset>[_params|_latency].md`. The cells use
`F` for Full XKV, `−RP` for no Receiver pooling, `−RX` for no raw Receiver-KV
cross-attention, and `LCFP` for exact LCF-X pooling.

`xkv-no-receiver-pooling` removes only the pooled Receiver slots from the joint
memory; raw Receiver KV still drives the position decoder.
`xkv-no-receiver-xattn` keeps both pools but broadcasts translated per-layer
memory across Receiver positions instead of using raw Receiver KV as
cross-attention queries. `xkv-lcfx-pooling` changes both side pools from direct
`N -> k` attention to LCF-X's original, intentionally non-vectorized
`N -> P -> 1` attention. It uses the same two query tensors, query
initialization, shared K/V weights, natural-paragraph boundaries, span
allocation, and batch/layer/span loops as LCF-X; `P=5` remains the fallback
for inputs without natural boundaries and can be changed with
`XKV_POOL_SPANS=`.

Key settings are `XKV_DMODEL=eighth` (`receiver_hidden//8`), `XKV_BLOCKS=1`,
`XKV_HEADS=8`, and `XKV_POOL_K` (`XKV_POOL_SLOTS` is an alias; `k` applies to
both caches). When unset, `k` is 1/2/3/4 for a 0.6B/1B/3B/8B Receiver; an explicit
`XKV_BLOCKS` or pool value overrides its respective default. The remaining
default is `XKV_DROPOUT=0.1`. The CLI accepts either `--pool-slots` or
`--pool-k`.
An explicit integer such as `XKV_DMODEL=256` still selects a fixed width.
Every `LOG_EVERY` optimizer steps, `training_log.jsonl` records normalized
attention entropy and, when `k>1`, inter-slot attention-map cosine for both pools as
`sharer_pool_*` and `receiver_pool_*`. Entropy or cosine near `1.0` indicates
nearly uniform or collapsed slots, respectively. Diagnostics
are disabled on other training steps and throughout normal inference.
On ROPES, XKV defaults to `BATCH_SIZE=32` and `GRAD_ACCUM=8` for the reference
pair (effective batch 256); the experiment sweep keeps its existing rule of
halving the batch and doubling accumulation as pair size increases.
Checkpoints and results are isolated under
`dh8_n1_h<XKV_HEADS>_do<XKV_DROPOUT>_k<resolved-k>_v11`
by default (`d256` replaces `dh8` for an explicit width of 256;
the dropout decimal point is rendered as `p`).

## Visual comparison table

Once you've got some results on disk, render a Markdown comparison table with
`make visual`, scoped with `DATASET=` and `ITEM=`:

```bash
make visual DATASET=hotpotqa_bridge                  # EM+F1 grid (default ITEM=scores)
make visual DATASET=hotpotqa_bridge ITEM=scores       # same, explicit
make visual DATASET=hotpotqa_bridge ITEM=params       # trainable-parameter grid (lcfx/xkv)
make visual DATASET=hotpotqa_bridge ITEM=latency      # average per-example latency grid
make visual-all DATASET=hotpotqa_bridge               # all three views, including 8B (4x4)
make visual ITEM=datasets                              # dataset overview (DATASET= ignored)
make visual DATASET=hotpotqa_bridge RUNS=1:5           # mean across runs 1..5
make visual DATASET=hotpotqa_bridge RUNS=0:5           # include existing unscoped results as run 0
make visual DATASET=hotpotqa_bridge RANGE='[1,3,5]'    # RANGE alias; selected runs only
make compare                                             # pairwise method differences across all datasets
make compare RUNS=1:5                                    # compare selected-run means
```

`make compare` writes three rows (`xkv-lcfx`, `xkv-t2t`, and `lcfx-t2t`)
with one column per dataset, preceded by an XKV/LCF-X/T2T method-summary table.
Each method summary reports `mean±sample SD` across run-level primary 3×3
Sharer/Receiver-grid averages. The difference table reports the difference
between method means and the sample SD of same-seed per-run differences.
Free-form datasets show `EM/F1` (or `ΔEM/ΔF1`) in percentage points;
qasc and strategyqa show `Δaccuracy`. With `RUNS=`, available model-pair cells
are averaged within each run and those run-level averages are summarized for
each method. Incomplete coverage is marked per method as
`n=available results/expected results`; a difference is blank only when one
method has no available result. The output defaults to
`results/comparison/compare.md`, or
`results/comparison/runs-<selected ids>/compare.md` for selected runs.

`make visual` remains the primary-model 3×3 view. `make visual-all` supports
the same `DATASET=`, `ITEM=`, `MODELS=`, and `OUT=` options, but always appends
the opt-in 8B model.

`ITEM=` picks what gets rendered:

- `scores` (default) — Sharer × Receiver grid (primary roster, 3×3; 4×4 with
  `make visual-all` or `EXTRA_MODELS=TRUE`, matching `exp`/`sweep`) for one
  `DATASET=`, with LCF-X / XKV / T2T EM+F1 per cell (best EM and best F1 in
  each cell bolded). Read from `results/<method>/<dataset>/<sharer>/<receiver>/metrics.json`.
- `params` — the same Sharer × Receiver grid, but trainable-parameter counts
  for LCF-X / XKV (T2T trains nothing, so it's omitted). Read from the
  `training_log.jsonl` under each checkpoint dir.
- `latency` — the same Sharer × Receiver grid, but LCF-X / XKV / T2T average
  per-example latency. LCF-X and XKV use fuser-or-translator/total format (for
  example, `6ms/141ms`), while T2T shows total only; the fastest component and
  total in each cell are bolded independently. Read from the `average.total_ms`
  and `average.fuser_ms` fields of
  `results/<method>/<dataset>/<sharer>/<receiver>/latency.json`.
- `datasets` — one overview table across *every* dataset (`DATASET=` is
  ignored): train/val split sizes plus the single-model-oracle EM+F1 for a
  representative 1B/3B/8B model (`make single`'s results).

Any method/checkpoint/result that hasn't been run yet is left blank rather
than erroring, so you can call `make visual` at any point mid-sweep. Tables
aggregating `RUNS=` keep the single-run table format but replace each result
with the mean across the selected runs. The companion table reports
completed/requested coverage and lists missing run ids.
Aggregation refuses to combine completed results whose `run_metadata.json`
configurations differ. Parameter reports verify that selected repetitions have
the same parameter count rather than computing a meaningless deviation.
Aggregated reports default to
`results/comparison/runs-<selected ids>/<dataset>...`.
Tables use the active LCF-X `SPANS` and XKV shape settings. XKV uses the exact active
`XKV_DMODEL`, `XKV_BLOCKS`, `XKV_HEADS`,
`XKV_POOL_K`/`XKV_POOL_SLOTS`, and `XKV_DROPOUT` configuration, which is also embedded in
the table caption. The
tables are written
to `results/comparison/<dataset>[_params|_latency].md` (or
`results/comparison/datasets.md` for `ITEM=datasets`) — override the path
with `OUT=`:

```bash
make visual DATASET=ropes OUT=results/comparison/ropes_custom.md
```

## Layout

```text
methods/      # one package per method (t2t, lcfx, eiyw, kvtok, xkv)
scripts/      # runnable entrypoints (prepare / run / train) + cache config
data/tasks/   # prepared task JSONL (reused across runs)
results/      # run outputs (one dir per run)
Makefile      # data / lcfx / eiyw / kvtok / xkv / t2t / exp / visual one-liners
```
