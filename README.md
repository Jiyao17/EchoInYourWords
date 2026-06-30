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

- The Hugging Face cache is auto-routed to `data/hf_cache` by
  `scripts/_hf_cache.py`. Prefix a command with `HF_HUB_OFFLINE=1` once assets are
  downloaded to skip Hub lookups.

## Quickstart (via the Makefile)

**Always drive experiments through `make`, and name every axis explicitly.**
`DATASET=` selects the data; **every method run takes `DATASET=` plus a
`SHARER=` / `RECEIVER=` pair** (two independent models — use a **heterogeneous**
pair). This keeps runs reproducible and self-documenting. Run `make help` for the
full target + knob list.

Data prep is a one-time idempotent step (skipped if present); then train and eval
the **same** dataset + pair so the model-stamped checkpoint matches:

```bash
make help                                          # targets + how to pick models

make data  DATASET=hotpotqa_bridge                 # prepare train (~70.7K) + val (5,899) once
make model MODEL=Qwen/Qwen2.5-0.5B-Instruct        # fetch each model into data/hf_cache (once)

# Methods: always pass DATASET + the SHARER/RECEIVER pair.
make lcfx-train DATASET=hotpotqa_bridge SHARER=Qwen/Qwen2.5-0.5B-Instruct RECEIVER=Qwen/Qwen3-0.6B
make lcfx-eval  DATASET=hotpotqa_bridge SHARER=Qwen/Qwen2.5-0.5B-Instruct RECEIVER=Qwen/Qwen3-0.6B
make lcfx       DATASET=hotpotqa_bridge SHARER=Qwen/Qwen2.5-0.5B-Instruct RECEIVER=Qwen/Qwen3-0.6B  # train -> eval

make eiyw-train DATASET=hotpotqa_bridge SHARER=Qwen/Qwen2.5-0.5B-Instruct RECEIVER=Qwen/Qwen3-0.6B
make eiyw-eval  DATASET=hotpotqa_bridge SHARER=Qwen/Qwen2.5-0.5B-Instruct RECEIVER=Qwen/Qwen3-0.6B
make eiyw       DATASET=hotpotqa_bridge SHARER=Qwen/Qwen2.5-0.5B-Instruct RECEIVER=Qwen/Qwen3-0.6B  # ours: train -> eval

make t2t        DATASET=hotpotqa_bridge SHARER=Qwen/Qwen2.5-0.5B-Instruct RECEIVER=Qwen/Qwen3-0.6B  # T2T baseline
```

Every run is isolated automatically by **dataset + pair** — results go to
`results/<method>/<dataset>/<sharer>/<receiver>/` and the checkpoint to
`data/checkpoints/lcfx/<dataset>/<sharer>/<receiver>/fuser.pt` (model id `/` → `--`),
so runs never overwrite each other. Just change `DATASET` / `SHARER` / `RECEIVER`:

```bash
make lcfx DATASET=hotpotqa_bridge SHARER=meta-llama/Llama-3.2-1B-Instruct RECEIVER=Qwen/Qwen3-0.6B
```

`SHARER` / `RECEIVER` move every method at once; the per-method `LCFX_*` /
`EIYW_*` / `T2T_*` vars override a single method. EIYW adds `SOFT_TOKENS` (K, the
number of soft pseudo-tokens; default 16) and otherwise reuses the LCF-X training
knobs. Its translator checkpoint lands at
`data/checkpoints/eiyw/<dataset>/<sharer>/<receiver>/translator.pt`.

### Available datasets (`DATASET=`)

- `hotpotqa_bridge` — cross-context HotpotQA-bridge: Sharer and Receiver hold
  disjoint evidence, so neither can answer alone.
- `musique` — cross-context MuSiQue-Ans 2-hop: each question's two supporting
  paragraphs are split one per side (harder, less shortcut-able than HotpotQA).

### Available models (`SHARER=` / `RECEIVER=`)

Any id below works as **either** the Sharer or the Receiver — mix families and
sizes freely (the LCF-X fuser bridges differing shapes; specs in
[Models](#models-sharer--receiver)). Fetch one with `make model MODEL=<id>` (or
let the first run download it); gated ids need `HF_TOKEN` exported first:

- `Qwen/Qwen3-0.6B` — ungated
- `Qwen/Qwen2.5-0.5B-Instruct` — ungated
- `Qwen/Qwen3-4B` — ungated
- `meta-llama/Llama-3.2-1B-Instruct` — gated; needs `HF_TOKEN`
- `google/gemma-3-1b-it` — gated; needs `HF_TOKEN`
- `unsloth/Llama-3.2-1B-Instruct` — ungated mirror of Llama-3.2-1B (no token)
- `unsloth/gemma-3-1b-it` — ungated mirror of Gemma-3-1B (no token)

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
.venv/bin/python scripts/prepare_hotpotqa_bridge.py --split train --num-examples -1 \
  --output data/tasks/hotpotqa_bridge_train.jsonl
.venv/bin/python scripts/prepare_hotpotqa_bridge.py --split validation --num-examples -1 \
  --output data/tasks/hotpotqa_bridge_val.jsonl
```

## Models (Sharer / Receiver)

The Sharer and Receiver are two independent models, set per run with
`--sharer-model` / `--receiver-model` (or `SHARER` / `RECEIVER` for the Makefile).
Use a **heterogeneous** pair: they may differ in family, size, hidden dim, head
dim, and depth — the LCF-X fuser's align stage bridges the differing shapes. Any
id works as either role.

Known-good ids (mix freely as Sharer/Receiver):

| Model | hidden | head_dim | layers | notes |
|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | 1024 | 128 | 28 | |
| `Qwen/Qwen2.5-0.5B-Instruct` | 896 | 64 | 24 | |
| `meta-llama/Llama-3.2-1B-Instruct` | 2048 | 64 | 16 | gated — needs `HF_TOKEN` |
| `google/gemma-3-1b-it` | 1152 | 256 | 26 | gated — needs `HF_TOKEN` |

`Qwen/Qwen3-4B` is another known-good (larger) Qwen option. The Llama/Gemma ids
are **gated** on the Hub — accept their license and export `HF_TOKEN`, or use the
ungated mirrors `unsloth/Llama-3.2-1B-Instruct` / `unsloth/gemma-3-1b-it`.

Trained fuser checkpoints are **stamped** with the Sharer/Receiver they were
trained for, and `run_lcfx.py` hard-errors if you load one against a different
pair (override with `--allow-model-mismatch`). Keep a distinct checkpoint per
pair — e.g. `make lcfx-hetero` runs Llama→Qwen and writes `fuser_hetero.pt`.

```bash
# heterogeneous example: Llama Sharer -> Qwen Receiver (needs HF_TOKEN)
make lcfx-hetero
# or mirror ids, no token needed:
make lcfx-hetero HETERO_SHARER=unsloth/Llama-3.2-1B-Instruct
```

## Run LCF-X (the method)

The scenario: Sharer and Receiver hold **different** context (disjoint evidence),
the Sharer's KV cache is pooled → projected → **fused into the Receiver's cache**,
and the Receiver answers. The two models are set independently (`--sharer-model` /
`--receiver-model`) and may differ; the fuser's align stage bridges different dims.

```bash
DATA=data/tasks/hotpotqa_bridge_val.jsonl
.venv/bin/python scripts/run_lcfx.py --dataset $DATA --num-spans 4 \
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
.venv/bin/python scripts/train_lcfx_fuser.py \
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
.venv/bin/python scripts/train_lcfx_fuser.py \
  --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --max-train-examples 200 --val-frac 0.1 --eval-max-examples 20 \
  --batch-size 4 --grad-accum 2 --max-steps 5 \
  --checkpoint-out data/checkpoints/lcfx/fuser_smoke.pt
.venv/bin/python scripts/run_lcfx.py \
  --dataset data/tasks/hotpotqa_bridge_val.jsonl --num-examples 50 \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --fuser-checkpoint data/checkpoints/lcfx/fuser_smoke.pt --output-dir results/lcfx_smoke
```

> Note: a full epoch on a single 24 GB GPU is compute-bound (~hours); the paper's
> ~1 h used a much faster GPU. Lowering `--batch-size` while raising `--grad-accum`
> keeps the *effective* batch (256) and optimization identical to the paper.

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
.venv/bin/python scripts/train_eiyw.py --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --num-soft-tokens 16 --num-spans 4 --checkpoint-out data/checkpoints/eiyw/translator.pt
.venv/bin/python scripts/run_eiyw.py --dataset data/tasks/hotpotqa_bridge_val.jsonl \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --num-soft-tokens 16 --translator-checkpoint data/checkpoints/eiyw/translator.pt \
  --output-dir results/eiyw
```

> `--no-fusion` runs the Receiver alone (the comparison floor / plumbing check).
> EIYW's prompt forward is grad-enabled (soft tokens lead the sequence), so it
> uses more memory than LCF-X — lower `--batch-size` and raise `--grad-accum` to
> keep the effective batch at 256. See
> [prompt/EIYW_IMPLEMENTATION_PLAN.md](prompt/EIYW_IMPLEMENTATION_PLAN.md).

## Layout

```text
methods/      # one package per method (t2t, lcfx, eiyw — all implemented)
scripts/      # runnable entrypoints (prepare / run / train) + cache config
data/tasks/   # prepared task JSONL (reused across runs)
results/      # run outputs (one dir per run)
Makefile      # data / lcfx{-train,-eval} / eiyw{-train,-eval} / t2t one-liners
```
