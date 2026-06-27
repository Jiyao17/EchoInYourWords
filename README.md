# Echo In Your Words

Research codebase for **flexible, cross-context communication between language
models** for multi-agent collaboration: a *Sharer* and *Receiver* hold different
evidence (and may differ in size/architecture), and we study how the Sharer can
help the Receiver answer. It compares three methods under one standardized
output contract: **T2T** (text message), **LCF-X** (position-free latent cache
communication), and a new **receiver-anchored** method.

- Architecture & contracts: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- Full run reference (every flag): [scripts/README.md](scripts/README.md)
- Design notes: [prompt/](prompt/)

## Setup

- Python 3.12 venv at `.venv` — run everything as `.venv/bin/python ...`.
- **torch must be the CUDA 11.8 build** (`torch==2.7.1+cu118`); the machine's
  driver is too old for the default CUDA wheels.
- The Hugging Face cache is auto-routed to `data/hf_cache` by
  `scripts/_hf_cache.py`. Prefix a command with `HF_HUB_OFFLINE=1` once assets are
  downloaded to skip Hub lookups.

## Quickstart (LCF-X, via the Makefile)

Data prep is a one-time idempotent step (file targets — skipped if present), so
after the first run training/eval are single commands:

```bash
make lcfx-data     # prepare bridge train (~70.7K) + full bridge-val (5,899) once
make lcfx-train    # train the fuser (paper recipe: Qwen3-0.6B, eff. batch 256, 1 epoch)
make lcfx-eval     # EM/F1 of the trained checkpoint + the no-fusion floor
make lcfx          # data -> train -> eval, end to end
```

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

## Run LCF-X (the method)

The scenario: Sharer and Receiver hold **different** context (disjoint evidence),
the Sharer's KV cache is pooled → projected → **fused into the Receiver's cache**,
and the Receiver answers. The two models may differ (`--sharer-model` /
`--receiver-model`); the fuser's align stage bridges different dims.

```bash
DATA=data/tasks/hotpotqa_bridge_val.jsonl
.venv/bin/python scripts/run_lcfx.py --dataset $DATA --num-spans 4 \
  --sharer-model Qwen/Qwen3-0.6B --receiver-model Qwen/Qwen3-0.6B \
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
**All defaults follow the paper**: Qwen3-0.6B both sides, AdamW, lr 1e-4,
200-step warmup, latent 128, 2.8% dev holdout, per-device batch 16 × grad-accum
16 = **effective batch 256**, 1 epoch (~276 steps), EM/F1.

```bash
# Paper defaults (needs a large GPU ~80GB). On smaller GPUs, see "Smaller GPUs".
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train_lcfx_fuser.py \
  --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --eval-data  data/tasks/hotpotqa_bridge_val.jsonl \
  --checkpoint-out data/checkpoints/lcfx/fuser.pt
# (model/dim/batch/warmup/etc. all default to the paper values above)
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
# train 5 steps on 200 examples, dev-eval on 20
make lcfx-train MAX_TRAIN=200 MAX_STEPS=5 EVAL_MAX=20 BATCH_SIZE=4 GRAD_ACCUM=2
# evaluate on 50 test examples (+ the no-fusion floor)
make lcfx-eval EVAL_N=50
```

Same with the scripts directly:

```bash
.venv/bin/python scripts/train_lcfx_fuser.py \
  --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --max-train-examples 200 --val-frac 0.1 --eval-max-examples 20 \
  --batch-size 4 --grad-accum 2 --max-steps 5 \
  --checkpoint-out data/checkpoints/lcfx/fuser_smoke.pt
.venv/bin/python scripts/run_lcfx.py \
  --dataset data/tasks/hotpotqa_bridge_val.jsonl --num-examples 50 \
  --fuser-checkpoint data/checkpoints/lcfx/fuser_smoke.pt --output-dir results/lcfx_smoke
```

> Note: a full epoch on a single 24 GB GPU is compute-bound (~hours); the paper's
> ~1 h used a much faster GPU. Lowering `--batch-size` while raising `--grad-accum`
> keeps the *effective* batch (256) and optimization identical to the paper.

## Layout

```text
methods/      # one package per method (t2t implemented; lcfx implemented; eiyw planned)
scripts/      # runnable entrypoints (prepare / run / train) + cache config
data/tasks/   # prepared task JSONL (reused across runs)
results/      # run outputs (one dir per run)
Makefile      # lcfx-data / lcfx-train / lcfx-eval / lcfx one-liners
```
