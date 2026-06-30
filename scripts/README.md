# Running Experiments

This folder holds the runnable entrypoints. Every script writes the same
standardized output files (`predictions.jsonl`, `predictions_brief.tsv`,
`metrics.json`, `latency.json`, `run_metadata.json`) into its `--output-dir`, so
all runs are directly comparable. `predictions_brief.tsv` is a quick-scan
companion to `predictions.jsonl` — one row per example with just `id`, `em`,
`f1`, `prediction`, `answer` (no prompts), e.g. `column -t -s$'\t' <file>` to
eyeball answer vs. prediction. For the project layout and data/output contracts,
see [../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md).

Each eval run also writes a persistent **`eval_log.jsonl`** into its
`--output-dir` (flushed per record, so it survives the session and a crash): a
`start` config record, a `progress` record every `--print-every` examples with
`processed`/`total`, `elapsed_s`, `rate_per_s`, `eta_s`, and the running
`accuracy`/`em`/`f1`, then a `done` summary. Tail it to watch a long run:
`tail -f <output-dir>/eval_log.jsonl`. (Training writes the analogous
`training_log.jsonl` beside its checkpoint.)

## Prerequisites

- Run everything with the project venv: `.venv/bin/python ...`.
- The Hugging Face cache is configured automatically by `_hf_cache.py`, which
  points `HF_HOME` at `data/hf_cache` (HF's native single-root layout: one
  `hub/` for all downloads, `datasets/` for built arrow data). You do **not**
  need to set `HF_HOME`.
- If assets are already downloaded, prefix a command with `HF_HUB_OFFLINE=1` to
  force offline use and skip Hub lookups.

The scripts:

| Script | Purpose |
|--------|---------|
| `prepare_hotpotqa_bridge.py` | Build the cross-context HotpotQA-bridge task file (LCF-X partitioning). |
| `run_text_to_text.py` | Run the T2T Sharer→Receiver pipeline. |
| `run_lcfx.py` | Run the LCF-X latent cache communication method (Sharer cache → pooled summary → fused into Receiver). |
| `run_eiyw.py` | Run EIYW, the receiver-anchored method (Sharer cache → K soft tokens conditioned on the Receiver's question → prepended to the Receiver input). |
| `run_single_model.py` | Run the single-model inference baseline (one model answers from the full context — no Sharer, no communication). |
| `train_lcfx_fuser.py` | Train the LCF-X fuser (frozen Sharer/Receiver; only the fuser learns). |
| `train_eiyw.py` | Train the EIYW translator (frozen Sharer/Receiver; only the translator learns). |
| `_hf_cache.py` | Helper imported by the others; not run directly. |

---

## Step 1 — Prepare the dataset (once)

Builds normalized cross-context examples and writes them to
`data/tasks/hotpotqa_bridge_cross.jsonl`. Filters to `bridge` questions and
splits the 10 paragraphs into two 5-paragraph partitions (1 supporting + 4
distractors each); deterministic given `--seed`.

```bash
.venv/bin/python scripts/prepare_hotpotqa_bridge.py --num-examples 200
```

Key options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--num-examples` | `200` | Number of bridge examples to write (`-1` for all). |
| `--split` | `validation` | HotpotQA split. |
| `--seed` | `0` | Seed for deterministic partitioning. |
| `--levels` | (all) | Difficulty filter, e.g. `--levels hard medium`. |
| `--output` | `data/tasks/hotpotqa_bridge_cross.jsonl` | Output JSONL path. |

The produced JSONL is reused by every run below for a fair comparison.

---

## Available tests / experiments

All runs below use the same dataset file and the same EM/F1 scoring (auto-
selected because HotpotQA answers are free-form). The communication methods (T2T,
LCF-X, EIYW) are compared against the single-model inference baseline.

### Test A — Text-to-Text communication (Sharer → Receiver)

The main method: Sharer writes a bounded message; Receiver answers using its own
context + that message.

```bash
.venv/bin/python scripts/run_text_to_text.py \
  --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct \
  --receiver-model Qwen/Qwen3-0.6B \
  --output-dir results/t2t_hotpotqa_bridge
```

Sharer and Receiver are independent; either flag left unset defaults to
`Qwen/Qwen3-0.6B` (see the [Models](../README.md#models-sharer--receiver) section
for the full list).

| Flag | Default | Meaning |
|------|---------|---------|
| `--sharer-model` | `Qwen/Qwen3-0.6B` | Sharer model id/path. |
| `--receiver-model` | `Qwen/Qwen3-0.6B` | Receiver model id/path (independent of Sharer). |
| `--sharer-max-new-tokens` | `128` | **Communication budget** — max tokens in the Sharer message. |
| `--receiver-max-new-tokens` | `64` | Max tokens in the Receiver answer. |
| `--num-examples` | all | Cap the number of examples. |
| `--scoring` | `auto` | `auto` / `choice` / `free_form`. |
| `--device`, `--dtype` | auto / `auto` | Device and weight dtype. |

**Limiting the shared message (LCF-X-style budget):** set
`--sharer-max-new-tokens`, e.g. `--sharer-max-new-tokens 32`. The *actual*
message length is also recorded per example as `token_counts.sharer_output`, so
you can sweep the budget and plot accuracy vs. message length.

### Test B — Single-model inference baseline

The reference point the communication methods are measured against: **one model**
answers each question from the **full** context (both partitions combined) — no
Sharer, no communication. It uses a dedicated single-model prompt, separate from
the method receiver prompts.

```bash
.venv/bin/python scripts/run_single_model.py \
  --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir results/single_model
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `Qwen/Qwen3-0.6B` | The single model that answers. |
| `--max-new-tokens` | `64` | Answer budget. |
| `--num-examples` | all | Cap the number of examples. |
| `--scoring` | `auto` | `auto` / `choice` / `free_form`. |
| `--device`, `--dtype` | auto / `auto` | Device and weight dtype. |

This full-context single model is the upper bound: a communication method's value
is how much of the gap up to it the method recovers using only a Sharer message /
latent summary / soft tokens.

### Test C — LCF-X latent cache communication

The Sharer encodes its own partition; its KV cache is pooled into a fixed-size,
position-free summary, fused into the Receiver's cache (via the bottleneck
fuser), and the Receiver answers from its own partition + that latent summary —
no text message, no token-position alignment.

```bash
.venv/bin/python scripts/run_lcfx.py \
  --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct \
  --receiver-model Qwen/Qwen2.5-0.5B-Instruct \
  --num-spans 4 \
  --output-dir results/lcfx_hotpotqa_zeroinit
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--sharer-model` / `--receiver-model` | `Qwen/Qwen3-0.6B` each | Independent frozen pair (may differ in size/arch). |
| `--fuser-checkpoint` | (none) | Trained fuser weights. **Omit → zero-init fuser.** |
| `--allow-model-mismatch` | off | Load a checkpoint whose stamped Sharer/Receiver differ from the pair (otherwise a hard error). |
| `--bottleneck-dim` | `128` | Joint KV bottleneck width (must match the checkpoint). |
| `--num-spans` | `1` | Sharer pooling spans (position-free summary granularity). |
| `--max-new-tokens` | `64` | Receiver answer budget. |

**Plumbing-correctness check (MVP):** with no `--fuser-checkpoint` the fuser is
**zero-initialized**, so LCF-X adds a zero residual — the Receiver answers from
its own partition only, i.e. the no-communication floor. Training the fuser is
what moves accuracy above that floor.

```bash
.venv/bin/python scripts/run_lcfx.py --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
  --num-examples 10 --output-dir results/lcfx_zeroinit            # zero-init fuser (floor)
```

### Test C (training) — train the LCF-X fuser

Frozen Sharer/Receiver; only the fuser learns (teacher-forced next-token CE on the
gold answer, with the Sharer summary fused into the Receiver's prompt cache).

```bash
.venv/bin/python scripts/train_lcfx_fuser.py \
  --train-data data/tasks/hotpotqa_bridge_cross.jsonl \
  --num-spans 4 --epochs 3 --lr 1e-3 \
  --checkpoint-out data/checkpoints/lcfx/fuser.pt

# then evaluate the trained fuser
.venv/bin/python scripts/run_lcfx.py \
  --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
  --num-spans 4 --fuser-checkpoint data/checkpoints/lcfx/fuser.pt \
  --output-dir results/lcfx_trained
```

> Note: `hotpotqa_bridge_cross.jsonl` is the validation file; using it for both
> train and eval is fine for a smoke test but leaks for real numbers — prepare a
> separate train split (or instruction data) for reportable results.

**Partial runs and batch control.** Two independent ways to shrink a run:

| Flag | Effect |
|------|--------|
| `--max-train-examples N` | Train on only the first `N` examples (smaller dataset). |
| `--max-steps N` | Stop after `N` optimizer steps (regardless of dataset size). |
| `--batch-size B` | Per-step micro-batch (memory-bound; ~4–8 for Qwen3-0.6B on 24 GB). |
| `--grad-accum A` | Micro-batches per optimizer step. **Effective batch = `B × A`.** |

```bash
# Partial run, direct script: 2k examples, effective batch 32, short warmup
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train_lcfx_fuser.py \
  --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --eval-data  data/tasks/hotpotqa_bridge_val.jsonl \
  --sharer-model Qwen/Qwen3-0.6B --receiver-model Qwen/Qwen3-0.6B \
  --bottleneck-dim 128 --num-spans 4 \
  --max-train-examples 2000 --batch-size 8 --grad-accum 4 --warmup 50 \
  --eval-max-examples 200 --eval-every 20 --save-every 20 \
  --checkpoint-out data/checkpoints/lcfx/fuser_partial.pt
```

Or via the Makefile (same knobs as overrides):

```bash
make lcfx-train MAX_TRAIN=2000 BATCH_SIZE=8 GRAD_ACCUM=4   # fewer samples
make lcfx-train MAX_STEPS=50   BATCH_SIZE=8 GRAD_ACCUM=4   # fewer optimizer steps
```

**Training progress is persisted** (survives the session). Every run writes a
`training_log.jsonl` next to its checkpoint — one JSON object per event: a
`start` record with the full config, a `train` record per `--log-every`
(`opt_step`, `train_loss`, `lr`), an `eval` record per eval (`eval_loss`, plus
`em`/`f1` when enabled, and a `best` flag), and an `end` summary. Override the
path with `--log-file`. Plot/inspect e.g.:

```bash
python -c "import json;[print(json.loads(l)) for l in open('data/checkpoints/lcfx/.../training_log.jsonl')]"
```

The teacher-forced **eval loss alone hides generation failures** (e.g. the model
not emitting EOS after the answer — loss keeps dropping while EM collapses). Add
generation-based EM/F1 to each eval with `--eval-em-examples N` (Makefile
`EVAL_EM=N`; `0` = off, the default, since generation is slower):

```bash
make lcfx-train EVAL_EM=100   # report EM/F1 on 100 eval examples each eval
make eiyw-train EVAL_EM=100
```

With `EVAL_EM > 0`, each eval also writes an **`eval_predictions_brief.tsv`**
beside the checkpoint — the same `id`/`em`/`f1`/`prediction`/`answer` TSV as the
inference `predictions_brief.tsv`, but for the dev set, overwritten each eval so
it always holds the latest snapshot. Handy for watching *what* the model predicts
as training progresses (e.g. spotting answers that are correct but don't stop).

### Test D — EIYW (our receiver-anchored method)

The Sharer encodes its own partition; its KV cache is translated into **K soft
pseudo-tokens, conditioned on the Receiver's question**, and **prepended to the
Receiver's input embeddings**. The Receiver then answers from its own partition
plus those soft tokens — no text message, and (unlike LCF-X) the Sharer's
information is added at the Receiver's *input* rather than broadcast into its KV
cache. See [../prompt/EIYW_IMPLEMENTATION_PLAN.md](../prompt/EIYW_IMPLEMENTATION_PLAN.md).

```bash
# Train the translator, then evaluate it.
.venv/bin/python scripts/train_eiyw.py \
  --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --num-soft-tokens 16 --num-spans 4 \
  --checkpoint-out data/checkpoints/eiyw/translator.pt

.venv/bin/python scripts/run_eiyw.py \
  --dataset data/tasks/hotpotqa_bridge_val.jsonl \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --num-soft-tokens 16 --translator-checkpoint data/checkpoints/eiyw/translator.pt \
  --output-dir results/eiyw_trained
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--sharer-model` / `--receiver-model` | `Qwen/Qwen3-0.6B` each | Independent frozen pair (may differ in size/arch). |
| `--translator-checkpoint` | (none) | Trained translator weights. **Omit → gentle-init translator (plumbing only).** |
| `--num-soft-tokens` | `16` | K soft tokens prepended to the Receiver input (must match the checkpoint). |
| `--bottleneck-dim` | `128` | Translator latent width (must match the checkpoint). |
| `--num-spans` | `4` | Sharer pooling spans. |
| `--allow-model-mismatch` | off | Load a checkpoint whose stamped Sharer/Receiver differ (otherwise a hard error). |
| `--max-new-tokens` | `64` | Receiver answer budget. |

**Plumbing-correctness check.** EIYW has no zero-init==Receiver-only equality
(zero-valued soft tokens are still extra positions, and exactly-zero embeddings
are numerically degenerate — the init is therefore a *gentle* small-scale start).
With no `--translator-checkpoint` the translator is gentle-initialized, so the run
is a plumbing pass (soft tokens near zero), not the method.

Memory note: EIYW's prompt forward is **grad-enabled** (the soft tokens lead the
sequence, so every prompt activation depends on them) — heavier than LCF-X, which
can prefill the prompt under `no_grad`. Gradient checkpointing on the frozen
Receiver is enabled automatically; lower `--batch-size` / raise `--grad-accum` as
needed. The same `MAX_TRAIN` / `MAX_STEPS` / `BATCH_SIZE` / `GRAD_ACCUM` knobs
work via `make eiyw-train`.

---

## Reading the results

After a run, the headline numbers are in `<output-dir>/metrics.json`:

```json
{
  "accuracy": 0.20,
  "scores": { "em": 0.20, "f1": 0.33 },
  "token_counts": { "average": { "sharer_output": 33.0, "receiver_output": 57.9 } }
}
```

- `scores.em` / `scores.f1` — HotpotQA exact-match and token-F1.
- `token_counts.average.sharer_output` — the actual mean message length (the
  realized communication cost).
- Per-example traces (prompts, message, prediction) are in `predictions.jsonl`
  for debugging.

> The small smoke runs so far (Qwen2.5-0.5B, 10 examples) are for plumbing
> validation only; report numbers from a larger sample and a stronger Receiver
> (e.g. Qwen3-0.6B) over `--num-examples 200`+.

---

## Suggested comparison sweep

```bash
DATA=data/tasks/hotpotqa_bridge_cross.jsonl
M=Qwen/Qwen2.5-0.5B-Instruct
N=200

.venv/bin/python scripts/run_single_model.py --dataset $DATA \
  --model $M --num-examples $N --output-dir results/single_model
.venv/bin/python scripts/run_text_to_text.py --dataset $DATA \
  --sharer-model $M --num-examples $N --output-dir results/t2t
```

Then compare the `scores` blocks across the `metrics.json` files (single-model
baseline vs. each communication method).
