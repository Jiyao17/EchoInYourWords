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

- Run everything with plain `python ...` — activate your own environment first
  (e.g. `source .venv/bin/activate`) so `python` resolves to the right interpreter.
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
| `prepare_musique.py` / `prepare_qasc.py` / `prepare_2wikimultihop.py` / `prepare_ropes.py` / `prepare_strategyqa.py` | Same idea for the other 5 datasets (see [../README.md](../README.md#available-datasets-dataset) for how each splits). |
| `run_text_to_text.py` | Run the T2T Sharer→Receiver pipeline. |
| `run_lcfx.py` | Run the LCF-X latent cache communication method (Sharer cache → pooled summary → fused into Receiver). |
| `run_eiyw.py` | Run EIYW (Sharer context → LCF-X's pooling unmodified → k·L_s slots cross-attend into the Receiver's own context *embeddings* (lookup only, no Receiver forward) → soft tokens in the Receiver's embedding width → prepended to the Receiver's input). |
| `run_kvtok.py` | Run KVTok (Sharer cache → Stage-1 pool → factorized feature/token-axis projection → Receiver embedding memory, with no initial Receiver forward/cache → `--num-tokens` soft tokens prepended to the Receiver's input). |
| `run_xkv.py` | Run XKV (dual expanded LCF-X `N→P→2` cache pools → layer map → bottleneck/attention translation → Receiver KV residual). |
| `run_single_model.py` | Run the single-model inference baseline (one model answers from the full context — no Sharer, no communication). |
| `train_lcfx_fuser.py` | Train the LCF-X fuser (frozen Sharer/Receiver; only the fuser learns). |
| `train_eiyw.py` | Train the EIYW translator (frozen Sharer/Receiver; only the translator learns). |
| `train_kvtok.py` | Train the KVTok translator (frozen Sharer/Receiver; only the translator learns). |
| `train_xkv.py` | Train the XKV translator (frozen Sharer/Receiver; only the translator learns). |
| `_hf_cache.py` | Helper imported by the others; not run directly. |

---

## Step 1 — Prepare the dataset (once)

Builds normalized cross-context examples and writes them to
`data/tasks/hotpotqa_bridge_cross.jsonl`. Filters to `bridge` questions and
splits the 10 paragraphs into two 5-paragraph partitions (1 supporting + 4
distractors each); deterministic given `--seed`.

```bash
python scripts/prepare_hotpotqa_bridge.py --num-examples 200
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
python scripts/run_text_to_text.py \
  --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
  --sharer-model unsloth/Llama-3.2-1B-Instruct \
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
python scripts/run_single_model.py \
  --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
  --model unsloth/Llama-3.2-1B-Instruct \
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

The Sharer encodes its own partition; each layer of its KV cache is pooled
(within-span then cross-span attention) into one position-free `[H,1,D]`
summary, fused into the Receiver's cache (via the bottleneck fuser), and the
Receiver answers from its own partition + that latent summary — no text
message, no token-position alignment.

```bash
python scripts/run_lcfx.py \
  --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
  --sharer-model Qwen/Qwen3-0.6B \
  --receiver-model Qwen/Qwen3-0.6B \
  --output-dir results/lcfx_hotpotqa_zeroinit
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--sharer-model` / `--receiver-model` | `Qwen/Qwen3-0.6B` each | Independent frozen pair (may differ in size/arch). |
| `--fuser-checkpoint` | (none) | Trained fuser weights. **Omit → zero-init fuser.** |
| `--allow-model-mismatch` | off | Load a checkpoint whose stamped Sharer/Receiver differ from the pair (otherwise a hard error). |
| `--bottleneck-dim` | `128` | Joint KV bottleneck width (must match the checkpoint). |
| `--span-scheme` | `paragraphs` | Span partition: natural blank-line blocks (paper's scheme) or `fixed` token chunks. Any trained checkpoint accepts either (paper E.4). |
| `--num-spans` | `5` | Span count for `--span-scheme fixed`. |
| `--max-new-tokens` | `64` | Receiver answer budget. |

**Plumbing-correctness check (MVP):** with no `--fuser-checkpoint` the fuser is
**zero-initialized**, so LCF-X adds a zero residual — the Receiver answers from
its own partition only, i.e. the no-communication floor. Training the fuser is
what moves accuracy above that floor.

```bash
python scripts/run_lcfx.py --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
  --num-examples 10 --output-dir results/lcfx_zeroinit            # zero-init fuser (floor)
```

### Test C (training) — train the LCF-X fuser

Frozen Sharer/Receiver; only the fuser learns (teacher-forced next-token CE on the
gold answer, with the Sharer summary fused into the Receiver's prompt cache).

```bash
python scripts/train_lcfx_fuser.py \
  --train-data data/tasks/hotpotqa_bridge_cross.jsonl \
  --epochs 3 --lr 1e-3 \
  --checkpoint-out data/checkpoints/lcfx/fuser.pt

# then evaluate the trained fuser
python scripts/run_lcfx.py \
  --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
  --fuser-checkpoint data/checkpoints/lcfx/fuser.pt \
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
python scripts/train_lcfx_fuser.py \
  --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --eval-data  data/tasks/hotpotqa_bridge_val.jsonl \
  --sharer-model Qwen/Qwen3-0.6B --receiver-model Qwen/Qwen3-0.6B \
  --bottleneck-dim 128 \
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
make kvtok-train EVAL_EM=100
```

With `EVAL_EM > 0`, each eval also writes an **`eval_predictions_brief.tsv`**
beside the checkpoint — the same `id`/`em`/`f1`/`prediction`/`answer` TSV as the
inference `predictions_brief.tsv`, but for the dev set, overwritten each eval so
it always holds the latest snapshot. Handy for watching *what* the model predicts
as training progresses (e.g. spotting answers that are correct but don't stop).

### Test D — EIYW (our method)

**Current iteration (v2): EIYW keeps LCF-X's paper-exact per-layer pooling on
the Sharer and replaces everything after it with a cross-attention translator
whose output is input tokens rather than a KV-cache residual.** The Sharer's
context is pooled per layer (up to `--queries-per-layer` k position-free
`[B,H,1,D]` slots per layer, Sharer geometry — the identical pool to LCF-X's,
so any gain over LCF-X is attributable to the translator). A `Linear`
projects the flattened `K*‖V*` slots to `--d-model` width with learned
layer/query tags; the Receiver's own context enters as raw input *embeddings*
(a table lookup — no Receiver forward pass) projected to the same width plus
sinusoidal PE (raw embeddings carry no order). `--num-blocks` pre-LN blocks
(self-attn over the slots, cross-attn into the Receiver memory, MLP) refine
the slots, and an RMS-calibrated head emits `k·L_s` soft tokens — count
independent of both context lengths, or adaptive in the Sharer's length with
`--adaptive-k-tokens`. The tokens are prepended to the Receiver's input
embeddings, so the Receiver attends to them like any other input position
(not reshaped into a KV delta, unlike LCF-X). See
[../prompt/EIYW_IMPLEMENTATION_PLAN.md](../prompt/EIYW_IMPLEMENTATION_PLAN.md)
for the original receiver-anchored design.

```bash
# Train the translator, then evaluate it.
python scripts/train_eiyw.py \
  --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --sharer-model unsloth/Llama-3.2-1B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --d-model 512 --queries-per-layer 1 --num-blocks 1 \
  --checkpoint-out data/checkpoints/eiyw/translator.pt

python scripts/run_eiyw.py \
  --dataset data/tasks/hotpotqa_bridge_val.jsonl \
  --sharer-model unsloth/Llama-3.2-1B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --translator-checkpoint data/checkpoints/eiyw/translator.pt \
  --output-dir results/eiyw_trained
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--sharer-model` / `--receiver-model` | `Qwen/Qwen3-0.6B` each | Independent frozen pair (may differ in size/arch). |
| `--translator-checkpoint` | (none) | Trained translator weights. **Omit → gentle-init translator (plumbing pass close to, but not guaranteed identical to, the Receiver-only floor).** |
| `--span-scheme` | `paragraphs` | Sharer pooling span partition: natural blank-line blocks or `fixed` token chunks (same knob as LCF-X's; any checkpoint accepts either). |
| `--num-spans` | `5` | Span count for `--span-scheme fixed`. |
| `--queries-per-layer` | `1` | Max Sharer pool queries per layer k (soft tokens = k·L_s). Zero-init only — ignored with `--translator-checkpoint`, which uses its own stamped value. |
| `--adaptive-k-tokens` | `0` (off) | w>0: active k = `clamp(ceil(N_s/w), 1, k)` — scales the token budget with the Sharer's context length. Zero-init only — the checkpoint's stamped value wins. |
| `--num-blocks` | `1` | Fusion depth (self-attn + cross-attn + MLP blocks). Zero-init only — ignored with `--translator-checkpoint`. |
| `--d-model` | `512` (receiver-independent) | Translator width (`half`/`hidden` as sweep options). Zero-init only — ignored with `--translator-checkpoint`. |
| `--allow-model-mismatch` | off | Load a checkpoint whose stamped Sharer/Receiver differ (otherwise a hard error). |
| `--max-new-tokens` | `64` | Receiver answer budget. |

**Plumbing-correctness check.** EIYW has no zero-init==Receiver-only
*equality* the way LCF-X's zero-init fuser does: an exact-zero soft embedding
is a numerical degeneracy (it makes the frozen Receiver's backward pass
non-finite, `0 * inf`), so the translator instead uses a *gentle* small-scale
init. With no `--translator-checkpoint` the translator is gentle-initialized,
so the run is a plumbing pass (soft tokens near zero), not an exact
Receiver-only equality check.

Memory note: EIYW's prompt forward is **grad-enabled** end to end (the soft
tokens lead the sequence, so every prompt activation depends on them) —
heavier than LCF-X, which can prefill the prompt under `no_grad`. Gradient
checkpointing on the frozen Receiver is enabled automatically; lower
`--batch-size` / raise `--grad-accum` as needed. The same `MAX_TRAIN` /
`MAX_STEPS` / `BATCH_SIZE` / `GRAD_ACCUM` knobs work via `make eiyw-train`.

### Test E — KVTok (baseline)

**KVTok reuses LCF-X's Fig. 3 Stage 1 (within-span attention) on the Sharer
only and drops Stage 2 (across-span attention).** It preserves the
`layers * num_spans` position axis, projects each flattened K/V feature row
with a shared `2·H·D → d_model` linear, then applies a shared token-axis
`layers·spans → num_tokens` linear. This factorization avoids the incorrect
billion-parameter dense Cartesian map. The Receiver supplies embedding-table
outputs only — processed by projection, sinusoidal PE, and LayerNorm — with no
initial Receiver forward/cache. One cross-attention step and the EIYW-v2
calibrated head produce soft tokens prepended to the Receiver's generation
prompt embeddings.

```bash
# Train the translator, then evaluate it.
python scripts/train_kvtok.py \
  --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --sharer-model unsloth/Llama-3.2-1B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --num-spans 5 --num-tokens 8 --d-model 512 --num-heads 8 \
  --checkpoint-out data/checkpoints/kvtok/translator.pt

python scripts/run_kvtok.py \
  --dataset data/tasks/hotpotqa_bridge_val.jsonl \
  --sharer-model unsloth/Llama-3.2-1B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --num-spans 5 --num-tokens 8 --d-model 512 --num-heads 8 \
  --translator-checkpoint data/checkpoints/kvtok/translator.pt \
  --output-dir results/kvtok_trained
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--sharer-model` / `--receiver-model` | `Qwen/Qwen3-0.6B` each | Independent frozen pair (may differ in size/arch). |
| `--translator-checkpoint` | (none) | Trained translator weights. **Omit → gentle-init translator (plumbing pass close to the Receiver-only floor).** |
| `--num-spans` | `5` | Sharer Stage-1 pooling spans (must match the checkpoint; same knob/default as LCF-X/EIYW). |
| `--num-tokens` | `8` | Soft tokens the down-projection outputs (must match the checkpoint — baked into its weight shape). |
| `--d-model` | `512` | Translator width (must match the checkpoint). |
| `--num-heads` | `8` | Cross-attention heads (must match after divisor-safe resolution). |
| `--allow-model-mismatch` | off | Load a checkpoint whose stamped Sharer/Receiver/arch differ (otherwise a hard error). |
| `--max-new-tokens` | `64` | Receiver answer budget. |

**Plumbing-correctness check.** Same rationale as EIYW: an exact-zero soft
embedding is a numerical degeneracy, so the translator uses a *gentle*
small-scale init instead. With no `--translator-checkpoint` the run is a
plumbing pass (soft tokens near zero), not an exact Receiver-only equality
check.

Unlike EIYW, `--num-spans`/`--num-tokens` are baked into the translator's
weight shapes at construction time. Width and resolved head count are also
part of its architecture, so all four values must be re-supplied identically
between `train_kvtok.py` and `run_kvtok.py` —
`load_translator` hard-errors on a mismatch. Same memory note as EIYW
(grad-enabled prompt forward; gradient checkpointing on automatically). The
same `MAX_TRAIN` / `MAX_STEPS` / `BATCH_SIZE` / `GRAD_ACCUM` knobs work via
`make kvtok-train`.

### Test F — XKV dual-cache KV update

XKV runs the frozen Sharer and Receiver prefix once, pools both caches with an
expanded LCF-X `N → P → 2` process, and applies the learned layer-axis map
when `L_s != L_r`. The two K/V summaries per layer and side double the joint
bottleneck input width while its output remains `[B,L_r,d]`. Raw Receiver K/V
positions then query the translated layer memory through shared
cross-attention. Two shared K/V up-projections produce distinct gated updates
at different cache positions.
For `DATASET=ropes`, `make xkv-train` defaults to batch 32 / accumulation 8;
full sweeps halve/double those values through the existing model-size tiers.

```bash
python scripts/train_xkv.py \
  --train-data data/tasks/hotpotqa_bridge_train.jsonl \
  --sharer-model unsloth/Llama-3.2-1B-Instruct \
  --receiver-model Qwen/Qwen3-0.6B \
  --d-model eighth --num-blocks 1 --pool-slots 2 \
  --checkpoint-out data/checkpoints/xkv/translator.pt

python scripts/run_xkv.py \
  --dataset data/tasks/hotpotqa_bridge_val.jsonl \
  --sharer-model unsloth/Llama-3.2-1B-Instruct \
  --receiver-model Qwen/Qwen3-0.6B \
  --d-model eighth --num-blocks 1 --pool-slots 2 \
  --translator-checkpoint data/checkpoints/xkv/translator.pt \
  --output-dir results/xkv
```

The Makefile equivalents are `make xkv`, `make xkv-train`, `make xkv-eval`,
and `make exp METHOD=xkv DATASET=<name>`.

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

> The small smoke runs so far (Qwen3-0.6B, 10 examples) are for plumbing
> validation only; report numbers from a larger sample and a stronger Receiver
> (e.g. `unsloth/Llama-3.2-1B-Instruct`) over `--num-examples 200`+.

---

## Suggested comparison sweep

```bash
DATA=data/tasks/hotpotqa_bridge_cross.jsonl
M=Qwen/Qwen3-0.6B
N=200

python scripts/run_single_model.py --dataset $DATA \
  --model $M --num-examples $N --output-dir results/single_model
python scripts/run_text_to_text.py --dataset $DATA \
  --sharer-model $M --num-examples $N --output-dir results/t2t
```

Then compare the `scores` blocks across the `metrics.json` files (single-model
baseline vs. each communication method).
