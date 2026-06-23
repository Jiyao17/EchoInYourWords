# Running Experiments

This folder holds the runnable entrypoints. Every script writes the same
standardized output files (`predictions.jsonl`, `metrics.json`, `latency.json`,
`run_metadata.json`) into its `--output-dir`, so all runs are directly
comparable. For the project layout and data/output contracts, see
[../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md).

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
| `run_receiver_baseline.py` | Run no-Sharer reference baselines (question-only / half / full context). |
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
selected because HotpotQA answers are free-form). They form the LCF-X comparison
ladder plus the T2T method.

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

If `--receiver-model` is omitted it reuses the Sharer model (single-model setup).

| Flag | Default | Meaning |
|------|---------|---------|
| `--sharer-model` | (required) | Sharer model id/path. |
| `--receiver-model` | = sharer | Receiver model id/path. |
| `--sharer-max-new-tokens` | `128` | **Communication budget** — max tokens in the Sharer message. |
| `--receiver-max-new-tokens` | `64` | Max tokens in the Receiver answer. |
| `--num-examples` | all | Cap the number of examples. |
| `--scoring` | `auto` | `auto` / `choice` / `free_form`. |
| `--device`, `--dtype` | auto / `auto` | Device and weight dtype. |

**Limiting the shared message (LCF-X-style budget):** set
`--sharer-max-new-tokens`, e.g. `--sharer-max-new-tokens 32`. The *actual*
message length is also recorded per example as `token_counts.sharer_output`, so
you can sweep the budget and plot accuracy vs. message length.

### Test B — Receiver baselines (no Sharer)

Reference points that bracket the T2T result. Same Receiver, varying how much
evidence it sees, with **no Sharer message**.

```bash
# Half-context: Receiver sees only its own 5 of 10 paragraphs
.venv/bin/python scripts/run_receiver_baseline.py \
  --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
  --context half \
  --receiver-model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir results/receiver_half_context
```

The `--context` flag selects the baseline:

| `--context` | Receiver sees | Paragraphs | Role |
|-------------|---------------|-----------|------|
| `question_only` | nothing | 0 / 10 | lower bound (no evidence) |
| `half` | its own partition | 5 / 10 | "no Sharer" cross-context floor |
| `full` | both partitions | 10 / 10 | upper bound (all evidence) |

Other flags mirror Test A (`--receiver-max-new-tokens` default `64`,
`--num-examples`, `--scoring`, `--device`, `--dtype`).

Expected ordering of accuracy: `question_only` ≤ `half` ≤ `T2T` ≤ `full`. T2T's
value is how much of the gap between *half* and *full* it recovers using only a
short text message.

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

.venv/bin/python scripts/run_receiver_baseline.py --dataset $DATA --context question_only \
  --receiver-model $M --num-examples $N --output-dir results/receiver_qonly
.venv/bin/python scripts/run_receiver_baseline.py --dataset $DATA --context half \
  --receiver-model $M --num-examples $N --output-dir results/receiver_half
.venv/bin/python scripts/run_receiver_baseline.py --dataset $DATA --context full \
  --receiver-model $M --num-examples $N --output-dir results/receiver_full
.venv/bin/python scripts/run_text_to_text.py --dataset $DATA \
  --sharer-model $M --num-examples $N --output-dir results/t2t
```

Then compare the `scores` blocks across the four `metrics.json` files.
