# Project Structure

This project studies **flexible communication between language models** for
multi-agent collaboration — how one model (the *Sharer*) can pass information to
another (the *Receiver*) so the Receiver answers better than it could alone, when
the two agents hold **different context** and may differ in **model size /
architecture / tokenizer**. It compares three methods under a single,
standardized experiment/output contract:

- **T2T (text-to-text)** — natural-language message baseline (*implemented*).
- **LCF-X (cross-context Latent Cache Flow)** — position-free latent cache
  communication; the existing-paper precedent for different-context agents
  (*implemented*).
- **EIYW (`methods/eiyw/`)** — our contribution: reuses LCF-X's pooling design
  unmodified on the Sharer (up to `k` slots per layer, optionally adaptive in
  the context length); the slots then cross-attend into the Receiver's raw
  context *embeddings* (a table lookup + sinusoidal PE — no Receiver forward
  pass) and become `k·L_s` soft tokens in the Receiver's embedding space,
  prepended to its input (*implemented*).
- **KVTok (`methods/kvtok/`)** — reuses only LCF-X's Stage 1 (within-span
  attention) on the Sharer, then uses factorized feature-axis and token-axis
  linear projections to produce `--num-tokens` slots. Processed Receiver
  embeddings condition those slots without an initial Receiver prefill/cache;
  an EIYW-v2-style head emits prepended soft tokens (*implemented*).
- **XKV (`methods/xkv/`)** — pools both Sharer and Receiver KV with vectorized
  learned-query attention (`N → k`, no spans, `k=2` by default), applies the
  current learned `L_s → L_r` layer-axis
  map, translates the concatenated layer summaries with configurable attention
  blocks, then uses
  raw Receiver positions as queries in a shared cross-attention decoder and
  emits a per-head-scaled, position-specific Receiver-native KV residual
  through two shared K/V up-projections, without layer/type Gumbel gates
  (*implemented*).

**Scope decision:** shared-context, token-position-aligned cache-transfer methods
(**C2C** and the original shared-context **LCF**) are **out of scope**. They
assume both agents encode the *same* prompt and fuse at aligned positions, which
is exactly the rigidity (same context, alignable tokenization) we want to avoid.
See [prompt/BASELINES_SUMMARY.md](prompt/BASELINES_SUMMARY.md) for the rationale.

The first implemented method is **T2T**: the Sharer reads its information and
writes a short natural-language message; the Receiver reads its own information
plus that message and produces the answer. T2T establishes the end-to-end
pipeline (load models → load task → run Sharer → run Receiver → score → save)
that LCF-X and the new method reuse.

## Communication setting: cross-context

All three methods target the **cross-context** setting — Sharer and Receiver see
*different* evidence, so the Receiver genuinely needs the Sharer's information to
answer. This is the realistic multi-agent collaboration case and the LCF-X
benchmark setting.

The first dataset is **partitioned HotpotQA-bridge**: each bridge question has 10
paragraphs (2 supporting + 8 distractors), split into two disjoint 5-paragraph
partitions, each holding one supporting paragraph and four distractors. The
Sharer gets one partition, the Receiver the other — so neither side can answer
alone.

## Directory layout

```text
EchoInYourWords/
├── PROJECT_STRUCTURE.md      # this file
├── README.md
├── prompt/                   # PLANNING DOCS (design notes, not code)
│   ├── PROJECT_PLAN.md
│   ├── T2T_IMPLEMENTATION_PLAN.md
│   ├── LCF_X_IMPLEMENTATION_PLAN.md # cross-context latent cache communication
│   ├── BASELINES_SUMMARY.md      # context settings + why C2C/LCF are out of scope
│   └── IMPLEMENTATION_INSIGHTS.md# framework / repo-shape decisions
├── methods/                  # METHOD IMPLEMENTATIONS (one package per method)
│   ├── common/                   # SHARED experiment layer (contract, scoring, IO)
│   ├── t2t/                      # text-to-text baseline (implemented)
│   ├── lcfx/                     # cross-context Latent Cache Flow (implemented)
│   ├── eiyw/                     # symmetric-pool, multi-soft-token method (implemented)
│   ├── kvtok/                    # Stage-1-only-pooling baseline (implemented)
│   └── xkv/                      # dual-cache KV-update translator (implemented)
├── scripts/                  # RUNNABLE ENTRYPOINTS + cache config
│   ├── README.md                 # how to run each test/experiment (commands)
│   ├── _hf_cache.py              # points HF_HOME at data/hf_cache
│   ├── prepare_hotpotqa_bridge.py# builds cross-context task JSONL
│   ├── run_text_to_text.py       # runs the T2T Sharer→Receiver pipeline
│   ├── run_lcfx.py / train_lcfx_fuser.py   # LCF-X eval / fuser training
│   ├── run_eiyw.py / train_eiyw.py         # EIYW eval / translator training
│   ├── run_kvtok.py / train_kvtok.py       # KVTok eval / translator training
│   ├── run_xkv.py / train_xkv.py             # XKV eval / translator training
│   └── run_single_model.py       # single-model inference baseline (full context)
├── data/
│   ├── hf_cache/                 # HuggingFace cache (its native single-root layout)
│   │   ├── hub/                  #   all Hub downloads — models--… and datasets--…
│   │   ├── datasets/             #   built (arrow) datasets
│   │   └── xet/                  #   dedup blob cache
│   ├── tasks/                    # processed/normalized task files (JSONL) — OURS
│   └── checkpoints/              # trained adapters/fusers (future methods)
├── results/                  # run outputs, one directory per run
└── .venv/                    # Python 3.12 environment (see Environment below)
```

> **Note on the word "prompt":** the top-level `prompt/` folder holds *planning
> documents*, not model prompts. The actual model prompts (Sharer / Receiver
> templates) live in code at [methods/common/prompts.py](methods/common/prompts.py).

## The shared layer (`methods/common/`)

The thin experiment layer every method consumes (so methods never import one
another). Import-light — no `torch`.

| File | Responsibility |
|------|----------------|
| [schema.py](methods/common/schema.py) | `Example` — the normalized cross-method data contract. Each method re-exports it under a local alias (`T2TExample` / `LCFXExample` / `EIYWExample`). |
| [scoring.py](methods/common/scoring.py) | HotpotQA/SQuAD `normalize_answer`, `exact_match`, `f1_score`, `score_free_form` (EM/F1). |
| [parsing.py](methods/common/parsing.py) | `parse_choice_answer` — extract a multiple-choice letter (returns raw text for free-form). |
| [prompts.py](methods/common/prompts.py) | **All** prompt builders: T2T's `build_sharer_prompt` / `build_receiver_message_prompt`, the latent methods' `build_sharer_encode_text` / `build_receiver_prompt` plus `build_receiver_encode_text` (used by EIYW/KVTok for Receiver embedding memory), the dedicated `build_single_model_prompt` (single-model baseline), and `format_choices`. |
| [io.py](methods/common/io.py) | Aggregate metrics/latency/tokens, `write_predictions_brief`, and the single `write_run_outputs(...)` used by every method (each binds its own `method=` default). |
| [progress.py](methods/common/progress.py) | `ProgressLogger` — persists training **and eval** progress to JSONL (`training_log.jsonl` beside the checkpoint; `eval_log.jsonl` in the run's output dir) so long runs survive the session. `eval_progress_fields` computes counts / ETA / running accuracy-EM-F1. |

## The T2T method package (`methods/t2t/`)

A model-agnostic library. The core never imports `torch`; the model backend uses
deferred imports, so `import methods.t2t` works without a GPU stack. Scoring,
parsing, the data contract, and IO come from `methods.common` (re-exported here).

| File | Responsibility |
|------|----------------|
| [schema.py](methods/t2t/schema.py) | `T2TExample` — alias of `methods.common.schema.Example`. |
| [model.py](methods/t2t/model.py) | `load_generator(...)` — turns a frozen HF causal LM into a text generator with a token budget; the only torch/transformers dependency. |
| [runner.py](methods/t2t/runner.py) | Per-example execution: `run_t2t_example` (Sharer→Receiver); `T2TResult`, `GenerationOutput`. |

### Prompts: what they do

All prompt builders are centralized in
[methods/common/prompts.py](methods/common/prompts.py) so methods share one copy
and the single-model baseline stays decoupled.

- **Sharer prompt** (`build_sharer_prompt`): tells the Sharer it may hold only
  *part* of the information, and asks it to **relay relevant facts from its own
  context, not guess the final answer**. This matters in cross-context: the
  Sharer cannot answer alone and should communicate evidence rather than
  hallucinate. Shows the Sharer's context, the question, and any choices.
- **T2T receiver prompt** (`build_receiver_message_prompt`): asks the Receiver to
  answer using *its own information plus the Sharer's message*. For
  multiple-choice it requests only the option letter; for free-form a short
  phrase (keeps EM/F1 meaningful).
- **Latent receiver prompt** (`build_receiver_prompt`) + **Sharer encode text**
  (`build_sharer_encode_text`): the LCF-X/EIYW pair — the Receiver answers from
  its own context with no text message (the Sharer's information arrives via the
  fused cache / soft tokens), and the Sharer encodes its own context + question.
- **Single-model prompt** (`build_single_model_prompt`): a *dedicated* prompt for
  the single-model inference baseline (one model answers from a given context),
  deliberately separate so it and the method receiver prompts never disturb each
  other.

### Data contract — `Example`

Every method consumes the same normalized example (`methods.common.schema.Example`),
so datasets are prepared once and reused for fair comparison:

```json
{
  "id": "example-id",
  "question": "Final question or task instruction",
  "choices": ["A", "B", "C", "D"],
  "answer": "A",
  "sharer_context": "Context available to the Sharer",
  "receiver_context": "Context available to the Receiver",
  "metadata": {}
}
```

`choices` is empty for free-form tasks (HotpotQA). In the cross-context setting
the Sharer and Receiver contexts differ (disjoint evidence partitions).

### Output contract — one directory per run

Every run writes four files into its `--output-dir` (under `results/`):

| File | Contents |
|------|----------|
| `predictions.jsonl` | one record per example (prompts, message, output, prediction, answer, `is_correct`, `scores`, `latency`, `token_counts`, `metadata`). |
| `predictions_brief.tsv` | quick-scan companion: one row per example with just `id`, `em`, `f1`, `prediction`, `answer` (no prompts). For eyeballing answer-vs-prediction fast. |
| `metrics.json` | aggregate accuracy, free-form `scores` (EM/F1), and `token_counts` totals/averages. |
| `latency.json` | per-stage and total latency totals/averages, using a uniform cross-method contract: `sharer_ms`, `fuser_ms` (translation/fusion — `0.0` for text-to-text, `>0` for cache/latent methods), `receiver_ms`, `total_ms`. |
| `run_metadata.json` | method, models, dataset, budgets, scoring mode, timestamp. |

Comparison/analysis depends only on these files, never on internal method
classes — so T2T, the receiver baselines, and the latent methods (LCF-X and
EIYW) are compared apples-to-apples.

## The LCF-X and EIYW packages (`methods/lcfx/`, `methods/eiyw/`)

Both mirror the T2T package shape (core import-light; only `model.py` /
`fuser.py` / `translator.py` touch torch, via deferred imports) and consume the
shared `methods.common` layer (the `Example` contract, scoring, parsing, prompt
primitives, IO). Both pool a KV cache with the same paper-exact per-layer
within-span / cross-span learned attention (Fig. 3; self-contained copies per
package); they differ in what gets pooled and how the result reaches the
Receiver:

- **LCF-X** (`fuser.py`, `cache.py`, `spans.py`): pools each Sharer layer's KV
  independently (within-span then cross-span attention, two learned queries per
  layer) into one position-free `[B,H,1,D]` summary per layer, layer-aligned to
  the Receiver (a linear layer-axis bridge only when `L_s ≠ L_r`), concatenates
  it with the Receiver's own real per-position KV, and fuses the result as a
  **gated KV residual added into the Receiver's own cache** (per-layer K/V
  gates: Gumbel-sigmoid in training, hard `{0,1}` at inference).
  Architecture/training detail:
  [prompt/LCF_X_IMPLEMENTATION_PLAN.md](prompt/LCF_X_IMPLEMENTATION_PLAN.md).
- **EIYW** (`translator.py`, `model.py`, `spans.py`): pools only the
  **Sharer's** context per layer (up to `--queries-per-layer` k position-free
  `[B,H,1,D]` slots per layer, Sharer geometry — the identical pool to
  LCF-X's, so any gain over LCF-X is attributable to the translator, not a
  richer pool; `--adaptive-k-tokens w` scales the *active* k with the
  Sharer's length, `clamp(ceil(N_s/w), 1, k)`). A `Linear` projects the
  flattened `K*‖V*` slots to `--d-model` width with learned layer/query
  tags; the Receiver's own context enters as raw input **embeddings** (a
  table lookup — no Receiver forward pass) projected to the same width plus
  **sinusoidal PE** (raw embeddings carry no order; the pooled slots are
  position-free and need none). `--num-blocks` decoder-shaped pre-LN blocks
  — self-attention over the slots, **cross-attention into the Receiver
  memory** (cost `k·L_s × N_r`, no `N_r²` term), MLP — refine the slots, and
  an RMS-calibrated head emits them as `k·L_s` **soft tokens prepended to
  the Receiver's input embeddings** (`inputs_embeds`), not a KV delta.
  Architecture/training detail:
  [prompt/EIYW_IMPLEMENTATION_PLAN.md](prompt/EIYW_IMPLEMENTATION_PLAN.md) (the
  original receiver-anchored design; the current iteration — see
  [README.md](README.md#run-eiyw-our-method)).

Both train only their adapter (fuser / translator) with frozen base models,
write **model-stamped** checkpoints (hard-error on a pair mismatch unless
`--allow-model-mismatch`), and support heterogeneous Sharer/Receiver pairs.

## The KVTok package (`methods/kvtok/`)

A third baseline against LCF-X/EIYW, following the same
per-package self-containment convention (own `model.py`/`cache.py`/etc. rather
than importing across method packages). It reuses LCF-X's Stage 1
(within-span attention) *only* — Stage 2 (across-span attention) is deleted
entirely — and is applied only to the Sharer's cache. It preserves the
`layers*spans` summary axis, projects each `K‖V` feature row into `d_model`
with shared weights, then linearly downsamples that position axis to
`--num-tokens`. The Receiver contributes only embedding-table outputs (plus
sinusoidal PE and LayerNorm), never an initial KV cache. Cross-attention and a
calibrated output head produce soft tokens prepended to `inputs_embeds`, the
same consumption mechanism as EIYW. `--num-spans`, `--num-tokens`,
`--d-model`, and the resolved `--num-heads` are part of
`arch_config()` and enforced by `load_translator`'s mismatch check. See
[README.md](README.md#run-kvtok-baseline).

## Environment

- Python **3.12** virtualenv at `.venv`. Tools are invoked as plain `python ...`
  (not `.venv/bin/python`) — activate the venv yourself first
  (`source .venv/bin/activate`) so `python` resolves to it.
- **torch must be the CUDA 11.8 build** (`torch==2.7.1+cu118`): the machine's
  NVIDIA driver (515.57 / CUDA 11.7) is too old for the default CUDA-13 wheels.
  Install with `--index-url https://download.pytorch.org/whl/cu118`.
- The Hugging Face cache uses its **native single-root layout**, pointed at
  `data/hf_cache` by [scripts/_hf_cache.py](scripts/_hf_cache.py) (which sets
  `HF_HOME`). All Hub downloads share one `hub/`, with models and datasets told
  apart by their `models--` / `datasets--` name prefix; built datasets live in
  `hf_cache/datasets/`. Our own processed task files stay separate in
  `data/tasks/`. No need to set `HF_HOME` manually when using the scripts.

See [scripts/README.md](scripts/README.md) for the exact commands to prepare
data and run each experiment.
