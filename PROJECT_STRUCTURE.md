# Project Structure

This project studies **communication between language models** — how one model
(the *Sharer*) can pass information to another (the *Receiver*) so the Receiver
answers better than it could alone. It compares several communication methods
(text-to-text, cache-to-cache, latent cache flow, and a new receiver-anchored
method) under a single, standardized experiment/output contract.

The first implemented method is **Text-to-Text (T2T)**: the Sharer reads its
information and writes a short natural-language message; the Receiver reads its
own information plus that message and produces the answer. T2T is the easiest
baseline to debug and establishes the end-to-end pipeline (load models → load
task → run Sharer → run Receiver → score → save) that later methods reuse.

## Two communication settings

- **Shared-context**: Sharer and Receiver see the *same* information. The
  message acts as background / distilled reasoning. (Used by C2C and
  shared-context LCF.)
- **Cross-context** (more general): Sharer and Receiver see *different* evidence.
  The Receiver genuinely needs the Sharer's information to answer. This is the
  LCF-X agent-communication setting, and the current focus.

The first dataset is **partitioned HotpotQA-bridge** (the LCF-X cross-context
benchmark): each bridge question has 10 paragraphs (2 supporting + 8
distractors), split into two disjoint 5-paragraph partitions, each holding one
supporting paragraph and four distractors. The Sharer gets one partition, the
Receiver the other — so neither side can answer alone. See
[prompt/BASELINES_SUMMARY.md](prompt/BASELINES_SUMMARY.md) for the full
paper-to-setting mapping.

## Directory layout

```text
EchoInYourWords/
├── PROJECT_STRUCTURE.md      # this file
├── README.md
├── prompt/                   # PLANNING DOCS (design notes, not code)
│   ├── PROJECT_PLAN.md
│   ├── T2T_IMPLEMENTATION_PLAN.md
│   ├── BASELINES_SUMMARY.md      # how each paper handles context / datasets
│   └── IMPLEMENTATION_INSIGHTS.md# framework / repo-shape decisions
├── methods/                  # METHOD IMPLEMENTATIONS (one package per method)
│   └── t2t/                      # text-to-text baseline (implemented)
├── scripts/                  # RUNNABLE ENTRYPOINTS + cache config
│   ├── README.md                 # how to run each test/experiment (commands)
│   ├── _hf_cache.py              # points HF_HOME at data/hf_cache
│   ├── prepare_hotpotqa_bridge.py# builds cross-context task JSONL
│   ├── run_text_to_text.py       # runs the T2T Sharer→Receiver pipeline
│   └── run_receiver_baseline.py  # runs no-Sharer reference baselines
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
> templates) live in code at [methods/t2t/prompts.py](methods/t2t/prompts.py).

## The T2T method package (`methods/t2t/`)

A model-agnostic library. The core never imports `torch`; the model backend uses
deferred imports, so `import methods.t2t` works without a GPU stack.

| File | Responsibility |
|------|----------------|
| [schema.py](methods/t2t/schema.py) | `T2TExample` — the normalized example consumed by every method (the data contract). |
| [prompts.py](methods/t2t/prompts.py) | Prompt builders: `build_sharer_prompt`, `build_receiver_prompt`, `build_receiver_only_prompt`, `format_choices`. |
| [model.py](methods/t2t/model.py) | `load_generator(...)` — turns a frozen HF causal LM into a text generator with a token budget; the only torch/transformers dependency. |
| [runner.py](methods/t2t/runner.py) | Per-example execution: `run_t2t_example` (Sharer→Receiver) and `run_receiver_only_example` (no-Sharer baselines); `T2TResult`, `GenerationOutput`. |
| [parsing.py](methods/t2t/parsing.py) | `parse_choice_answer` — extract a multiple-choice letter (returns raw text for free-form). |
| [scoring.py](methods/t2t/scoring.py) | HotpotQA/SQuAD `normalize_answer`, `exact_match`, `f1_score`, `score_free_form` (EM/F1). |
| [io.py](methods/t2t/io.py) | Aggregate metrics/latency/tokens and write the standardized output files. |

### Prompts: what they do

All prompts are centralized in [prompts.py](methods/t2t/prompts.py).

- **Sharer prompt** (`build_sharer_prompt`): tells the Sharer it may hold only
  *part* of the information, and asks it to **relay relevant facts from its own
  context, not guess the final answer**. This matters in cross-context: the
  Sharer cannot answer alone and should communicate evidence rather than
  hallucinate. Shows the Sharer's context, the question, and any choices.
- **Receiver prompt** (`build_receiver_prompt`): asks the Receiver to answer
  using *its own information plus the Sharer's message*. For multiple-choice it
  requests only the option letter; for free-form it requests a short answer
  phrase (keeps EM/F1 meaningful).
- **Receiver-only prompt** (`build_receiver_only_prompt`): the no-Sharer variant
  used for the LCF-X reference baselines — same answer formatting, but no
  "message from another model" block. Omits the context section entirely for the
  question-only baseline.

### Data contract — `T2TExample`

Every method consumes the same normalized example, so datasets are prepared once
and reused for fair comparison:

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

`choices` is empty for free-form tasks (HotpotQA). In shared-context tasks the
two contexts may be identical; in cross-context they differ.

### Output contract — one directory per run

Every run writes four files into its `--output-dir` (under `results/`):

| File | Contents |
|------|----------|
| `predictions.jsonl` | one record per example (prompts, message, output, prediction, answer, `is_correct`, `scores`, `latency`, `token_counts`, `metadata`). |
| `metrics.json` | aggregate accuracy, free-form `scores` (EM/F1), and `token_counts` totals/averages. |
| `latency.json` | per-stage and total latency totals/averages. |
| `run_metadata.json` | method, models, dataset, budgets, scoring mode, timestamp. |

Comparison/analysis depends only on these files, never on internal method
classes — so T2T, the receiver baselines, and future methods (C2C, LCF, LCF-X,
receiver-anchored) are compared apples-to-apples.

## Environment

- Python **3.12** virtualenv at `.venv` (run tools as `.venv/bin/python ...`).
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
