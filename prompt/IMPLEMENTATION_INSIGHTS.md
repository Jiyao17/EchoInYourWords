# Implementation Framework Insights

## Main Decision

Do not force text-to-text, LCF-X, and the new receiver-anchored method into one deep shared implementation framework.

Instead, keep method implementations mostly independent and compare them through a thin shared experiment layer.

This is likely the cleanest path because each method has different internal assumptions:

- Text-to-text: natural-language message generation and re-ingestion.
- LCF-X: cross-context communication through pooled, position-free Sharer cache summaries.
- New method: Sharer context translated into vectors or pseudo-tokens attached to Receiver-side input tokens or latent slots.

(Shared-context, token-position-aligned methods — C2C and the original LCF — are out of scope; see [BASELINES_SUMMARY.md](BASELINES_SUMMARY.md).)

Trying to hide all of these behind one model wrapper too early would create brittle abstractions and make faithful baseline reproduction harder.

## What Should Be Shared

Share only the parts needed for reproducible comparison:

- dataset download and local asset checks;
- common task/example metadata where practical;
- model and checkpoint registry;
- run configuration format;
- output schemas;
- metric computation;
- latency and parameter reporting;
- result aggregation;
- visualization and comparison scripts.

These shared pieces should not dictate how a method performs cache extraction, cache injection, adapter training, pooling, or text-message generation.

## What Can Stay Method-Specific

Each method can own its own implementation details:

- model wrapper;
- communication architecture;
- training loop;
- tokenizer alignment logic;
- cache extraction and injection logic;
- checkpoint format;
- inference runner;
- baseline-specific preprocessing;
- method-specific debugging utilities.

This lets each baseline stay close to its original paper or codebase.

## Recommended Repository Shape

Use a method-module layout with a thin shared experiment layer:

```text
methods/
  t2t/
  lcfx/
  eiyw/                # new receiver-anchored method

data/
  tasks/
  preprocessing/

experiments/
  configs/
  manifests/

scripts/
  run_method.py
  evaluate_outputs.py
  compare_results.py
  visualize_results.py

results/
  predictions/
  metrics/
  figures/
```

The exact names can change later, but the separation should remain:

- `methods/` contains how each communication method works.
- `scripts/` contains how methods are run and compared.
- `results/` contains standardized outputs for comparison.

## Method Contract

Each method should expose a small external contract rather than share internals.

Inputs:

- method config;
- task config;
- model-pair config;
- checkpoint or training config;
- output directory.

Outputs:

- `predictions.jsonl`: one record per example;
- `metrics.json`: aggregate scores;
- `latency.json`: timing measurements, using a uniform per-stage contract so
  methods stay comparable — `sharer_ms`, `fuser_ms` (translation/fusion stage,
  `0.0` for methods like text-to-text that do not fuse, `>0` for cache/latent
  methods), `receiver_ms`, and `total_ms`;
- `run_metadata.json`: method, models, dataset, checkpoint, seed, versions, and command metadata.

The comparison scripts should depend on these outputs, not on internal method classes.

## Comparison Workflow

Example workflow:

```bash
python scripts/run_text_to_text.py --task cross_context --config experiments/configs/t2t.yaml
python scripts/run_lcfx.py        --task cross_context --config experiments/configs/lcfx.yaml
python scripts/run_eiyw.py        --task cross_context --config experiments/configs/ours.yaml

python scripts/compare_results.py \
  --runs results/t2t_run results/lcfx_run results/ours_run
```

The important point is not the exact command syntax. The important point is that each method can run differently internally, while comparison happens over standardized result files.

## Why Not A C2C-Style Framework

C2C-style codebases are naturally built around **aligned input sections** (same
prompt, source-token-aligned projection). We deliberately avoid adopting that as
the project framework:

- LCF-X is position-free and cross-context — it does not align Sharer/Receiver tokens.
- The new method's Receiver-side anchoring should not be forced into a C2C-style
  source-token-aligned projector interface.

C2C and shared-context LCF are out of scope as implementations (see
[BASELINES_SUMMARY.md](BASELINES_SUMMARY.md)); we keep only their conceptual
contrast as motivation for a more flexible design.

## Role Of OpenHermes

OpenHermes is a candidate training dataset, not the framework.

General instruction/chat data in that style is useful for training the
fuser/adapter modules (LCF-X and the new method) with frozen base models.

It should be treated as one dataset option inside the training pipeline, not as
the organizing principle of the whole project.

## Design Principle

Build independent method implementations plus a strict experiment/output contract.

This keeps the project:

- faithful to each baseline;
- easier to debug;
- flexible for the new method;
- easier to compare across the cross-context methods;
- less likely to accumulate premature abstraction debt.
