# Implementation Framework Insights

## Main Decision

Do not force C2C, LCF, LCF-X, text-to-text, and the new receiver-anchored method into one deep shared implementation framework.

Instead, keep method implementations mostly independent and compare them through a thin shared experiment layer.

This is likely the cleanest path because each method has different internal assumptions:

- C2C: source-token-aligned KV-cache projection.
- LCF: shared-context cache compression through a latent bottleneck.
- LCF-X: cross-context communication through pooled, position-free Sharer cache summaries.
- Text-to-text: natural-language message generation and re-ingestion.
- New method: Sharer context translated into vectors or pseudo-tokens attached to Receiver-side input tokens or latent slots.

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
  text_to_text/
  c2c/
  lcf/
  lcfx/
  receiver_anchored/

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
- `latency.json`: timing measurements;
- `run_metadata.json`: method, models, dataset, checkpoint, seed, versions, and command metadata.

The comparison scripts should depend on these outputs, not on internal method classes.

## Comparison Workflow

Example workflow:

```bash
python scripts/run_method.py --method c2c --task shared_context --config experiments/configs/c2c.yaml
python scripts/run_method.py --method lcf --task shared_context --config experiments/configs/lcf.yaml
python scripts/run_method.py --method receiver_anchored --task shared_context --config experiments/configs/ours.yaml

python scripts/compare_results.py \
  --runs results/c2c_run results/lcf_run results/ours_run
```

The important point is not the exact command syntax. The important point is that each method can run differently internally, while comparison happens over standardized result files.

## Role Of C2C/Rosetta Code

The C2C/Rosetta codebase is useful as:

- a reference implementation for C2C;
- a possible source of dataset adapters and evaluation patterns;
- a practical starting point for shared-context cache-transfer baselines.

However, it should not become the required framework for all methods.

Reasons:

- C2C/Rosetta is naturally built around aligned input sections.
- LCF-X and the new method may need different input structures.
- The new method's Receiver-side anchoring should not be forced into a C2C-style source-token-aligned projector interface.

Use C2C/Rosetta where it helps reproduce C2C faithfully. Wrap its outputs into the shared result schema for comparison.

## Role Of OpenHermes

OpenHermes is a training dataset, not the framework.

It is useful for training shared-context adapter/fuser baselines because both C2C and LCF use general instruction/chat data in that style.

It should be treated as one dataset option inside the training pipeline, not as the organizing principle of the whole project.

## Design Principle

Build independent method implementations plus a strict experiment/output contract.

This keeps the project:

- faithful to each baseline;
- easier to debug;
- flexible for the new method;
- easier to compare across shared-context and cross-context settings;
- less likely to accumulate premature abstraction debt.
