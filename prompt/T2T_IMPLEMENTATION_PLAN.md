# Text-To-Text Baseline Implementation Plan

## Goal

Implement the simplest communication baseline first:

1. The Sharer reads its available input or context.
2. The Sharer generates a short natural-language message.
3. The Receiver reads its own input plus the Sharer's message.
4. The Receiver produces the final answer.

This baseline should be inference-only. No adapter training is required.

## Design Direction

Build a general text-to-text method, not a reproduction-specific pipeline.

The core implementation should know only about:

- a normalized task example;
- a Sharer prompt;
- a Receiver prompt;
- model generation;
- answer parsing or scoring;
- standardized result output.

If a paper baseline needs a special prompt format or input split, add that later as a thin wrapper or task adapter around the general runner.

## Why Start Here

Text-to-text is the easiest communication baseline to debug because it uses normal language generation rather than cache extraction or latent injection.

It also gives the project an early end-to-end pipeline:

- load models;
- load task examples;
- run Sharer;
- run Receiver;
- save predictions;
- compute metrics;
- record latency, token counts, and metadata.

The same result contract can later be reused by latent/cache methods.

## Scope

The first implementation should support:

- frozen Hugging Face causal language models;
- one Sharer model and one Receiver model;
- deterministic generation by default;
- configurable Sharer message token budget;
- configurable Receiver answer token budget;
- shared-context and cross-context examples through the same schema;
- generation-based answer parsing for the first version;
- standardized output files for comparison.

The first version does not need:

- training;
- batching;
- distributed inference;
- prompt search;
- multiple Sharers;
- streaming generation;
- cache reuse between stages;
- paper-specific compatibility logic inside the core runner.

## Method Behavior

For each example:

1. Build the Sharer prompt from the Sharer's available context and the task question.
2. Generate one Sharer message.
3. Build the Receiver prompt from the Receiver's available context, the task question, and the Sharer message.
4. Generate or score the Receiver's answer.
5. Save the full trace.

Important rule:

- Generate the Sharer message exactly once per example.
- Reuse that same message for Receiver inference, logging, scoring, and analysis.

## Shared-Context Mode

In shared-context mode:

- Sharer and Receiver receive the same task information.
- The Sharer message acts as background, explanation, hint, or distilled reasoning.
- This mode is useful for comparing text communication against shared-context latent/cache methods.

Basic flow:

```text
Sharer input:
  shared context
  question/task
  optional choices

Sharer output:
  concise helpful message

Receiver input:
  shared context
  question/task
  optional choices
  Sharer message

Receiver output:
  final answer
```

## Cross-Context Mode

In cross-context mode:

- Sharer and Receiver receive different context, evidence, memory, or subtask information.
- The Sharer message communicates useful information from its side.
- The Receiver combines its own information with the Sharer message to answer.

Basic flow:

```text
Sharer input:
  sharer context
  question/task
  optional choices

Sharer output:
  concise message containing useful information

Receiver input:
  receiver context
  question/task
  optional choices
  Sharer message

Receiver output:
  final answer
```

## Minimal Data Contract

Use a task example shape that can represent both shared-context and cross-context settings:

```json
{
  "id": "example-id",
  "question": "Final question or task instruction",
  "choices": ["A", "B", "C", "D"],
  "answer": "A",
  "sharer_context": "Context available to Sharer",
  "receiver_context": "Context available to Receiver",
  "metadata": {}
}
```

Notes:

- `choices` can be empty or omitted for free-form tasks.
- In shared-context mode, `sharer_context` and `receiver_context` may contain the same information.
- Dataset-specific fields should be converted into this normalized format before calling the method.
- Paper-specific formats should be handled by adapters, not by changing the core runner.

## Output Contract

Each run should write a dedicated output directory containing:

```text
predictions.jsonl
metrics.json
latency.json
run_metadata.json
```

Each `predictions.jsonl` row should include:

```json
{
  "id": "example-id",
  "method": "text_to_text",
  "sharer_prompt": "...",
  "sharer_message": "...",
  "receiver_prompt": "...",
  "receiver_output": "...",
  "prediction": "...",
  "answer": "...",
  "is_correct": true,
  "latency": {
    "sharer_ms": 0.0,
    "receiver_ms": 0.0,
    "total_ms": 0.0
  },
  "token_counts": {
    "sharer_input": 0,
    "sharer_output": 0,
    "receiver_input": 0,
    "receiver_output": 0
  },
  "metadata": {}
}
```

The exact fields can evolve, but the first version should preserve enough information to compare methods and debug examples.

## Prompt Templates

Keep prompts simple and centralized.

Sharer prompt:

```text
You are helping another model answer a task.
Read the information below and write a concise message containing only the most useful information.

Information:
{sharer_context}

Question:
{question}

Choices:
{choices}

Message:
```

Receiver prompt:

```text
Answer the task using your own information and the message from another model.

Your information:
{receiver_context}

Message from the other model:
{sharer_message}

Question:
{question}

Choices:
{choices}

Answer:
```

For multiple-choice tasks, the prompt builder should add a clear final-answer instruction, such as asking for only the option letter.

## Proposed Files

Initial implementation can be small:

```text
methods/text_to_text/
  prompts.py
  runner.py
  parsing.py

scripts/
  run_text_to_text.py

experiments/configs/text_to_text/
  smoke.yaml
```

Responsibilities:

- `prompts.py`: Sharer and Receiver prompt builders.
- `runner.py`: model loading, generation, per-example execution, latency/token counting, output writing.
- `parsing.py`: answer extraction from generated Receiver outputs.
- `run_text_to_text.py`: command-line entrypoint.
- `smoke.yaml`: tiny reproducible config for the first run.

## Implementation Steps

1. Define the normalized task example structure.
2. Implement general Sharer and Receiver prompt builders.
3. Implement model loading and deterministic generation helpers.
4. Implement `run_example` with one Sharer generation and one Receiver generation.
5. Write `predictions.jsonl`.
6. Compute simple metrics.
7. Add latency and token-count reporting.
8. Add a tiny smoke dataset/config.
9. Run the smoke test with very small local or downloaded models.
10. Connect real dataset adapters after the runner is stable.

## Evaluation Notes

For multiple-choice tasks:

- parse the Receiver output into an answer choice;
- prefer deterministic prompting and generation for the first version;
- later, optionally add logit-based Receiver scoring if generation parsing is too noisy.

For free-form QA:

- preserve raw output;
- use task-specific metrics later;
- do not overfit parsing in the first version.

## Future Compatibility Wrappers

Some paper baselines may expect specific fields, prompts, or answer formats. Handle those outside the core T2T runner.

Future wrappers can:

- map benchmark-specific examples into the normalized schema;
- generate paper-specific prompt variants;
- split a question into special Sharer/Receiver fields;
- convert this runner's outputs into a paper-style report;
- call the same core `run_example` function with matched parameters.

The core method should remain general.

## Acceptance Criteria

The first T2T implementation is acceptable when:

- it can run end to end on a tiny shared-context smoke task;
- it can run end to end on a tiny cross-context smoke task;
- it writes `predictions.jsonl`, `metrics.json`, `latency.json`, and `run_metadata.json`;
- each prediction stores Sharer prompt, Sharer message, Receiver prompt, and Receiver output;
- Sharer generation happens once per example;
- it supports configurable communication and response token budgets;
- it can be compared later against other methods using only standardized output files.

## Follow-Up After MVP

After the basic version works:

- add real shared-context dataset adapters;
- add real cross-context dataset adapters;
- add configurable prompt variants;
- add paired comparison scripts;
- add visualization for accuracy, latency, and token cost;
- decide per benchmark whether generation parsing or logit scoring should be the final reporting method.
