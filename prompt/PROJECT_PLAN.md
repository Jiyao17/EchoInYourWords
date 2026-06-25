# Echo In Your Words: Project Plan

## Summary

- Goal: build a research codebase for agent-to-agent communication without text, where a Sharer model's internal context is translated into Receiver-native latent/cache representations.
- Core idea: let the Receiver use information from the Sharer without requiring the Sharer to generate natural-language messages.
- Main novelty: translate the Sharer's context into vectors/tokens attached to the Receiver's own input, rather than transferring source-token-aligned cache vectors.
- Scope: compare against text-based communication (T2T) and the existing cross-context latent/cache baseline (LCF-X) in the cross-context setting, with flexibility across different context and different model size/architecture.

## Research Focus: cross-context, flexible communication

The project targets **cross-context** communication only:

- Sharer and Receiver process different context, evidence, memory, or subtask
  information while working toward the same final objective.
- This is the realistic multi-agent collaboration case, where one agent has
  information the other does not.
- The methods must be **flexible**: tolerant of **different context** and of
  **different model size / architecture / tokenizer** between the two agents.

**Out of scope:** shared-context, token-position-aligned cache transfer (C2C and
the original shared-context LCF). They require both agents to encode the same
prompt and fuse at aligned positions, which is precisely the rigidity this
project avoids. See [BASELINES_SUMMARY.md](BASELINES_SUMMARY.md).

## Methods (three only)

- Text-to-text (baseline, implemented):
  - Sharer reads its context and sends a concise natural-language message.
  - The main human-readable communication baseline.
- LCF-X-style cross-context latent transfer (baseline):
  - Sharer cache is pooled/summarized into a fixed-size, position-free
    representation the Receiver conditions on — no token-position alignment.
  - The existing-paper precedent for different-context communication.
- New receiver-anchored method (our contribution):
  - Sharer context translated into vectors / pseudo-tokens anchored to the
    Receiver's own input (see New Method).
- Diagnostic references:
  - Receiver-only (question-only / half-context / full-context) references
    bracket the methods, but are not the central contribution.

## New Method

- Build a receiver-input-anchored communicator:
  - the Sharer first processes its available context and exposes internal states;
  - the Receiver processes its own input, which provides the anchor points for communication;
  - a translator converts the Sharer's context into vectors or pseudo-token representations attached to the Receiver's own input tokens or Receiver-side latent slots;
  - the Receiver consumes these attached vectors as native additional context during inference.
- The main contribution is the anchoring strategy:
  - existing cache-transfer methods generally move information along Sharer/source token positions or through position-free summaries;
  - this method asks what information should be added at the Receiver's own tokens or latent positions;
  - the communication should therefore be shaped around how the Receiver represents its task, not only around how the Sharer encoded its context.
- Keep the design high level for now:
  - do not commit this plan to a specific injection mechanism yet;
  - later method notes can decide whether the attached representations are implemented as cache residuals, soft tokens, adapter states, or another Receiver-native interface.
- Keep the method flexible enough to support:
  - cross-context communication (different evidence per agent);
  - different Sharer/Receiver model pairs (size / architecture / tokenizer);
  - future multi-sharer extensions.

## Implementation Areas

- Asset management:
  - datasets, models, checkpoints, and generated results should have reproducible local storage rules;
  - downloads should use existence checks, pinned revisions where possible, and offline reuse.
- Dataset processing:
  - cross-context datasets should be represented by clean task abstractions;
  - dataset-specific formatting and answer scoring should live behind stable interfaces.
- Model-pair support:
  - model-pair metadata should capture tokenizer, architecture, cache shape, and compatibility information;
  - adapters should validate shape compatibility before training or inference.
- Training:
  - train only communication/adaptation modules unless an experiment explicitly requires otherwise;
  - keep frozen-model training as the default assumption.
- Inference:
  - support Receiver-only, text-to-text, the LCF-X cross-context baseline, and the new method through a common evaluation interface.
- Evaluation:
  - report quality, latency, trainable parameters, adapter size, and per-example outputs;
  - keep paired per-question results so methods can be compared directly.

## Test Plan

- Unit tests:
  - dataset/model asset checks;
  - tokenizer and cache-shape compatibility;
  - adapter forward-shape checks;
  - cache injection behavior;
  - task-specific answer scoring.
- Smoke tests:
  - tiny-subset training and evaluation before full experiments;
  - at least one cross-context end-to-end run.
- Reproduction checks:
  - text-to-text baseline runs through the common evaluator;
  - LCF-X cross-context baseline runs through the common evaluator.
- Acceptance checks:
  - the new method can be compared fairly against T2T and LCF-X;
  - quality and latency are both included in the final comparison.

## Future Planning Documents

- Baseline details: exact T2T and LCF-X configurations.
- Dataset details: selected datasets, preprocessing, splits, prompts, and scoring.
- Model-pair details: chosen Sharer/Receiver pairs and compatibility metadata.
- Method details: architecture, injection mechanism, training objective, and ablations.
- Experiment details: run matrix, metrics, reporting format, and expected tables.

## Assumptions

- Treat this as a research-code project, not only a paper note.
- Keep this file high level; move implementation-specific choices into focused planning documents.
- The main contribution should be framed as receiver-native latent communication, not merely KV compression.
