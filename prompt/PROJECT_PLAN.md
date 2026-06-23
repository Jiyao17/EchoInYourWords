# Echo In Your Words: Project Plan

## Summary

- Goal: build a research codebase for agent-to-agent communication without text, where a Sharer model's internal context is translated into Receiver-native latent/cache representations.
- Core idea: let the Receiver use information from the Sharer without requiring the Sharer to generate natural-language messages.
- Main novelty: translate the Sharer's context into vectors/tokens attached to the Receiver's own input, rather than transferring source-token-aligned cache vectors.
- Scope: compare against text-based communication and existing latent/cache communication baselines in both shared-context and cross-context settings.

## Research Tracks

- Shared-context communication:
  - Sharer and Receiver process the same prompt or question.
  - This track tests whether latent/cache transfer improves the Receiver's answer quality, efficiency, or robustness when both agents have access to the same information.
  - Relevant baselines include text-to-text communication, C2C-style cache transfer, and LCF-style compressed latent/cache transfer.
- Cross-context communication:
  - Sharer and Receiver process different context, evidence, memory, or subtask information while working toward the same final objective.
  - This track is closer to realistic multi-agent collaboration, where one agent has information the other does not.
  - Relevant baselines include text-to-text communication and LCF-X-style position-free latent/cache communication.

## Baseline Families

- Text-to-text:
  - Sharer reads its available context and sends a concise natural-language message to the Receiver.
  - This is the main human-readable communication baseline.
- C2C-style cache transfer:
  - Sharer and Receiver process aligned or corresponding inputs.
  - The Sharer's KV cache is projected into the Receiver's cache space.
  - This is a shared-context cache-transfer baseline.
- LCF-style latent/cache transfer:
  - Sharer cache information is compressed before transfer.
  - This is a shared-context compressed-cache baseline.
- LCF-X-style cross-context transfer:
  - Sharer information is pooled or summarized into a position-free latent representation.
  - This is a cross-context latent communication baseline.
- Diagnostic references:
  - Receiver-only, Sharer-only, full-context, and oracle-style references can be used where helpful, but they are not the central contribution.

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
  - shared-context communication;
  - cross-context communication;
  - different Sharer/Receiver model pairs;
  - future multi-sharer extensions.

## Implementation Areas

- Asset management:
  - datasets, models, checkpoints, and generated results should have reproducible local storage rules;
  - downloads should use existence checks, pinned revisions where possible, and offline reuse.
- Dataset processing:
  - shared-context and cross-context datasets should be represented by clean task abstractions;
  - dataset-specific formatting and answer scoring should live behind stable interfaces.
- Model-pair support:
  - model-pair metadata should capture tokenizer, architecture, cache shape, and compatibility information;
  - adapters should validate shape compatibility before training or inference.
- Training:
  - train only communication/adaptation modules unless an experiment explicitly requires otherwise;
  - keep frozen-model training as the default assumption.
- Inference:
  - support Receiver-only, text-to-text, existing latent/cache baselines, and the new method through a common evaluation interface.
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
  - at least one shared-context and one cross-context end-to-end run.
- Reproduction checks:
  - text-to-text baseline runs through the common evaluator;
  - C2C/LCF-style shared-context baselines run where applicable;
  - LCF-X-style cross-context baseline runs where applicable.
- Acceptance checks:
  - the new method can be compared fairly against the relevant baseline families;
  - results are reported separately for shared-context and cross-context settings;
  - quality and latency are both included in the final comparison.

## Future Planning Documents

- Baseline details: exact C2C, LCF, LCF-X, and text-to-text configurations.
- Dataset details: selected datasets, preprocessing, splits, prompts, and scoring.
- Model-pair details: chosen Sharer/Receiver pairs and compatibility metadata.
- Method details: architecture, injection mechanism, training objective, and ablations.
- Experiment details: run matrix, metrics, reporting format, and expected tables.

## Assumptions

- Treat this as a research-code project, not only a paper note.
- Keep this file high level; move implementation-specific choices into focused planning documents.
- The main contribution should be framed as receiver-native latent communication, not merely KV compression.
