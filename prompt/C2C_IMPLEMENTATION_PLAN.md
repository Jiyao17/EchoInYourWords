# Cache-to-Cache (C2C) Baseline Implementation Plan

## Goal

Implement a **C2C-style shared-context cache-transfer baseline**.

In C2C, the Sharer and Receiver encode the **same prompt**. A trained fuser maps the Sharer's KV cache into the Receiver's cache space and fuses it into aligned Receiver cache positions. The Receiver then scores the answer using the fused cache.

This is a baseline, not the project's main method. The first milestone is **upstream-first**: reuse the official C2C fuser and checkpoint if available, wrap it into this project's runner and output contract, and produce a reliable C2C baseline number.

Reference:

* Paper: https://arxiv.org/html/2510.03215
* Code: https://github.com/thu-nics/C2C

## Boundary

C2C is **shared-context only**.

In the normalized schema:

```text
sharer_context == receiver_context
```

C2C should be evaluated on shared-context multiple-choice datasets, not on partitioned HotpotQA-bridge. HotpotQA-bridge is the LCF-X / cross-context setting.

A cross-context cache-transfer variant would be a separate method, such as `c2c-cross`, and should not be reported as standard C2C.

## Strategy

### MVP: upstream-first integration

The MVP does not reimplement C2C internals. It integrates the upstream C2C fuser and runs it through this project's standard evaluation pipeline.

MVP choices:

* **Fuser:** upstream C2C fuser.
* **Checkpoint:** use the released upstream checkpoint if available; otherwise re-train with upstream code/configs.
* **Model pair:** use the upstream-checkpoint-compatible pair. Public paper text suggests a Qwen2.5-0.5B Sharer and Qwen3-0.6B Receiver, but exact HF model IDs, chat template, tokenizer assumptions, cache shapes, and checkpoint compatibility must be confirmed from upstream configs/checkpoint metadata.
* **Datasets:** use the upstream C2C MC evaluation datasets. Confirm the exact list from upstream configs; expected candidates include MMLU-Redux, ARC-Challenge, OpenBookQA, and MMLU-Pro or C-Eval.
* **Scoring:** label-token logit-MC scoring.
* **Output:** this project's four standard result files.

### Post-MVP

After the upstream C2C baseline works, we can add:

* `OurFuser` behind the same interface;
* our own model pairs;
* our own datasets;
* heterogeneous tokenizer alignment;
* C2C ablations such as C2C vs C2C-C, gate temperature, and layer mapping.

## Data Contract

Reuse the existing normalized example schema.

```json
{
  "id": "example-id",
  "question": "Question stem",
  "choices": ["A. London", "B. Paris", "C. Berlin", "D. Rome"],
  "answer": "A",
  "sharer_context": "Shared context",
  "receiver_context": "Same as sharer_context in shared-context mode",
  "metadata": {"dataset": "mmlu_redux"}
}
```

Rules:

* `choices` are labeled option strings.
* `answer` is the gold label.
* `scripts/prepare_shared_context_mc.py` builds the shared-context MC JSONL files.
* For the upstream-fuser MVP, keep the prompt as close as practical to the upstream C2C evaluation prompt.
* Record prompt-format deviations in `run_metadata.json`.

## Architecture

C2C uses an independent method package:

```text
methods/c2c/
```

The MVP owns:

* dataset preparation;
* prompt building;
* logit-MC scoring;
* runner;
* output writing;
* the `Fuser` interface.

The MVP does **not** reimplement upstream alignment, layer mapping, cache injection, or fuser internals. Those stay behind the upstream adapter.

### Fuser interface

`methods/c2c/fuser.py` defines a small interface:

```text
Fuser(sharer_cache, receiver_cache, alignment, layer_mapping) -> fused_receiver_cache
```

Implementations:

* `BypassFuser`: returns the Receiver cache unchanged. Used for wiring sanity.
* `UpstreamC2CFuser`: wraps the upstream C2C fuser and checkpoint.
* `OurFuser`: future project method, not part of the MVP.

The runner depends only on this interface.

## Scoring

Use **label-token logit-MC** scoring.

The prompt asks the model to answer with an option letter. The scorer compares:

```text
P("A" | prompt)
P("B" | prompt)
P("C" | prompt)
P("D" | prompt)
```

and chooses the highest-probability label.

Requirements:

* Record `scoring="logit_mc"`.
* Record `scoring_policy="label_token"`.
* Verify that option labels are single tokens for the tokenizer.
* If not, fall back to first-token or answer-text scoring and record the deviation.

Add the same label-token logit-MC scorer to T2T for shared-context MC evaluation so C2C and T2T are compared under the same scoring mode. Cross-context HotpotQA T2T keeps free-form EM/F1.

## Evaluation Ladder

Shared-context C2C uses this ladder:

| Rung               | Description                                                     |
| ------------------ | --------------------------------------------------------------- |
| Direct Receiver    | Receiver answers the shared prompt alone.                       |
| T2T shared-context | Sharer writes a natural-language message; Receiver reads it.    |
| C2C bypass         | Full C2C path with fusion disabled. Must match Direct Receiver. |
| C2C upstream fuser | Actual upstream C2C baseline.                                   |

The key reported value is:

```text
C2C upstream fuser accuracy - Direct Receiver accuracy
```

Also report:

* accuracy;
* trainable fuser parameters;
* fuser size on disk;
* latency.

## Experiment Matrix

Default first comparison:

```text
4 datasets × 4 rungs = 16 inference runs
```

Commands:

```text
run_c2c.py --direct-receiver
run_c2c.py --bypass
run_c2c.py --checkpoint <upstream_or_retrained>
run_text_to_text.py --scoring logit_mc
```

A training run is needed only if no usable upstream checkpoint exists.

`compare_results.py` reads the 16 `metrics.json` files and produces per-dataset comparison tables. It must group by scoring mode and never mix different scoring modes in the same accuracy column.

## Output Contract

Each run writes:

```text
predictions.jsonl
metrics.json
latency.json
run_metadata.json
```

C2C predictions use C2C-native fields:

```text
id
shared_prompt
choice_scores
prediction
answer
is_correct
latency
token_counts
metadata
```

C2C has no natural-language Sharer message, so do not write `sharer_message`.

`run_metadata.json` must include:

```text
method=c2c
sharer_model_id
receiver_model_id
checkpoint
checkpoint_source
fuser_variant
dataset
scoring=logit_mc
scoring_policy=label_token
prompt_format
deviations
upstream_commit
torch_version
transformers_version
cuda_version
timestamp
```

Any deviation from the upstream setting must be recorded.

## Training

Training is fallback only.

If a usable upstream checkpoint exists, skip training and run inference.

If no checkpoint exists, re-train the upstream fuser using upstream code/configs. Do not re-author the training loop for the MVP.

Training assumptions:

* Sharer frozen.
* Receiver frozen.
* Fuser trainable.
* Objective: next-token loss through the fused-cache path.
* Training data: upstream C2C training data, expected to be OpenHermes-style instruction data; confirm from upstream configs.

Important gradient note:

```text
Frozen LM parameters do not mean Receiver forward should run under torch.no_grad().
```

The Receiver parameters stay frozen, but gradient must flow through the Receiver computation into the fused cache and then into the fuser.

## Proposed Files

```text
methods/c2c/
  __init__.py
  model.py           # thin model-pair loader/config; cache details delegated to upstream adapter in MVP
  fuser.py           # Fuser interface + BypassFuser + UpstreamC2CFuser (+ OurFuser later)
  upstream/          # vendored or pinned upstream C2C code
  runner.py          # Direct Receiver, bypass, upstream C2C run paths
  scoring.py         # label-token logit-MC scoring
  io.py              # reuse/extend t2t.io writers

scripts/
  prepare_shared_context_mc.py
  run_c2c.py
  compare_results.py
  train_c2c.py       # optional fallback only if no upstream checkpoint exists

methods/t2t/scoring.py
scripts/run_text_to_text.py

data/checkpoints/c2c/<pair>/
```

`alignment.py`, `mapping.py`, and a custom `train.py` are post-MVP files. Add them only when building `OurFuser`.

## Implementation Steps

### MVP

1. Inspect upstream C2C repo.
2. Confirm checkpoint availability.
3. Confirm exact model IDs, tokenizer/chat template, cache shapes, fuser config, eval commands, and dataset list.
4. Prepare a tiny shared-context MC smoke dataset.
5. Vendor or pin upstream C2C code behind `methods/c2c/upstream/`.
6. Implement `Fuser`, `BypassFuser`, and `UpstreamC2CFuser`.
7. Implement Direct Receiver scoring.
8. Implement C2C bypass path.
9. Verify bypass predictions and accuracy match Direct Receiver under deterministic settings; logits should match within tolerance.
10. Implement upstream fuser path.
11. Verify fused Receiver caches are actually consumed by the Receiver scoring path.
12. Write the four standard output files.
13. Run the upstream fuser on the paper-compatible pair and datasets.
14. Produce the C2C baseline table.

### Post-MVP

1. Add `OurFuser`.
2. Add our own model pairs and datasets.
3. Add heterogeneous tokenizer alignment if needed.
4. Add upstream fuser ablations.
5. Add custom training loop only when upstream training is no longer sufficient.

## Risks

### Upstream dependency drift

The top risk is upstream integration. The upstream code may depend on specific `torch`, `transformers`, CUDA, or cache-class behavior. Keep it isolated behind the adapter and record exact versions.

### Checkpoint compatibility

A fuser checkpoint is valid only for its trained model pair and cache configuration. Before reportable inference, verify:

```text
model IDs
tokenizer/chat template
layer counts
hidden sizes
head counts
head dimensions
cache format
fuser config
layer mapping
```

### Prompt mismatch

The upstream fuser may be sensitive to prompt/template differences. Keep the MVP prompt close to upstream evaluation prompt and record deviations.

### Scoring mismatch

Do not mix logit-MC and generation-parse results in the same accuracy table.

## Acceptance Criteria

The MVP is complete when:

* the paper-compatible Sharer and Receiver load successfully;
* upstream C2C code is importable behind the adapter;
* Direct Receiver runs on a smoke MC task;
* `BypassFuser` reproduces Direct Receiver predictions and accuracy exactly under deterministic settings;
* logits match within a small numerical tolerance;
* `UpstreamC2CFuser` produces fused Receiver caches;
* those fused caches are actually used by the Receiver scoring path;
* all four output files are written;
* metadata records model IDs, checkpoint source, upstream commit, library versions, scoring policy, and deviations;
* a real C2C upstream-fuser number is produced on the reference ladder.

If no upstream checkpoint exists, a re-trained fuser is acceptable only when it reaches accuracy comparable to the released or paper-reported number.

## Next Action

Do upstream inspection before writing more implementation code.

Checklist:

```text
[ ] clone or inspect https://github.com/thu-nics/C2C
[ ] find whether a pretrained fuser checkpoint is released
[ ] confirm exact model IDs
[ ] confirm checkpoint compatibility metadata
[ ] confirm eval command and configs
[ ] confirm dataset list
[ ] confirm torch / transformers / CUDA requirements
[ ] test loading the paper-compatible model pair
[ ] run a tiny upstream smoke eval if possible
```

These answers determine whether MVP is inference-only or requires upstream re-training first.
