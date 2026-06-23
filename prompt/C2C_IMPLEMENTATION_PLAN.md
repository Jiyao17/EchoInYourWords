# Cache-to-Cache (C2C) Baseline Implementation Plan

## Goal

Implement a C2C-style shared-context cache-transfer baseline. The Sharer and
Receiver encode the same prompt; a trained cache fuser projects the Sharer's KV
cache into the Receiver's cache and fuses it at aligned token positions; the
Receiver then scores the answer under the fused cache rather than its own cache
alone.

This is the project's second communication method, and a larger lift than
Text-to-Text. Where T2T is inference-only and reuses the ordinary generation
path, C2C touches KV-cache extraction and injection, tokenizer alignment, a
trained fuser, and logit-based multiple-choice scoring. To keep the first
milestone tractable we do not reimplement the fuser; we reuse the upstream one,
as described below. C2C here is a C2C-style baseline rather than a strict
reproduction: the model pair, datasets, and scoring are configurable, but the
first milestone deliberately pins them to the paper's values.

The reference paper is at <https://arxiv.org/html/2510.03215> and the code at
<https://github.com/thu-nics/C2C>. In their method the fuser is a per-layer
residual, `fused = receiver_cache + F(receiver_cache, aligned_sharer_cache)`,
built from a projection, an input-aware head-modulation layer, and a learnable
gate; both language models are frozen and only the fuser is trained. We treat
that mechanism as a black box.

## Strategy: Upstream First

The first milestone reuses the upstream C2C fuser together with the paper's own
model pair and datasets, run through this project's runner and output contract.
It pins three things. The fuser is the upstream C2C module: we prefer their
released checkpoint for an immediate inference result, and re-train via their
code only if no checkpoint is released. The model pair is the paper's pair,
Sharer `Qwen/Qwen2.5-0.5B-Instruct` and Receiver `Qwen/Qwen3-0.6B`, which share
the Qwen tokenizer family, so alignment is near-identity. The datasets are the
paper's multiple-choice sets; we confirm the exact list from the upstream
configs, but expect MMLU-Redux, ARC-Challenge, OpenBookQA, and MMLU-Pro or
C-Eval.

Starting from upstream is the fastest path to a faithful, working C2C number. It
avoids re-deriving a subtle fuser and training stack, and it gives us a trusted
reference point before we change anything.

The part we design now is the seam for later. Everything upstream sits behind a
small Fuser interface, so that when we want our own fuser, model pair, or
datasets, we can swap them in without touching the runner, the scoring, or the
output contract. The guiding idea is to use theirs now, keep the seam, and build
ours later.

## Setting and Boundary

C2C requires the Sharer and Receiver to encode the same prompt, so in the
normalized schema `sharer_context` equals `receiver_context` and the task carries
choices. It is therefore evaluated on shared-context multiple-choice datasets,
not on the partitioned HotpotQA-bridge, which is LCF-X's cross-context benchmark.
Cache transfer under different-evidence conditions would be a separate extension
(call it `c2c-cross`), reported on its own and never presented as a drop-in C2C
result.

## Scope

The first milestone is an integration, not a reimplementation. It wires the
upstream fuser to the paper's model pair and datasets, runs it through our runner
and output contract, and produces a real C2C number. It should:

- load the paper's model pair, frozen;
- prepare the paper's multiple-choice datasets, with a tiny subset for smoke and
  the full sets for reporting;
- integrate the upstream fuser behind the Fuser interface, reusing its alignment,
  layer mapping, and cache injection;
- obtain a fuser checkpoint, preferring the upstream released one and otherwise
  re-training via their code;
- score multiple-choice answers by logit (label-token);
- write the four standardized output files;
- pass the bypass sanity check described under Evaluation;
- yield a real C2C number on the reference ladder.

Building our own fuser, model pair, or datasets is deliberately out of scope for
this milestone, as is reimplementing alignment, mapping, or the fuser ourselves;
those wait until we build our own fuser. The milestone also does not need
cross-context data, free-form generation scoring, multi-Sharer fusion,
distributed training, or long-context experiments.

## Data Contract

C2C reuses the normalized example from
[methods/t2t/schema.py](../methods/t2t/schema.py); no new schema is needed.

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

The choices are labeled option strings, combining the label and the option text,
and the answer is the matching label, which is also the label-token the scorer
ranks. A new script, `scripts/prepare_shared_context_mc.py`, builds these JSONL
files in the same spirit as `prepare_hotpotqa_bridge.py`, and dataset-specific
formatting stays inside that script.

Because the upstream fuser was trained under a particular prompt distribution,
the MVP prompt should stay as close as practical to the upstream C2C evaluation
prompt, and any prompt-format deviation should be recorded in `run_metadata.json`.

## Architecture

For the first milestone the division of labor is simple. We own dataset
preparation, the prompt and scoring, the runner, the output contract, and the
Fuser interface. The dual-model cache wrapper, token alignment, layer mapping,
and cache injection are provided by the upstream code, which we wrap rather than
reimplement. We write our own versions of those only later, when we build our own
fuser.

The Fuser interface, in `methods/c2c/fuser.py`, is the seam the runner depends
on. Given the Sharer cache, the Receiver cache, the alignment, and the layer
mapping, an implementation returns the fused Receiver cache to inject. Three
implementations matter:

- `UpstreamC2CFuser` is the MVP implementation. It adapts the upstream fuser
  module and its checkpoint, released or re-trained, to the interface.
- `BypassFuser` is a literal identity passthrough that returns the Receiver cache
  unchanged. Because it is bit-identical to Direct Receiver, it gives us the
  wiring sanity check.
- `OurFuser` is a placeholder for our own method behind the same interface. It is
  not built now, but the seam keeps it open.

## Scoring

We score multiple-choice answers by likelihood over the option letters: we
compare `P("A" | prompt)`, `P("B" | prompt)`, and so on, and take the argmax.
This is the standard C2C/LCF scoring. It assumes each label is a single token for
the tokenizer, which holds for bare letters on Qwen and Llama; if it does not, we
fall back to first-token or answer-text scoring. The policy is recorded as
`scoring_policy` in `run_metadata.json`.

So that the Text-to-Text rung is scored the same way, we add a label-token
logit-MC path to T2T — a small scorer in
[methods/t2t/scoring.py](../methods/t2t/scoring.py) and a `--scoring logit_mc`
flag on [scripts/run_text_to_text.py](../scripts/run_text_to_text.py). This keeps
the whole shared-context ladder on one scoring mode. The cross-context HotpotQA
T2T runs keep their free-form EM/F1.

## Evaluation

The cross-context `half ≤ T2T ≤ full` ladder does not apply here, because both
sides see the same context. The shared-context ladder, all scored by logit-MC, is:

| Rung | What it is |
|------|-----------|
| Direct Receiver | The Receiver answers the shared prompt alone, with no fusion. |
| T2T (shared-context) | The Sharer writes a natural-language message on the same context, which the Receiver reads. |
| C2C bypass | The full C2C path with fusion disabled — the wiring sanity. |
| C2C (upstream fuser) | The method itself; its value is the accuracy it gains over Direct Receiver. |

The bypass rung is the sanity check: under deterministic decoding it must
reproduce Direct Receiver's predictions and accuracy exactly, with logits
matching within a small numerical tolerance. Alongside accuracy we report the
fuser's parameter count, its size on disk, and latency.

## Experiment Matrix

The default first comparison, which stays configurable, runs the paper pair
across the four datasets and the four rungs above, giving sixteen inference runs.
Inference uses `run_c2c.py` with `--direct-receiver`, `--bypass`, or
`--checkpoint`, and the T2T rung uses `run_text_to_text.py --scoring logit_mc`. A
re-training run is needed only if no upstream checkpoint is released. The bypass
run must reproduce Direct Receiver's predictions and accuracy on every dataset.
Finally, `scripts/compare_results.py` rolls the sixteen `metrics.json` files into
per-dataset tables, grouped by scoring mode, and the headline is the C2C
(upstream fuser) rung's lift over Direct Receiver.

## Output Contract

Each run writes the four standard files into its output directory. Because C2C has
no natural-language message, `predictions.jsonl` uses C2C-native fields rather
than T2T's.

```text
predictions.jsonl   # id, shared_prompt, choice_scores, prediction, answer,
                    #   is_correct, latency, token_counts, metadata (no sharer_message)
metrics.json        # accuracy + trainable_params + adapter_size_bytes + token totals
latency.json        # sharer_encode / receiver_encode / alignment / fuse / score + total
run_metadata.json   # method=c2c, models, checkpoint, fuser_variant=upstream_c2c,
                    #   scoring=logit_mc, scoring_policy=label_token, deviations,
                    #   upstream commit, torch/transformers/CUDA versions,
                    #   checkpoint source, timestamp
```

Any deviation from the paper's setting — pair, dataset, corpus, or scoring — is
recorded in `run_metadata.json`, `train_metadata.json`, or the checkpoint record,
so every run describes itself and comparison stays honest. We reuse and extend the
writers in [methods/t2t/io.py](../methods/t2t/io.py), adding `trainable_params`
and `adapter_size_bytes`, and comparison depends only on these files, never on a
method's internals.

## Training (only if no released checkpoint)

If a usable upstream checkpoint exists, we skip training and go straight to
inference. Otherwise we re-train the upstream fuser through their own code and
configs, rather than re-authoring the loop: both language models stay frozen,
only the fuser is optimized, on the paper's data (about 500k OpenHermes-2.5
samples, with a smaller subset for smoke). One detail is easy to get wrong:
"frozen" means `requires_grad` is false on the language models, not that the
Receiver forward runs under `torch.no_grad` — gradient must still flow through the
Receiver into the fuser. The resulting checkpoint carries a model-pair record of
ids and shapes that is validated at load.

## Proposed Files

```text
methods/c2c/
  __init__.py
  model.py           # thin loader/config for the paper pair; cache logic delegated to the upstream adapter
  fuser.py           # Fuser interface + UpstreamC2CFuser + BypassFuser (OurFuser later)
  upstream/          # vendored or pinned upstream C2C code, isolated behind the adapter
  runner.py          # run_c2c_example, the direct-receiver and bypass paths, C2CResult
  scoring.py         # logit-MC label-token scoring
  io.py              # reuse and extend the t2t.io output writers
  # alignment.py / mapping.py / train.py — only when we build our own fuser (post-MVP)

scripts/
  prepare_shared_context_mc.py   # build the four MC JSONL files
  run_c2c.py                     # inference + four output files (--direct-receiver / --bypass / --checkpoint)
  compare_results.py             # roll the metrics.json files into per-dataset tables
  train_c2c.py                   # fallback: re-train the upstream fuser if no checkpoint is released

methods/t2t/scoring.py + scripts/run_text_to_text.py   # add the label-token logit-MC path
data/checkpoints/c2c/<pair>/                           # upstream or re-trained fuser + model-pair record
```

## Implementation Steps

The MVP integrates the upstream fuser and produces a real C2C number.

1. Build `prepare_shared_context_mc.py` and produce the paper's MC datasets, with
   a tiny subset for smoke.
2. Vendor or pin the upstream C2C code, get it importable in our environment, and
   isolate it behind the adapter.
3. Define the Fuser interface and `BypassFuser` in `fuser.py`, and have
   `model.py` load the frozen paper pair.
4. Implement `UpstreamC2CFuser` to load the upstream fuser and its checkpoint
   (released, or re-trained), adapting the cache I/O. Before any reportable
   inference, verify that the checkpoint's model-pair metadata, cache shapes,
   tokenizer assumptions, and fuser config match
   `Qwen2.5-0.5B-Instruct → Qwen3-0.6B`.
5. Implement `runner.py` with logit-MC scoring, and `run_c2c.py` to write the
   four output files.
6. Run the bypass sanity gate: `BypassFuser` must reproduce Direct Receiver's
   predictions and accuracy exactly, with logits within tolerance.
7. Run the upstream fuser on the paper pair and datasets to produce a real C2C
   number on the ladder. This ends the MVP.

After the MVP the work turns to our own extensions: an `OurFuser` behind the same
interface with its own training, our own model pairs and datasets, heterogeneous
tokenizer alignment, and ablations on the upstream fuser (`C2C` versus `C2C-C`,
gate temperature, layer mapping).

## Risks and Open Questions

The largest risk is the upstream integration itself, and the open question at its
center. Reusing their code means matching its `transformers`, `torch`, and
cache-class behavior, all of which we keep behind the adapter. The open question
is whether a pretrained fuser checkpoint is released for the paper pair: if it is,
the MVP is inference-only; if not, we must re-train first. Either way we verify
that a checkpoint matches the pair before trusting its numbers.

Three smaller risks follow. The `past_key_values` format varies across
transformers versions, so we isolate it in `model.py` and the adapter. Tokenizer
alignment is upstream's concern on the same-family paper pair, and becomes ours
only for our own fuser or for heterogeneous pairs. And scoring comparability
depends on always recording `scoring` and `scoring_policy` and never mixing
scoring modes in one comparison column.

## Acceptance Criteria

The MVP is done when it runs end to end on a smoke multiple-choice task with the
paper pair; when the upstream fuser produces fused Receiver caches that are
actually consumed by the Receiver scoring path; when the four output files are
written with the C2C-native fields and `scoring_policy` set to `label_token`; when
the `BypassFuser` reproduces Direct Receiver's predictions and accuracy exactly
under deterministic settings, with logits within a small numerical tolerance; and
when it yields a real C2C number on the ladder that other methods can compare
against using the standardized files alone. A re-train, if one is needed, is
acceptable once it reaches accuracy comparable to the released or paper number.

## Next Action

The next concrete task is upstream inspection, not more design. We should inspect
<https://github.com/thu-nics/C2C> to learn whether a pretrained fuser checkpoint
is released and whether it matches the paper pair, to find the exact evaluation
command, config, and dataset list, to confirm the `transformers`, `torch`, and
cache-class requirements, and to test that the paper pair loads in our
environment. Those answers settle the open checkpoint question and make
Implementation Steps 2 through 4 concrete.
