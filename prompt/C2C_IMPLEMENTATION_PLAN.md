# Cache-to-Cache (C2C) Baseline Implementation Plan

## Goal

A C2C-style shared-context cache-transfer baseline. Per example:

1. Sharer and Receiver encode the same prompt.
2. A trained fuser fuses the Sharer KV cache into the Receiver's at aligned
   positions (`fused = receiver + F(receiver, aligned_sharer)`).
3. The Receiver scores the answer under the fused cache.

Notes:

- We **reuse the upstream C2C fuser**, not reimplement it.
- C2C-style baseline, not a strict reproduction: model pair, datasets, and
  scoring are configurable, but pinned to the paper's values for the first milestone.
- Paper <https://arxiv.org/html/2510.03215>, code <https://github.com/thu-nics/C2C>.
  Their fuser = per-layer residual (projection + head modulation + learnable gate);
  both LMs frozen, only the fuser trained. We treat it as a black box.

## Strategy: Upstream First

First milestone pins:

1. **Fuser** — upstream C2C module; prefer their released checkpoint, else
   re-train via their code.
2. **Model pair** — `Qwen/Qwen2.5-0.5B-Instruct` → `Qwen/Qwen3-0.6B` (same Qwen
   family, so alignment is near-identity).
3. **Datasets** — the paper's MC sets; confirm from upstream configs (expect
   MMLU-Redux, ARC-Challenge, OpenBookQA, MMLU-Pro/C-Eval).

All of it sits behind a `Fuser` interface, so we can swap in our own fuser / model
/ dataset later without touching the runner, scoring, or output contract.

## Setting & Boundary

- Shared-context only: `sharer_context == receiver_context`, task has `choices`.
- Evaluated on shared-context MC datasets, **not** the partitioned HotpotQA-bridge
  (that is LCF-X).
- Different-evidence cache transfer is a separate `c2c-cross` extension, reported
  apart — never a drop-in C2C result.

## Scope

First milestone (integration, not reimplementation):

1. Load the paper pair, frozen.
2. Prepare the paper MC datasets (smoke subset + full).
3. Integrate the upstream fuser behind the `Fuser` interface (reuse its
   alignment / mapping / injection).
4. Obtain a checkpoint (released, else re-train).
5. Logit-MC (label-token) scoring.
6. Write the four standardized output files.
7. Pass the bypass sanity check.
8. Produce a real C2C number on the ladder.

Out of scope until we build our own fuser: our own fuser / model / dataset;
reimplementing alignment / mapping / the fuser; cross-context data; generation
scoring; multi-Sharer; distributed or long-context training.

## Data Contract

Reuse `T2TExample` ([methods/t2t/schema.py](../methods/t2t/schema.py)); no new
schema.

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

- `choices` = labeled option strings; `answer` = matching label (the label-token
  the scorer ranks).
- `scripts/prepare_shared_context_mc.py` builds the JSONL (like
  `prepare_hotpotqa_bridge.py`); dataset-specific formatting stays there.
- Keep the MVP prompt close to the upstream C2C eval prompt; record any
  prompt-format deviation in `run_metadata.json`.

## Architecture

- **We own:** dataset prep, scoring, runner, output contract, the `Fuser` interface.
- **Upstream-provided (MVP):** cache wrapper, alignment, layer mapping, cache
  injection. We reimplement these only for our own fuser.

`Fuser` interface (`methods/c2c/fuser.py`) — the runner depends only on this:

1. `UpstreamC2CFuser` — adapts the upstream fuser + checkpoint (MVP).
2. `BypassFuser` — identity passthrough, bit-identical to Direct Receiver; the
   wiring sanity.
3. `OurFuser` — placeholder for our method, later.

## Scoring

- Label-token logit-MC: rank the option letters by `P(letter | prompt)`, argmax.
- Assumes each label is a single token (true for Qwen/Llama bare letters; else
  fall back to first-token or answer-text). Record `scoring_policy=label_token`.
- Add the same label-token path to T2T
  ([methods/t2t/scoring.py](../methods/t2t/scoring.py) + `--scoring logit_mc` on
  [scripts/run_text_to_text.py](../scripts/run_text_to_text.py)) so the ladder is
  one scoring mode. Cross-context HotpotQA T2T keeps its EM/F1.

## Evaluation (reference ladder)

| Rung | What it is |
|------|-----------|
| Direct Receiver | Receiver answers the shared prompt alone, no fusion. |
| T2T (shared-context) | Sharer writes an NL message on the same context. |
| C2C bypass | Full C2C path, fusion off — the wiring sanity. |
| C2C (upstream fuser) | The method; value = lift over Direct Receiver. |

- Bypass rung: under deterministic decoding must reproduce Direct Receiver
  predictions/accuracy exactly (logits within a small numerical tolerance).
- Report accuracy + fuser parameter count + size on disk + latency.

## Experiment Matrix

- Paper pair × 4 datasets × 4 rungs = **16 inference runs** (+ a re-training run
  only if no checkpoint is released).
- Commands: `run_c2c.py --direct-receiver / --bypass / --checkpoint`; T2T via
  `run_text_to_text.py --scoring logit_mc`.
- The bypass run must reproduce Direct Receiver on every dataset.
- `scripts/compare_results.py` → per-dataset tables grouped by scoring mode;
  headline = C2C (upstream fuser) lift over Direct Receiver.

## Output Contract

Four files per run. C2C has no NL message, so `predictions.jsonl` uses C2C-native
fields:

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

- Record any deviation (pair/dataset/corpus/scoring) in `run_metadata.json` /
  `train_metadata.json` / the checkpoint record.
- Reuse/extend [methods/t2t/io.py](../methods/t2t/io.py) writers (add
  `trainable_params`, `adapter_size_bytes`); comparison reads only these files.

## Training (only if no released checkpoint)

1. If a usable checkpoint exists, skip training and go to inference.
2. Else re-train the upstream fuser via their code/configs (not a re-authored
   loop): both LMs frozen, optimize only the fuser, ~500k OpenHermes-2.5 (subset
   for smoke).
3. Caveat: "frozen" = `requires_grad=False`, **not** `torch.no_grad` on the
   Receiver forward — gradient must flow through the Receiver into the fuser.
4. Save a checkpoint with a model-pair record (ids, shapes), validated at load.

## Proposed Files

```text
methods/c2c/
  __init__.py
  model.py           # thin loader/config for the paper pair; cache logic via the upstream adapter
  fuser.py           # Fuser interface + UpstreamC2CFuser + BypassFuser (OurFuser later)
  upstream/          # vendored or pinned upstream C2C code, behind the adapter
  runner.py          # run_c2c_example, direct-receiver + bypass paths, C2CResult
  scoring.py         # logit-MC label-token scoring
  io.py              # reuse/extend the t2t.io writers
  # alignment.py / mapping.py / train.py — only for our own fuser (post-MVP)

scripts/
  prepare_shared_context_mc.py   # build the four MC JSONL files
  run_c2c.py                     # inference + four output files (--direct-receiver/--bypass/--checkpoint)
  compare_results.py             # roll metrics.json into per-dataset tables
  train_c2c.py                   # fallback: re-train the upstream fuser if no checkpoint

methods/t2t/scoring.py + scripts/run_text_to_text.py   # add the label-token logit-MC path
data/checkpoints/c2c/<pair>/                           # upstream or re-trained fuser + record
```

## Implementation Steps

MVP:

1. `prepare_shared_context_mc.py` → paper datasets (smoke subset).
2. Vendor/pin the upstream C2C code; make it importable; isolate behind the adapter.
3. `fuser.py`: define the `Fuser` interface + `BypassFuser`; `model.py` loads the
   frozen paper pair.
4. `UpstreamC2CFuser`: load the upstream fuser + checkpoint. Before reportable
   inference, verify the checkpoint's model-pair metadata, cache shapes, tokenizer
   assumptions, and fuser config match the pair.
5. `runner.py` (logit-MC) + `run_c2c.py` (four output files).
6. Bypass sanity gate: `BypassFuser` == Direct Receiver (predictions/accuracy
   exact, logits within tolerance).
7. Run the upstream fuser → a real C2C number on the ladder. (End of MVP.)

Post-MVP: `OurFuser` behind the interface (+ training); our own pairs/datasets;
heterogeneous alignment; upstream ablations (`C2C` vs `C2C-C`, gate temperature,
layer mapping).

## Risks & Open Questions

1. **Upstream integration + the checkpoint question (top risk).** Match
   transformers/torch/cache-class behind the adapter. Is a checkpoint released for
   the pair? Yes → inference-only; no → re-train first. Verify it matches the pair
   before trusting numbers.
2. **Cache API drift.** `past_key_values` formats vary by version — isolate in
   `model.py`/the adapter.
3. **Tokenizer alignment.** Upstream's concern for the MVP; ours only for our own
   fuser or heterogeneous pairs.
4. **Scoring comparability.** Always record `scoring`/`scoring_policy`; never mix
   modes in one comparison column.

## Acceptance

MVP done when:

1. Runs end-to-end on a smoke MC task with the paper pair.
2. The upstream fuser's fused caches are actually consumed by the Receiver scoring
   path.
3. The four output files are written with the C2C-native fields and
   `scoring_policy=label_token`.
4. `BypassFuser` reproduces Direct Receiver predictions/accuracy exactly (logits
   within tolerance).
5. It yields a real C2C number on the ladder, comparable via the standardized
   files alone.

Re-train (if needed) is acceptable once accuracy ≈ the released/paper number.

## Next Action

Inspect <https://github.com/thu-nics/C2C>:

1. Is a pretrained fuser checkpoint released, and does it match the paper pair?
2. Exact eval command/config + dataset list.
3. transformers/torch/cache-class requirements.
4. Test that the paper pair loads in our environment.

These settle the open checkpoint question and make Implementation Steps 2–4 concrete.
