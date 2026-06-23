# Cache-to-Cache (C2C) Baseline Implementation Plan

## Goal

Implement a **C2C-style shared-context cache-transfer baseline**: the Sharer and
Receiver encode the **same** prompt; a trained **cache fuser** projects the
Sharer's KV cache into the Receiver's cache and fuses it at aligned token
positions; the Receiver then scores the answer under the **fused** cache.

Unlike T2T (inference-only text), C2C touches KV-cache extraction/injection,
tokenizer alignment, a trained fuser, and logit-based MC scoring — so it is a
larger lift. **We do not reimplement the fuser: we reuse the upstream one** (see
*Integration Strategy*). This is a C2C-*style* baseline, not a strict
reproduction; model pair, datasets, and scoring are configurable, but the first
milestone deliberately pins them to the paper's values.

Reference: paper <https://arxiv.org/html/2510.03215>, code
<https://github.com/thu-nics/C2C>. The C2C fuser is a per-layer residual
(`fused = receiver_cache + F(receiver_cache, aligned_sharer_cache)`) with a
projection, input-aware head modulation, and a learnable gate; both LMs are
frozen and only the fuser is trained. We treat that mechanism as a black box.

## Setting & Boundary (shared-context only)

C2C requires the Sharer and Receiver to encode the **same** prompt, so
`sharer_context == receiver_context` and the task carries `choices`. It is
therefore evaluated on **shared-context MC datasets**, **not** the partitioned
HotpotQA-bridge (that is LCF-X's cross-context benchmark). Cache transfer under
different-evidence conditions would be a separate extension (`c2c-cross`),
reported apart — never as a drop-in C2C result.

## Integration Strategy (upstream-first, extensible)

**First milestone: reuse the upstream C2C fuser with the paper's own model pair
and datasets**, run through this project's runner and output contract:

- **Fuser:** the upstream C2C module. Prefer their **released checkpoint** for an
  immediate inference result; if none is released, re-train via their code (see
  *Training, fallback*).
- **Model pair:** the paper's pair — Sharer `Qwen/Qwen2.5-0.5B-Instruct` →
  Receiver `Qwen/Qwen3-0.6B` (same Qwen family, so alignment is near-identity).
- **Datasets:** the paper's MC sets — **confirm the exact list from the upstream
  configs**; expected to include MMLU-Redux, ARC-Challenge, OpenBookQA, and
  MMLU-Pro/C-Eval.

**Why:** fastest path to a faithful, working C2C number; avoids re-deriving a
subtle fuser/training stack; gives a trusted reference before we change anything.

**Extensibility (designed now):** everything upstream sits behind a small
**`Fuser` interface**. That seam lets us later swap in our **own fuser / model /
dataset** without touching the runner, scoring, or output contract. *Use theirs
now, keep the seam, build ours later.*

## Data Contract

Reuse the normalized example ([methods/t2t/schema.py](../methods/t2t/schema.py));
no new schema:

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

`choices` are **labeled option strings** (label + text); `answer` is the matching
**label** (also the label-token the scorer ranks). A new
`scripts/prepare_shared_context_mc.py` builds these JSONL files (mirroring
`prepare_hotpotqa_bridge.py`); dataset-specific formatting stays in that script.

For the upstream-fuser MVP, keep the prompt format **as close as practical to the
upstream C2C evaluation prompt** (the fuser was trained under that distribution);
record any prompt-format deviation in `run_metadata.json`.

## Architecture

**What we own** (from day one): dataset prep, prompt/scoring, the runner, the
output contract, and the **`Fuser` interface**. **Upstream-provided** for the
first milestone (we wrap, not reimplement): the dual-model cache wrapper, token
alignment, layer mapping, and cache injection. We reimplement those ourselves
only when building our own fuser, later.

**`Fuser` interface (`methods/c2c/fuser.py`).** The runner depends only on this
protocol — given the Sharer cache, Receiver cache, alignment, and layer mapping,
return the fused Receiver cache to inject. Implementations:

- **`UpstreamC2CFuser` (MVP):** adapts the upstream fuser module + checkpoint
  (released or re-trained) to the interface.
- **`BypassFuser` (sanity):** a literal identity passthrough returning the
  Receiver cache unchanged — bit-identical to Direct Receiver, used for the
  wiring sanity.
- **`OurFuser` (later):** a placeholder for our own method behind the same
  interface; not built now, but the seam keeps it open.

## Scoring (logit-MC, label-token)

Score MC by comparing next-token probabilities over the option **letters**
(`P("A"|prompt)`, `P("B"|prompt)`, …) and taking the argmax — the
C2C/LCF-standard scoring. This assumes each label is a **single token** for the
tokenizer (true for bare letters on Qwen/Llama; otherwise fall back to
first-token or answer-text scoring). Record `scoring_policy="label_token"`.

**T2T same-scoring (prerequisite).** Add a label-token logit-MC path to T2T
([methods/t2t/scoring.py](../methods/t2t/scoring.py) + `--scoring logit_mc` on
[scripts/run_text_to_text.py](../scripts/run_text_to_text.py)) so the whole
shared-context ladder is scored `logit_mc` rather than mixing scoring modes.
Cross-context HotpotQA T2T keeps its free-form EM/F1.

## Evaluation (reference ladder)

The cross-context `half ≤ T2T ≤ full` ladder does **not** apply (both sides see
the same context). The shared-context ladder is:

| Rung | What it is | Scoring |
|------|-----------|---------|
| **Direct Receiver** | Receiver answers the shared prompt alone, no fusion. | `logit_mc` |
| **T2T (shared-context)** | Sharer writes an NL message on the same context. | `logit_mc` |
| **C2C bypass** | `BypassFuser`: fusion wired but inert — must equal Direct Receiver predictions/accuracy (logits within numerical tolerance). Wiring sanity. | `logit_mc` |
| **C2C (upstream fuser)** | The method; gain = lift over Direct Receiver. | `logit_mc` |

Report accuracy plus fuser parameter count, fuser size on disk, and latency.

## Experiment Matrix

Default first comparison (configurable): the **paper pair** × the **four MC
datasets** × the four rungs above = **16 inference runs** (`run_c2c.py` with
`--direct-receiver` / `--bypass` / `--checkpoint <upstream_or_retrained>`; T2T via
`run_text_to_text.py --scoring logit_mc`), plus a re-training run **only if** no
upstream checkpoint is released. The C2C bypass run must reproduce Direct Receiver
predictions/accuracy on every dataset (logits within tolerance).
`scripts/compare_results.py` rolls the 16 `metrics.json` into
per-dataset tables, **grouped by scoring mode**; the headline is the C2C (upstream
fuser) rung's lift over Direct Receiver.

## Output Contract

Each run writes the four standard files into `--output-dir`. C2C has **no NL
message**, so `predictions.jsonl` uses C2C-native fields:

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

Any deviation from the paper's setting (pair, dataset, corpus, scoring) is
**recorded** in `run_metadata.json` / `train_metadata.json` / the checkpoint
record, so runs are self-describing and comparison stays honest. Reuse/extend
[methods/t2t/io.py](../methods/t2t/io.py) writers (add `trainable_params` /
`adapter_size_bytes`); comparison depends only on these files, never on internals.

## Training (fallback — only if no released checkpoint)

If a usable upstream checkpoint exists, **skip training** and go straight to
inference. Otherwise re-train the upstream fuser via **their code/configs** (not a
re-authored loop): both LMs frozen, optimize only the fuser, on the paper's data
(~500k OpenHermes-2.5; configurable subset for smoke). One correctness note:
"frozen" = `requires_grad=False` on the LMs, **not** `torch.no_grad()` on the
Receiver forward — gradient must still flow through the Receiver into the fuser.
Save a checkpoint with a model-pair record (ids, shapes) validated at load.

## Proposed Files

```text
methods/c2c/
  __init__.py
  model.py           # thin loader/config for the paper pair; cache logic delegated to the upstream adapter in MVP
  fuser.py           # Fuser interface + UpstreamC2CFuser + BypassFuser (+ OurFuser later)
  upstream/          # vendored/pinned upstream C2C code, isolated behind the adapter
  runner.py          # run_c2c_example, direct-receiver + bypass paths, C2CResult
  scoring.py         # logit-MC label-token scoring
  io.py              # reuse/extend t2t.io output writers
  # alignment.py / mapping.py / train.py — only when we build our own fuser (post-MVP)

scripts/
  prepare_shared_context_mc.py   # build the four MC JSONL files
  run_c2c.py                     # inference + four output files (--direct-receiver/--bypass/--checkpoint)
  compare_results.py             # roll metrics.json into per-dataset tables (grouped by scoring)
  train_c2c.py                   # fallback: re-train upstream fuser (only if no released checkpoint)

methods/t2t/scoring.py + scripts/run_text_to_text.py   # + label-token logit-MC path
data/checkpoints/c2c/<pair>/                           # upstream (or re-trained) fuser + record
```

## Implementation Steps

**MVP — integrate the upstream fuser, produce a real C2C number:**

1. `prepare_shared_context_mc.py` → the paper's MC datasets (tiny subset for smoke).
2. Vendor/pin the upstream C2C code; get it importable; isolate behind the adapter.
3. `fuser.py`: define the `Fuser` interface + `BypassFuser`; `model.py` loads the
   paper pair (frozen).
4. `UpstreamC2CFuser`: load the upstream fuser + checkpoint (released, else
   re-trained), adapting cache I/O. **Before reportable inference, verify the
   checkpoint's model-pair metadata, cache shapes, tokenizer assumptions, and
   fuser config match `Qwen2.5-0.5B-Instruct → Qwen3-0.6B`.**
5. `runner.py` (logit-MC) + `run_c2c.py` writing the four output files.
6. **Sanity gate:** `BypassFuser` reproduces Direct Receiver predictions/accuracy
   exactly (logits within tolerance) — wiring correct.
7. Run the upstream fuser on the paper pair + datasets → a real C2C number on the
   ladder. **(End of MVP.)**

**Post-MVP:** `OurFuser` behind the same interface (our method + training); our
own model pairs / datasets; heterogeneous tokenizer alignment; upstream ablations
(`C2C` vs `C2C-C`, gate temperature, layer mapping).

## Risks & Open Questions

- **Upstream integration & the checkpoint question (top risk).** Reusing their
  code means matching its transformers/torch/cache-class versions and cache I/O;
  keep it behind the adapter. **Open:** is a pretrained fuser checkpoint released
  for the paper pair? Yes → inference-only MVP; no → re-train first. Verify any
  checkpoint matches the pair before trusting numbers.
- **Cache injection API drift.** `past_key_values` formats vary by transformers
  version; isolate in `model.py` / the adapter.
- **Tokenizer alignment.** Upstream's concern on the same-family paper pair for
  the MVP; becomes ours only for our own fuser or heterogeneous pairs.
- **Scoring comparability.** Keep `scoring` / `scoring_policy` recorded and never
  mix scoring modes in one comparison column.

## Acceptance

The MVP is done when: it runs end-to-end on a smoke MC task with the paper pair;
the upstream fuser produces fused Receiver caches that are **actually consumed by
the Receiver scoring path**; the four output files are written with the C2C-native
fields and `scoring_policy="label_token"`; the `BypassFuser` reproduces Direct
Receiver predictions and accuracy **exactly under deterministic settings (logits
within a small numerical tolerance)**; and it yields a real C2C number on the
ladder, comparable via the standardized files alone. A re-train (only if needed)
is acceptable when it reaches accuracy comparable to the released/paper number.

## Next Action

The next concrete task is **upstream inspection, not more design**: clone/inspect
<https://github.com/thu-nics/C2C> to (1) check whether a pretrained fuser
checkpoint is released and whether it matches the paper pair; (2) identify the
exact eval command/config and the dataset list; (3) confirm the
`transformers`/`torch`/cache-class requirements; (4) test that the paper pair
loads in our environment. The answers fix the open checkpoint question and make
Implementation Steps 2–4 concrete.
