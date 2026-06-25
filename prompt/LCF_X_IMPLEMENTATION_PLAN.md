# LCF-X (Cross-Context Latent Cache Flow) Implementation Plan

## Context

The project's focus is **flexible model-to-model communication** for multi-agent
collaboration: agents that hold **different context**, and may differ in **model
size / architecture / tokenizer**. Shared-context, token-position-aligned
cache-transfer methods (C2C and the original shared-context LCF) are explicitly
**out of scope** — they assume both agents encode the *same* prompt and fuse at
aligned positions, which is exactly the rigidity we want to avoid (see
[BASELINES_SUMMARY.md](BASELINES_SUMMARY.md)).

We implement three methods only:

- **T2T** (text-to-text) — natural-language message baseline (done).
- **LCF-X** — cross-context, **position-free** latent cache communication (this plan).
- **New method** (`methods/eiyw/`) — our receiver-anchored communicator.

**LCF-X** is the existing-paper precedent for different-context communication.
The Sharer reads its own context; its KV cache is **pooled into a fixed-size,
position-free summary**, projected into the Receiver's cache space, and fused
into the Receiver as a gated residual. The Receiver then answers from its own
context plus that summary — no token-position alignment, so the two sides can
hold genuinely different evidence.

### Decisions for this deliverable

1. **Scaffold/MVP first.** Build the cache extract → pool → project → fuse →
   generate pipeline end-to-end with an **untrained (zero-init) fuser**,
   validated on a smoke task. The training loop is the next step (interface
   defined now, body deferred).
2. **Reuse existing cross-context infra.** Target **partitioned HotpotQA-bridge**
   (already built by [scripts/prepare_hotpotqa_bridge.py](../scripts/prepare_hotpotqa_bridge.py)),
   free-form **EM/F1** scoring ([methods/t2t/scoring.py](../methods/t2t/scoring.py)),
   and the existing receiver-baseline ladder (question-only / half / full) plus
   T2T as reference points.
3. **Clean from-scratch, flexible by design.** Implement pooling / projection /
   injection ourselves in `methods/lcfx/`. The projector must **not** assume
   matched feature dims between Sharer and Receiver (see Cross-model handling).

## Key MVP Insight — zero-init fuser must equal Receiver-only

A **zero-initialized** fuser (zero up-projection ⇒ zero gated residual) leaves
the Receiver cache unchanged, so LCF-X must reproduce the **Receiver-only
(half-context) baseline** exactly. This is the central plumbing-correctness
check: extraction, pooling, projection, and injection do not corrupt the
Receiver forward pass. Training then moves accuracy above that floor, toward the
full-context upper bound.

## Architecture (from the paper, with our gap-filling)

Per-Receiver-layer flow. **Bold** = the LCF-X-specific step.

1. **Extract** the Sharer's KV cache from a forward pass over the Sharer's context.
2. **Pool (position-free):** collapse the Sharer's `N_S` token positions into a
   fixed-size summary — divide the context into `P` spans, pool within each span,
   then across spans → `K*, V* ∈ ℝ^(B×H_s×1×D_s)` (still in the Sharer's
   head dims). **Broadcast** the summary across the Receiver's `N_R` positions so
   every Receiver position has a Sharer counterpart to fuse with.
3. **Project (joint KV bottleneck):** concatenate the (broadcast) Sharer KV with
   the Receiver KV and flatten; down-project to bottleneck dim `d ∈ {64,128,256,512}`,
   a small MLP, split the latent into `[z_K, z_V]`, compute per-head gates from
   the full latent, and up-project to Receiver-cache shape. One shared pipeline
   for K and V (keys/values are projections of the same hidden state).
4. **Fuse:** gated residual add into the Receiver cache (`K' = K + α_K·ΔK`, etc.).
5. **Generate:** the Receiver answers from its own context + fused cache; score
   free-form with EM/F1.

### Cross-model handling (the paper's gap, our choice)

The paper does **not** specify how it reconciles different head counts, head
dims, or layer counts — it lets the learned linear absorb the mismatch. We make
this explicit so different model sizes work:

- **Feature dim:** `W_down` takes the **concatenation** `[sharer_flat ⊕ receiver_flat]`
  (input width `d_s + d_r`, not a shared width); `W_up` outputs Receiver shape.
  So matched per-head dims are **not** required.
- **Layer count:** an explicit Sharer→Receiver **layer map** (e.g. proportional
  index, configurable) handles differing depths (24 vs 28, etc.).
- `check_pair_compatible(...)` **records** dims/heads/layers to size the projector,
  rather than requiring equality.

> Note: pooling matches the **sequence/position** axis (via broadcast); the
> projector matches the **feature** axis. These are independent — that is what
> makes LCF-X tolerant of different context *and* different model size.

## Relationship to LCF (why no separate flag/mode)

The original shared-context LCF is LCF-X **without pooling** (1:1 position pairing
from the shared input) and is out of scope. We therefore implement only the
position-free path; we do **not** add an aligned-fusion mode. If a shared-context
sanity comparison is ever wanted, it is a thin configuration on the same
projector, not new core code — but it is not a deliverable here.

## New module layout — `methods/lcfx/`

Mirror the T2T package shape; core stays import-light, only `model.py`/`fuser.py`
touch torch (deferred imports).

| File | Responsibility |
|------|----------------|
| `schema.py` | `LCFXExample` — cross-context example (`id, question, choices, answer, sharer_context, receiver_context, metadata`). Mirrors `T2TExample`. |
| `prompts.py` | Build the Sharer encode prompt (its context) and the Receiver answer prompt (its context + fused summary); reuse T2T answer formatting. |
| `cache.py` | `extract_kv_cache`, `pool_summary` (spans→1 slot, position-free), `broadcast_to`, `inject_kv_cache` (gated residual). Normalizes legacy tuple / `DynamicCache`. |
| `fuser.py` | `LCFXFuser(nn.Module)` — joint KV bottleneck (`W_down` on concatenated sharer⊕receiver, MLP, latent split, per-head gates, `W_up` to receiver shape), `zero_init` for the equality check, explicit layer map. Only trainable params. |
| `model.py` | `load_lcfx_pair(...)` (frozen Sharer + Receiver, deferred torch), `encode`, `generate_with_cache`, `check_pair_compatible`. |
| `runner.py` | `run_lcfx_example` (extract→pool→project→fuse→generate) and `run_receiver_only_example` (no fusion, for the equality check); `LCFXResult` (`method="lcf_x"`). |
| `io.py` | Mirror of T2T `io.py` for `LCFXResult`; same four output files; uniform latency contract `sharer_ms` (encode+pool), `fuser_ms` (project+fuse), `receiver_ms` (generate), `total_ms`. |
| `training.py` | **Interface now, body next step.** `train_fuser(...)`: frozen base models, optimize only the fuser, teacher-forced next-token loss through the fused-cache path. Gradients must flow through the (frozen) Receiver into the fuser — do **not** wrap the Receiver forward in `no_grad`. |
| `__init__.py` | Public exports. |

Reuse from `methods/t2t/`: scoring (EM/F1), the example/JSONL pattern, the IO
aggregation/contract, and the receiver-baseline ladder semantics.

## Entrypoints — `scripts/`

- `scripts/run_lcfx.py` — mirrors [run_text_to_text.py](../scripts/run_text_to_text.py):
  `--dataset data/tasks/hotpotqa_bridge_cross.jsonl`, `--sharer-model`, `--receiver-model`,
  `--fuser-checkpoint` (zero-init when omitted), `--bottleneck-dim`, `--num-spans`,
  `--no-fusion` (Receiver-only check), `--output-dir`, `--device/--dtype`.
- `scripts/train_lcfx_fuser.py` — training CLI (**next step**): wires `train_fuser`;
  argument surface defined now, optimization deferred.
- Dataset prep already exists (`prepare_hotpotqa_bridge.py`); no new prep needed for MVP.

## Docs to update alongside the code

[PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) (`methods/lcfx/` row, three-method
framing) and [scripts/README.md](../scripts/README.md) (LCF-X run + the
zero-init-equals-receiver sanity check, slotting into the existing ladder).

## Verification

1. **Shapes/compat:** `LCFXFuser` produces receiver-shaped residuals from a
   different-dim Sharer; `check_pair_compatible` reports the bridge for the
   Qwen2.5-0.5B / Qwen3-0.6B pair.
2. **Zero-init equality (core check):** zero-init fuser ⇒ `run_lcfx_example`
   predictions equal `run_receiver_only_example` (half-context) on the smoke set.
3. **Smoke run:** tiny cross-context JSONL → `run_lcfx.py` writes all four
   standardized files; accuracy == receiver half-context.
4. **End-to-end:** HotpotQA-bridge subset → `run_lcfx.py` produces `metrics.json`
   with EM/F1; `--no-fusion` matches the zero-init run. It slots into the
   existing ladder: `question_only ≤ half ≤ {T2T, LCF-X} ≤ full`.

Commands:

```bash
.venv/bin/python scripts/prepare_hotpotqa_bridge.py --num-examples 200   # already available
.venv/bin/python scripts/run_lcfx.py \
  --dataset data/tasks/hotpotqa_bridge_cross.jsonl \
  --sharer-model Qwen/Qwen2.5-0.5B-Instruct --receiver-model Qwen/Qwen3-0.6B \
  --output-dir results/lcfx_hotpotqa_zeroinit          # zero-init baseline == receiver half
```

## Out of scope (this deliverable)

Fuser training-loop body (interface only; next step), shared-context LCF / C2C
(dropped), heterogeneous-tokenizer text alignment beyond what pooling needs, and
batched/distributed training.
