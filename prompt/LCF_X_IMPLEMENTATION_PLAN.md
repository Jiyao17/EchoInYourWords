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

1. **Scaffold first, then training.** The cache extract → pool → align → project
   → fuse → generate pipeline was validated end-to-end with an **untrained
   (zero-init) fuser** (== Receiver-only). The **fuser training loop is now
   implemented** (`methods.lcfx.training.train_fuser`): frozen base models,
   gradients flow only into the fuser.
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

1. **Extract** the Sharer's raw KV cache (`[B, H_s, N_S, D_s]` per layer).
2. **Pool — 2-stage learned attention (paper-faithful):**
   - *Within-span* (stage 1): a shared learned query attends over the tokens of
     each of `P` spans → `[B, H_s, P, D_s]` (keeps Sharer geometry), over each
     example's **real** tokens (length-masked in the batched path).
   - *Across-span + cross-model* (stage 2): Receiver-sized learnable queries (one
     per Receiver layer) attend over the `L_s·P` span tokens, and the value
     projection emits Receiver head geometry → a **position-free** summary
     `ℝ^(B×H_r×1×D_r)` per Receiver layer. Note the axes are converted by *two*
     mechanisms (see Cross-model handling): sequence `P→1` and layers `L_s→L_r`
     by the **attention queries**; heads `H_s→H_r` and head-dim `D_s→D_r` by the
     **value projection**. (The paper's pool keeps the `1` slot and keeps `H, D`;
     we remap `H/D` in the value projection because we allow different models.)
4. **Project (the paper's joint KV bottleneck) — per Receiver layer, Receiver-aware:**
   the projector assumes Sharer and Receiver caches share shape, which step 3
   guarantees. Faithful to the paper, per layer:
   `x = concat&flatten(aS_K, aS_V, R_K, R_V)` (the `4·H·D` Concat & Flatten) →
   `z = W_down x` → **2-layer MLP** → split `[z_K, z_V]` → **input-dependent
   per-head gates** `α_i = σ(W_gate z)` → **separate per-layer up-projections**
   `Δ_i = W_up^i z_i` → **per-layer (Gumbel-)sigmoid gate** `g`. The residual is
   position-dependent and Receiver-aware. (~26.6M trainable params for the Qwen3
   pair, matching the paper's ~25.9M.)
5. **Fuse:** gated residual add into the Receiver cache: `R_i' = R_i + g·(α_i⊙Δ_i)`.
6. **Generate:** the Receiver answers from its own context + fused cache; score
   free-form with EM/F1.

### Cross-model handling — how each axis is converted (precise mechanism)

The LCF projector assumes the Sharer and Receiver caches share shape, so `align`
does all cross-model conversion *first*. **But not every axis is converted the
same way** — this matters for understanding/extending the code:

| Axis | Conversion | Mechanism |
|---|---|---|
| Sequence `N` | `N_S → P → 1` (broadcast to `N_R` in `project`) | **attention** — query count sets the output count (within-span query pools each span; the result is one position-free slot) |
| Layers `L` | `L_s → L_r` | **attention** — one learnable query per Receiver layer attends over all Sharer-layer span tokens |
| Heads `H` | `H_s → H_r` | **value projection** — `value_proj: 2·H_s·D_s → 2·H_r·D_r`, then reshaped into `H_r` heads (NOT query-driven) |
| Head-dim `D` | `D_s → D_r` | **value projection** (the feature dim; in attention the output feature dim always comes from the value, so this is a projection even in the paper, which keeps `D`) |

So `N` and `L` are handled by the attention queries (the "N way"), while `H` and
`D` are folded into the stage-2 `value_proj` linear. `H` *could* be made
query-driven too (per-`(layer, head)` queries, like `N_R` queries produce `N_R`
outputs); we deliberately keep it as a value projection — it's simpler, and `D`
is a value projection regardless. The projector then concatenates the aligned
Sharer cache with the Receiver cache (now identical shape) and emits the residual.

`check_pair_compatible(...)` **records** dims/heads/layers (used to size the
projections); it never *requires* equality, so different model sizes work.

> Honest caveat: the paper's pool keeps `H, D` (its Sharer and Receiver are both
> Qwen3-0.6B, so they match) and leaves any mismatch to the learned projections.
> Converting `H/D` in the value projection (and `N/L` by the queries) is **our**
> principled generalization for different model sizes.

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
| `cache.py` | `extract_kv` (raw per-layer KV) + `add_residual_inplace` (gated residual add). All pooling/alignment lives in the fuser. Targets transformers-5 `DynamicCache` (`.layers[i].keys/.values`). |
| `fuser.py` | `LCFXFuser(nn.Module)` — `align` (**2-stage learned-attention pooling**: within-span `N_S→P` keeping Sharer geom, then across-span + cross-model `L_s·P→1` with Receiver-sized queries → Receiver geometry; converts `N/H/D/L`) + `project` (the paper's **per-layer** joint KV bottleneck: `W_down` → 2-layer MLP → latent split `z_K/z_V` → input-dependent per-head gates `α(z)` → separate `W_up^K/W_up^V` → per-layer Gumbel-sigmoid gate `g`). `zero_init` zeroes the up-projections (Δ=0 ⇒ residual 0; gates stay nonzero so up still gets gradient). ~26.9M params (≈ paper's 25.9M). |
| `model.py` | `load_lcfx_pair(...)` (frozen Sharer + Receiver, deferred torch), `sharer_summary` (encode → raw KV), `decode_receiver` (greedy; computes the residual from the prefilled Receiver cache via a `residual_fn` callback, then injects), `check_pair_compatible`. |
| `runner.py` | `run_lcfx_example` (encode→align[2-stage pool]→project[with Receiver cache]→fuse→generate) and `run_receiver_only_example` (no fusion, for the equality check); `LCFXResult` (`method="lcf_x"`). |
| `io.py` | Mirror of T2T `io.py` for `LCFXResult`; same four output files; uniform latency contract `sharer_ms` (encode), `fuser_ms` (align[pool]+project), `receiver_ms` (generate), `total_ms`. |
| `training.py` | **Implemented.** `train_fuser(...)`: frozen base models, optimize only the fuser, teacher-forced next-token CE on the gold answer with the residual fused into the Receiver prompt cache. Prompt prefill is `no_grad`; the answer forward is grad-enabled so gradients flow through the (frozen) Receiver into the fuser. |
| `__init__.py` | Public exports. |

Reuse from `methods/t2t/`: scoring (EM/F1), the example/JSONL pattern, the IO
aggregation/contract, and the receiver-baseline ladder semantics.

## Entrypoints — `scripts/`

- `scripts/run_lcfx.py` — mirrors [run_text_to_text.py](../scripts/run_text_to_text.py):
  `--dataset data/tasks/hotpotqa_bridge_cross.jsonl`, `--sharer-model`, `--receiver-model`,
  `--fuser-checkpoint` (zero-init when omitted), `--bottleneck-dim`, `--num-spans`,
  `--no-fusion` (Receiver-only check), `--output-dir`, `--device/--dtype`.
- `scripts/train_lcfx_fuser.py` — training CLI: wires `train_fuser`
  (`--train-data`, `--num-spans`, `--lr`, `--epochs`, `--max-steps`,
  `--checkpoint-out`); writes a fuser checkpoint that `run_lcfx.py` loads.
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

A proper train/eval split and larger-scale training runs (the loop is
implemented; only smoke-trained so far), shared-context LCF / C2C
(dropped), heterogeneous-tokenizer text alignment beyond what pooling needs, and
batched/distributed training.
