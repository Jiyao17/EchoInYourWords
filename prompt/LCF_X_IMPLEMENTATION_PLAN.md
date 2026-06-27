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

1. **Extract** the Sharer's KV cache from a forward pass over the Sharer's context.
2. **Span pre-reduction (pooling, cheap/parameter-free):** mean-pool the Sharer's
   `N_S` positions into `P` span means per layer → `K, V ∈ ℝ^(B×H_s×P×D_s)`
   (still in the Sharer's head dims). This just produces the token set for the
   resampler.
3. **Align (attention resampler) — bring the Sharer to Receiver shape:**
   Receiver-sized learnable queries (one per Receiver layer) attend over the
   `L_s · P` Sharer tokens (joint `[K; V]`) and emit a Receiver-shaped,
   position-free summary `ℝ^(B×H_r×1×D_r)` per Receiver layer. One learned
   attention converts every cross-model axis: **sequence** (`P → 1`), **heads**
   (`H_s → H_r`), **head_dim** (`D_s → D_r`, via the value projection), and
   **layers** (`L_s → L_r`, via per-Receiver-layer queries).
4. **Project (joint KV bottleneck, the LCF projector) — same-shape, Receiver-aware:**
   the projector assumes the Sharer and Receiver caches share shape, which step 3
   guarantees. Broadcast the aligned summary across the Receiver's `N_R` positions
   and **concatenate it with the Receiver's own cache** at each position
   (`[aK; aV; rK; rV]` — the paper's `4·H·D` Concat & Flatten), down-project to
   `d ∈ {64,128,256,512}`, a small MLP, per-(layer, head) tanh gate, up-project to
   Receiver shape. The residual is therefore **position-dependent and
   Receiver-aware**, not a pure Sharer broadcast.
5. **Fuse:** gated residual add into the Receiver cache (`K' = K + α_K·ΔK`, etc.).
6. **Generate:** the Receiver answers from its own context + fused cache; score
   free-form with EM/F1.

### Cross-model handling — align converts every axis; the projector fuses same-shape caches

The LCF projector assumes the Sharer and Receiver caches share shape. So the
**align** resampler does all cross-model conversion *first*, and only then does the
projector fuse the two same-shaped caches. The LCF-X pool is a *learned attention*
whose query count is already a free hyperparameter; we extend that one mechanism
so its queries are sized to the **Receiver**:

- **Sequence (`N`):** Sharer tokens → 1 summary slot (broadcast to `N_R`).
- **Heads / head-dim (`H, D`):** the value projection maps `2·H_s·D_s → 2·H_r·D_r`,
  so the resampler output is already in Receiver head geometry.
- **Layers (`L`):** one learnable query per Receiver layer attends over tokens
  from all Sharer layers (`L_s → L_r`).
- The **projector** then concatenates the aligned Sharer cache with the Receiver
  cache (now identical shape) and emits a Receiver-aware residual.
- `check_pair_compatible(...)` **records** dims/heads/layers (used to size the
  resampler); it never *requires* equality, so different model sizes work.

> Honest caveat: the paper's pool keeps `H, D` (its Sharer and Receiver are both
> Qwen3-0.6B, so they match) and leaves any mismatch to the downstream projector.
> Sizing the resampler's queries/outputs to the Receiver is **our** principled
> generalization — the same attention method, now also remapping `H/D/L`.

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
| `cache.py` | `extract_kv`, `pool_summary` (span pre-reduction `N_S→P` per layer, keeps Sharer `H_s,D_s`), `add_residual_inplace` (gated residual, broadcast over `N_R`). Targets transformers-5 `DynamicCache` (`.layers[i].keys/.values`). |
| `fuser.py` | `LCFXFuser(nn.Module)` — `align` (attention resampler: Receiver-sized queries over the `L_s·P` Sharer tokens → Receiver-shaped summary, converting `N/H/D/L`) + `project` (joint KV bottleneck that concatenates the aligned Sharer **and** Receiver caches, same shape, → Receiver-aware residual). `zero_init` zeroes the projector's up-projection (gate also starts at 0) for the equality check. |
| `model.py` | `load_lcfx_pair(...)` (frozen Sharer + Receiver, deferred torch), `sharer_summary` (encode + span pre-reduction), `decode_receiver` (greedy; computes the residual from the prefilled Receiver cache via a `residual_fn` callback, then injects), `check_pair_compatible`. |
| `runner.py` | `run_lcfx_example` (encode→pool→align→project[with Receiver cache]→fuse→generate) and `run_receiver_only_example` (no fusion, for the equality check); `LCFXResult` (`method="lcf_x"`). |
| `io.py` | Mirror of T2T `io.py` for `LCFXResult`; same four output files; uniform latency contract `sharer_ms` (encode+pool), `fuser_ms` (align+project), `receiver_ms` (generate), `total_ms`. |
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
