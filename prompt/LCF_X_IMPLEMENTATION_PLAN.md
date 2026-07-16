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
2. **Pool — per-layer 2-stage attention (paper Fig. 3 / §3.2 / E.3):** each
   Sharer layer is pooled independently with **two learned queries per layer**
   (`q_base^(l)`, `q_layer^(l)`; `2·H·D` params/layer, 57,344 for a 28-layer
   Qwen3-0.6B pair — matching Appendix E.3):
   - *Within-span* (stage 1, `q_base^(l)`): the sharer text is divided into
     `P` contiguous spans (natural `\n\n` paragraph blocks by default — the
     paper trains with HotpotQA's 5-paragraph structure, E.1 — or fixed
     near-equal token chunks). The layer's shared base query pools each span:
     `[B, H_s, N_S, D_s] → K̄,V̄ [B, H_s, P, D_s]`, over each example's
     **real** tokens only. `P` is not a parameter shape, so it can vary freely
     at inference (span-scheme invariance, E.4).
   - *Cross-span* (stage 2, `q_layer^(l)`): a second attention step aggregates
     **that layer's own** `P` span summaries into one per-head sharer summary
     `K*, V* ∈ ℝ^(B×H_s×1×D_s)` — no cross-layer mixing. The summaries are
     **layer-aligned**: sharer layer `l` feeds receiver layer `l` (the paper's
     LCF-X pair is symmetric, `L_s == L_r`). `H_s`/`D_s` are never remapped.
3. **Project (the paper's joint KV bottleneck) — per Receiver layer:** the
   pooled summary is broadcast across the Receiver sequence and takes the place
   of the sharer's matched-position cache in LCF; the projector is unchanged
   and does **not** require Sharer and Receiver caches to share shape:
   `x = concat&flatten(S_K, S_V, R_K, R_V)` (Concat & Flatten, width
   `2·H_s·D_s + 2·H_r·D_r` — the paper's own shared-context pair is asymmetric,
   joint width 2304) → `z = W_down x` → **2-layer MLP** (intermediate `4d`) →
   **input-dependent per-head gates** `α_i = σ(W_gate z)` (from the full latent)
   → **separate per-layer up-projections** `Δ_i = W_up^i z` (always emitting
   Receiver geometry) → **per-layer per-type gate** `g_i` (`i ∈ {K,V}`,
   Gumbel-sigmoid in training, **hard `{0,1}` at inference**). The paper's prose
   describes a latent split `z → (z_K, z_V)` feeding the up-projections, but its
   own reported parameter counts (693K/layer for LCF-128, 25.87M for LCF-X) only
   reproduce with the **full** `d`-dim latent feeding each `W_up^i`
   (693,010/layer exactly); we match the reported counts.
4. **Fuse:** gated residual add into the Receiver cache: `R_i' = R_i + g_i·(α_i⊙Δ_i)`.
5. **Generate:** the Receiver answers from its own context + fused cache; score
   free-form with EM/F1.

### Cross-model handling — how each axis is converted (precise mechanism)

The pool converts only `N`; `H`/`D` pass through untouched and `L` maps 1:1,
matching the paper's Fig. 3 (its Sharer and Receiver are both Qwen3-0.6B, so
`H, D, L` trivially match). We support different-sized Sharer/Receiver pairs
anyway:

| Axis | Conversion | Mechanism |
|---|---|---|
| Sequence `N` | `N_S → P → 1` (broadcast to `N_R` in `project`) | **attention** — within-span (`q_base^(l)`) then cross-span (`q_layer^(l)`), per layer |
| Layers `L` | 1:1 (`l → l`); `L_s → L_r` only when counts differ | layer-aligned handoff; a single **linear projection across the layer axis** (`layer_map`, shared by K and V, created only when `L_s ≠ L_r`) — our one gap-fill beyond the paper |
| Heads `H` | untouched (`H_s` stays `H_s`) | pooled summary stays in Sharer geometry; absorbed by the projector's `W_down` (input width `2·H_s·D_s + 2·H_r·D_r`) |
| Head-dim `D` | untouched (`D_s` stays `D_s`) | same — absorbed by `W_down`, not by the pool |

The projector concatenates the (still Sharer-shaped) pooled summary with the
Receiver cache — no shape match required — and its `W_up^K`/`W_up^V` always
emit Receiver geometry, so the residual comes out Receiver-shaped regardless
of how different the Sharer was.

`check_pair_compatible(...)` **records** dims/heads/layers (used to size the
projections); it never *requires* equality, so different model sizes work.

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
| `fuser.py` | `LCFXFuser(nn.Module)` — `align` (**per-layer 2-stage pooling**: within-span `N_S→P` with `q_base^(l)`, then cross-span `P→1` with `q_layer^(l)` over that layer's own span summaries; Sharer geometry throughout, layer-aligned 1:1, `layer_map` linear bridge only when `L_s≠L_r`) + `project` (the paper's **per-layer** joint KV bottleneck on `[S* (Sharer geom, broadcast) ; R (Receiver geom)]`: `W_down` (input width `2·H_s·D_s + 2·H_r·D_r`) → 2-layer MLP → input-dependent per-head gates `α(z)` → separate `W_up^K/W_up^V` from the full latent (always Receiver geometry) → per-layer per-type gate `g_K/g_V`, Gumbel-sigmoid in training / hard `{0,1}` at eval). `zero_init` zeroes the up-projections (Δ=0 ⇒ residual 0; gates stay nonzero so up still gets gradient). |
| `spans.py` | `paragraph_span_starts(tok, text)` — token start offsets of the sharer text's `\n\n` blocks (the paper's natural-paragraph span scheme; E.1 trains with HotpotQA's 5-paragraph structure). |
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
