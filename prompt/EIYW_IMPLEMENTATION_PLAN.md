# EIYW (receiver-anchored soft pseudo-tokens) Implementation Plan

## Context

The project compares three cross-context Sharer→Receiver communication methods
under one output contract: **T2T** (text message), **LCF-X** (position-free
pooled KV residual), and **EIYW** — the project's namesake contribution
(`methods/eiyw/`). EIYW is the **receiver-anchored** method: instead of moving
information along the *Sharer's* token positions or a single position-free
summary, it asks *what should be added at the Receiver's own input*, shaped by
how the Receiver represents its task (see
[PROJECT_PLAN.md](PROJECT_PLAN.md) "New Method").

**Mechanism (chosen):** translate the Sharer's KV cache into **K soft pseudo-
tokens, conditioned on the Receiver's question**, and **prepend them to the
Receiver's input embeddings** (`inputs_embeds`). The Receiver attends to them
natively. Only the translator is trained; both base models stay frozen.

Contrast with the LCF-X baseline (deliberately the most distinct mechanism):

| | LCF-X | EIYW |
|---|---|---|
| What is sent | one position-free pooled summary per layer | K soft tokens in Receiver embedding space |
| Where injected | gated KV **residual**, broadcast to *every* Receiver position | **prepended to the Receiver input** (sequence front) |
| Shaped by | how the Sharer encoded its context | the Receiver's own question (the *anchor*) |
| Trained module | fuser (`fuser.py`) | translator (`translator.py`) |

## Architecture

`EIYWTranslator` (`methods/eiyw/translator.py`) maps the raw Sharer KV +
a Receiver anchor → `[B, K, hidden_r]` soft tokens:

1. **Pool the Sharer cache** (learned attention, reused from the LCF-X design):
   within-span `N_S → P` keeping Sharer geometry, giving `L_s·P` span tokens of
   feature dim `2·H_s·D_s`. Length-masked over each example's real tokens.
2. **Receiver-anchored queries (the novelty):** a learnable query bank
   `Q_base ∈ ℝ^{K×d}` is modulated by the Receiver-question anchor
   (`Q = Q_base + W_a·anchor`, FiLM-style). The anchor is the **mean-pooled
   embedding of the Receiver prompt** (`EIYWPair.receiver_anchor`), so *what gets
   communicated depends on the Receiver's task*.
3. **Cross-attention** of the K queries over the `L_s·P` span tokens; `value_proj`
   emits `hidden_r` features, and `out_proj` produces the K soft tokens. This
   bridges arbitrary `(H_s, D_s, L_s) → hidden_r`, so heterogeneous pairs work
   (`check_pair_compatible` records dims, never requires equality).

`zero_init=True` uses a **gentle small-scale** init of `out_proj` (not exact
zero). Exactly-zero soft embeddings are a numerical degeneracy — feeding all-zero
vectors to the frozen Receiver makes its backward non-finite (`0 * inf`) — and
EIYW has no zero-init==Receiver-only property to preserve anyway.

### Decode path (`EIYWPair.decode_receiver`)

Prefill uses `inputs_embeds = concat([soft_tokens, embed(prompt_ids)])`, then a
manual greedy loop feeds new `input_ids` against the growing cache. With no soft
tokens this reduces to ordinary greedy decoding (verified: `inputs_embeds`
prefill reproduces the `input_ids` last-logit exactly), which is the
`--no-fusion` Receiver-only floor.

### Training (`train_translator`) — key difference from LCF-X

Frozen Sharer + Receiver; teacher-forced next-token CE on the gold answer.
Because the soft tokens lead the sequence, every prompt/answer activation depends
on them, so the forward over `[soft ; prompt ; answer]` is **grad-enabled
end-to-end** (LCF-X could `no_grad` its prompt prefill and inject the residual
after). To keep batching safe, each example's full sequence is built with no
internal padding and the batch is **right-padded** — left-padding would put
fully-masked leading tokens through a grad-enabled softmax (all `-inf` → NaN);
right-pad tokens always attend causally to real tokens. Each example's answer
logits are gathered at its own offset. Gradient checkpointing on the frozen
Receiver is enabled to offset the extra memory.

Checkpoints are **model-stamped** (`save_translator`/`load_translator`): payload
carries `sharer_model`/`receiver_model`/`arch` (incl. `num_soft_tokens`,
`bottleneck_dim`, `receiver_hidden`); `run_eiyw.py` hard-errors on a mismatch
unless `--allow-model-mismatch`.

## Module layout — `methods/eiyw/`

Mirrors `methods/lcfx/`:

| File | Responsibility |
|------|----------------|
| `schema.py` | `EIYWExample = T2TExample` alias. |
| `prompts.py` | `build_sharer_encode_text` (Sharer context+question) + `build_receiver_prompt` (Receiver's own context only; reused T2T no-Sharer prompt — also the anchor text). |
| `cache.py` | `extract_kv` (raw per-layer Sharer KV). No residual injection — EIYW injects at the input. |
| `translator.py` | `EIYWTranslator` — pooling + receiver-anchored cross-attention → K soft tokens. `from_specs` / `arch_config` / `from_arch_config` / `trainable_parameters`. |
| `model.py` | `ModelSpec`, `EIYWPair` (`sharer_summary`, `receiver_anchor`, `decode_receiver` via `inputs_embeds`), `load_eiyw_pair`, `check_pair_compatible`. |
| `runner.py` | `EIYWResult` (`method="eiyw"`), `run_eiyw_example`, `run_receiver_only_example`. Latency: `sharer_ms` (encode), `fuser_ms` (translation), `receiver_ms` (decode), `total_ms`. |
| `training.py` | `example_loss`, `batched_example_loss`, `evaluate`, `train_translator`, `save_translator`/`load_translator`. |
| `io.py` | `write_run_outputs` reusing T2T aggregators; same four files. |

Entrypoints: `scripts/run_eiyw.py`, `scripts/train_eiyw.py`. Makefile:
`eiyw-train` / `eiyw-eval` / `eiyw` / `eiyw-hetero` (knobs reuse the LCF-X set;
`SOFT_TOKENS` adds K). Results land in `results/eiyw/<dataset>/<sharer>/<receiver>/`,
checkpoints in `data/checkpoints/eiyw/<slug>/translator.pt`.

## Verification (all passing on Qwen2.5-0.5B → Qwen3-0.6B)

1. **Shapes/compat:** `EIYWTranslator` emits `[B, K, hidden_r]` from a different-
   dim Sharer; `check_pair_compatible` reports the bridge.
2. **Decode-path equivalence:** `inputs_embeds`-prefill (no soft tokens)
   reproduces the `input_ids` greedy decode (last-logit diff `0.0`).
3. **Single==batched loss parity**; finite gradients into all translator params.
4. **Smoke end-to-end:** tiny train → stamped checkpoint → eval writes all four
   standardized files with EM/F1; `--no-fusion` floor runs; mismatch guard fires.
5. **Ladder placement (to run at scale):** `question_only ≤ half ≤ {T2T, LCF-X,
   EIYW} ≤ full`.

## Out of scope

Full-scale training runs, multi-Sharer extension, alternate injection mechanisms
(receiver-as-query KV residual / latent slots were considered, not chosen),
distributed training.
