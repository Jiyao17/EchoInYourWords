# Cache-to-Cache (C2C) Baseline Implementation Plan

## Goal

Implement the **Cache-to-Cache** shared-context cache-transfer baseline:

1. The Sharer and Receiver both encode the **same** prompt (context + question).
2. A trained **cache fuser** projects the Sharer's KV cache into the Receiver's
   cache space and fuses it into the Receiver's cache at aligned token positions.
3. The Receiver generates (or scores) the final answer conditioned on the
   **fused** cache rather than on its own cache alone.

Unlike Text-to-Text, C2C **requires training** a small fusion module while both
models stay frozen. It is the first method in the project that touches KV-cache
extraction/injection, tokenizer alignment, and an adapter training loop.

Reference paper: <https://arxiv.org/html/2510.03215> · Reference code:
<https://github.com/thu-nics/C2C>. Use the reference implementation to stay
faithful to the C2C *mechanism*, but wrap its outputs into this project's result
contract (see [IMPLEMENTATION_INSIGHTS.md](IMPLEMENTATION_INSIGHTS.md)).

**This is a C2C-*style* baseline, not a strict reproduction.** We implement a
C2C-style **shared-context cache-transfer** method under this project's pipeline
and output contract. The original C2C paper/repo is a **methodological
reference**, not a fixed experimental setting: the model pair, tokenizer family,
training corpus, evaluation datasets, prompt format, and scoring details are all
project choices (see *Project Adaptation Policy* below).

## Background: How C2C Works

Grounded summary of the original C2C mechanism we adapt (see
[BASELINES_SUMMARY.md](BASELINES_SUMMARY.md) for the setting):

- **Shared context.** Both models encode the same input. C2C assumes
  **token-level correspondence** between the two inputs; it is *not* a
  different-evidence agent-communication method (that is LCF-X).
- **Cache fuser (residual integration).** For aligned layers the fused cache is
  `C_n(X) + F_n( C_n(X), C^S_{G(n)}(X) )` — the Receiver cache plus a fuser
  output, added as a residual. The fuser has three parts:
  - **Projection module**: concatenate Receiver and (projected) Sharer KV, apply
    a projection + feature fusion. The complex variant `C2C-C` first maps the
    Sharer cache into the Receiver's hidden dimensionality with a 3-layer MLP.
  - **Dynamic weighting**: an input-aware **head modulation** layer reweights the
    projected information per attention head.
  - **Learnable gate**: a per-layer gate trained with a **Gumbel-sigmoid** and
    temperature annealing (differentiable during training → binary at inference).
- **Layer mapping `G`.** Models of different depth are aligned by **terminal
  alignment**: align final layers first, then penultimate, and so on backward
  until the shallower model's first layer.
- **Token alignment.** Different tokenizers are reconciled by decoding each
  Receiver token to its string and re-encoding with the Sharer tokenizer; for
  one-to-many mappings, pick the Sharer token with maximal string coverage.
  Template/structural tokens are padded to length; content tokens align
  semantically.
- **Fused vs Receiver-only positions.** Fusion replaces the **context/question**
  KV cache `C^F(X)`; the generated-response prefix `C(Y)` stays Receiver-only.
  A `kv_cache_index` (the reference repo's term) marks which positions use the
  fused cache.
- **Trained vs frozen.** Both Sharer and Receiver are **frozen**; only the C2C
  fuser + gate are trained, via next-token prediction on the Receiver's
  responses but conditioned on the fused cache.
- **Reference defaults (not requirements).** The original paper uses Qwen3-0.6B
  as Receiver, several Sharer models (Qwen2.5-0.5B, Llama3.2-1B, Qwen3-4B),
  OpenHermes-2.5 for training, and MC benchmarks such as OpenBookQA, MMLU-Redux,
  ARC-Challenge, and C-Eval with logit-based zero-shot scoring. In this project
  these are **reference defaults / candidates**; the actual model pair, training
  corpus, and evaluation datasets are configurable (see *Project Adaptation
  Policy*).

## Project Adaptation Policy

The items below are **configurable project choices**, not requirements inherited
from the original paper. Each is recorded in run/train metadata (see *Recording
deviations* under the Experiment Plan):

- **Sharer / Receiver model pair** — default Qwen2.5-0.5B-Instruct → Qwen3-0.6B.
- **Tokenizer family** — same-family first; heterogeneous is post-MVP.
- **Training corpus and subset size** — default OpenHermes-2.5; staged subset → full.
- **Evaluation datasets** — default the four MC sets below.
- **Prompt format** — the shared-context MC prompt builder.
- **Scoring policy details** — logit-MC label-token scoring (see *Logit-MC
  scoring policy*).
- **Output contract and experiment runner** — this project's four-file contract.

Reference-compatible defaults let us sanity-check against the paper where useful,
but any deviation is legitimate **as long as it is recorded**.

## Design Direction

Follow the project rule: **independent method module + thin shared experiment
layer** ([IMPLEMENTATION_INSIGHTS.md](IMPLEMENTATION_INSIGHTS.md)).

- C2C lives in its own package `methods/c2c/` and owns its model wrapper, fuser
  architecture, token alignment, cache injection, training loop, and checkpoint
  format. **Do not** force it through the T2T runner or a generic model wrapper.
- It **reuses only the shared contract**: the normalized example schema, the
  four standardized output files, metric computation, and latency/parameter
  reporting — so C2C results **can be compared through the same result
  contract**, with the benchmark setting and scoring mode explicitly recorded.
  Methods may differ in setting/scoring, so comparison is contract-level, not a
  claim that every number is directly equivalent.
- The C2C/Rosetta reference code is a faithfulness reference and a possible
  source of dataset adapters; it must not become the framework for other methods.

## Why C2C Next, and Why It Is Harder Than T2T

C2C is the natural second baseline: it is the representative **shared-context**
cache-transfer method and the direct contrast to text communication. But it is a
significantly larger lift than T2T because it adds, for the first time:

- a **dual-model forward** that exposes and swaps KV caches mid-sequence;
- **tokenizer alignment** between heterogeneous Sharer/Receiver pairs;
- a **trainable fusion module** with a real training loop and checkpoints;
- **logit-based** multiple-choice evaluation (not generation parsing).

T2T established the load → run → score → save pipeline and the output contract.
C2C reuses that contract but introduces the cache-and-training machinery that
LCF and the new method will later build on.

## Scope

The MVP is deliberately small: get the **cache extraction → fusion → scoring**
pipeline correct on the easy case before adding the full C2C machinery. Heavy
components (heterogeneous tokenizer alignment, `C2C-C`, head modulation, learnable
gates, large-scale training) are explicitly **post-MVP**.

### MVP — must support

1. a **same-tokenizer or same-family** model pair (alignment near-identity);
2. a **shared-context MC smoke dataset**;
3. **frozen** Sharer and Receiver;
4. **KV-cache extraction** from both models on the same prompt;
5. **terminal layer mapping** for differing depths;
6. a **simple fuser or zero-gate fuser** (residual add; gate forced off/identity);
7. **logit-based** zero-shot multiple-choice scoring;
8. the standardized **four output files**;
9. a **zero-gate sanity check**: greedy/deterministic decoding makes predictions
   and accuracy match Direct Receiver **exactly**; logits match within a small
   numerical tolerance (from fp/kernel nondeterminism in the *forward*, not from
   the decoder). Implementing zero-gate as a **literal identity passthrough**
   (Receiver cache returned unchanged) makes the caches bit-identical and the
   match exact by construction. Proves the cache path is wired but inert before
   training.

### Post-MVP — added as later milestones

- a frozen-model training loop that trains **only** the fuser/gate;
- the full fuser: input-aware **head modulation** + **learnable gate**
  (Gumbel-sigmoid + temperature annealing);
- the **`C2C-C`** projection variant (3-layer MLP to the Receiver dimension);
- **heterogeneous tokenizer** alignment (Receiver→Sharer, cross-family);
- checkpoint save/load with a model-pair compatibility record;
- larger **OpenHermes-2.5** training and the four full MC datasets.

The first version does **not** need (any milestone):

- cross-context / partitioned data (that is LCF-X — see boundary above);
- free-form generation scoring (use logit MC first; add generation later);
- multi-Sharer fusion;
- distributed or multi-GPU training;
- long-context (LongBench) scaling experiments;
- depth-normalized layer mapping (terminal alignment only at first);
- streaming generation or cache reuse beyond the single fused forward.

## Setting: Shared-Context Only (Boundary)

C2C is shared-context by construction. In the normalized schema this means
`sharer_context == receiver_context` and the task carries `choices`. We therefore
**do not** reuse the partitioned HotpotQA-bridge file; instead we prepare the
four multiple-choice datasets as shared-context examples.

**Explicit boundary.** This C2C-style baseline is **shared-context only by
construction**. Because the C2C-style cache-transfer **mechanism** relies on the
Sharer and Receiver encoding the *same* prompt, token-level alignment and cache
fusion are only meaningful there. This shared-context C2C baseline must **not**
be evaluated on the partitioned HotpotQA-bridge (LCF-X cross-context) setting —
that is a different method's benchmark. If we later want cache transfer under
different-evidence / cross-context conditions, that is a **separate research
extension** (`c2c-cross` or similar), reported apart from the shared-context C2C
baseline, never as a drop-in C2C-style result.

Flow per example:

```text
Sharer input:   shared context + question + choices   -> Sharer KV cache
Receiver input: shared context + question + choices   -> Receiver KV cache
Fuser:          fused cache = Receiver cache + F(Receiver cache, aligned Sharer cache)
Receiver:       score each choice's logits under the fused cache -> prediction
```

## Data Contract

Reuse the existing normalized example
([methods/t2t/schema.py](../methods/t2t/schema.py)); no new schema is needed:

```json
{
  "id": "example-id",
  "question": "Question stem",
  "choices": ["A. London", "B. Paris", "C. Berlin", "D. Rome"],
  "answer": "A",
  "sharer_context": "Shared context (may be empty for knowledge MC)",
  "receiver_context": "Same as sharer_context in shared-context mode",
  "metadata": {"dataset": "mmlu_redux", "subject": "..."}
}
```

`choices` are **labeled option strings** — option label + option text, e.g.
`"B. Paris"` — so the prompt renders real answer text. The gold `answer` is the
matching **label** (`"A"`), which is also the label-token the scorer ranks (see
*Logit-MC scoring policy*).

A new `scripts/prepare_shared_context_mc.py` builds these JSONL files for
MMLU-Redux, ARC-Challenge, OpenBookQA, and MMLU-Pro (and optionally C-Eval),
mirroring how `prepare_hotpotqa_bridge.py` produces the cross-context file.
Dataset-specific formatting stays in the prepare script, not in the runner.

## Method Architecture

### 1. Dual-model cache wrapper (`methods/c2c/model.py`)

- Load Sharer and Receiver as frozen HF causal LMs (deferred torch/transformers
  imports, mirroring [methods/t2t/model.py](../methods/t2t/model.py)).
- Run a forward pass that returns `past_key_values` (the KV cache) for a given
  prompt, plus the per-token input ids needed for alignment.
- Expose hooks to **inject** a fused cache into the Receiver before the answer is
  scored/generated.
- Record per-model layer count, hidden size, head count/dim for compatibility.

### 2. Token alignment (`methods/c2c/alignment.py`)

- Given the Sharer and Receiver tokenizations of the same prompt string, build a
  `Receiver position -> Sharer position` map by decode-then-re-encode; for
  one-to-many, choose the Sharer token with maximal string coverage.
- Produce a `kv_cache_index` marking which Receiver positions (context/question)
  receive fused cache vs which (response prefix) stay Receiver-only.
- Validate alignment coverage; warn/skip examples where alignment is degenerate.

### 3. Layer mapping (`methods/c2c/mapping.py`)

- Implement **terminal alignment** `G(n)`: map Receiver layer n to a Sharer layer
  from the top down. Keep the interface pluggable so depth-normalized mapping can
  be added later.

### 4. Cache fuser (`methods/c2c/fuser.py`)

Build in two stages matching the Scope split.

**MVP fuser — zero-gate / simple residual wrapper.**

- A minimal `CacheFuser` whose forward returns the Receiver cache with the gate
  **forced off** — a **literal identity passthrough** so the fused cache is
  bit-identical to Direct Receiver. Its purpose is to exercise the
  extract → align → map → inject → score path, **not** to add information.
- With the gate off it must be **inert** — the zero-gate sanity check (reproduces
  Direct Receiver; see tolerance note in Scope).
- Expose `trainable_parameter_count()` and `state_dict` from the start, so the
  output/parameter contract is in place before the full fuser exists.

**Full fuser — projection + head modulation + learnable gate (post-MVP).**

- Per aligned layer: concatenate Receiver and (projected) Sharer KV, apply
  projection + feature fusion, apply input-aware **head modulation**, and add the
  result as a **residual** to the Receiver cache.
- **Learnable gate** per layer via Gumbel-sigmoid with temperature annealing
  (differentiable in training, binarized at inference).
- Provide `C2C` (direct concat) and `C2C-C` (3-layer MLP projection to Receiver
  dim) variants behind a config flag.

## Training

### Three-stage forward (per the paper)

1. **Forward**: Sharer and Receiver each encode the shared prompt → two caches.
2. **Fusion**: replace the Receiver's context cache with the fused cache.
3. **Supervision**: Receiver prefills the target response conditioned on the
   fused cache; next-token loss flows back **through the fuser only**.

### Loop (`methods/c2c/train.py`)

- Freeze both models; optimize only `CacheFuser` parameters.
- Objective: next-token cross-entropy on the Receiver's response tokens.
- Data: first ~500k **OpenHermes-2.5** samples (configurable subset for smoke).
- Anneal the gate temperature on a schedule from soft to near-binary.

**Gradient-flow caveat (easy to get wrong).** "Frozen" means the model
*parameters* do not update — set `requires_grad=False` on all Sharer and
Receiver parameters. It does **not** mean the Receiver forward is run under
`torch.no_grad()`. During fuser training, gradient must flow **from the loss,
back through the Receiver's forward computation, into the fused cache, and into
the fuser**. Wrapping the Receiver forward in `torch.no_grad()` (or
`inference_mode`) severs that path and the fuser never learns. Concretely: the
Sharer encode that *produces* its cache can be `no_grad` (its cache is a fixed
input to the fuser), but the Receiver's fusion-and-supervision forward must run
**with grad enabled**; only its parameters are frozen.
- Save a checkpoint containing: fuser `state_dict`, gate state, fuser config,
  layer-mapping choice, and a **model-pair record** (Sharer/Receiver ids,
  tokenizer hashes, layer/hidden/head shapes) for compatibility checks at load.

### Checkpoint format

Store under `data/checkpoints/c2c/<pair>/`. Loading validates the model-pair
record against the live models before inference and refuses on shape mismatch.

## Training Run Plan (project-scale trained fuser)

Target (default scale): a **~500k OpenHermes-2.5** training run for the default
primary pair, producing the reportable trained fuser. This is the post-MVP
milestone that turns the verified cache path into the actual method. The corpus,
subset size, and pair are project choices (see *Project Adaptation Policy*); the
~500k default is **reference-compatible** with the paper, not mandatory.

**Source of truth for hyperparameters.** Copy concrete values from the C2C
reference repo configs (<https://github.com/thu-nics/C2C>); the values below are
**proposed defaults to confirm against that repo**, not invented final numbers.

| Knob | Proposed default | Notes |
|------|------------------|-------|
| Optimizer | AdamW on **fuser/gate params only** | both LMs `requires_grad=False` |
| Learning rate | `1e-3` | small module; cosine decay |
| Warmup | 1–3% of steps | stabilizes early gate/projection |
| Effective batch | 32–64 (via grad-accum) | tune to VRAM |
| Max sequence length | 1024 | cap OpenHermes samples for memory |
| Epochs | ~1 over 500k | ≈ the paper's budget |
| Precision | bf16/fp16 (CUDA 11.8 build) | matches repo env constraint |
| Gate temperature τ | anneal `~1.0 → ~0.1` | soft→near-binary over training |

**Gate annealing.** Schedule τ from soft to near-binary across the run (linear or
exponential); log τ alongside loss so the soft→hard transition is visible.

**Checkpoint selection.** Evaluate on a small held-out **dev MC set** (and/or
held-out OpenHermes loss) every `N` steps; keep **best-by-dev-accuracy** plus the
**last** checkpoint. Record the selected step in the checkpoint's model-pair
record.

**Compute & resumption.** Single pair (Qwen2.5-0.5B-Instruct → Qwen3-0.6B) on the
RunPod GPU (torch CUDA-11.8 build, per
[PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)). Support `--resume <ckpt>` and
`--max-train-samples` (smoke → subset → full). Estimate and record wall-clock and
peak VRAM in the train log.

**Staged training ladder (run in order):**

1. **Smoke** — a few hundred steps on a tiny subset: loss decreases; at step 0 the
   zero-gate equivalence (== Direct Receiver) still holds. Plumbing only.
2. **Subset** — ~50–100k samples: confirm dev MC accuracy rises **above Direct
   Receiver**. This is the "the method actually helps" gate before paying for full.
3. **Full** — ~500k samples: the reportable run; produces the checkpoint used in
   the experiment matrix.

**Training logs.** Emit per-interval loss, gate values/τ, lr, throughput to a
train log, and write a `train_metadata.json` (data subset size, seed, steps,
selected checkpoint, commit hash, library versions) — **no silent truncation**:
if a subset is used, record the realized sample count.

## Inference Runner (`methods/c2c/runner.py`)

Mirror the structure of [methods/t2t/runner.py](../methods/t2t/runner.py): keep
the runner thin and torch-light where possible, returning a structured result.

Per example:

1. Encode the shared prompt in both models → Sharer + Receiver caches.
2. Align tokens, build `kv_cache_index`, map layers.
3. Fuse: `fused = receiver_cache + F(receiver_cache, aligned_sharer_cache)`.
4. **Score** each choice with logit-based MC scoring under the fused cache.
5. Parse the predicted choice; score against the gold answer.
6. Emit a `C2CResult`. C2C has **no natural-language Sharer message**, so the
   record drops T2T's `sharer_message` / `sharer_prompt` and uses C2C-native
   fields instead:

   ```json
   {
     "id": "example-id",
     "method": "c2c",
     "shared_prompt": "...",
     "choice_scores": [-12.3, -9.8, -15.1, -11.0],
     "prediction": "B",
     "answer": "B",
     "is_correct": true,
     "latency": {
       "sharer_encode_ms": 0.0,
       "receiver_encode_ms": 0.0,
       "alignment_ms": 0.0,
       "fuse_ms": 0.0,
       "score_ms": 0.0,
       "total_ms": 0.0
     },
     "token_counts": {"sharer_input": 0, "receiver_input": 0},
     "metadata": {"fuser_variant": "zero_gate", "scoring": "logit_mc"}
   }
   ```

Provide a `run_c2c_example(...)` analogous to `run_t2t_example(...)` and a
no-fusion **Direct Receiver** path for the shared-context floor baseline (the
Direct Receiver rung; "Receiver-only" is reserved for the positional-cache sense).

## Evaluation

- Primary metric: **logit-based zero-shot multiple-choice accuracy** (compute the
  Receiver's likelihood of each choice and pick the argmax). This is the **default
  scoring approach** used in C2C/LCF-style work, applied here to the project's MC
  evaluation sets (default: the four below).
- Report alongside accuracy: **trainable parameter count** and **adapter (fuser)
  size on disk**, plus latency — these are first-class comparison axes in the
  project ([PROJECT_PLAN.md](PROJECT_PLAN.md), Evaluation).

### Logit-MC scoring policy

There are two ways to score MC by likelihood:

- **label-token** — compare next-token probabilities `P("A" | prompt)`,
  `P("B" | prompt)`, … over the option **letters**;
- **answer-text** — compare the (length-normalized) sequence likelihood of each
  option's full text, e.g. `P("Paris" | prompt)`.

**MVP uses label-token scoring.** It is simpler, free of length bias, and
trivially shared with the T2T rung; the prompt ends by asking for the option
letter, so the next-token distribution over letters is the decision. Answer-text
scoring is an optional post-MVP variant. Whichever is used is recorded as
`scoring_policy` in `run_metadata.json` so the policy is never ambiguous.

Label-token scoring assumes each option label (`A`…`D`) is a **single token** at
the decision position for the given tokenizer; verify this per tokenizer (most
Qwen/Llama tokenizers satisfy it for bare letters), otherwise fall back to
first-token or answer-text scoring.

### Reference ladder (shared-context)

The `half ≤ T2T ≤ full` ladder is a **cross-context** story and does **not**
apply here: because `sharer_context == receiver_context`, "Direct Receiver"
already sees the full shared context, so there is no half/full gap to recover.
The correct shared-context ladder is:

| Rung | What it is |
|------|-----------|
| **Direct Receiver** | Receiver answers the shared prompt alone, no fusion. |
| **T2T (shared-context)** | Sharer writes an NL message on the *same* context; Receiver reads context + message. |
| **C2C zero-gate** | Fusion wired but inert — must **equal** Direct Receiver predictions/accuracy (logits within tolerance), sanity. |
| **C2C trained-fuser** | The actual method; gain = lift over Direct Receiver. |

C2C's value is the accuracy lift of the trained fuser over Direct Receiver,
with T2T as the *text-communication* point of comparison at equal context.

### Scoring comparability

C2C uses **logit-based MC**; the current T2T baseline uses **generation
parsing**. These must not be silently compared as if identical. Required:

- every run records its mode in `run_metadata.json` as
  `scoring="logit_mc"` or `scoring="generation_parse"`;
- the comparison/visualization scripts **group and report by scoring mode**,
  never mixing the two on one axis;
- to make the C2C-vs-T2T row in the shared-context ladder fair, **add a
  logit-MC scoring path to T2T** for shared-context MC datasets (so both methods
  are scored `logit_mc`). **Decision: we add this T2T path up front** (see the
  Experiment Plan below), so the whole shared-context ladder is scored
  `logit_mc` from the start rather than mixing scoring modes.

## Experiment Plan (Run Matrix)

**Default first reportable comparison, subject to project model/dataset
selection** (see *Project Adaptation Policy*). The pair and dataset list below are
defaults, not fixed requirements:

- **Model pair:** single primary pair — Sharer `Qwen/Qwen2.5-0.5B-Instruct`,
  Receiver `Qwen/Qwen3-0.6B`. Heterogeneous / additional pairs are post-MVP.
- **Datasets (all four):** MMLU-Redux, ARC-Challenge, OpenBookQA, MMLU-Pro —
  prepared once each via `prepare_shared_context_mc.py`.
- **Scoring:** `logit_mc` for **every** rung (see the T2T path below), so the
  table is internally comparable.

**Rungs per dataset (the shared-context ladder):**

| Rung | Command | Scoring |
|------|---------|---------|
| Direct Receiver (no fusion) | `run_c2c.py --no-fusion` | `logit_mc` |
| T2T (shared-context) | `run_text_to_text.py --scoring logit_mc` | `logit_mc` |
| C2C zero-gate (sanity) | `run_c2c.py --fuser zero_gate` | `logit_mc` |
| C2C trained-fuser | `run_c2c.py --checkpoint <ckpt>` | `logit_mc` |

→ **4 rungs × 4 datasets = 16 inference runs**, plus the one training run that
produces `<ckpt>`. Each writes its own `--output-dir` with the four standard
files. The zero-gate run must reproduce Direct Receiver predictions/accuracy on
every dataset (logits within tolerance) — the cross-dataset sanity check.

### Shared-context T2T logit-MC path (prerequisite)

To make the T2T rung same-scoring as C2C, add a logit-MC option to the existing
T2T method (a small, contained change, not a rewrite):

- extend [methods/t2t/scoring.py](../methods/t2t/scoring.py) (or a new
  `score_choices`) with a **label-token** logit-MC scorer (matching C2C's policy)
  that ranks the option **letters** by Receiver next-token likelihood under the
  T2T Receiver prompt (context + Sharer message + question);
- add `--scoring logit_mc` to [scripts/run_text_to_text.py](../scripts/run_text_to_text.py)
  and record `scoring="logit_mc"` in `run_metadata.json`;
- this only affects shared-context MC datasets; the cross-context HotpotQA T2T
  runs keep `generation_parse` (free-form EM/F1) as before.

### Reporting

A `scripts/compare_results.py` reads the 16 `metrics.json` files and emits one
table per dataset plus an average row: **rows = rungs, columns = accuracy,
trainable_params, adapter_size_bytes, mean latency**. It **groups by
`run_metadata.scoring`** and refuses to place different scoring modes in the same
accuracy column. Output a markdown/CSV table under `results/figures/` (or
`results/compare/`). The headline read is C2C trained-fuser's **lift over Direct
Receiver**, with T2T as the equal-context text-communication reference.

### Reproducibility

- Fixed `--seed` for dataset prep and any sampling; pin dataset revisions.
- Each `run_metadata.json` records commit hash, library versions, model ids,
  checkpoint path, fuser variant, layer-mapping, dataset, and scoring mode.
- Re-running a cell with the same seed + checkpoint reproduces its `metrics.json`.

### Recording deviations

Any difference from the original C2C setting (model pair, tokenizer family,
training corpus/subset, datasets, prompt format, scoring policy) is **recorded,
not hidden**:

- inference runs → in `run_metadata.json`;
- training runs → in `train_metadata.json`;
- trained fusers → in the checkpoint's model-pair record.

This keeps comparisons honest and makes every run self-describing.

## Output Contract

Unchanged from the project standard. Each run writes into `--output-dir`:

```text
predictions.jsonl   # per example: shared_prompt, choice_scores, prediction,
                    #   answer, is_correct, latency, token_counts, metadata
                    #   (NO sharer_message — C2C has no NL message)
metrics.json        # accuracy + trainable_params + adapter_size_bytes + token totals
latency.json        # per-stage: sharer_encode / receiver_encode / alignment /
                    #   fuse / score + total
run_metadata.json   # method=c2c, sharer/receiver ids, checkpoint, fuser variant,
                    #   layer-mapping, dataset, seed, scoring=logit_mc,
                    #   scoring_policy=label_token, deviations, timestamp
```

Comparison/visualization scripts depend only on these files, never on C2C
internals — so T2T, C2C, and future runs are **comparable through the same
contract**, with each run's setting and scoring mode recorded for honest reading.
Reuse [methods/t2t/io.py](../methods/t2t/io.py) helpers where their shape already
fits; extend (do not fork) the summary helpers to add `trainable_params` /
`adapter_size_bytes`.

## Proposed Files

```text
methods/c2c/
  __init__.py
  schema.py          # reuse T2TExample, or re-export it (shared-context view)
  model.py           # dual frozen LMs; KV-cache extract + inject hooks
  alignment.py       # tokenizer alignment + kv_cache_index
  mapping.py         # terminal layer mapping G(n)
  fuser.py           # CacheFuser: projection + head modulation + learnable gate
  train.py           # frozen-model fuser training loop (3-stage forward)
  runner.py          # run_c2c_example, receiver-only path, C2CResult
  scoring.py         # logit-based multiple-choice scoring (reuse normalize/EM later)
  io.py              # reuse/extend t2t.io output writers

scripts/
  prepare_shared_context_mc.py   # build MMLU-Redux / ARC-C / OBQA / MMLU-Pro JSONL
  train_c2c.py                   # train a fuser for a model pair (staged ladder)
  run_c2c.py                     # inference + standardized outputs (--no-fusion, --fuser, --checkpoint)
  compare_results.py             # roll 16 metrics.json into per-dataset tables (groups by scoring)

methods/t2t/scoring.py           # + logit-MC scorer (shared-context T2T rung)
scripts/run_text_to_text.py      # + --scoring logit_mc

data/checkpoints/c2c/<pair>/     # trained fuser + model-pair record
```

## Implementation Steps

**MVP (cache path correct, no training):**

1. Add `prepare_shared_context_mc.py`; produce a tiny shared-context MC JSONL.
2. Build `model.py`: load two frozen LMs; extract `past_key_values`; add an
   inject hook for a replacement Receiver cache.
3. Implement `alignment.py` on a **same-tokenizer** pair first (identity map +
   `kv_cache_index`); defer heterogeneous alignment to post-MVP.
4. Implement `mapping.py` terminal alignment.
5. Implement `fuser.py` with a **zero-gate / simple** fuser first (residual add,
   gate forced off); expose trainable-parameter count.
6. Implement `runner.py` inference path with logit MC scoring; wire `run_c2c.py`
   to write the four standardized output files.
7. **Sanity gate:** the untrained zero-gate fuser must reproduce **Direct
   Receiver** predictions and accuracy exactly (logits within numerical
   tolerance) — proving fusion is wired but inert. (End of MVP.)

**Post-MVP (the actual method):**

8. Add the full fuser: `C2C-C` MLP projection, input-aware head modulation, and
   the Gumbel-sigmoid learnable gate with temperature annealing.
9. Implement `train.py` (3-stage forward, gate annealing) honoring the
   gradient-flow caveat; train on a small OpenHermes subset, then scale up.
10. Add checkpoint save/load with model-pair validation; extend to heterogeneous
    tokenizer pairs; run the full shared-context comparison vs the reference ladder.

## Smoke Test & Acceptance Criteria

Smoke (plumbing only, tiny models/subset):

- same-tokenizer pair (e.g. two Qwen models) so alignment is near-identity;
- untrained zero-gate fuser run reproduces **Direct Receiver** accuracy (sanity);
- (post-MVP) a few-hundred-step train run on OpenHermes lowers training loss.

The **MVP** is acceptable when:

- it runs end-to-end on a tiny shared-context MC smoke task;
- it extracts both caches, aligns tokens (identity case), maps layers, and
  injects a fused cache;
- inference writes `predictions.jsonl`, `metrics.json`, `latency.json`, and
  `run_metadata.json` with the C2C-native fields and `scoring="logit_mc"`;
- `metrics.json` reports accuracy and trainable parameter count;
- the **zero-gate fuser reproduces Direct Receiver predictions and accuracy
  exactly** under deterministic settings (logits within a small numerical
  tolerance) — fusion wired but inert;
- results are comparable to other methods using only the standardized files.

The **full C2C** is additionally acceptable when:

- it trains **only** the fuser/gate with both models frozen (gradient flows
  through the Receiver forward, parameters do not update) and saves a loadable
  checkpoint with a model-pair record;
- `metrics.json` also reports adapter (fuser) size on disk;
- the trained fuser is compared against the shared-context reference ladder
  (**Direct Receiver**, **T2T shared-context**, **C2C zero-gate**), with the
  gain reported as lift over Direct Receiver.

## Risks & Open Questions

- **Tokenizer alignment fidelity (treat as unsolved, not assumed).**
  Decode-and-re-encode alignment breaks easily around **whitespace, special
  tokens, and chat-template structure** (system/role markers, BOS/EOS,
  added-generation prompts). Do **not** assume it works. Staged plan: (1) MVP
  uses a **same-tokenizer or near-identity** pair so the map is effectively the
  identity and the cache path can be verified in isolation; (2) only after the
  extraction → fusion → scoring path is proven do we extend to **heterogeneous**
  tokenizers (e.g. Llama Sharer → Qwen Receiver), with explicit alignment-coverage
  checks and the ability to skip/flag degenerate examples.
- **Cache injection API drift.** `past_key_values` formats differ across
  transformers versions / cache classes; isolate this in `model.py` behind a
  small adapter so version changes do not leak into the fuser.
- **Training cost.** Full 500k-sample training is out of scope for a smoke run;
  expose a `--max-train-samples` knob and report the realized subset in
  `run_metadata.json` (no silent truncation).
- **Logit MC vs generation scoring.** C2C reports logit-based MC; T2T reports
  generation parsing. Keep both clearly labeled in `run_metadata.json.scoring`
  so comparisons are not silently mixing scoring modes.

## Follow-Up After MVP

- Add C-Eval and MMLU-Pro; add depth-normalized layer mapping as an ablation.
- Add the `C2C-C` MLP projection and gate-temperature ablations.
- Add heterogeneous Sharer/Receiver pairs and report per-pair tables.
- Share the cache wrapper / alignment utilities with the LCF plan where the
  shared-context assumptions overlap (LCF differs mainly in the compressed
  bottleneck), without coupling the two method packages.
```
