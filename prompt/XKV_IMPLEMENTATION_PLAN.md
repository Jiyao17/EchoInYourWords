# XKV: Dual-Cache Attention Translator — Architecture and Implementation Plan

## 1. Objective

XKV is a Sharer-to-Receiver communication method that reads the **KV caches of
both models** and predicts a **Receiver-native KV-cache update**. It combines
three ideas:

1. use the same vectorized learned-query attention pool on both caches;
2. translate the resulting joint representation with attention; and
3. emit a position-specific Receiver residual through shared K/V projections
   with deterministic per-head scaling.

Both language models remain frozen. Only the XKV pools, bottleneck, translator,
and update head are trained.

![XKV architecture](../xkv_architecture-1.svg)

## 2. Notation and cache boundary

- `B`: batch size.
- `L_s`, `L_r`: Sharer and Receiver layer counts.
- `H_s`, `H_r`: Sharer and Receiver KV-head counts.
- `D_s`, `D_r`: Sharer and Receiver head dimensions.
- `N_s^l`, `N_r^l`: real per-layer Sharer and Receiver cache lengths. These are
  normally constant over layers; Gemma-3 mixes sliding and full attention, so
  they may differ.
- `k`: direct-attention summaries retained per layer and KV head on both sides
  (Receiver-tier default `k=1/2/3/4` for 0.6B/1B/4B/8B).
- `d`: XKV bottleneck/translator width (default: Receiver hidden width divided
  by 8).

The frozen Sharer prefills its context and produces
`C_s = {(K_s^l,V_s^l)}_(l=1..L_s)`. The frozen Receiver prefills **all but the
last prompt token** and produces
`C_r = {(K_r^l,V_r^l)}_(l=1..L_r)`. XKV reads both caches once, emits a residual
`Delta C_r`, adds it to that Receiver prefix cache, and then normal decoding
continues with the held-out last prompt token. This is the same cache boundary
used by `LCFXPair.decode_receiver`, preserving causal decoding and the
zero-update equivalence check.

## 3. Architecture

### 3.1 Symmetric direct dual-cache pooling

For each side `a in {s,r}`, learn `k` queries per layer and KV head,
`Q_a in R^(L_a x H_a x k x D_a)`. Apply one vectorized scaled dot-product
attention directly over all retained raw cache positions:

```text
A_a = softmax(Q_a K_a^T / sqrt(D_a))         # [B,L_a,H_a,k,N_a]
K_a* = A_a K_a,  V_a* = A_a V_a           # [B,L_a,H_a,k,D_a]
```

Pool queries use the v9 Kaiming-normal initialization followed by `x0.02`.
There is no learned attention temperature.

The K-derived attention weights reduce K and V consistently. There is no
intermediate span axis and no Python loop over layers or slots. Each learned
query has its own parameters, so the default two slots can retain different
information rather than broadcasting one pooled value. Padding is masked.
The raw Receiver K/V positions remain available to the position decoder in
Section 3.4.

For Gemma-3 only, define `N_a* = min_l N_a^l` and pool the most recent `N_a*`
positions from every layer and crop each side's mask to the same suffix.
Non-Gemma caches keep the strict requirement that every layer has the same width.

### 3.2 Layer alignment, concatenation, and bottleneck

Map the Sharer layer axis to `L_r` when `L_s != L_r`, using the same linear
layer-axis bridge as LCF-X. For each Receiver layer `l`, flatten heads and K/V,
then concatenate the two pooled sides:

```text
x_l = [flat(K_s*^l), flat(V_s*^l), flat(K_r*^l), flat(V_r*^l)]
      in R^(2 k H_s D_s + 2 k H_r D_r)

z_l = LN(MLP(W_down x_l)) in R^d
```

The result `Z=[z_1,...,z_Lr]` has shape `[B,L_r,d]`. `W_down` is the explicit
heterogeneous-model bridge; XKV must not require matching layer, head, head-dim,
sequence, or tokenizer geometry.

The fixed concatenation order preserves side identity; optionally apply
side-specific normalization before concatenation to balance cache scales. Add a
learned Receiver-layer embedding to `z_l` before attention so the translator
does not treat the layer tokens as an unordered set.

### 3.3 Attention translator

Process `Z` with a stack of `n` pre-norm Transformer encoder blocks. Use
`n=1` by default for every Receiver and expose it as `--num-blocks` for depth
ablations. Each block has its
own parameters unless an explicit weight-tied recurrent ablation is selected:

```text
Z <- Z + MHA(LN(Z), LN(Z), LN(Z))
Z <- Z + FFN(LN(Z))
```

Repeat this update `n` times. Self-attention lets every Receiver-layer update
use evidence from all pooled Sharer and Receiver layers instead of assuming a
strict `l -> l` relationship. Use a padding mask only if a future variant
supports a variable number of layer tokens; sequence padding has already
disappeared in the pooling stage.

### 3.4 Receiver-position cross-attention decoder and KV head

Full pooling removes token positions, so recover position-specific updates by
using the raw Receiver cache as queries into the translated layer memory:

```text
Q_l = W_q [flat(K_r^l); flat(V_r^l)] + e_l  # [B,N_r,d]
U_l = Q_l + CrossAttn(LN(Q_l), K=Z, V=Z)
U_l = U_l + FFN(LN(U_l))
alpha_K, alpha_V = sigmoid(W_gate U_l)       # per Receiver KV head
Delta_K = reshape(W_up,K U_l)                # [B,H_r,N_r,D_r]
Delta_V = reshape(W_up,V U_l)                # [B,H_r,N_r,D_r]

K_r' = K_r + alpha_K * Delta_K
V_r' = V_r + alpha_V * Delta_V
```

Use separate `W_up,K` and `W_up,V`, but **share both projections and
`W_gate` across Receiver layers**. The query projection, cross-attention, and
decoder FFN are shared as well; learned Receiver-layer embeddings and
raw position queries retain layer/position identity. For Gemma-3, decode the
recent common window and left-pad full-attention layers with zeros so older
cache positions are preserved. Padding positions also receive zero updates.

Do not use layer/type Gumbel gates: completed runs left every logit at its
positive initialization, so the gates introduced stochastic fractional scaling
during training but became all-on at evaluation. Zero-initialize the two
up-projections so an untrained XKV produces exactly zero residual and therefore
reproduces Receiver-only decoding.

## 4. End-to-end execution

1. Tokenize the two private contexts independently.
2. Run frozen Sharer prefill under `no_grad`; retain raw `C_s`.
3. Run frozen Receiver prefix prefill under `no_grad`; retain raw `C_r`.
4. Pool both caches with independent vectorized `N -> k` attention modules.
5. Align the layer axes, concatenate the pooled side features, and bottleneck to
   `[B,L_r,d]`.
6. Run self-attention translation over the joint layer memory.
7. Use raw Receiver K/V positions as queries in the shared cross-attention
   decoder, then generate position-specific `(Delta K, Delta V)` through the
   two shared full-width up-projections.
8. Add the per-head-scaled residual to `C_r`.
9. Feed the held-out last Receiver prompt token, then decode autoregressively.

The update is computed once at the prompt boundary. Generated-token KV entries
are appended normally and are not repeatedly rewritten. A recurrent/recomputed
update can be studied later, but it should not be part of the MVP because it
changes latency, causality, and the comparison with LCF-X.

## 5. Training objective

- Freeze the Sharer and Receiver parameters; put both models in evaluation mode.
- Optimize only XKV with teacher-forced next-token cross-entropy on the gold
  Receiver answer.
- Produce the two prefix caches under `no_grad` and detach them. Apply the XKV
  residual, then run the frozen Receiver over the answer with gradients enabled
  through the cache values. Frozen Receiver weights need no optimizer state, but
  the forward graph must remain differentiable with respect to `Delta C_r`.
- Mask prompt and padding labels; supervise answer tokens plus the resolved stop
  token, following the existing LCF-X training path.
- Use AdamW, gradient clipping, mixed precision, validation loss for checkpoint
  selection, and the same effective-batch protocol as the baselines.
- Regularize update magnitude with a small optional term
  `lambda_delta * mean(||Delta K||_2 + ||Delta V||_2)` and log per-layer update
  RMS/per-head-alpha activity. Start with `lambda_delta=0`; enable it only if
  update norms become unstable.

## 6. Proposed repository layout

Create an independent `methods/xkv/` package, mirroring `methods/lcfx/` rather
than coupling the two methods through inheritance:

| File | Responsibility |
|---|---|
| `translator.py` | Symmetric vectorized direct-attention pools, layer bridge, joint bottleneck, self-attention translator, Receiver-position cross-attention decoder, and shared KV update head. |
| `cache.py` | Reuse or wrap `extract_kv` and `add_residual_inplace`; validate layer/shape/dtype/device invariants. |
| `model.py` | Frozen pair loading and Receiver prefix-cache decode boundary. |
| `training.py` | Batched loss, evaluation, checkpoint save/load, and training loop. |
| `runner.py` | One-example/batched inference and standardized timing/result records. |
| `schema.py` | Alias the common cross-context example schema. |
| `__init__.py` | Public exports. |

Add `scripts/train_xkv.py`, `scripts/run_xkv.py`, Makefile targets
`xkv-train`/`xkv-eval`/`xkv`, and model-stamped checkpoints under
`data/checkpoints/xkv/<dataset>/<sharer>/<receiver>/<architecture>/translator.pt`.
The architecture stamp must include all cache geometry, `d`, attention heads,
`n`, `k`, and whether the blocks/heads are shared.

## 7. Implementation phases

### Phase A — shape-safe MVP

1. Extract and batch both caches, with correct real lengths.
2. Implement one vectorized `N -> k` learned-query attention pool and apply an
   independent instance to each cache.
3. Implement layer alignment, concatenation, bottleneck, translator blocks,
   position cross-attention decoder, and zero-initialized shared update head.
4. Integrate the Receiver prefix-cache injection/decode path.

### Phase B — training and checkpointing

1. Implement single-example loss, then batched loss with length masking.
2. Add frozen-model training, evaluation, logging, checkpoint metadata, and
   mismatch guards.
3. Add CLI and Makefile integration using the common result contract.

### Phase C — evaluation and ablation

Compare XKV with Receiver-only, T2T, LCF-X, EIYW, KVTok, and the full-context
upper bound. Report EM/F1, trainable parameters, communication size, peak memory,
prefill/translator/decode latency, time to first token, and update RMS.

Required ablations:

- Sharer-only pool versus dual-cache pool;
- direct pooling versus legacy two-stage pooling;
- concatenation+MLP versus attention translator;
- shared versus per-layer update decoder;
- shared position MLP versus position cross-attention;
- split K/V latents versus full-latent up-projections;
- learned per-head scaling versus no output scaling;
- one, two, or four direct summaries (`k=1,2,4`);
- bottleneck width and translator depth.

## 8. Verification gates

1. **Shape matrix:** test homogeneous and heterogeneous pairs, including
   `L_s != L_r`, `H_s != H_r`, `D_s != D_r`, unequal sequence lengths, and
   batch rows with different real lengths. Include Gemma-3 hybrid caches with
   unequal sliding/full layer widths on either side.
2. **Pool correctness:** padded tokens receive zero attention; both outputs
   contain `k` finite, independently parameterized summaries per layer/head.
3. **Zero-update equivalence:** zero-init XKV logits and greedy output match the
   Receiver-only cache path within dtype tolerance.
4. **Gradient isolation:** finite nonzero gradients reach both pools, bottleneck,
   attention blocks, per-head scaling, and up-projections; no base-model
   parameter receives a gradient or optimizer state.
5. **Cache safety:** updates exactly match every Receiver cache layer's
   shape/dtype/device and never alter the Sharer cache.
6. **Single/batched parity:** per-example loss and predictions agree between
   single and padded batched execution.
7. **Checkpoint guards:** reject incompatible model geometry or architecture
   unless an explicit unsafe override is supplied.
8. **End-to-end smoke:** tiny train/eval writes the standard predictions,
   metrics, latency, and metadata files.

## 9. Main risks and controls

- **Receiver double-prefill:** use the first Receiver prefix prefill as the
  generation cache itself; do not run a separate Receiver encoding pass.
- **Translator cost:** never self-attend over all `L_r*N_r` raw cache positions.
  Self-attention sees only `L_r` pooled tokens; the position decoder performs
  cross-attention from `N_r` queries to `L_r` memory tokens per layer.
- **Pooling removes useful locality:** log normalized attention entropy and
  inter-slot attention-map cosine separately for the Sharer and Receiver. If
  slots collapse, add an attention-map diversity regularizer before restoring
  span pooling.
- **Updates overpower the frozen Receiver:** zero-init the up-projections, log
  update/cache RMS ratios, clip gradients, and optionally enable update-norm
  regularization.
- **Attention learns identity instead of communication:** compare against the
  Receiver-only-pool and Sharer-only-pool ablations and inspect cross-side
  attention/per-head-alpha activity.

### Recommended design choices

1. **Use `n=1` by default and sweep depth only as an ablation.** Use independent
   blocks when testing `n>1`; tied recurrence is useful only as a
   parameter-control ablation. Implement the independent case as a `ModuleList`
   of `n` blocks.
2. **Share the position decoder and two up-projections across layers.** One
   projection emits K and one emits V from the full decoded latent. Raw Receiver
   K/V provides position identity; layer embeddings and layer-specific K/V
   gates retain layer identity without independent per-layer heads.
3. **Use the joint bottleneck normalization first.** The existing
   `W_down -> MLP -> LayerNorm` controls the concatenated scale. Add side-specific
   normalization only if measured cache-scale imbalance warrants it.
4. **Use a Receiver-tier `k` default.** Use `k=1/2/3/4` for 0.6B/1B/4B/8B
   Receivers, respectively. This scales pooled information with Receiver size
   while retaining a single vectorized attention operation and an explicit
   fixed-`k` override for ablations.
5. **Keep only deterministic per-head sigmoid scaling.** Do not restore the
   layer/type Gumbel gates unless a separate ablation demonstrates learned
   sparsity and removes the train/evaluation scaling mismatch.
6. **Write updates directly in the Receiver cache's stored key space.** Cached
   keys have already received the Receiver's positional transform; do not apply
   RoPE a second time to `Delta K`. The position-query path supplies the relevant
   position information.

## 10. Parameter estimate

Let `F_s=H_s*D_s`, `F_r=H_r*D_r`, and let `k` be the direct-attention slot
count used on both sides. With PyTorch-style linear/MHA biases,
FFN width `4d`, a shared position decoder, two shared full-width up-projections,
and no optional side normalization, the trainable count is:

| Component | Parameters |
|---|---:|
| Dual direct-attention pools | `k*(L_s*F_s + L_r*F_r)` |
| Optional layer map (`L_s != L_r`) | `L_s*L_r + L_r` |
| Joint down-projection | `2*k*(F_s + F_r)*d + d` |
| Joint bottleneck MLP + LayerNorm | `8*d^2 + 7*d` |
| Shared Receiver-layer embedding | `L_r*d` |
| Each independent translator block | `12*d^2 + 13*d` |
| Position query projection | `2*F_r*d + d` |
| Position cross-attention + FFN + norms | `12*d^2 + 13*d` |
| Shared update head | `2*H_r*d + 2*H_r + 2*(d*F_r + F_r)` |

For the repository default `Qwen3-0.6B -> Qwen3-0.6B`
(`L_s=L_r=28`, `H_s=H_r=8`, `D_s=D_r=128`) with `k=1` and
`d=1024//8=128`:

| Component | Parameters |
|---|---:|
| Dual direct-attention pools | 57,344 |
| Joint bottleneck, norm, and layer embedding | 659,968 |
| `n=1` attention translator | 198,272 |
| Shared Receiver-position decoder | 460,544 |
| Shared update head | 266,256 |
| **Total (`n=1`)** | **1,642,384 (~1.64M)** |

With independent attention blocks, each extra repetition adds **198,272**
parameters:

```text
P_shared(n) = 1,444,112 + 198,272*n
```

Thus `n=2` is 1,840,656 parameters and `n=4` is 2,237,200. If the same block is
recurrently applied `n` times with tied weights, the parameter count stays at
1,642,384 while attention compute grows approximately linearly with `n`.

The current symmetric direct-pool design has 5,379,856 parameters at fixed
`d=256,k=2` for this pair. Changing `k` affects only the learned pool queries
and the joint down-projection input; it does not alter the translator sequence
length or the `[B,L_r,d]` bottleneck output.

These totals exclude frozen Sharer/Receiver weights, optimizer state, and
activation memory. Different model cache geometries change the pool, input, and
output projection terms, so the implementation should report the exact count
from `sum(p.numel() for p in xkv.parameters() if p.requires_grad)`.

## 11. MVP defaults

Use `d=receiver_hidden//8`, Receiver-tier `k=1/2/3/4` for 0.6B/1B/4B/8B, `n=1`, with
independent direct-attention queries per layer/head and side, 8 attention heads
(or the largest divisor of `d` not exceeding 8), independently parameterized
self-attention blocks, a shared position
cross-attention decoder, two shared full-width up-projections, FFN width `4d`,
dropout `0.1`, deterministic per-head sigmoid scaling, and no layer/type gate.
Treat these as
sweep defaults, not claims of optimality. An optional tied-block implementation
can separate depth from parameter count. The current CLI names are `--d-model`
(`eighth` by default, or an explicit integer), `--num-blocks`, and `--pool-k`
(`--pool-slots` remains an alias)
(`XKV_DMODEL`, `XKV_BLOCKS`, and `XKV_POOL_K` in the Makefile;
`XKV_POOL_SLOTS` is an alias). `k` is
checkpoint-shape metadata and must match at training and evaluation.
