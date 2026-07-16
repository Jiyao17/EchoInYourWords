"""EIYW translator (v2): LCF-X pooled Sharer slots cross-attend into the
Receiver's raw context embeddings -> soft tokens in the Receiver's embedding
space.

Pipeline:

1. **Sharer pool (LCF-X Fig. 3; paper-exact at ``k = 1``; unchanged).** The
   Sharer's context is encoded (``no_grad``) and pooled layer-by-layer:
   within-span attention (``q_base^(l)``, ``N_s -> P`` over natural paragraph
   blocks or fixed chunks) then cross-span attention with up to ``k`` queries
   (``q_layer^(l,j)``) over that layer's own ``P`` span summaries — ``k``
   position-free ``K*, V* [B, H_s, 1, D_s]`` slots per layer, in Sharer
   geometry. Flattening ``K*‖V*`` per slot gives ``[B, k·L_s, 2·H_s·D_s]``.
   Because the pool is identical to LCF-X's, any gain over LCF-X is
   attributable to what happens *after* pooling.

2. **Slot queries.** ``W_S: 2·H_s·D_s -> d_model`` projects the slots into the
   translator width; a learned **layer embedding** (which layer) and, for
   ``k > 1``, a **query-index embedding** (which cross-span query) tag each
   slot. No side embedding — the two streams are structurally distinct now.
   No sequence PE either: the slots are position-free summaries by design.

3. **Receiver memory.** The Receiver's own context is a raw **embedding
   lookup** ``[B, N_r, d_hidden]`` — no Receiver forward pass (v1 ran the full
   frozen Receiver over its context just to pool the cache; v2 deletes that
   model call). ``W_R: d_hidden -> d_model``, then **sinusoidal positional
   encoding** is added: unlike a KV cache, raw input embeddings carry no
   order, so without PE the cross-attention could not distinguish "A before B"
   from "B before A" in the Receiver's context. Sinusoidal (not learned)
   keeps it length-agnostic — ``N_r`` varies freely at inference, matching
   the pool's span-scheme invariance. This is the only place sequence order
   enters the translator.

4. **Decoder-shaped fusion blocks.** ``num_blocks`` pre-LN residual blocks,
   each ``{self-attn over the slots, cross-attn from slots into the receiver
   memory, MLP}``. The slots are the query stream, so the output token count
   is ``k·L_s`` — bounded and independent of both context lengths — and the
   cross-attention cost is ``S·N_r`` (no ``N_r²`` term, which a
   concat-then-self-attend design would pay). Padded memory positions are
   excluded via ``receiver_lengths`` -> key padding mask.

5. **Output head.** LayerNorm then ``W_out: d_model -> receiver_hidden``
   produces the soft tokens ``[B, k·L_s, receiver_hidden]``, prepended to the
   Receiver's input embeddings (see ``model.EIYWPair.decode_receiver``). The
   Receiver's own RoPE assigns them positions 0..S-1 at prefill, so they need
   no extra PE.

**Adaptive token budget** (``adaptive_k_tokens = w > 0``): a fixed slot count
compresses a 100-token and a 4000-token Sharer context into the same budget —
a capacity ceiling on long inputs and wasted prefill on short ones. With
``w > 0`` the number of *active* cross-span queries scales with the input:
``k(N_s) = clamp(ceil(N_s / w), 1, queries_per_layer)``, i.e. one query per
``w`` Sharer tokens up to the trained maximum. ``q_layer``'s ``k`` axis holds
independent queries, so using a prefix of them is well-defined; training over
mixed-length data exercises every prefix. The count is resolved **per batch**
from the batch's longest real Sharer length (keeping ``translate``'s output a
single rectangular tensor); at inference batches are single examples, so the
budget is exact. ``w = 0`` (default) always uses all ``queries_per_layer``
queries — the fixed-budget setting, paper-exact at ``queries_per_layer = 1``.

The output magnitude is **calibrated to the Receiver's measured embedding
RMS** (``embed_scale`` buffer x learnable scalar ``out_gain``, init 0.05):
real input-embedding loudness differs ~100x across model families (Llama-3.2
~0.01 vs Gemma-3 ~1.0 after its internal sqrt(hidden) rescale), so any fixed
init is simultaneously louder than every real token for one receiver and
inaudible for another — uncalibrated, Llama receivers started at train loss
9.7 (vs the ~3 receiver-only floor) and collapsed below chance. Calibrated,
every receiver starts at ~5% of its own embedding loudness — near, but not
at, zero: exact-zero soft embeddings are a numerical degeneracy that makes
the frozen Receiver's backward pass non-finite (``0 * inf``), whereas LCF-X's
exact-zero init is safe because it lands on an *additive KV residual*, not an
embedding fed through the model.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _span_starts(n: int, num_spans: int, boundaries: "list[int] | None") -> list[int]:
    """Span start offsets within ``n`` real tokens (same rule as LCF-X's pool)."""

    if boundaries:
        starts = sorted({min(max(int(b), 0), n - 1) for b in boundaries})
        if starts[0] != 0:
            starts.insert(0, 0)
        return starts
    p = max(1, min(int(num_spans), n))
    base, extra = divmod(n, p)
    starts, pos = [], 0
    for i in range(p):
        starts.append(pos)
        pos += base + (1 if i < extra else 0)
    return starts


def _resolve_d_model(spec: "str | int", receiver_hidden: int) -> int:
    """Resolve a ``--d-model`` spec against the Receiver's hidden width.

    An int (or int-like string; the default is ``512``) -> that exact width:
    a **fixed, receiver-independent hyperparameter**. The old ``"half"``
    default scaled the translator quadratically with the Receiver
    (16·N·d² per block: 40M params for a 2048-hidden receiver vs that
    pair's ~11M LCF-X fuser), which both bloated the adapter and made it
    harder to train on the small-dataset step budgets. ``"half"``
    (``receiver_hidden // 2``) and ``"hidden"``/``"full"``
    (``receiver_hidden``) remain available as sweep options.
    """

    if isinstance(spec, bool):  # guard: bool is an int subclass
        raise TypeError("d_model spec must be str or int, not bool")
    if isinstance(spec, int):
        return spec
    s = str(spec).strip().lower()
    if s == "":
        return 512
    if s == "half":
        return max(1, receiver_hidden // 2)
    if s in ("hidden", "full", "receiver"):
        return receiver_hidden
    return int(s)


def _safe_num_heads(d_model: int, num_heads: int) -> int:
    """Largest head count <= ``num_heads`` that divides ``d_model`` (>=1)."""

    for h in range(min(num_heads, d_model), 0, -1):
        if d_model % h == 0:
            return h
    return 1


def _sinusoidal_pe(n: int, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Standard sin/cos positional encoding ``[n, d]`` (length-agnostic, no params)."""

    position = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    half = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    div = torch.exp(half * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)[:, : d // 2]  # d odd: cos gets one fewer column
    return pe.to(dtype)


class _LCFXPool(nn.Module):
    """LCF-X pool (Fig. 3) for the Sharer: per-layer, paper-exact at ``k = 1``.

    Per layer ``l``: within-span attention with ``q_base^(l)`` (``N -> P``),
    then cross-span attention over that layer's own ``P`` span summaries with
    up to ``k`` (= ``queries_per_layer``) queries ``q_layer^(l,j)``
    (``P -> k``) — ``k`` position-free ``K*, V* [B, H, 1, D]`` slots per
    layer, no cross-layer mixing, ``H``/``D`` untouched. Scores are a direct
    dot product against the raw K (no key/value projection) and the same
    weights pool K and V; ``P`` is not a parameter shape, so the span scheme
    can vary freely at inference. ``k_active`` (adaptive token budget) pools
    with only the first ``k_active`` cross-span queries — each query is an
    independent parameter row, so a prefix is well-defined.

    Output: the flattened slot matrix ``[B, k_active·L, 2·H·D]`` (``K*‖V*``
    per slot; layer-major, query-minor: slot index = ``l·k_active + j``).
    """

    def __init__(
        self, layers: int, kv_heads: int, head_dim: int, queries_per_layer: int = 1
    ) -> None:
        super().__init__()
        self.layers = layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.queries_per_layer = max(1, int(queries_per_layer))

        self.q_base = nn.Parameter(torch.empty(layers, kv_heads, head_dim))
        self.q_layer = nn.Parameter(
            torch.empty(layers, self.queries_per_layer, kv_heads, head_dim)
        )
        for query in (self.q_base, self.q_layer):
            nn.init.kaiming_normal_(query)
            with torch.no_grad():
                query.mul_(0.02)

    def _pool(
        self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One attention pool: ``query [H,D]`` over ``K,V [1,H,n,D]`` -> ``[1,H,D]`` x2."""

        scores = torch.einsum("hd,bhnd->bhn", query, keys) / self.head_dim**0.5
        weight = scores.softmax(dim=-1).to(keys.dtype)
        pooled_k = torch.einsum("bhn,bhnd->bhd", weight, keys)
        pooled_v = torch.einsum("bhn,bhnd->bhd", weight, values)
        return pooled_k, pooled_v

    def forward(
        self,
        summary: list[tuple[torch.Tensor, torch.Tensor]],
        lengths: "torch.Tensor | None" = None,
        num_spans: int = 5,
        span_boundaries: "list[list[int] | None] | None" = None,
        k_active: "int | None" = None,
    ) -> torch.Tensor:
        """``L`` x ``(K, V)`` each ``[B, H, N, D]`` (raw, possibly padded) ->
        slot matrix ``[B, k_active·L, 2·H·D]``."""

        k = self.queries_per_layer if k_active is None else max(
            1, min(int(k_active), self.queries_per_layer)
        )
        weight_dtype = self.q_base.dtype
        batch = summary[0][0].shape[0]
        full_n = summary[0][0].shape[2]

        rows = []
        for b in range(batch):
            n = int(lengths[b]) if lengths is not None else full_n
            n = max(1, n)
            bounds = span_boundaries[b] if span_boundaries is not None else None
            starts = _span_starts(n, num_spans, bounds)
            ends = starts[1:] + [n]
            slots = []
            for layer_idx, (key, value) in enumerate(summary):
                kb = key[b : b + 1, :, :n, :].to(weight_dtype)
                vb = value[b : b + 1, :, :n, :].to(weight_dtype)
                spans = [
                    self._pool(self.q_base[layer_idx], kb[:, :, s:e, :], vb[:, :, s:e, :])
                    for s, e in zip(starts, ends)
                ]
                spans_k = torch.stack([sk for sk, _ in spans], dim=2)  # [1, H, P, D]
                spans_v = torch.stack([sv for _, sv in spans], dim=2)
                for j in range(k):
                    star_k, star_v = self._pool(
                        self.q_layer[layer_idx, j], spans_k, spans_v
                    )
                    slots.append(
                        torch.cat([star_k.reshape(1, -1), star_v.reshape(1, -1)], dim=-1)
                    )
            rows.append(torch.stack(slots, dim=1))  # [1, k*L, 2*H*D]
        return torch.cat(rows, dim=0)  # [B, k*L, 2*H*D]


class _XBlock(nn.Module):
    """Pre-LN residual block: self-attn over slots, cross-attn into the memory, MLP."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        h = self.norm_self(x)
        attn, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + attn
        h = self.norm_cross(x)
        attn, _ = self.cross_attn(
            h, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + attn
        x = x + self.mlp(self.norm_mlp(x))
        return x


class EIYWTranslator(nn.Module):
    """Pools the Sharer per layer, then its slots cross-attend into the Receiver's
    context embeddings to become soft tokens."""

    def __init__(
        self,
        *,
        sharer_layers: int,
        sharer_kv_heads: int,
        sharer_head_dim: int,
        receiver_hidden: int,
        d_model: int,
        queries_per_layer: int = 1,
        adaptive_k_tokens: int = 0,
        num_blocks: int = 1,
        num_heads: int = 8,
        receiver_embed_rms: float = 1.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.sharer_layers = sharer_layers
        self.sharer_kv_heads = sharer_kv_heads
        self.sharer_head_dim = sharer_head_dim
        self.receiver_hidden = receiver_hidden
        self.d_model = int(d_model)
        self.queries_per_layer = max(1, int(queries_per_layer))
        self.adaptive_k_tokens = max(0, int(adaptive_k_tokens))
        self.num_blocks = max(1, int(num_blocks))
        self.num_heads = _safe_num_heads(self.d_model, num_heads)

        d_model = self.d_model

        # 1. Sharer pool, paper-exact at k=1; k>1 keeps k distinct cross-span
        # queries per layer, so the extra slots carry real information out of
        # the cache instead of being post-hoc copies of one collapsed slot.
        self.sharer_pool = _LCFXPool(
            sharer_layers, sharer_kv_heads, sharer_head_dim, self.queries_per_layer
        )

        # 2. Slot queries: width projection + identity tags. No sequence PE
        # (the slots are position-free) and no side embedding (the query and
        # memory streams are structurally distinct in this design).
        self.sharer_proj = nn.Linear(2 * sharer_kv_heads * sharer_head_dim, d_model)
        self.layer_emb = nn.Parameter(torch.empty(sharer_layers, d_model))
        nn.init.normal_(self.layer_emb, std=0.02)
        if self.queries_per_layer > 1:
            self.query_emb = nn.Parameter(torch.empty(self.queries_per_layer, d_model))
            nn.init.normal_(self.query_emb, std=0.02)
        else:
            self.register_parameter("query_emb", None)
        self.norm_slots = nn.LayerNorm(d_model)

        # 3. Receiver memory: raw context embeddings -> d_model, + sinusoidal
        # PE (raw embeddings are order-free; see module docstring), normed once.
        self.receiver_proj = nn.Linear(receiver_hidden, d_model)
        self.norm_mem = nn.LayerNorm(d_model)

        # 4. Decoder-shaped fusion blocks (self-attn + cross-attn + MLP).
        self.blocks = nn.ModuleList(
            [_XBlock(d_model, self.num_heads) for _ in range(self.num_blocks)]
        )

        # 5. Output head into the Receiver's embedding space, calibrated to the
        # Receiver's measured embedding loudness. Soft tokens are emitted as
        #   out(norm_out(x)) * embed_scale * out_gain
        # where ``embed_scale`` is the Receiver's real input-embedding RMS
        # (a buffer: measured once through the embedding module at pair load,
        # then persisted with the checkpoint) and ``out_gain`` is a learnable
        # scalar starting at 0.05 (~5% of the receiver's own embedding
        # loudness at step 0 — near, but safely not at, zero; see module
        # docstring for the collapse this prevents).
        self.norm_out = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, receiver_hidden)
        self.register_buffer("embed_scale", torch.tensor(float(receiver_embed_rms)))
        self.out_gain = nn.Parameter(torch.tensor(0.05))
        if zero_init:
            nn.init.normal_(self.out.weight, std=0.02)
            nn.init.zeros_(self.out.bias)

    @property
    def num_tokens(self) -> int:
        """Maximum soft-token count = ``queries_per_layer * sharer_layers``.

        With ``adaptive_k_tokens > 0`` the emitted count is
        ``active_queries(N_s) * sharer_layers <= num_tokens``.
        """

        return self.queries_per_layer * self.sharer_layers

    def active_queries(self, sharer_tokens: int) -> int:
        """Cross-span queries active for a Sharer context of ``sharer_tokens``.

        ``adaptive_k_tokens = w > 0``: one query per ``w`` Sharer tokens,
        clamped to ``[1, queries_per_layer]``; ``w = 0``: always the full
        ``queries_per_layer`` (fixed budget).
        """

        if self.adaptive_k_tokens <= 0:
            return self.queries_per_layer
        need = -(-max(1, int(sharer_tokens)) // self.adaptive_k_tokens)  # ceil
        return max(1, min(self.queries_per_layer, need))

    def translate(
        self,
        sharer_summary: list[tuple[torch.Tensor, torch.Tensor]],
        receiver_embeds: torch.Tensor,
        *,
        sharer_lengths: "torch.Tensor | None" = None,
        receiver_lengths: "torch.Tensor | None" = None,
        num_spans: int = 5,
        sharer_span_boundaries: "list[list[int] | None] | None" = None,
    ) -> torch.Tensor:
        """Raw Sharer cache + Receiver context embeddings -> soft tokens.

        ``sharer_summary``: ``L_s`` x ``(K, V)`` each ``[B, H_s, N_s, D_s]``
        (raw, possibly padded; ``sharer_lengths[b]`` = real token count).
        ``receiver_embeds``: ``[B, N_r, receiver_hidden]`` raw input
        embeddings of the Receiver's own context (possibly right-padded;
        ``receiver_lengths`` masks pad positions out of the cross-attention).

        Returns ``[B, k·L_s, receiver_hidden]`` where ``k`` is resolved once
        per call from the batch's longest real Sharer length (see
        :meth:`active_queries`) — exact per example at batch size 1.
        """

        full_n = sharer_summary[0][0].shape[2]
        n_max = int(sharer_lengths.max()) if sharer_lengths is not None else full_n
        k = self.active_queries(n_max)

        slots = self.sharer_pool(
            sharer_summary,
            lengths=sharer_lengths,
            num_spans=num_spans,
            span_boundaries=sharer_span_boundaries,
            k_active=k,
        )  # [B, k*L_s, 2*H_s*D_s]

        x = self.sharer_proj(slots) + self.layer_emb.repeat_interleave(k, dim=0)
        if self.query_emb is not None:
            x = x + self.query_emb[:k].repeat(self.sharer_layers, 1)
        x = self.norm_slots(x)  # [B, k*L_s, d_model]

        weight_dtype = self.receiver_proj.weight.dtype
        memory = self.receiver_proj(receiver_embeds.to(weight_dtype))  # [B, N_r, d_model]
        n_r = memory.shape[1]
        memory = memory + _sinusoidal_pe(n_r, self.d_model, memory.device, memory.dtype)
        memory = self.norm_mem(memory)

        pad_mask = None
        if receiver_lengths is not None:
            positions = torch.arange(n_r, device=memory.device)
            pad_mask = positions.unsqueeze(0) >= receiver_lengths.to(memory.device).clamp(
                min=1
            ).unsqueeze(1)  # [B, N_r], True = padding (masked out)

        for block in self.blocks:
            x = block(x, memory, memory_key_padding_mask=pad_mask)
        # [B, k*L_s, receiver_hidden], scaled to the Receiver's real embedding
        # loudness (see the output-head comment in __init__).
        return self.out(self.norm_out(x)) * (self.embed_scale * self.out_gain)

    def forward(
        self,
        sharer_summary: list[tuple[torch.Tensor, torch.Tensor]],
        receiver_embeds: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.translate(sharer_summary, receiver_embeds, **kwargs)

    @classmethod
    def from_specs(
        cls,
        sharer_spec: "object",
        receiver_spec: "object",
        *,
        d_model: "str | int" = 512,
        queries_per_layer: int = 1,
        adaptive_k_tokens: int = 0,
        num_blocks: int = 1,
        num_heads: int = 8,
        zero_init: bool = True,
    ) -> "EIYWTranslator":
        return cls(
            sharer_layers=sharer_spec.layers,
            sharer_kv_heads=sharer_spec.kv_heads,
            sharer_head_dim=sharer_spec.head_dim,
            receiver_hidden=receiver_spec.hidden,
            d_model=_resolve_d_model(d_model, receiver_spec.hidden),
            queries_per_layer=queries_per_layer,
            adaptive_k_tokens=adaptive_k_tokens,
            num_blocks=num_blocks,
            num_heads=num_heads,
            receiver_embed_rms=getattr(receiver_spec, "embed_rms", 1.0),
            zero_init=zero_init,
        )

    def arch_config(self) -> dict:
        """Self-describing architecture: every weight shape plus the runtime
        knobs (``adaptive_k_tokens``) that must match between train and eval."""

        return {
            "sharer_layers": self.sharer_layers,
            "sharer_kv_heads": self.sharer_kv_heads,
            "sharer_head_dim": self.sharer_head_dim,
            "receiver_hidden": self.receiver_hidden,
            "d_model": self.d_model,
            "queries_per_layer": self.queries_per_layer,
            "adaptive_k_tokens": self.adaptive_k_tokens,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
        }

    @classmethod
    def from_arch_config(cls, cfg: dict, *, zero_init: bool = True) -> "EIYWTranslator":
        """Rebuild a translator from an :meth:`arch_config` dict."""

        return cls(
            sharer_layers=cfg["sharer_layers"],
            sharer_kv_heads=cfg["sharer_kv_heads"],
            sharer_head_dim=cfg["sharer_head_dim"],
            receiver_hidden=cfg["receiver_hidden"],
            d_model=cfg["d_model"],
            queries_per_layer=cfg.get("queries_per_layer", 1),
            adaptive_k_tokens=cfg.get("adaptive_k_tokens", 0),
            num_blocks=cfg.get("num_blocks", 1),
            num_heads=cfg.get("num_heads", 8),
            zero_init=zero_init,
        )

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
