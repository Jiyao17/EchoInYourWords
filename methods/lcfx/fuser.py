"""LCF-X fuser: paper-faithful 2-stage pooling + joint key-value bottleneck projector.

Pooling (paper Fig. 3 top, Section 3.2, Appendix E.3) — runs **per layer**, with
two learned query vectors per layer (``q_base^(l)``, ``q_layer^(l)``; 2*H*D
parameters per layer, 57,344 for a 28-layer Qwen3-0.6B pair):

1. **Within-span attention** (``q_base^(l)``): the sharer's context is divided
   into ``P`` contiguous token spans ("a paragraph, passage, or fixed-size
   chunk"). Within each span the layer's shared base query attends over that
   span's raw K and pools both K and V with the same weights:
   ``[B, H, N_s, D] -> K̄, V̄ [B, H, P, D]``. The same query is reused for every
   span, so ``P`` is not a parameter shape — it can vary freely at inference
   (paper Appendix E.4, span-scheme invariance).
2. **Cross-span attention** (``q_layer^(l)``): a second attention step
   aggregates that layer's ``P`` span summaries into one per-head sharer
   summary ``K*, V* [B, H, 1, D]``.

The pooled summary stays in **Sharer** geometry (``H_s``/``D_s`` never
remapped) and is layer-aligned: sharer layer ``l``'s summary feeds receiver
layer ``l``'s projector. The paper's LCF-X pair is symmetric (two Qwen3-0.6B,
``L_s == L_r``); when layer counts differ we bridge with a single linear
projection across the layer axis (``layer_map``, created only when needed —
the one deliberate gap-fill beyond the paper).

Projector (paper Fig. 3 bottom, Section 3 steps 1-7), per receiver layer. The
summary is broadcast across the receiver sequence and takes the place of the
sharer's matched-position cache in LCF; the projector itself is unchanged:

    x    = concat&flatten(S_K, S_V, R_K, R_V)          # 2*H_s*D_s + 2*H_r*D_r
    z    = W_down x ; z = 2-layer MLP(z)               # -> d, intermediate 4d
    a_i  = sigmoid(W_gate z)                           # per-head gates from full z
    Δ_i  = W_up^i z,  i in {K,V}                       # separate per-layer up-proj
    R_i' = R_i + g_i · (a_i ⊙ Δ_i)                     # per-layer per-type gate

``W_up^i`` outputs Receiver geometry, so any Sharer/Receiver H/D mismatch is
absorbed by the concat width + ``W_down`` (the paper's own shared-context LCF
pair is asymmetric: joint width 2304 = 2*2*64 + 2*8*128). The paper's prose
describes a latent split ``z -> (z_K, z_V)`` (each d/2) feeding ``W_up^i``, but
that undershoots the paper's own reported adapter sizes (693K/layer for LCF-128
— Table 5 — and 25.87M total for LCF-X — Appendix E.3) by exactly one
up-projection's worth per K/V; feeding each ``W_up^i`` the *full* d-dim latent
reproduces both figures (693,010/layer; 25.83M), so we match the reported
parameter counts rather than the prose.

Gating: ``g_i`` is a per-layer, per-type (K and V separate) Gumbel-sigmoid gate
— the paper's layer-pruning audit reads trained gate logits "for both K and V".
Training samples a Gumbel-sigmoid relaxation (temperature ``gate_tau``, annealed
1.0 -> 0.5 over the run, Table 11); at inference the gate is **hard**,
``g_i ∈ {0,1}`` (Section 3 / Table 6 "hard at eval"): ``logit > 0 -> 1``.

``zero_init=True`` zeroes the up-projections, so Δ = 0 ⇒ residual = 0 and LCF-X
reproduces the Receiver-only run (the plumbing check), while the gates stay
nonzero so the up-projections still receive gradient (ReZero-style).

Paper-faithful init (Table 11): pool queries Kaiming-normal x 0.02;
up-projections Kaiming-normal x 0.06; gate logits +1.0 (Appendix E.1:
sigmoid(1.0)=0.73 gives the residual branch a meaningful gain at step 0,
where a 0.0 init stalls near sigmoid(0)=0.5 with weak gate gradients).
"""

from __future__ import annotations

import torch
from torch import nn


def _gumbel_sigmoid(logit: torch.Tensor, training: bool, tau: float = 1.0) -> torch.Tensor:
    """Per-layer per-type gate. Training: Gumbel-sigmoid sample; eval: hard {0,1}."""

    if training:
        u = torch.rand_like(logit).clamp_(1e-6, 1 - 1e-6)
        return torch.sigmoid((logit + torch.log(u) - torch.log1p(-u)) / tau)
    return (logit > 0).to(logit.dtype)


def _span_starts(n: int, num_spans: int, boundaries: "list[int] | None") -> list[int]:
    """Span start offsets within ``n`` real tokens.

    With ``boundaries`` (token offsets of natural blocks, e.g. paragraphs) they
    are clamped/deduped to valid starts; otherwise the tokens are divided into
    ``num_spans`` near-equal contiguous chunks (the paper's "fixed-size chunk"
    span scheme).
    """

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


class _LayerProjector(nn.Module):
    """Paper's per-layer joint KV bottleneck (Fig. 3 bottom, steps 2-7).

    Input is `[S_K ; S_V ; R_K ; R_V]` — the broadcast pooled Sharer summary
    (still in *Sharer* geometry) concatenated with the Receiver cache (Receiver
    geometry) — so the down-projection's input width is
    `2*H_s*D_s + 2*H_r*D_r`, not `4*H_r*D_r`. Output (`W_up^i`) is always
    Receiver geometry, so this is where any Sharer/Receiver H/D mismatch
    resolves.
    """

    def __init__(
        self,
        sharer_kv_heads: int,
        sharer_head_dim: int,
        kv_heads: int,
        head_dim: int,
        bottleneck_dim: int,
        zero_init: bool,
    ) -> None:
        super().__init__()
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.bottleneck_dim = bottleneck_dim
        sharer_feat = sharer_kv_heads * sharer_head_dim  # H_s * D_s
        receiver_feat = kv_heads * head_dim  # H_r * D_r

        self.down = nn.Linear(2 * sharer_feat + 2 * receiver_feat, bottleneck_dim)
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, 4 * bottleneck_dim),
            nn.GELU(),
            nn.Linear(4 * bottleneck_dim, bottleneck_dim),
        )
        # Input-dependent per-head gates a_K, a_V (from the full latent z).
        self.gate = nn.Linear(bottleneck_dim, 2 * kv_heads)
        # Separate per-type up-projections, each reading the full latent z (see
        # module docstring for why, vs. the paper's prose-described d/2 split).
        self.up_key = nn.Linear(bottleneck_dim, receiver_feat)
        self.up_value = nn.Linear(bottleneck_dim, receiver_feat)
        if zero_init:
            for up in (self.up_key, self.up_value):
                nn.init.zeros_(up.weight)
                nn.init.zeros_(up.bias)
        else:
            # Paper (Table 11): "Up-proj Kaiming scale: 0.06".
            for up in (self.up_key, self.up_value):
                nn.init.kaiming_normal_(up.weight)
                with torch.no_grad():
                    up.weight.mul_(0.06)
                nn.init.zeros_(up.bias)

    def forward(
        self,
        aligned_key: torch.Tensor,
        aligned_value: torch.Tensor,
        recv_key: torch.Tensor,
        recv_value: torch.Tensor,
        layer_gate: torch.Tensor,
        training: bool,
        tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # aligned_*: [B, H_s, 1, D_s] (Sharer geometry, broadcast over N);
        # recv_*: [B, H_r, N, D_r] (Receiver geometry);
        # layer_gate: [2] — this layer's (K, V) gate logits.
        weight_dtype = self.down.weight.dtype
        batch, _, seq, _ = recv_key.shape

        def _flat(x: torch.Tensor) -> torch.Tensor:
            return x.permute(0, 2, 1, 3).reshape(batch, x.shape[2], -1).to(weight_dtype)

        joint = torch.cat(
            [
                _flat(aligned_key.expand(-1, -1, seq, -1)),
                _flat(aligned_value.expand(-1, -1, seq, -1)),
                _flat(recv_key),
                _flat(recv_value),
            ],
            dim=-1,
        )  # [B, N, 2*sharer_feat + 2*receiver_feat]

        z = self.mlp(self.down(joint))  # [B, N, d]

        alpha = torch.sigmoid(self.gate(z)).view(batch, seq, 2, self.kv_heads)
        alpha_key, alpha_value = alpha[:, :, 0, :], alpha[:, :, 1, :]  # [B, N, H_r]

        delta_key = self.up_key(z).view(batch, seq, self.kv_heads, self.head_dim)
        delta_value = self.up_value(z).view(batch, seq, self.kv_heads, self.head_dim)

        gate = _gumbel_sigmoid(layer_gate, training, tau)  # [2] in (0,1); {0,1} at eval
        d_key = gate[0] * (alpha_key.unsqueeze(-1) * delta_key)  # [B, N, H_r, D_r]
        d_value = gate[1] * (alpha_value.unsqueeze(-1) * delta_value)
        return d_key.permute(0, 2, 1, 3), d_value.permute(0, 2, 1, 3)  # [B, H_r, N, D_r]


class LCFXFuser(nn.Module):
    """Paper-faithful per-layer 2-stage pooling + per-layer KV-bottleneck projector.

    ``align`` pools each Sharer layer independently (within-span then
    cross-span) into one Sharer-shaped ``[B, H_s, 1, D_s]`` summary per layer;
    `H`/`D` are never remapped, and layers map 1:1 onto receiver layers
    (``layer_map`` bridges only when the counts differ). ``project`` resolves
    any H/D mismatch: its down-projection input width is
    `2*H_s*D_s + 2*H_r*D_r`, and its up-projection always emits Receiver
    geometry.
    """

    def __init__(
        self,
        *,
        sharer_layers: int,
        sharer_kv_heads: int,
        sharer_head_dim: int,
        receiver_layers: int,
        receiver_kv_heads: int,
        receiver_head_dim: int,
        bottleneck_dim: int = 128,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        # Sharer dims (kept so the fuser is self-describing for checkpointing).
        self.sharer_layers = sharer_layers
        self.sharer_kv_heads = sharer_kv_heads
        self.sharer_head_dim = sharer_head_dim
        self.receiver_layers = receiver_layers
        self.receiver_kv_heads = receiver_kv_heads
        self.receiver_head_dim = receiver_head_dim
        self.bottleneck_dim = bottleneck_dim

        # Paper pool (Appendix E.3): two learned query vectors per layer, each
        # in RAW Sharer head/dim space — no key/value projection. Attention
        # scores come from a direct dot product against the raw Sharer K, and
        # the same weights pool both K and V. 2*H*D parameters per layer
        # (57,344 for a 28-layer Qwen3-0.6B pair, matching E.3).
        # Init: Kaiming-normal x 0.02 (Table 11: "Pool query init: Kaiming x0.02").
        # --- Stage 1: within-span attention (N_s -> P), one shared base query
        # per Sharer layer (q_base^(l)), reused across every span of that layer.
        self.q_base = nn.Parameter(torch.empty(sharer_layers, sharer_kv_heads, sharer_head_dim))
        # --- Stage 2: cross-span attention (P -> 1) within the same layer, one
        # layer query per Sharer layer (q_layer^(l)); still Sharer geometry.
        self.q_layer = nn.Parameter(torch.empty(sharer_layers, sharer_kv_heads, sharer_head_dim))
        for query in (self.q_base, self.q_layer):
            nn.init.kaiming_normal_(query)
            with torch.no_grad():
                query.mul_(0.02)

        # Layer-count bridge (our gap-fill; the paper's LCF-X pair is symmetric
        # so it never needs one): a single linear projection across the layer
        # axis, shared by K and V, created only when L_s != L_r.
        self.layer_map = (
            nn.Linear(sharer_layers, receiver_layers)
            if sharer_layers != receiver_layers
            else None
        )

        # --- Paper's per-layer joint KV bottleneck projector ---
        self.projectors = nn.ModuleList(
            [
                _LayerProjector(
                    sharer_kv_heads,
                    sharer_head_dim,
                    receiver_kv_heads,
                    receiver_head_dim,
                    bottleneck_dim,
                    zero_init,
                )
                for _ in range(receiver_layers)
            ]
        )
        # Per-layer, per-type (K, V) Gumbel-sigmoid gate logits. Paper
        # (Appendix E.1): init +1.0, not 0.0 — sigmoid(1.0)=0.73 gives the
        # residual branch a meaningful gain at step 0. The layer-pruning audit
        # (Appendix D) reads these "for both K and V", hence the pair per layer.
        self.layer_gate = nn.Parameter(torch.ones(receiver_layers, 2))
        # Gumbel-sigmoid temperature; the training loop anneals this 1.0 -> 0.5
        # over the run (Table 11: "Gate temperature anneal: 1.0 -> 0.5"). Not a
        # Parameter/buffer — it's a training-schedule knob, not a learned value.
        self.gate_tau = 1.0

    # -- pooling ---------------------------------------------------------------
    def _pool(
        self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One attention pool: ``query [H,D]`` over ``K,V [1,H,n,D]`` -> ``[1,H,D]`` x2.

        Scores are a direct dot product against the raw K (no key/value
        projection, matching the paper's pool); the same weights pool K and V.
        """

        scores = torch.einsum("hd,bhnd->bhn", query, keys) / self.sharer_head_dim**0.5
        weight = scores.softmax(dim=-1).to(keys.dtype)
        pooled_k = torch.einsum("bhn,bhnd->bhd", weight, keys)
        pooled_v = torch.einsum("bhn,bhnd->bhd", weight, values)
        return pooled_k, pooled_v

    def align(
        self,
        summary: list[tuple[torch.Tensor, torch.Tensor]],
        lengths: "torch.Tensor | None" = None,
        num_spans: int = 5,
        span_boundaries: "list[list[int] | None] | None" = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer 2-stage pooling: raw Sharer cache -> one summary per layer.

        Input:  ``L_s`` x ``(K, V)`` each ``[B, H_s, N_s, D_s]`` (raw, possibly padded).
        ``lengths[b]`` = real token count for row ``b`` (None => use full N_s).
        ``span_boundaries[b]`` = token start offsets of row ``b``'s natural spans
        (e.g. paragraph blocks — the paper's training scheme); None => divide
        the real tokens into ``num_spans`` near-equal chunks.

        Per Sharer layer ``l``: within-span attention with ``q_base^(l)``
        (``N_s -> P``), then cross-span attention with ``q_layer^(l)`` over that
        layer's own ``P`` span summaries (``P -> 1``) — Fig. 3's
        ``K*, V* ∈ R^{B x H x 1 x D}``. No cross-layer mixing.

        Output: ``L_r`` x ``(K*, V*)`` each ``[B, H_s, 1, D_s]`` — layer-aligned
        1:1 when ``L_s == L_r`` (the paper's setting); otherwise bridged by the
        ``layer_map`` linear projection across the layer axis.
        """

        weight_dtype = self.q_base.dtype
        batch = summary[0][0].shape[0]
        full_n = summary[0][0].shape[2]

        # out[l][b]: [1, H_s, 1, D_s]
        out_k: list[list[torch.Tensor]] = [[] for _ in summary]
        out_v: list[list[torch.Tensor]] = [[] for _ in summary]
        for b in range(batch):
            n = int(lengths[b]) if lengths is not None else full_n
            bounds = span_boundaries[b] if span_boundaries is not None else None
            starts = _span_starts(n, num_spans, bounds)
            ends = starts[1:] + [n]
            for layer_idx, (k, v) in enumerate(summary):
                kb = k[b : b + 1, :, :n, :].to(weight_dtype)
                vb = v[b : b + 1, :, :n, :].to(weight_dtype)
                # Stage 1: q_base pools each span -> [1, H_s, P, D_s].
                spans = [
                    self._pool(self.q_base[layer_idx], kb[:, :, s:e, :], vb[:, :, s:e, :])
                    for s, e in zip(starts, ends)
                ]
                spans_k = torch.stack([sk for sk, _ in spans], dim=2)
                spans_v = torch.stack([sv for _, sv in spans], dim=2)
                # Stage 2: q_layer pools this layer's span summaries -> [1, H_s, 1, D_s].
                star_k, star_v = self._pool(self.q_layer[layer_idx], spans_k, spans_v)
                out_k[layer_idx].append(star_k.unsqueeze(2))
                out_v[layer_idx].append(star_v.unsqueeze(2))

        aligned = [
            (torch.cat(layer_k, dim=0), torch.cat(layer_v, dim=0))
            for layer_k, layer_v in zip(out_k, out_v)
        ]  # L_s x ([B, H_s, 1, D_s], [B, H_s, 1, D_s])

        if self.layer_map is None:
            return aligned

        # L_s != L_r: linear projection across the layer axis (shared by K and V).
        stack_k = torch.cat([k.squeeze(2).unsqueeze(-1) for k, _ in aligned], dim=-1)
        stack_v = torch.cat([v.squeeze(2).unsqueeze(-1) for _, v in aligned], dim=-1)
        mapped_k = self.layer_map(stack_k)  # [B, H_s, D_s, L_r]
        mapped_v = self.layer_map(stack_v)
        return [
            (
                mapped_k[..., i].unsqueeze(2),
                mapped_v[..., i].unsqueeze(2),
            )
            for i in range(self.receiver_layers)
        ]

    # -- projection ------------------------------------------------------------
    def project(
        self,
        aligned: list[tuple[torch.Tensor, torch.Tensor]],
        receiver_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer joint KV bottleneck on ``[S* (Sharer geom) ; R]`` -> residual."""

        residuals: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, ((a_key, a_value), (r_key, r_value)) in enumerate(
            zip(aligned, receiver_kv)
        ):
            d_key, d_value = self.projectors[layer_idx](
                a_key, a_value, r_key, r_value, self.layer_gate[layer_idx], self.training, self.gate_tau
            )
            residuals.append((d_key, d_value))
        return residuals

    def forward(
        self,
        summary: list[tuple[torch.Tensor, torch.Tensor]],
        receiver_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self.project(self.align(summary), receiver_kv)

    @classmethod
    def from_specs(
        cls,
        sharer_spec: "object",
        receiver_spec: "object",
        *,
        bottleneck_dim: int = 128,
        zero_init: bool = True,
    ) -> "LCFXFuser":
        return cls(
            sharer_layers=sharer_spec.layers,
            sharer_kv_heads=sharer_spec.kv_heads,
            sharer_head_dim=sharer_spec.head_dim,
            receiver_layers=receiver_spec.layers,
            receiver_kv_heads=receiver_spec.kv_heads,
            receiver_head_dim=receiver_spec.head_dim,
            bottleneck_dim=bottleneck_dim,
            zero_init=zero_init,
        )

    def arch_config(self) -> dict:
        """Self-describing architecture (the dims that determine every weight shape).

        Persisted in the checkpoint so a trained fuser can be rebuilt and validated
        against the Sharer/Receiver pair it was trained for.
        """

        return {
            "sharer_layers": self.sharer_layers,
            "sharer_kv_heads": self.sharer_kv_heads,
            "sharer_head_dim": self.sharer_head_dim,
            "receiver_layers": self.receiver_layers,
            "receiver_kv_heads": self.receiver_kv_heads,
            "receiver_head_dim": self.receiver_head_dim,
            "bottleneck_dim": self.bottleneck_dim,
        }

    @classmethod
    def from_arch_config(cls, cfg: dict, *, zero_init: bool = True) -> "LCFXFuser":
        """Rebuild a fuser from an :meth:`arch_config` dict."""

        return cls(
            sharer_layers=cfg["sharer_layers"],
            sharer_kv_heads=cfg["sharer_kv_heads"],
            sharer_head_dim=cfg["sharer_head_dim"],
            receiver_layers=cfg["receiver_layers"],
            receiver_kv_heads=cfg["receiver_kv_heads"],
            receiver_head_dim=cfg["receiver_head_dim"],
            bottleneck_dim=cfg["bottleneck_dim"],
            zero_init=zero_init,
        )

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
