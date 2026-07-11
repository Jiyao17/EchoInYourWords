"""LCF-X fuser: paper-faithful projector + a 2-stage learned-attention pooling.

Two stages:

1. **Pooling / align (learned attention).** Following the paper (Fig. 3), the
   pool is a learned attention whose query count determines the output count,
   and it leaves `H` and `D` untouched — only `N` (sequence) and `L` (layers)
   are converted:
     - within-span (`N_S -> P`, keeps Sharer geometry) then across-span
       (`L_s*P -> 1`), both driven by the **attention queries** (one query per
       Receiver layer gives `L_s -> L_r`).
   Output: a *Sharer*-shaped Sharer summary `[B, H_s, 1, D_s]` per Receiver
   layer — `H_s`/`D_s` are never remapped here. The Sharer's KV shape doesn't
   need to match the Receiver's at all: the projector below concatenates the
   (still Sharer-shaped) pooled summary with the Receiver cache and lets the
   down-projection absorb the size difference; the up-projection then emits a
   Receiver-shaped residual regardless of how the Sharer was shaped.

2. **Projector (paper's joint key-value bottleneck), per Receiver layer.** Faithful
   to the paper:
       x    = concat&flatten(aS_K, aS_V, R_K, R_V)        # 2*H_s*D_s + 2*H_r*D_r
       z    = W_down x ; z = 2-layer MLP(z)               # -> d, intermediate 4d
       a_i  = sigmoid(W_gate z)                           # input-dependent per-head gate
       Δ_i  = W_up^i z,  i in {K,V}                       # separate per-layer up-proj
       R_i' = R_i + g · (a_i ⊙ Δ_i)                       # g = per-layer (Gumbel-)sigmoid gate
   Each Receiver layer has its own down / MLP / gate / up (per-layer projectors);
   `W_up^i` outputs Receiver geometry, so this is also where any Sharer/Receiver
   shape mismatch resolves. The paper's prose describes an intermediate "latent
   split" into z_K/z_V halves (each d/2) feeding W_up^i, but that undershoots its
   own reported adapter sizes (693K/layer for the base-LCF pair, ~25.8M for
   LCF-X) by almost exactly one up-projection's worth of parameters per K/V;
   feeding each W_up^i the *full* d-dim latent z reproduces both reported
   figures to within rounding, so we match the paper's reported parameter
   counts here rather than its (apparently unfaithful) prose.

``zero_init=True`` zeroes the up-projections, so Δ = 0 ⇒ residual = 0 and LCF-X
reproduces the Receiver-only run (the plumbing check), while the gates stay
nonzero so the up-projections still receive gradient (ReZero-style).

Paper-faithful init/schedule (Appendix E.1, Table 11): pool queries and
up-projections use Kaiming-normal init scaled by 0.02 / 0.06 respectively; the
per-layer gate logit starts at +1.0 (sigmoid(1.0)=0.73 — a meaningful residual
gain at step 0, unlike a 0.0 init which stalls near sigmoid(0)=0.5 with weak
gradients); and the Gumbel-sigmoid temperature anneals 1.0 -> 0.5 over training
(``LCFXFuser.gate_tau``, set externally by the training loop).
"""

from __future__ import annotations

import torch
from torch import nn


def _gumbel_sigmoid(logit: torch.Tensor, training: bool, tau: float = 1.0) -> torch.Tensor:
    """Per-layer soft gate. Training: Gumbel-sigmoid sample; eval: plain sigmoid."""

    if training:
        u = torch.rand_like(logit).clamp_(1e-6, 1 - 1e-6)
        logit = (logit + torch.log(u) - torch.log1p(-u)) / tau
    return torch.sigmoid(logit)


class _LayerProjector(nn.Module):
    """Paper's per-layer joint KV bottleneck.

    Input is `[aS_K ; aS_V ; R_K ; R_V]` — the pooled Sharer summary (still in
    *Sharer* geometry) concatenated with the Receiver cache (Receiver
    geometry) — so the down-projection's input width is
    `2*H_s*D_s + 2*H_r*D_r`, not `4*H_r*D_r`. Output (`W_up^i`) is always
    Receiver geometry, so this is where any Sharer/Receiver shape mismatch
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
        # recv_*: [B, H_r, N, D_r] (Receiver geometry).
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

        gate = _gumbel_sigmoid(layer_gate, training, tau)  # scalar in (0,1)
        d_key = gate * (alpha_key.unsqueeze(-1) * delta_key)  # [B, N, H_r, D_r]
        d_value = gate * (alpha_value.unsqueeze(-1) * delta_value)
        return d_key.permute(0, 2, 1, 3), d_value.permute(0, 2, 1, 3)  # [B, H_r, N, D_r]


class LCFXFuser(nn.Module):
    """Paper-faithful learned-attention pooling + per-layer KV-bottleneck projector.

    The pool (``align``) never remaps `H`/`D`; the Sharer summary stays in
    Sharer geometry. Cross-model shape differences are resolved entirely by
    the projector (``project``): its down-projection input width is
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

        # Paper-faithful pool (Appendix E.3): two learned query vectors per layer,
        # each in RAW Sharer head/dim space -- no key/value projection. Attention
        # scores come from a direct dot product against the raw Sharer K, and the
        # same weights pool both K and V. This keeps the pool to `2*H_s*D_s*(L_s+L_r)`
        # parameters (57,344 for a 28-layer Qwen3-0.6B/Qwen3-0.6B pair, matching the
        # paper), instead of routing through learned key/value projection matrices.
        # Init: Kaiming-normal x 0.02 (Table 11: "Pool query init: Kaiming x0.02").
        # --- Stage 1: within-span attention (N_S -> P), one query per Sharer layer
        # (q_base^(l) in the paper), reused across every span of that layer.
        self.q_base = nn.Parameter(torch.empty(sharer_layers, sharer_kv_heads, sharer_head_dim))
        nn.init.kaiming_normal_(self.q_base)
        with torch.no_grad():
            self.q_base.mul_(0.02)
        # --- Stage 2: across-span attention (L_s*P tokens -> 1), one query per
        # Receiver layer (q_layer^(l)); still Sharer geometry (H_s, D_s untouched).
        self.q_layer = nn.Parameter(torch.empty(receiver_layers, sharer_kv_heads, sharer_head_dim))
        nn.init.kaiming_normal_(self.q_layer)
        with torch.no_grad():
            self.q_layer.mul_(0.02)

        # --- Stage 2: paper's per-layer joint KV bottleneck projector ---
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
        # Per-layer (Gumbel-)sigmoid gate logit. Paper (Appendix E.1): init +1.0,
        # not 0.0 -- sigmoid(1.0)=0.73 gives the residual branch a meaningful gain
        # at step 0; a 0.0 init stalls near sigmoid(0)=0.5 with weak gate gradients.
        self.layer_gate = nn.Parameter(torch.ones(receiver_layers))
        # Gumbel-sigmoid temperature; the training loop anneals this 1.0 -> 0.5
        # over the run (Table 11: "Gate temperature anneal: 1.0 -> 0.5"). Not a
        # Parameter/buffer -- it's a training-schedule knob, not a learned value.
        self.gate_tau = 1.0

    # -- Stage 1: learned-attention pooling -------------------------------------
    def _within_span(
        self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor, num_spans: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Within-span attention for one Sharer layer: raw ``K,V [1,H,n,D]`` -> span
        summaries ``[1,H,P,D]`` x2 (keeps Sharer geometry).

        ``query`` (that layer's ``q_base``, ``[H,D]``) attends directly on the raw K
        within each span -- no key/value projection, matching the paper's pool -- and
        the same attention weights pool both K and V. Pads to exactly ``num_spans``
        rows if the span count falls short (e.g. more spans requested than tokens).
        """

        n = key.shape[2]
        p = max(1, min(int(num_spans), n))
        scale = self.sharer_head_dim**0.5
        pooled_k, pooled_v = [], []
        for k_span, v_span in zip(torch.tensor_split(key, p, dim=2), torch.tensor_split(value, p, dim=2)):
            scores = torch.einsum("hd,bhnd->bhn", query, k_span) / scale
            weight = scores.softmax(dim=-1).to(k_span.dtype)
            pooled_k.append(torch.einsum("bhn,bhnd->bhd", weight, k_span))
            pooled_v.append(torch.einsum("bhn,bhnd->bhd", weight, v_span))
        spans_k = torch.stack(pooled_k, dim=2)  # [1, H, p, D]
        spans_v = torch.stack(pooled_v, dim=2)
        if p < num_spans:  # pad to a fixed P so the across-span token set is uniform
            pad_k = spans_k.new_zeros(1, spans_k.shape[1], num_spans - p, spans_k.shape[-1])
            pad_v = spans_v.new_zeros(1, spans_v.shape[1], num_spans - p, spans_v.shape[-1])
            spans_k = torch.cat([spans_k, pad_k], dim=2)
            spans_v = torch.cat([spans_v, pad_v], dim=2)
        return spans_k, spans_v

    def align(
        self,
        summary: list[tuple[torch.Tensor, torch.Tensor]],
        lengths: "torch.Tensor | None" = None,
        num_spans: int = 4,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """2-stage learned pooling: raw Sharer cache -> position-free Sharer-shaped summary.

        Input:  ``L_s`` x ``(K, V)`` each ``[B, H_s, N_S, D_s]`` (raw, possibly padded).
        ``lengths[b]`` = real token count for row ``b`` (None => use full N_S).
        Stage 1: within-span attention ``N_S -> P`` (per example, real tokens only,
        one ``q_base`` query per Sharer layer). Stage 2: across-span attention
        ``L_s*P -> 1`` (one ``q_layer`` query per Receiver layer). Both stages attend
        directly on the raw Sharer K/V (no key/value projection, paper-faithful);
        `H_s`/`D_s` are untouched throughout.
        Output: ``L_r`` x ``(aK, aV)`` each ``[B, H_s, 1, D_s]``.
        """

        weight_dtype = self.q_base.dtype
        batch = summary[0][0].shape[0]
        full_n = summary[0][0].shape[2]

        # Stage 1: per example, per Sharer layer, within-span attention over real tokens.
        token_k_sets, token_v_sets = [], []
        for b in range(batch):
            n = int(lengths[b]) if lengths is not None else full_n
            layer_k, layer_v = [], []
            for layer_idx, (k, v) in enumerate(summary):
                spans_k, spans_v = self._within_span(
                    k[b : b + 1, :, :n, :].to(weight_dtype),
                    v[b : b + 1, :, :n, :].to(weight_dtype),
                    self.q_base[layer_idx].to(weight_dtype),
                    num_spans,
                )
                layer_k.append(spans_k)
                layer_v.append(spans_v)
            token_k_sets.append(torch.cat(layer_k, dim=2))  # [1, H_s, L_s*P, D_s]
            token_v_sets.append(torch.cat(layer_v, dim=2))
        token_k = torch.cat(token_k_sets, dim=0)  # [B, H_s, L_s*P, D_s]
        token_v = torch.cat(token_v_sets, dim=0)

        # Stage 2: one query per Receiver layer attends over the L_s*P span summaries
        # (still Sharer geometry, no key/value projection).
        scale = self.sharer_head_dim**0.5
        aligned: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(self.receiver_layers):
            query = self.q_layer[layer_idx].to(weight_dtype)  # [H_s, D_s]
            scores = torch.einsum("hd,bhnd->bhn", query, token_k) / scale
            weight = scores.softmax(dim=-1).to(token_k.dtype)
            out_k = torch.einsum("bhn,bhnd->bhd", weight, token_k).unsqueeze(2)  # [B,H_s,1,D_s]
            out_v = torch.einsum("bhn,bhnd->bhd", weight, token_v).unsqueeze(2)
            aligned.append((out_k, out_v))
        return aligned

    # -- Stage 2 --------------------------------------------------------------
    def project(
        self,
        aligned: list[tuple[torch.Tensor, torch.Tensor]],
        receiver_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer joint KV bottleneck on ``[aligned-S (Sharer geom) ; R]`` -> residual."""

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
