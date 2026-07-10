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
       x      = concat&flatten(aS_K, aS_V, R_K, R_V)        # 2*H_s*D_s + 2*H_r*D_r
       z      = W_down x ; z = 2-layer MLP(z)               # -> d, intermediate 4d
       z_K,z_V = split(z)                                   # each d/2
       a_i    = sigmoid(W_gate z)                           # input-dependent per-head gate
       Δ_i    = W_up^i z_i,  i in {K,V}                     # separate per-layer up-proj
       R_i'   = R_i + g · (a_i ⊙ Δ_i)                       # g = per-layer (Gumbel-)sigmoid gate
   Each Receiver layer has its own down / MLP / gate / up (per-layer projectors);
   `W_up^i` outputs Receiver geometry, so this is also where any Sharer/Receiver
   shape mismatch resolves.

``zero_init=True`` zeroes the up-projections, so Δ = 0 ⇒ residual = 0 and LCF-X
reproduces the Receiver-only run (the plumbing check), while the gates stay
nonzero so the up-projections still receive gradient (ReZero-style).
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
        half = bottleneck_dim // 2

        self.down = nn.Linear(2 * sharer_feat + 2 * receiver_feat, bottleneck_dim)
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, 4 * bottleneck_dim),
            nn.GELU(),
            nn.Linear(4 * bottleneck_dim, bottleneck_dim),
        )
        # Input-dependent per-head gates a_K, a_V (from the full latent z).
        self.gate = nn.Linear(bottleneck_dim, 2 * kv_heads)
        # Separate per-type up-projections from the half-latents.
        self.up_key = nn.Linear(half, receiver_feat)
        self.up_value = nn.Linear(half, receiver_feat)
        if zero_init:
            for up in (self.up_key, self.up_value):
                nn.init.zeros_(up.weight)
                nn.init.zeros_(up.bias)

    def forward(
        self,
        aligned_key: torch.Tensor,
        aligned_value: torch.Tensor,
        recv_key: torch.Tensor,
        recv_value: torch.Tensor,
        layer_gate: torch.Tensor,
        training: bool,
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
        half = self.bottleneck_dim // 2
        z_key, z_value = z[..., :half], z[..., half:]

        alpha = torch.sigmoid(self.gate(z)).view(batch, seq, 2, self.kv_heads)
        alpha_key, alpha_value = alpha[:, :, 0, :], alpha[:, :, 1, :]  # [B, N, H_r]

        delta_key = self.up_key(z_key).view(batch, seq, self.kv_heads, self.head_dim)
        delta_value = self.up_value(z_value).view(batch, seq, self.kv_heads, self.head_dim)

        gate = _gumbel_sigmoid(layer_gate, training)  # scalar in (0,1)
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
        if bottleneck_dim % 2 != 0:
            raise ValueError("bottleneck_dim must be even (latent is split into z_K, z_V)")
        # Sharer dims (kept so the fuser is self-describing for checkpointing).
        self.sharer_layers = sharer_layers
        self.sharer_kv_heads = sharer_kv_heads
        self.sharer_head_dim = sharer_head_dim
        self.receiver_layers = receiver_layers
        self.receiver_kv_heads = receiver_kv_heads
        self.receiver_head_dim = receiver_head_dim
        self.bottleneck_dim = bottleneck_dim
        self.attn_dim = bottleneck_dim
        sharer_token_dim = 2 * sharer_kv_heads * sharer_head_dim

        # --- Stage 1a: within-span learned attention (N_S -> P), keeps Sharer geometry.
        # A shared learned query pools the tokens within each span (paper: "the same
        # pooling query is reused for every span").
        self.span_key = nn.Linear(sharer_token_dim, bottleneck_dim)
        self.span_query = nn.Parameter(torch.randn(bottleneck_dim) * 0.02)
        # --- Stage 1b: across-span attention (L_s*P tokens -> 1), still Sharer
        # geometry (paper: pool keeps H, D). One learnable query per Receiver
        # layer attends over the span tokens to convert L_s -> L_r; the value
        # projection stays in Sharer feature space (no H/D remap here) -- the
        # projector below absorbs any Sharer/Receiver shape mismatch instead.
        self.key_proj = nn.Linear(sharer_token_dim, bottleneck_dim)
        self.value_proj = nn.Linear(sharer_token_dim, sharer_token_dim)
        self.queries = nn.Parameter(torch.randn(receiver_layers, bottleneck_dim) * 0.02)

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
        # Per-layer (Gumbel-)sigmoid gate logit; 0 -> sigmoid 0.5 (open at start).
        self.layer_gate = nn.Parameter(torch.zeros(receiver_layers))

    # -- Stage 1: learned-attention pooling -------------------------------------
    @staticmethod
    def _flatten_kv(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """[1, H, n, D] x2 -> [1, n, 2*H*D] joint token features."""

        batch, _, n, _ = key.shape
        kf = key.permute(0, 2, 1, 3).reshape(batch, n, -1)
        vf = value.permute(0, 2, 1, 3).reshape(batch, n, -1)
        return torch.cat([kf, vf], dim=2)

    def _within_span(self, tokens: torch.Tensor, num_spans: int) -> torch.Tensor:
        """Within-span learned attention: [1, n, F] -> [1, P, F] (keeps Sharer geom).

        Splits the n real tokens into P spans; a shared learned query attends within
        each span. Pads to exactly ``num_spans`` rows if n < num_spans.
        """

        n = tokens.shape[1]
        p = max(1, min(int(num_spans), n))
        key = self.span_key(tokens.to(self.span_key.weight.dtype))  # [1, n, d]
        scores = torch.einsum("bnd,d->bn", key, self.span_query) / (self.attn_dim**0.5)
        out = []
        for tok_span, score_span in zip(
            torch.tensor_split(tokens, p, dim=1), torch.tensor_split(scores, p, dim=1)
        ):
            weight = score_span.softmax(dim=1).to(tok_span.dtype)  # [1, s]
            out.append(torch.einsum("bs,bsf->bf", weight, tok_span))  # [1, F]
        spans = torch.stack(out, dim=1)  # [1, p, F]
        if p < num_spans:  # pad to a fixed P so the across-span token set is uniform
            pad = spans.new_zeros(1, num_spans - p, spans.shape[-1])
            spans = torch.cat([spans, pad], dim=1)
        return spans

    def align(
        self,
        summary: list[tuple[torch.Tensor, torch.Tensor]],
        lengths: "torch.Tensor | None" = None,
        num_spans: int = 4,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """2-stage learned pooling: raw Sharer cache -> position-free Sharer-shaped summary.

        Input:  ``L_s`` x ``(K, V)`` each ``[B, H_s, N_S, D_s]`` (raw, possibly padded).
        ``lengths[b]`` = real token count for row ``b`` (None => use full N_S).
        Stage 1: within-span learned attention ``N_S -> P`` (per example, real tokens
        only). Stage 2: across-span attention ``L_s*P -> 1``, one query per Receiver
        layer (``L_s -> L_r``); `H_s`/`D_s` are untouched (paper-faithful pool).
        Output: ``L_r`` x ``(aK, aV)`` each ``[B, H_s, 1, D_s]``.
        """

        weight_dtype = self.key_proj.weight.dtype
        batch = summary[0][0].shape[0]
        full_n = summary[0][0].shape[2]

        # Stage 1: per example, within-span attention over real tokens -> L_s*P tokens.
        token_sets = []
        for b in range(batch):
            n = int(lengths[b]) if lengths is not None else full_n
            layer_spans = [
                self._within_span(self._flatten_kv(k[b : b + 1, :, :n, :], v[b : b + 1, :, :n, :]), num_spans)
                for k, v in summary
            ]
            token_sets.append(torch.cat(layer_spans, dim=1))  # [1, L_s*P, F_s]
        token_set = torch.cat(token_sets, dim=0).to(weight_dtype)  # [B, L_s*P, F_s]

        # Stage 2: Receiver-sized queries attend over the span tokens -> L_s -> L_r,
        # values stay in Sharer feature space (H_s, D_s untouched).
        keys = self.key_proj(token_set)
        values = self.value_proj(token_set)
        queries = self.queries.to(weight_dtype)
        scores = torch.einsum("ld,btd->blt", queries, keys) / (self.attn_dim**0.5)
        pooled = torch.einsum("blt,btf->blf", scores.softmax(dim=-1), values)

        aligned: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(self.receiver_layers):
            out = pooled[:, layer_idx].view(
                batch, 2, self.sharer_kv_heads, self.sharer_head_dim
            )
            aligned.append((out[:, 0].unsqueeze(2), out[:, 1].unsqueeze(2)))
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
                a_key, a_value, r_key, r_value, self.layer_gate[layer_idx], self.training
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
