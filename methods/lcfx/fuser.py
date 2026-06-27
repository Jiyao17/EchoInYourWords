"""LCF-X fuser: align the Sharer to Receiver shape, then project jointly with it.

Two stages, matching Figure 3 of the paper:

1. **Align (attention resampler).** Following the paper's insight that pooling is a
   learned attention with free-size queries, we size the queries to the
   *Receiver*. Receiver-sized learnable queries (one per Receiver layer) attend
   over the Sharer's pooled KV tokens and emit a Receiver-shaped, position-free
   summary `[B, H_r, 1, D_r]` per Receiver layer. This single attention converts
   every cross-model axis at once: sequence (`L_s·P → 1`), heads (`H_s → H_r`),
   head_dim (`D_s → D_r`, via the value projection), and layers (`L_s → L_r`).

2. **Project (joint KV bottleneck, the LCF projector).** The projector assumes the
   Sharer and Receiver caches have the **same shape**. The align stage guarantees
   that, so here we broadcast the aligned summary across the Receiver's `N_R`
   positions and **concatenate it with the Receiver's own cache** at each position
   (`[aK; aV; rK; rV]`, the paper's `4·H·D` Concat & Flatten), down-project to the
   bottleneck, up-project per layer, gate, and add to the Receiver cache. So the
   residual is Receiver-aware and position-dependent, not a pure Sharer broadcast.

``zero_init=True`` zeroes the projector's up-projection (and the gate starts at 0),
so the residual is exactly zero and LCF-X reproduces the Receiver-only run (the
MVP plumbing-correctness check). Sharer/Receiver may differ in heads, head_dim,
and layer count.
"""

from __future__ import annotations

import torch
from torch import nn


class LCFXFuser(nn.Module):
    """Align the Sharer summary to Receiver shape, then jointly project a residual."""

    def __init__(
        self,
        *,
        sharer_layers: int,
        sharer_kv_heads: int,
        sharer_head_dim: int,
        receiver_layers: int,
        receiver_kv_heads: int,
        receiver_head_dim: int,
        bottleneck_dim: int = 64,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.receiver_layers = receiver_layers
        self.receiver_kv_heads = receiver_kv_heads
        self.receiver_head_dim = receiver_head_dim
        self.attn_dim = bottleneck_dim

        sharer_token_dim = 2 * sharer_kv_heads * sharer_head_dim  # joint [K; V]
        receiver_feat = receiver_kv_heads * receiver_head_dim
        self.receiver_feat = receiver_feat

        # --- Align stage: attention resampler -> Receiver-shaped summary ---
        self.key_proj = nn.Linear(sharer_token_dim, bottleneck_dim)
        self.value_proj = nn.Linear(sharer_token_dim, 2 * receiver_feat)
        self.queries = nn.Parameter(torch.randn(receiver_layers, bottleneck_dim) * 0.02)

        # --- Project stage: joint KV bottleneck on [aligned-Sharer ; Receiver] ---
        # Input per position concatenates aK, aV, rK, rV (the paper's 4*H*D).
        self.down = nn.Linear(4 * receiver_feat, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.ModuleList(
            [nn.Linear(bottleneck_dim, 2 * receiver_feat) for _ in range(receiver_layers)]
        )
        # Per-(layer, head) gate. The residual is ``up_output * tanh(gate)``, so a
        # zero gate gives an exactly-zero residual (the equality check) while the
        # up-projection stays normally initialized. This is the ReZero trick: it
        # keeps the gate's gradient nonzero (∝ up_output), so training can start —
        # zeroing *both* factors would make both gradients zero and never move.
        self.gate = nn.Parameter(torch.zeros(receiver_layers, receiver_kv_heads))
        if not zero_init:
            nn.init.normal_(self.gate, std=0.1)

    # -- align stage --------------------------------------------------------
    def align(
        self, summary: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Resample span-pooled Sharer tokens into Receiver-shaped 1-slot summaries.

        Input:  ``L_s`` x ``(K, V)`` each ``[B, H_s, P, D_s]`` (from cache.pool_summary).
        Output: ``L_r`` x ``(aK, aV)`` each ``[B, H_r, 1, D_r]``.
        """

        weight_dtype = self.key_proj.weight.dtype
        batch = summary[0][0].shape[0]

        tokens = []
        for key, value in summary:
            spans = key.shape[2]
            key_flat = key.permute(0, 2, 1, 3).reshape(batch, spans, -1)
            value_flat = value.permute(0, 2, 1, 3).reshape(batch, spans, -1)
            tokens.append(torch.cat([key_flat, value_flat], dim=2))
        token_set = torch.cat(tokens, dim=1).to(weight_dtype)  # [B, L_s*P, sharer_token_dim]

        keys = self.key_proj(token_set)  # [B, T, attn_dim]
        values = self.value_proj(token_set)  # [B, T, 2*receiver_feat]
        queries = self.queries.to(weight_dtype)  # [L_r, attn_dim]

        scores = torch.einsum("ld,btd->blt", queries, keys) / (self.attn_dim**0.5)
        attn = scores.softmax(dim=-1)  # [B, L_r, T]
        pooled = torch.einsum("blt,btf->blf", attn, values)  # [B, L_r, 2*receiver_feat]

        aligned: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(self.receiver_layers):
            out = pooled[:, layer_idx].view(
                batch, 2, self.receiver_kv_heads, self.receiver_head_dim
            )
            a_key = out[:, 0].unsqueeze(2)  # [B, H_r, 1, D_r]
            a_value = out[:, 1].unsqueeze(2)
            aligned.append((a_key, a_value))
        return aligned

    # -- project stage ------------------------------------------------------
    def project(
        self,
        aligned: list[tuple[torch.Tensor, torch.Tensor]],
        receiver_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Joint KV bottleneck on matched-shape ``[aligned-Sharer ; Receiver]`` caches.

        ``aligned[i]``: ``(aK, aV)`` each ``[B, H_r, 1, D_r]`` (broadcast over N_R).
        ``receiver_kv[i]``: ``(rK, rV)`` each ``[B, H_r, N_R, D_r]``.
        Returns per layer ``(dK, dV)`` each ``[B, H_r, N_R, D_r]``.
        """

        weight_dtype = self.down.weight.dtype
        residuals: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, ((a_key, a_value), (r_key, r_value)) in enumerate(
            zip(aligned, receiver_kv)
        ):
            batch, _, seq, _ = r_key.shape

            def _flat(x: torch.Tensor) -> torch.Tensor:
                # [B, H_r, N, D_r] -> [B, N, H_r*D_r]
                return x.permute(0, 2, 1, 3).reshape(batch, x.shape[2], -1).to(weight_dtype)

            a_key_b = a_key.expand(-1, -1, seq, -1)
            a_value_b = a_value.expand(-1, -1, seq, -1)
            joint = torch.cat(
                [_flat(a_key_b), _flat(a_value_b), _flat(r_key), _flat(r_value)], dim=-1
            )  # [B, N, 4*receiver_feat]

            latent = self.act(self.down(joint))  # [B, N, d]
            out = self.up[layer_idx](latent).view(
                batch, seq, 2, self.receiver_kv_heads, self.receiver_head_dim
            )
            gate = torch.tanh(self.gate[layer_idx]).view(1, 1, self.receiver_kv_heads, 1)
            d_key = (out[:, :, 0] * gate).permute(0, 2, 1, 3)  # [B, H_r, N, D_r]
            d_value = (out[:, :, 1] * gate).permute(0, 2, 1, 3)
            residuals.append((d_key, d_value))
        return residuals

    def forward(
        self,
        summary: list[tuple[torch.Tensor, torch.Tensor]],
        receiver_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Align the Sharer summary, then jointly project with the Receiver cache."""

        return self.project(self.align(summary), receiver_kv)

    @classmethod
    def from_specs(
        cls,
        sharer_spec: "object",
        receiver_spec: "object",
        *,
        bottleneck_dim: int = 64,
        zero_init: bool = True,
    ) -> "LCFXFuser":
        """Build a fuser from two model specs (objects with layers/kv_heads/head_dim)."""

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

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
