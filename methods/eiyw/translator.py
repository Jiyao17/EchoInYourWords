"""EIYW translator: Sharer KV -> receiver-anchored soft pseudo-tokens.

The receiver-anchored method's contribution. Where LCF-X pools the Sharer cache
into one position-free summary per layer and broadcasts it as a gated KV residual
to *every* Receiver position, EIYW translates the Sharer cache into **K soft
tokens in the Receiver's embedding space**, conditioned on the Receiver's own
question, and **prepends them to the Receiver input**. The Receiver then attends
to them natively. Only this module is trained; both base models stay frozen.

Pipeline (one ``translate`` call):

1. **Pool the Sharer cache** (learned attention, reused from the LCF-X design):
   within-span ``N_S -> P`` keeping Sharer geometry, giving ``L_s*P`` span tokens
   of feature dim ``2*H_s*D_s``.
2. **Receiver-anchored queries (the novelty):** a learnable query bank
   ``Q_base in R^{K x d}`` is modulated by the Receiver-question anchor
   (``Q = Q_base + W_a . anchor``, FiLM-style), so *what gets communicated depends
   on the Receiver's task*, not only on how the Sharer encoded its context.
3. **Cross-attention** of the K queries over the ``L_s*P`` span tokens; the value
   projection emits ``hidden_r`` features, and an output projection produces the
   K soft tokens ``[B, K, hidden_r]``. This bridges arbitrary ``(H_s, D_s, L_s)``
   to ``hidden_r``, so heterogeneous Sharer/Receiver pairs work unchanged.

``zero_init=True`` zeros the output projection so the soft tokens start near zero
(a gradient-safe init). Note this does **not** reduce EIYW to the Receiver
answering alone — zero-valued soft tokens are still extra positions the Receiver
attends to. A gentle-init run (no ``--translator-checkpoint``) is therefore a
plumbing pass, not a zero-init equality (see ``runner.py`` / ``scripts/run_eiyw.py``).
"""

from __future__ import annotations

import torch
from torch import nn


class EIYWTranslator(nn.Module):
    """Pools the Sharer KV into K receiver-anchored soft tokens (Receiver embed space)."""

    def __init__(
        self,
        *,
        sharer_layers: int,
        sharer_kv_heads: int,
        sharer_head_dim: int,
        receiver_hidden: int,
        num_soft_tokens: int = 16,
        bottleneck_dim: int = 128,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if num_soft_tokens < 1:
            raise ValueError("num_soft_tokens must be >= 1")
        # Sharer dims + receiver hidden are kept so the module is self-describing
        # for checkpointing (they determine every weight shape).
        self.sharer_layers = sharer_layers
        self.sharer_kv_heads = sharer_kv_heads
        self.sharer_head_dim = sharer_head_dim
        self.receiver_hidden = receiver_hidden
        self.num_soft_tokens = num_soft_tokens
        self.bottleneck_dim = bottleneck_dim
        self.attn_dim = bottleneck_dim
        sharer_token_dim = 2 * sharer_kv_heads * sharer_head_dim

        # --- Stage 1: within-span learned attention (N_S -> P), keeps Sharer geom.
        # A shared learned query pools the tokens within each span.
        self.span_key = nn.Linear(sharer_token_dim, bottleneck_dim)
        self.span_query = nn.Parameter(torch.randn(bottleneck_dim) * 0.02)

        # --- Stage 2: receiver-anchored cross-attention (L_s*P -> K soft tokens).
        self.key_proj = nn.Linear(sharer_token_dim, bottleneck_dim)
        self.value_proj = nn.Linear(sharer_token_dim, receiver_hidden)
        # Learnable query bank (one query per soft token) ...
        self.query_bank = nn.Parameter(torch.randn(num_soft_tokens, bottleneck_dim) * 0.02)
        # ... modulated by the Receiver-question anchor (the receiver-anchoring).
        self.anchor_proj = nn.Linear(receiver_hidden, bottleneck_dim)

        # --- Output projection into Receiver embedding space.
        # ``zero_init`` uses a *gentle* small-scale init (not exact zero): the soft
        # tokens start tiny but nonzero, a near-minimal perturbation of the
        # Receiver. Exact-zero soft embeddings are a numerical degeneracy — feeding
        # all-zero vectors to the frozen Receiver makes its backward non-finite
        # (``0 * inf``) — and EIYW has no zero-init==Receiver-alone property to
        # preserve anyway (a gentle-init run is just a plumbing pass).
        self.out_proj = nn.Linear(receiver_hidden, receiver_hidden)
        if zero_init:
            nn.init.normal_(self.out_proj.weight, std=1e-3)
            nn.init.zeros_(self.out_proj.bias)

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

    def translate(
        self,
        summary: list[tuple[torch.Tensor, torch.Tensor]],
        anchor: torch.Tensor,
        lengths: "torch.Tensor | None" = None,
        num_spans: int = 4,
    ) -> torch.Tensor:
        """Raw Sharer cache + Receiver anchor -> K soft tokens in Receiver embed space.

        Input:  ``summary`` = ``L_s`` x ``(K, V)`` each ``[B, H_s, N_S, D_s]`` (raw,
        possibly padded); ``anchor`` = ``[B, hidden_r]`` (pooled Receiver-question
        embedding); ``lengths[b]`` = real Sharer token count for row ``b``.
        Output: ``[B, K, hidden_r]`` soft tokens.
        """

        weight_dtype = self.key_proj.weight.dtype
        batch = summary[0][0].shape[0]
        full_n = summary[0][0].shape[2]

        # Stage 1: per example, within-span attention over real tokens -> L_s*P tokens.
        token_sets = []
        for b in range(batch):
            n = int(lengths[b]) if lengths is not None else full_n
            n = max(1, n)
            layer_spans = [
                self._within_span(
                    self._flatten_kv(k[b : b + 1, :, :n, :], v[b : b + 1, :, :n, :]),
                    num_spans,
                )
                for k, v in summary
            ]
            token_sets.append(torch.cat(layer_spans, dim=1))  # [1, L_s*P, F_s]
        token_set = torch.cat(token_sets, dim=0).to(weight_dtype)  # [B, L_s*P, F_s]

        # Stage 2: receiver-anchored queries attend over the span tokens.
        keys = self.key_proj(token_set)  # [B, L_s*P, d]
        values = self.value_proj(token_set)  # [B, L_s*P, hidden_r]
        anchor_q = self.anchor_proj(anchor.to(weight_dtype))  # [B, d]
        queries = self.query_bank.to(weight_dtype)[None] + anchor_q[:, None, :]  # [B, K, d]
        scores = torch.einsum("bkd,btd->bkt", queries, keys) / (self.attn_dim**0.5)
        pooled = torch.einsum("bkt,btf->bkf", scores.softmax(dim=-1), values)  # [B, K, hidden_r]
        return self.out_proj(pooled)  # [B, K, hidden_r]

    def forward(
        self,
        summary: list[tuple[torch.Tensor, torch.Tensor]],
        anchor: torch.Tensor,
        lengths: "torch.Tensor | None" = None,
        num_spans: int = 4,
    ) -> torch.Tensor:
        return self.translate(summary, anchor, lengths=lengths, num_spans=num_spans)

    @classmethod
    def from_specs(
        cls,
        sharer_spec: "object",
        receiver_spec: "object",
        *,
        num_soft_tokens: int = 16,
        bottleneck_dim: int = 128,
        zero_init: bool = True,
    ) -> "EIYWTranslator":
        return cls(
            sharer_layers=sharer_spec.layers,
            sharer_kv_heads=sharer_spec.kv_heads,
            sharer_head_dim=sharer_spec.head_dim,
            receiver_hidden=receiver_spec.hidden,
            num_soft_tokens=num_soft_tokens,
            bottleneck_dim=bottleneck_dim,
            zero_init=zero_init,
        )

    def arch_config(self) -> dict:
        """Self-describing architecture (the dims that determine every weight shape).

        Persisted in the checkpoint so a trained translator can be rebuilt and
        validated against the Sharer/Receiver pair it was trained for.
        """

        return {
            "sharer_layers": self.sharer_layers,
            "sharer_kv_heads": self.sharer_kv_heads,
            "sharer_head_dim": self.sharer_head_dim,
            "receiver_hidden": self.receiver_hidden,
            "num_soft_tokens": self.num_soft_tokens,
            "bottleneck_dim": self.bottleneck_dim,
        }

    @classmethod
    def from_arch_config(cls, cfg: dict, *, zero_init: bool = True) -> "EIYWTranslator":
        """Rebuild a translator from an :meth:`arch_config` dict."""

        return cls(
            sharer_layers=cfg["sharer_layers"],
            sharer_kv_heads=cfg["sharer_kv_heads"],
            sharer_head_dim=cfg["sharer_head_dim"],
            receiver_hidden=cfg["receiver_hidden"],
            num_soft_tokens=cfg["num_soft_tokens"],
            bottleneck_dim=cfg["bottleneck_dim"],
            zero_init=zero_init,
        )

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
