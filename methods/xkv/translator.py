"""XKV dual-cache attention translator.

Both raw caches are independently reduced by an expanded LCF-X
``N -> P -> Q`` pool, with ``Q=2`` independent cross-span summaries by default.
A learned linear map aligns the Sharer layer axis to the Receiver layer count,
the two pooled cache vectors are concatenated and bottlenecked, and
``num_blocks`` self-attention blocks translate the resulting joint layer
memory. Raw Receiver cache positions then query that memory through a shared
cross-attention decoder. A shared gated output head emits Receiver-shaped
key/value residuals through separate K and V up-projections.
"""

from __future__ import annotations

import torch
from torch import nn


def _resolve_d_model(spec: "str | int", receiver_hidden: int) -> int:
    """Resolve a fixed width or Receiver-relative ``hidden//8`` width."""

    if isinstance(spec, bool):
        raise TypeError("d_model spec must be str or int, not bool")
    if isinstance(spec, int):
        width = spec
    else:
        value = str(spec).strip().lower().replace(" ", "")
        if value in {"eighth", "hidden/8", "hidden//8", "receiver/8"}:
            width = int(receiver_hidden) // 8
        else:
            width = int(value)
    if width < 1:
        raise ValueError("resolved d_model must be >= 1")
    return width


def _safe_num_heads(d_model: int, num_heads: int) -> int:
    for heads in range(min(int(num_heads), int(d_model)), 0, -1):
        if d_model % heads == 0:
            return heads
    return 1


def _span_starts(n: int, num_spans: int, boundaries: "list[int] | None") -> list[int]:
    if boundaries:
        starts = sorted({min(max(int(value), 0), n - 1) for value in boundaries})
        if not starts or starts[0] != 0:
            starts.insert(0, 0)
        return starts
    count = max(1, min(int(num_spans), n))
    base, extra = divmod(n, count)
    starts, pos = [], 0
    for index in range(count):
        starts.append(pos)
        pos += base + (1 if index < extra else 0)
    return starts


def _window_boundaries(
    boundaries: "list[int] | None", dropped_tokens: int, retained_tokens: int
) -> "list[int] | None":
    """Move full-sequence span starts onto a retained suffix window."""

    if not boundaries:
        return boundaries
    shifted = {
        min(max(int(boundary) - dropped_tokens, 0), retained_tokens - 1)
        for boundary in boundaries
        if int(boundary) >= dropped_tokens
    }
    shifted.add(0)
    return sorted(shifted)


def _cache_width(
    cache: list[tuple[torch.Tensor, torch.Tensor]], *, windowed_cache: bool
) -> int:
    """Common working width, with unequal layers allowed only for Gemma-3."""

    widths = [int(key.shape[2]) for key, _ in cache]
    if not widths or min(widths) < 1:
        raise ValueError(f"cache has invalid layer widths: {widths}")
    if not windowed_cache and len(set(widths)) != 1:
        raise ValueError(
            "unequal per-layer cache widths are supported only for a declared "
            f"Gemma-3 hybrid cache; got {widths}"
        )
    return min(widths)


def _gumbel_sigmoid(logit: torch.Tensor, training: bool, tau: float) -> torch.Tensor:
    if training:
        uniform = torch.rand_like(logit).clamp_(1e-6, 1 - 1e-6)
        noise = torch.log(uniform) - torch.log1p(-uniform)
        return torch.sigmoid((logit + noise) / tau)
    return (logit > 0).to(logit.dtype)


class TwoStagePool(nn.Module):
    """One side of the LCF-X pool, returning pooled KV slots per layer."""

    def __init__(
        self, layers: int, kv_heads: int, head_dim: int, output_slots: int = 2
    ) -> None:
        super().__init__()
        if output_slots < 1:
            raise ValueError("output_slots must be >= 1")
        self.layers = int(layers)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.output_slots = int(output_slots)
        self.q_base = nn.Parameter(torch.empty(layers, kv_heads, head_dim))
        self.q_layer = nn.Parameter(
            torch.empty(layers, self.output_slots, kv_heads, head_dim)
        )
        for query in (self.q_base, self.q_layer):
            nn.init.kaiming_normal_(query)
            with torch.no_grad():
                query.mul_(0.02)

    def _pool(
        self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.einsum("hd,bhnd->bhn", query, keys) / self.head_dim**0.5
        weights = scores.softmax(dim=-1).to(keys.dtype)
        return (
            torch.einsum("bhn,bhnd->bhd", weights, keys),
            torch.einsum("bhn,bhnd->bhd", weights, values),
        )

    def forward(
        self,
        cache: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        attention_mask: "torch.Tensor | None" = None,
        num_spans: int = 5,
        span_boundaries: "list[list[int] | None] | None" = None,
        windowed_cache: bool = False,
    ) -> torch.Tensor:
        """Pool ``L x (K,V)[B,H,N,D]`` into ``[B,L,2*Q*H*D]``.

        ``attention_mask`` may be left- or right-padded; real cache positions
        are selected before span offsets are applied. ``Q=output_slots`` is the
        number of independent cross-span summaries retained per layer.
        """

        if len(cache) != self.layers:
            raise ValueError(f"cache has {len(cache)} layers, expected {self.layers}")
        batch = cache[0][0].shape[0]
        common_n = _cache_width(cache, windowed_cache=windowed_cache)
        original_n = max(int(key.shape[2]) for key, _ in cache)
        dtype = self.q_base.dtype
        rows: list[torch.Tensor] = []
        for batch_index in range(batch):
            if attention_mask is None:
                position_mask = None
                n = common_n
                dropped_tokens = max(0, original_n - common_n)
            else:
                full_mask = attention_mask[batch_index].to(torch.bool)
                if not windowed_cache and full_mask.numel() != common_n:
                    raise ValueError(
                        f"attention mask width {full_mask.numel()} != cache width {common_n}"
                    )
                if full_mask.numel() < common_n:
                    raise ValueError(
                        f"attention mask width {full_mask.numel()} < cache width {common_n}"
                    )
                position_mask = full_mask[-common_n:]
                n = int(position_mask.sum().item())
                dropped_tokens = max(0, int(full_mask.sum().item()) - n)
            if n < 1:
                raise ValueError("cache window contains no unmasked token positions")
            boundaries = span_boundaries[batch_index] if span_boundaries else None
            boundaries = _window_boundaries(boundaries, dropped_tokens, n)
            starts = _span_starts(n, num_spans, boundaries)
            ends = starts[1:] + [n]
            layer_rows: list[torch.Tensor] = []
            for layer_index, (key, value) in enumerate(cache):
                key_row = key[batch_index : batch_index + 1, :, -common_n:, :]
                value_row = value[batch_index : batch_index + 1, :, -common_n:, :]
                if position_mask is not None:
                    key_row = key_row[:, :, position_mask, :]
                    value_row = value_row[:, :, position_mask, :]
                key_row = key_row[:, :, :n, :].to(dtype)
                value_row = value_row[:, :, :n, :].to(dtype)
                spans = [
                    self._pool(
                        self.q_base[layer_index],
                        key_row[:, :, start:end, :],
                        value_row[:, :, start:end, :],
                    )
                    for start, end in zip(starts, ends)
                ]
                span_keys = torch.stack([item[0] for item in spans], dim=2)
                span_values = torch.stack([item[1] for item in spans], dim=2)
                pooled = [
                    self._pool(query, span_keys, span_values)
                    for query in self.q_layer[layer_index]
                ]
                pooled_key = torch.stack([item[0] for item in pooled], dim=2)
                pooled_value = torch.stack([item[1] for item in pooled], dim=2)
                layer_rows.append(
                    torch.cat(
                        [pooled_key.reshape(1, -1), pooled_value.reshape(1, -1)],
                        dim=-1,
                    )
                )
            rows.append(torch.stack(layer_rows, dim=1))
        return torch.cat(rows, dim=0)


class _TranslatorBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm_attn(x)
        attended, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        x = x + attended
        return x + self.ffn(self.norm_ffn(x))


class _PositionDecoder(nn.Module):
    """Shared Receiver-position queries cross-attending to joint layer memory."""

    def __init__(
        self, receiver_feature: int, d_model: int, num_heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.query = nn.Linear(2 * receiver_feature, d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        receiver_rows: torch.Tensor,
        memory: torch.Tensor,
        layer_embedding: torch.Tensor,
    ) -> torch.Tensor:
        # receiver_rows: [B,Lr,N,2Fr], memory: [B,Lr,d]
        batch, layers, positions, _ = receiver_rows.shape
        queries = self.query(receiver_rows)
        queries = queries + layer_embedding.view(1, layers, 1, -1)
        flat_queries = queries.reshape(batch * layers, positions, -1)
        flat_memory = (
            memory[:, None, :, :]
            .expand(batch, layers, memory.shape[1], memory.shape[2])
            .reshape(batch * layers, memory.shape[1], memory.shape[2])
        )
        normalized = self.norm_cross(flat_queries)
        attended, _ = self.cross_attn(
            normalized, flat_memory, flat_memory, need_weights=False
        )
        output = flat_queries + attended
        output = output + self.ffn(self.norm_ffn(output))
        return output.reshape(batch, layers, positions, -1)


class XKVTranslator(nn.Module):
    """Dual-cache pooled attention translator producing Receiver KV residuals."""

    def __init__(
        self,
        *,
        sharer_layers: int,
        sharer_kv_heads: int,
        sharer_head_dim: int,
        receiver_layers: int,
        receiver_kv_heads: int,
        receiver_head_dim: int,
        d_model: int = 256,
        num_blocks: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        pool_slots: int = 2,
        output_mode: str = "position_cross_attention",
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be >= 1")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if pool_slots < 1:
            raise ValueError("pool_slots must be >= 1")
        if output_mode != "position_cross_attention":
            raise ValueError(f"unsupported XKV output_mode: {output_mode!r}")
        self.sharer_layers = int(sharer_layers)
        self.sharer_kv_heads = int(sharer_kv_heads)
        self.sharer_head_dim = int(sharer_head_dim)
        self.receiver_layers = int(receiver_layers)
        self.receiver_kv_heads = int(receiver_kv_heads)
        self.receiver_head_dim = int(receiver_head_dim)
        self.d_model = int(d_model)
        self.num_blocks = int(num_blocks)
        self.num_heads = _safe_num_heads(self.d_model, num_heads)
        self.dropout = float(dropout)
        self.pool_slots = int(pool_slots)
        self.output_mode = output_mode

        sharer_feature = self.sharer_kv_heads * self.sharer_head_dim
        receiver_feature = self.receiver_kv_heads * self.receiver_head_dim
        self.sharer_pool = TwoStagePool(
            self.sharer_layers,
            self.sharer_kv_heads,
            self.sharer_head_dim,
            self.pool_slots,
        )
        self.receiver_pool = TwoStagePool(
            self.receiver_layers,
            self.receiver_kv_heads,
            self.receiver_head_dim,
            self.pool_slots,
        )
        self.layer_map = (
            nn.Linear(self.sharer_layers, self.receiver_layers)
            if self.sharer_layers != self.receiver_layers
            else None
        )

        joint_width = 2 * self.pool_slots * (sharer_feature + receiver_feature)
        self.down = nn.Linear(joint_width, self.d_model)
        self.bottleneck_mlp = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.GELU(),
            nn.Linear(4 * self.d_model, self.d_model),
        )
        self.norm_joint = nn.LayerNorm(self.d_model)
        self.layer_embedding = nn.Parameter(
            torch.empty(self.receiver_layers, self.d_model)
        )
        nn.init.normal_(self.layer_embedding, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _TranslatorBlock(self.d_model, self.num_heads, self.dropout)
                for _ in range(self.num_blocks)
            ]
        )
        self.position_decoder = _PositionDecoder(
            receiver_feature, self.d_model, self.num_heads, self.dropout
        )
        self.head_gate = nn.Linear(self.d_model, 2 * self.receiver_kv_heads)
        self.up_key = nn.Linear(self.d_model, receiver_feature)
        self.up_value = nn.Linear(self.d_model, receiver_feature)
        self.layer_gate = nn.Parameter(torch.ones(self.receiver_layers, 2))
        self.gate_tau = 1.0

        if zero_init:
            for projection in (self.up_key, self.up_value):
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
        else:
            for projection in (self.up_key, self.up_value):
                nn.init.kaiming_normal_(projection.weight)
                with torch.no_grad():
                    projection.weight.mul_(0.06)
                nn.init.zeros_(projection.bias)

    def _map_sharer_layers(self, slots: torch.Tensor) -> torch.Tensor:
        if self.layer_map is None:
            return slots
        return self.layer_map(slots.transpose(1, 2)).transpose(1, 2)

    def _receiver_rows(
        self,
        cache: list[tuple[torch.Tensor, torch.Tensor]],
        common_n: int,
    ) -> torch.Tensor:
        rows = []
        for key, value in cache:
            key = key[:, :, -common_n:, :]
            value = value[:, :, -common_n:, :]
            batch, _, positions, _ = key.shape
            key_rows = key.permute(0, 2, 1, 3).reshape(batch, positions, -1)
            value_rows = value.permute(0, 2, 1, 3).reshape(batch, positions, -1)
            rows.append(torch.cat([key_rows, value_rows], dim=-1))
        return torch.stack(rows, dim=1).to(self.down.weight.dtype)

    def translate(
        self,
        sharer_cache: list[tuple[torch.Tensor, torch.Tensor]],
        receiver_cache: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        sharer_attention_mask: "torch.Tensor | None" = None,
        receiver_attention_mask: "torch.Tensor | None" = None,
        num_spans: int = 5,
        sharer_span_boundaries: "list[list[int] | None] | None" = None,
        receiver_span_boundaries: "list[list[int] | None] | None" = None,
        sharer_windowed_cache: bool = False,
        receiver_windowed_cache: bool = False,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return Receiver-shaped per-layer cache residuals."""

        sharer_slots = self.sharer_pool(
            sharer_cache,
            attention_mask=sharer_attention_mask,
            num_spans=num_spans,
            span_boundaries=sharer_span_boundaries,
            windowed_cache=sharer_windowed_cache,
        )
        receiver_slots = self.receiver_pool(
            receiver_cache,
            attention_mask=receiver_attention_mask,
            num_spans=num_spans,
            span_boundaries=receiver_span_boundaries,
            windowed_cache=receiver_windowed_cache,
        )
        sharer_slots = self._map_sharer_layers(sharer_slots)
        joint = torch.cat([sharer_slots, receiver_slots], dim=-1)
        memory = self.norm_joint(self.bottleneck_mlp(self.down(joint)))
        memory = memory + self.layer_embedding.unsqueeze(0)
        for block in self.blocks:
            memory = block(memory)

        receiver_common_n = _cache_width(
            receiver_cache, windowed_cache=receiver_windowed_cache
        )
        receiver_rows = self._receiver_rows(receiver_cache, receiver_common_n)
        latent = self.position_decoder(
            receiver_rows, memory, self.layer_embedding
        )
        batch, layers, positions, _ = latent.shape
        alpha = torch.sigmoid(self.head_gate(latent)).view(
            batch, layers, positions, 2, self.receiver_kv_heads
        )
        delta_key = self.up_key(latent).view(
            batch,
            layers,
            positions,
            self.receiver_kv_heads,
            self.receiver_head_dim,
        )
        delta_value = self.up_value(latent).view_as(delta_key)
        gates = _gumbel_sigmoid(
            self.layer_gate, self.training, max(float(self.gate_tau), 1e-6)
        )
        delta_key = delta_key * alpha[..., 0, :].unsqueeze(-1)
        delta_value = delta_value * alpha[..., 1, :].unsqueeze(-1)
        delta_key = delta_key * gates[None, :, None, 0, None, None]
        delta_value = delta_value * gates[None, :, None, 1, None, None]

        if receiver_attention_mask is not None:
            real = receiver_attention_mask[:, -receiver_common_n:].to(
                device=delta_key.device, dtype=delta_key.dtype
            )[:, None, :, None, None]
            delta_key = delta_key * real
            delta_value = delta_value * real

        residuals = []
        for layer in range(layers):
            recent_key = delta_key[:, layer].permute(0, 2, 1, 3).contiguous()
            recent_value = delta_value[:, layer].permute(0, 2, 1, 3).contiguous()
            layer_n = int(receiver_cache[layer][0].shape[2])
            left_pad = layer_n - receiver_common_n
            if left_pad:
                recent_key = torch.nn.functional.pad(recent_key, (0, 0, left_pad, 0))
                recent_value = torch.nn.functional.pad(
                    recent_value, (0, 0, left_pad, 0)
                )
            residuals.append((recent_key, recent_value))
        return residuals

    def forward(self, *args, **kwargs):
        return self.translate(*args, **kwargs)

    @classmethod
    def from_specs(
        cls,
        sharer_spec: object,
        receiver_spec: object,
        **kwargs,
    ) -> "XKVTranslator":
        d_model = _resolve_d_model(kwargs.pop("d_model", "eighth"), receiver_spec.hidden)
        return cls(
            sharer_layers=sharer_spec.layers,
            sharer_kv_heads=sharer_spec.kv_heads,
            sharer_head_dim=sharer_spec.head_dim,
            receiver_layers=receiver_spec.layers,
            receiver_kv_heads=receiver_spec.kv_heads,
            receiver_head_dim=receiver_spec.head_dim,
            d_model=d_model,
            **kwargs,
        )

    def arch_config(self) -> dict:
        return {
            "sharer_layers": self.sharer_layers,
            "sharer_kv_heads": self.sharer_kv_heads,
            "sharer_head_dim": self.sharer_head_dim,
            "receiver_layers": self.receiver_layers,
            "receiver_kv_heads": self.receiver_kv_heads,
            "receiver_head_dim": self.receiver_head_dim,
            "d_model": self.d_model,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "pool_slots": self.pool_slots,
            "output_mode": self.output_mode,
        }

    @classmethod
    def from_arch_config(
        cls, config: dict, *, zero_init: bool = True
    ) -> "XKVTranslator":
        return cls(**config, zero_init=zero_init)

    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


__all__ = ["TwoStagePool", "XKVTranslator"]
