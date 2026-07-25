"""XKV dual-cache attention translator.

Both caches are reduced directly from ``N -> k`` by vectorized learned-query
attention. Production defaults select ``k=1/2/3/4`` for 0.6B/1B/3B/8B Receivers
and one translation block for every Receiver.
Pool queries use the v9 Kaiming-normal x0.02 initialization.
A learned linear map aligns the Sharer layer axis to the Receiver layer count,
the pooled cache vectors are concatenated and bottlenecked, and
``num_blocks`` self-attention blocks translate the resulting joint layer
memory. Raw Receiver cache positions then query that memory through a shared
cross-attention decoder. A shared output head emits Receiver-shaped key/value
residuals through position-dependent per-head scaling and separate K/V
up-projections.

The XKV ablations share this implementation. They selectively remove Receiver
pooling, replace raw Receiver-KV cross-attention with broadcast layer memory,
or replace direct pooling with the exact LCF-X ``N -> P -> 1`` pool.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from methods.lcfx.fuser import _span_starts

XKV_FULL = "xkv"
XKV_NO_RECEIVER_POOLING = "xkv-no-receiver-pooling"
XKV_NO_RECEIVER_XATTN = "xkv-no-receiver-xattn"
XKV_LCFX_POOLING = "xkv-lcfx-pooling"
XKV_VARIANTS = (
    XKV_FULL,
    XKV_NO_RECEIVER_POOLING,
    XKV_NO_RECEIVER_XATTN,
    XKV_LCFX_POOLING,
)


def _variant_options(variant: str) -> tuple[str, str, bool]:
    """Return ``(pooling_mode, output_mode, receiver_pooling)`` for a preset."""

    if variant == XKV_FULL:
        return "direct_attention_k", "position_cross_attention", True
    if variant == XKV_NO_RECEIVER_POOLING:
        return "direct_attention_k", "position_cross_attention", False
    if variant == XKV_NO_RECEIVER_XATTN:
        return "direct_attention_k", "broadcast_layer_memory", True
    if variant == XKV_LCFX_POOLING:
        return "lcfx_exact", "position_cross_attention", True
    raise ValueError(
        f"unsupported XKV variant: {variant!r}; expected one of {XKV_VARIANTS}"
    )


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


def _default_pool_slots(receiver_model: str) -> int:
    """Receiver-size default: 0.6B/1B/3B/8B -> 1/2/3/4 slots."""

    tokens = str(receiver_model).lower().replace("/", "-").replace("_", "-").split("-")
    for size, slots in (("0.6b", 1), ("1b", 2), ("3b", 3), ("4b", 3), ("8b", 4)):
        if size in tokens:
            return slots
    return 2


def _cache_width(
    cache: list[tuple[torch.Tensor, torch.Tensor]], *, windowed_cache: bool
) -> int:
    """Common suffix width for uniform or declared hybrid-attention caches."""

    widths = [int(key.shape[2]) for key, _ in cache]
    if not widths or min(widths) < 1:
        raise ValueError(f"cache has invalid layer widths: {widths}")
    if not windowed_cache and len(set(widths)) != 1:
        raise ValueError(
            "unequal per-layer cache widths are supported only for a declared "
            f"hybrid full/sliding-attention cache; got {widths}"
        )
    return min(widths)


class AttentionPool(nn.Module):
    """Vectorized learned-query attention reducing raw positions to ``k`` slots."""

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
        # [L,H,k,D] follows the pooled-cache layout requested by XKV.
        self.query = nn.Parameter(
            torch.empty(layers, kv_heads, self.output_slots, head_dim)
        )
        nn.init.kaiming_normal_(self.query)
        with torch.no_grad():
            self.query.mul_(0.02)
        self._diagnostics_enabled = False
        self._last_diagnostics: dict[str, torch.Tensor] | None = None

    def set_diagnostics(self, enabled: bool) -> None:
        """Enable one-forward attention diagnostics without changing outputs."""

        self._diagnostics_enabled = bool(enabled)

    def diagnostics(self) -> dict[str, float] | None:
        """Return the most recently collected scalar diagnostics."""

        if self._last_diagnostics is None:
            return None
        return {
            name: float(value.detach().float().cpu().item())
            for name, value in self._last_diagnostics.items()
        }

    def _record_diagnostics(
        self,
        weights: torch.Tensor,
        valid: "torch.Tensor | None",
    ) -> None:
        """Record normalized entropy and mean off-diagonal slot cosine."""

        with torch.no_grad():
            probabilities = weights.detach().float()
            entropy = -(
                probabilities
                * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
            ).sum(dim=-1)
            if valid is None:
                valid_count = torch.full(
                    (probabilities.shape[0],),
                    probabilities.shape[-1],
                    device=probabilities.device,
                    dtype=probabilities.dtype,
                )
            else:
                valid_count = valid.sum(dim=-1).to(probabilities.dtype)
            entropy_denominator = valid_count.clamp_min(2).log()[:, None, None, None]
            normalized_entropy = torch.where(
                valid_count[:, None, None, None] > 1,
                entropy / entropy_denominator,
                torch.zeros_like(entropy),
            ).mean()

            diagnostics = {"entropy": normalized_entropy}
            if self.output_slots > 1:
                unit_weights = F.normalize(probabilities, dim=-1)
                similarity = torch.einsum(
                    "blhkn,blhjn->blhkj", unit_weights, unit_weights
                )
                off_diagonal = ~torch.eye(
                    self.output_slots,
                    device=similarity.device,
                    dtype=torch.bool,
                )
                diagnostics["slot_cosine"] = similarity[..., off_diagonal].mean()

            self._last_diagnostics = diagnostics

    def forward(
        self,
        cache: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        attention_mask: "torch.Tensor | None" = None,
        windowed_cache: bool = False,
    ) -> torch.Tensor:
        """Pool ``L x (K,V)[B,H,N,D]`` into ``[B,L,2*H*k*D]``.

        Keys determine the attention weights. The same weights reduce K and V,
        yielding separate pooled tensors ``[B,L,H,k,D]`` before flattening.
        ``attention_mask`` may be left- or right-padded.
        """

        if len(cache) != self.layers:
            raise ValueError(f"cache has {len(cache)} layers, expected {self.layers}")
        common_n = _cache_width(cache, windowed_cache=windowed_cache)
        keys = torch.stack([key[:, :, -common_n:, :] for key, _ in cache], dim=1)
        values = torch.stack([value[:, :, -common_n:, :] for _, value in cache], dim=1)
        keys = keys.to(self.query.dtype)
        values = values.to(self.query.dtype)
        scores = torch.einsum("lhkd,blhnd->blhkn", self.query, keys)
        scores = scores / self.head_dim**0.5
        valid = None
        if attention_mask is not None:
            if attention_mask.shape[-1] < common_n:
                raise ValueError(
                    f"attention mask width {attention_mask.shape[-1]} < cache width {common_n}"
                )
            if not windowed_cache and attention_mask.shape[-1] != common_n:
                raise ValueError(
                    f"attention mask width {attention_mask.shape[-1]} != cache width {common_n}"
                )
            valid = attention_mask[:, -common_n:].to(
                device=scores.device, dtype=torch.bool
            )
            scores = scores.masked_fill(~valid[:, None, None, None, :], float("-inf"))
        weights = scores.softmax(dim=-1).to(keys.dtype)
        if self._diagnostics_enabled:
            self._record_diagnostics(weights, valid)
        pooled_key = torch.einsum("blhkn,blhnd->blhkd", weights, keys)
        pooled_value = torch.einsum("blhkn,blhnd->blhkd", weights, values)
        batch = keys.shape[0]
        return torch.cat(
            [
                pooled_key.reshape(batch, self.layers, -1),
                pooled_value.reshape(batch, self.layers, -1),
            ],
            dim=-1,
        )


class LCFXExactAttentionPool(nn.Module):
    """The exact LCF-X two-query ``N -> P -> 1`` attention pool."""

    def __init__(
        self,
        layers: int,
        kv_heads: int,
        head_dim: int,
        num_spans: int = 5,
    ) -> None:
        super().__init__()
        if num_spans < 1:
            raise ValueError("num_spans must be >= 1")
        self.layers = int(layers)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.output_slots = 1
        self.num_spans = int(num_spans)
        # Names, shapes, and initialization match LCFXFuser exactly.
        self.q_base = nn.Parameter(torch.empty(layers, kv_heads, head_dim))
        self.q_layer = nn.Parameter(torch.empty(layers, kv_heads, head_dim))
        for query in (self.q_base, self.q_layer):
            nn.init.kaiming_normal_(query)
            with torch.no_grad():
                query.mul_(0.02)
        self._diagnostics_enabled = False
        self._last_diagnostics: dict[str, torch.Tensor] | None = None

    def set_diagnostics(self, enabled: bool) -> None:
        self._diagnostics_enabled = bool(enabled)

    def diagnostics(self) -> dict[str, float] | None:
        if self._last_diagnostics is None:
            return None
        return {
            name: float(value.detach().float().cpu().item())
            for name, value in self._last_diagnostics.items()
        }

    def _pool(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = torch.einsum("hd,bhnd->bhn", query, keys)
        scores = scores / self.head_dim**0.5
        weights = scores.softmax(dim=-1).to(keys.dtype)
        pooled_key = torch.einsum("bhn,bhnd->bhd", weights, keys)
        pooled_value = torch.einsum("bhn,bhnd->bhd", weights, values)
        return pooled_key, pooled_value, weights

    def forward(
        self,
        cache: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        attention_mask: "torch.Tensor | None" = None,
        windowed_cache: bool = False,
        span_boundaries: "list[list[int] | None] | None" = None,
    ) -> torch.Tensor:
        """Pool raw cache positions with LCF-X's exact two-stage equations."""

        if len(cache) != self.layers:
            raise ValueError(f"cache has {len(cache)} layers, expected {self.layers}")
        common_n = _cache_width(cache, windowed_cache=windowed_cache)
        batch = cache[0][0].shape[0]
        cache_device = cache[0][0].device

        if attention_mask is None:
            valid = torch.ones(
                batch, common_n, device=cache_device, dtype=torch.bool
            )
            full_valid_count = valid.sum(dim=-1)
        else:
            if attention_mask.shape[-1] < common_n:
                raise ValueError(
                    f"attention mask width {attention_mask.shape[-1]} < cache width {common_n}"
                )
            if not windowed_cache and attention_mask.shape[-1] != common_n:
                raise ValueError(
                    f"attention mask width {attention_mask.shape[-1]} != cache width {common_n}"
                )
            valid = attention_mask[:, -common_n:].to(
                device=cache_device, dtype=torch.bool
            )
            full_valid_count = attention_mask.to(
                device=cache_device, dtype=torch.bool
            ).sum(dim=-1)
        common_valid_count = valid.sum(dim=-1)
        if (common_valid_count < 1).any():
            raise ValueError("LCF-X pooling received an empty cache row")

        out_key: list[list[torch.Tensor]] = [[] for _ in range(self.layers)]
        out_value: list[list[torch.Tensor]] = [[] for _ in range(self.layers)]
        entropy: list[torch.Tensor] = []
        for row in range(batch):
            positions = valid[row].nonzero(as_tuple=False).flatten()
            n = int(positions.numel())
            dropped = int(full_valid_count[row] - common_valid_count[row])
            raw_boundaries = (
                span_boundaries[row] if span_boundaries is not None else None
            )
            boundaries = None
            if raw_boundaries:
                boundaries = [
                    int(start) - dropped
                    for start in raw_boundaries
                    if int(start) >= dropped
                ]
                if not boundaries or boundaries[0] != 0:
                    boundaries.insert(0, 0)
            starts = _span_starts(n, self.num_spans, boundaries)
            ends = starts[1:] + [n]
            for layer, (key, value) in enumerate(cache):
                # Deliberately retain LCF-X's original row/layer/span loops.
                # Do not stack/vectorize this path: its latency is part of the
                # ablation being measured.
                row_key = key[row : row + 1, :, -common_n:, :].index_select(
                    2, positions
                ).to(self.q_base.dtype)
                row_value = value[
                    row : row + 1, :, -common_n:, :
                ].index_select(2, positions).to(self.q_base.dtype)
                spans = [
                    self._pool(
                        self.q_base[layer],
                        row_key[:, :, start:end, :],
                        row_value[:, :, start:end, :],
                    )
                    for start, end in zip(starts, ends)
                ]
                span_keys = torch.stack([key for key, _, _ in spans], dim=2)
                span_values = torch.stack([value for _, value, _ in spans], dim=2)
                pooled_key, pooled_value, weights = self._pool(
                    self.q_layer[layer], span_keys, span_values
                )
                out_key[layer].append(pooled_key.unsqueeze(2))
                out_value[layer].append(pooled_value.unsqueeze(2))
                if self._diagnostics_enabled:
                    probabilities = weights.detach().float()
                    item_entropy = -(
                        probabilities
                        * probabilities.clamp_min(
                            torch.finfo(probabilities.dtype).tiny
                        ).log()
                    ).sum(dim=-1)
                    if len(starts) > 1:
                        item_entropy = item_entropy / torch.tensor(
                            len(starts),
                            device=item_entropy.device,
                            dtype=item_entropy.dtype,
                        ).log()
                    else:
                        item_entropy = torch.zeros_like(item_entropy)
                    entropy.append(item_entropy.mean())

        pooled_key = torch.stack(
            [torch.cat(layer_rows, dim=0) for layer_rows in out_key], dim=1
        )
        pooled_value = torch.stack(
            [torch.cat(layer_rows, dim=0) for layer_rows in out_value], dim=1
        )
        if self._diagnostics_enabled:
            self._last_diagnostics = {"entropy": torch.stack(entropy).mean()}
        return torch.cat(
            [
                pooled_key.reshape(batch, self.layers, -1),
                pooled_value.reshape(batch, self.layers, -1),
            ],
            dim=-1,
        )


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
        variant: str = XKV_FULL,
        pooling_mode: "str | None" = None,
        output_mode: "str | None" = None,
        receiver_pooling: "bool | None" = None,
        pool_spans: int = 5,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be >= 1")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if pool_slots < 1:
            raise ValueError("pool_slots must be >= 1")
        if pool_spans < 1:
            raise ValueError("pool_spans must be >= 1")
        preset = _variant_options(variant)
        pooling_mode = preset[0] if pooling_mode is None else pooling_mode
        output_mode = preset[1] if output_mode is None else output_mode
        receiver_pooling = (
            preset[2] if receiver_pooling is None else bool(receiver_pooling)
        )
        if (pooling_mode, output_mode, receiver_pooling) != preset:
            raise ValueError(
                f"XKV variant {variant!r} requires pooling/output/receiver "
                f"settings {preset}, got "
                f"{(pooling_mode, output_mode, receiver_pooling)}"
            )
        if pooling_mode == "lcfx_exact" and pool_slots != 1:
            raise ValueError(
                "xkv-lcfx-pooling uses LCF-X's exact P->1 second stage, "
                "so pool_slots must be 1"
            )
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
        self.variant = variant
        self.pooling_mode = pooling_mode
        self.output_mode = output_mode
        self.receiver_pooling = receiver_pooling
        self.pool_spans = int(pool_spans)

        sharer_feature = self.sharer_kv_heads * self.sharer_head_dim
        receiver_feature = self.receiver_kv_heads * self.receiver_head_dim
        if self.pooling_mode == "lcfx_exact":
            self.sharer_pool = LCFXExactAttentionPool(
                self.sharer_layers,
                self.sharer_kv_heads,
                self.sharer_head_dim,
                num_spans=self.pool_spans,
            )
        else:
            self.sharer_pool = AttentionPool(
                self.sharer_layers,
                self.sharer_kv_heads,
                self.sharer_head_dim,
                self.pool_slots,
            )
        self.receiver_pool = (
            (
                LCFXExactAttentionPool(
                    self.receiver_layers,
                    self.receiver_kv_heads,
                    self.receiver_head_dim,
                    num_spans=self.pool_spans,
                )
                if self.pooling_mode == "lcfx_exact"
                else AttentionPool(
                    self.receiver_layers,
                    self.receiver_kv_heads,
                    self.receiver_head_dim,
                    self.pool_slots,
                )
            )
            if self.receiver_pooling
            else None
        )
        self.layer_map = (
            nn.Linear(self.sharer_layers, self.receiver_layers)
            if self.sharer_layers != self.receiver_layers
            else None
        )

        joint_width = 2 * self.pool_slots * sharer_feature
        if self.receiver_pooling:
            joint_width += 2 * self.pool_slots * receiver_feature
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
        self.position_decoder = (
            _PositionDecoder(
                receiver_feature, self.d_model, self.num_heads, self.dropout
            )
            if self.output_mode == "position_cross_attention"
            else None
        )
        self.head_gate = nn.Linear(self.d_model, 2 * self.receiver_kv_heads)
        self.up_key = nn.Linear(self.d_model, receiver_feature)
        self.up_value = nn.Linear(self.d_model, receiver_feature)

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

    def set_pool_diagnostics(self, enabled: bool) -> None:
        """Collect pooling statistics on the next translation when enabled."""

        self.sharer_pool.set_diagnostics(enabled)
        if self.receiver_pool is not None:
            self.receiver_pool.set_diagnostics(enabled)

    def pool_diagnostics(self) -> dict[str, float]:
        """Return flattened Sharer/Receiver statistics for progress logging."""

        result: dict[str, float] = {}
        for side, pool in (
            ("sharer", self.sharer_pool),
            ("receiver", self.receiver_pool),
        ):
            if pool is None:
                continue
            diagnostics = pool.diagnostics()
            if diagnostics is not None:
                result.update(
                    {
                        f"{side}_pool_{name}": value
                        for name, value in diagnostics.items()
                    }
                )
        return result

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
        sharer_span_boundaries: "list[list[int] | None] | None" = None,
        receiver_span_boundaries: "list[list[int] | None] | None" = None,
        sharer_windowed_cache: bool = False,
        receiver_windowed_cache: bool = False,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return Receiver-shaped per-layer cache residuals."""

        sharer_slots = self.sharer_pool(
            sharer_cache,
            attention_mask=sharer_attention_mask,
            windowed_cache=sharer_windowed_cache,
            **(
                {"span_boundaries": sharer_span_boundaries}
                if self.pooling_mode == "lcfx_exact"
                else {}
            ),
        )
        sharer_slots = self._map_sharer_layers(sharer_slots)
        if self.receiver_pool is None:
            joint = sharer_slots
        else:
            receiver_slots = self.receiver_pool(
                receiver_cache,
                attention_mask=receiver_attention_mask,
                windowed_cache=receiver_windowed_cache,
                **(
                    {"span_boundaries": receiver_span_boundaries}
                    if self.pooling_mode == "lcfx_exact"
                    else {}
                ),
            )
            joint = torch.cat([sharer_slots, receiver_slots], dim=-1)
        memory = self.norm_joint(self.bottleneck_mlp(self.down(joint)))
        memory = memory + self.layer_embedding.unsqueeze(0)
        for block in self.blocks:
            memory = block(memory)

        receiver_common_n = _cache_width(
            receiver_cache, windowed_cache=receiver_windowed_cache
        )
        if self.position_decoder is None:
            latent = memory.unsqueeze(2).expand(-1, -1, receiver_common_n, -1)
        else:
            receiver_rows = self._receiver_rows(receiver_cache, receiver_common_n)
            latent = self.position_decoder(receiver_rows, memory, self.layer_embedding)
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
        delta_key = delta_key * alpha[..., 0, :].unsqueeze(-1)
        delta_value = delta_value * alpha[..., 1, :].unsqueeze(-1)

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
        d_model = _resolve_d_model(
            kwargs.pop("d_model", "eighth"), receiver_spec.hidden
        )
        pool_slots = kwargs.pop("pool_slots", None)
        if pool_slots is None:
            pool_slots = (
                1
                if kwargs.get("variant", XKV_FULL) == XKV_LCFX_POOLING
                else _default_pool_slots(receiver_spec.name)
            )
        num_blocks = kwargs.pop("num_blocks", 1)
        return cls(
            sharer_layers=sharer_spec.layers,
            sharer_kv_heads=sharer_spec.kv_heads,
            sharer_head_dim=sharer_spec.head_dim,
            receiver_layers=receiver_spec.layers,
            receiver_kv_heads=receiver_spec.kv_heads,
            receiver_head_dim=receiver_spec.head_dim,
            d_model=d_model,
            pool_slots=pool_slots,
            num_blocks=num_blocks,
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
            "variant": self.variant,
            "pooling_mode": self.pooling_mode,
            "output_mode": self.output_mode,
            "receiver_pooling": self.receiver_pooling,
            "pool_spans": self.pool_spans,
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


__all__ = [
    "AttentionPool",
    "LCFXExactAttentionPool",
    "XKVTranslator",
    "XKV_FULL",
    "XKV_NO_RECEIVER_POOLING",
    "XKV_NO_RECEIVER_XATTN",
    "XKV_LCFX_POOLING",
    "XKV_VARIANTS",
]
