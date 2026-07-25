"""CPU-only architecture checks for XKV (no model downloads)."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from methods.lcfx.fuser import LCFXFuser  # noqa: E402
from methods.xkv.training import load_translator, save_translator  # noqa: E402
from methods.xkv.translator import (  # noqa: E402
    LCFXExactAttentionPool,
    XKVTranslator,
    _default_pool_slots,
    _resolve_d_model,
)


def _cache(batch: int, layers: int, heads: int, positions: int, dim: int):
    return [
        (
            torch.randn(batch, heads, positions, dim),
            torch.randn(batch, heads, positions, dim),
        )
        for _ in range(layers)
    ]


def _variable_cache(batch: int, heads: int, positions: list[int], dim: int):
    return [
        (
            torch.randn(batch, heads, width, dim),
            torch.randn(batch, heads, width, dim),
        )
        for width in positions
    ]


def main() -> None:
    assert _default_pool_slots("Qwen/Qwen3-0.6B") == 1
    assert _default_pool_slots("unsloth/gemma-3-1b-it") == 2
    assert _default_pool_slots("unsloth/Llama-3.2-3B-Instruct") == 3
    assert _default_pool_slots("unsloth/gemma-3-4b-it") == 3
    assert _default_pool_slots("mistralai/Ministral-8B-Instruct-2410") == 4
    assert _default_pool_slots("unknown/receiver") == 2
    assert _resolve_d_model("eighth", 1024) == 128
    assert _resolve_d_model("hidden//8", 1152) == 144
    assert _resolve_d_model("receiver/8", 3072) == 384
    assert _resolve_d_model(256, 1024) == 256
    default = XKVTranslator(
        sharer_layers=28,
        sharer_kv_heads=8,
        sharer_head_dim=128,
        receiver_layers=28,
        receiver_kv_heads=8,
        receiver_head_dim=128,
        d_model=256,
        num_blocks=1,
    )
    assert default.trainable_parameters() == 5_379_856
    assert default.position_decoder is not None
    assert default.output_mode == "position_cross_attention"
    assert default.pooling_mode == "direct_attention_k"
    assert default.pool_slots == 2
    assert tuple(default.sharer_pool.query.shape) == (28, 8, 2, 128)
    assert tuple(default.receiver_pool.query.shape) == (28, 8, 2, 128)
    assert default.sharer_pool.query.square().mean().sqrt().item() < 0.01
    assert default.receiver_pool.query.square().mean().sqrt().item() < 0.01
    assert tuple(default.down.weight.shape) == (256, 8192)
    assert default.up_key.in_features == 256
    assert default.up_value.in_features == 256
    assert not hasattr(default, "layer_gate")
    assert not hasattr(default, "gate_tau")
    qwen_spec = SimpleNamespace(
        name="Qwen/Qwen3-0.6B",
        layers=28,
        kv_heads=8,
        head_dim=128,
        hidden=1024,
    )
    dynamic_default = XKVTranslator.from_specs(qwen_spec, qwen_spec)
    assert dynamic_default.d_model == 128
    assert dynamic_default.pool_slots == 1
    assert dynamic_default.num_blocks == 1
    assert dynamic_default.trainable_parameters() == 1_642_384
    gemma_spec = SimpleNamespace(
        name="unsloth/gemma-3-1b-it",
        layers=26,
        kv_heads=1,
        head_dim=256,
        hidden=1152,
    )
    llama_spec = SimpleNamespace(
        name="unsloth/Llama-3.2-3B-Instruct",
        layers=28,
        kv_heads=8,
        head_dim=128,
        hidden=3072,
    )
    gemma_default = XKVTranslator.from_specs(qwen_spec, gemma_spec)
    llama_default = XKVTranslator.from_specs(qwen_spec, llama_spec)
    assert gemma_default.pool_slots == 2 and gemma_default.num_blocks == 1
    assert llama_default.pool_slots == 3 and llama_default.num_blocks == 1
    explicit = XKVTranslator.from_specs(
        qwen_spec, llama_spec, pool_slots=1, num_blocks=1
    )
    assert explicit.pool_slots == 1 and explicit.num_blocks == 1
    deeper = XKVTranslator(
        sharer_layers=28,
        sharer_kv_heads=8,
        sharer_head_dim=128,
        receiver_layers=28,
        receiver_kv_heads=8,
        receiver_head_dim=128,
        d_model=256,
        num_blocks=2,
    )
    assert deeper.trainable_parameters() - default.trainable_parameters() == 789_760
    print("default/depth parameter counts OK")

    kwargs = dict(
        sharer_layers=3,
        sharer_kv_heads=2,
        sharer_head_dim=4,
        receiver_layers=2,
        receiver_kv_heads=1,
        receiver_head_dim=6,
        d_model=8,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
    )
    sharer = _cache(2, 3, 2, 7, 4)
    receiver = _cache(2, 2, 1, 5, 6)
    sharer_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]])
    receiver_mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]])

    no_receiver_pool = XKVTranslator(
        **kwargs, variant="xkv-no-receiver-pooling", zero_init=False
    )
    assert no_receiver_pool.receiver_pool is None
    assert no_receiver_pool.position_decoder is not None
    assert no_receiver_pool.down.in_features == 2 * 2 * (2 * 4)

    no_receiver_xattn = XKVTranslator(
        **kwargs, variant="xkv-no-receiver-xattn", zero_init=False
    )
    assert no_receiver_xattn.receiver_pool is not None
    assert no_receiver_xattn.position_decoder is None
    assert no_receiver_xattn.output_mode == "broadcast_layer_memory"

    lcfx_pooling = XKVTranslator(
        **kwargs,
        variant="xkv-lcfx-pooling",
        pool_slots=1,
        pool_spans=3,
        zero_init=False,
    )
    assert isinstance(lcfx_pooling.sharer_pool, LCFXExactAttentionPool)
    assert isinstance(lcfx_pooling.receiver_pool, LCFXExactAttentionPool)
    assert lcfx_pooling.pooling_mode == "lcfx_exact"
    assert lcfx_pooling.pool_slots == 1
    assert lcfx_pooling.pool_spans == 3

    # The ablation deliberately retains LCF-X's original non-vectorized
    # batch/layer/span loop. With identical queries, its pooled tensors must be
    # numerically identical to LCFXFuser.align, including natural boundaries.
    exact_cache = _cache(2, 2, 2, 7, 4)
    exact_pool = LCFXExactAttentionPool(2, 2, 4, num_spans=3)
    reference = LCFXFuser(
        sharer_layers=2,
        sharer_kv_heads=2,
        sharer_head_dim=4,
        receiver_layers=2,
        receiver_kv_heads=2,
        receiver_head_dim=4,
        bottleneck_dim=8,
        dropout=0.0,
    )
    with torch.no_grad():
        reference.q_base.copy_(exact_pool.q_base)
        reference.q_layer.copy_(exact_pool.q_layer)
    boundaries = [[0, 2, 5], [0, 3]]
    exact_output = exact_pool(exact_cache, span_boundaries=boundaries)
    reference_output = reference.align(
        exact_cache,
        num_spans=3,
        span_boundaries=boundaries,
    )
    reference_key = torch.stack(
        [key for key, _ in reference_output], dim=1
    ).reshape(2, 2, -1)
    reference_value = torch.stack(
        [value for _, value in reference_output], dim=1
    ).reshape(2, 2, -1)
    assert torch.allclose(
        exact_output,
        torch.cat([reference_key, reference_value], dim=-1),
    )

    for variant in (no_receiver_pool, no_receiver_xattn, lcfx_pooling):
        residual = variant.translate(
            sharer,
            receiver,
            sharer_attention_mask=sharer_mask,
            receiver_attention_mask=receiver_mask,
        )
        assert residual[0][0].shape == (2, 1, 5, 6)
        loss = sum(
            key.square().mean() + value.square().mean() for key, value in residual
        )
        loss.backward()
        missing = [
            name
            for name, parameter in variant.named_parameters()
            if parameter.requires_grad
            and (parameter.grad is None or not torch.isfinite(parameter.grad).all())
        ]
        assert not missing, (variant.variant, missing)
    print("three XKV ablation architecture/gradient checks OK")

    zero = XKVTranslator(**kwargs, zero_init=True).eval()
    three_slot = XKVTranslator(**kwargs, pool_slots=3, zero_init=True).eval()
    assert tuple(three_slot.sharer_pool.query.shape) == (3, 2, 3, 4)
    assert tuple(three_slot.receiver_pool.query.shape) == (2, 1, 3, 6)
    assert three_slot.down.in_features == 2 * 3 * (2 * 4 + 1 * 6)
    pooled = zero.sharer_pool(
        sharer,
        attention_mask=sharer_mask,
    )
    assert pooled.shape == (2, 3, 2 * 2 * 2 * 4)
    pooled_keys = pooled[..., : 2 * 2 * 4].reshape(2, 3, 2, 2, 4)
    assert not torch.allclose(pooled_keys[:, :, :, 0], pooled_keys[:, :, :, 1])
    zero.set_pool_diagnostics(True)
    zero.translate(
        sharer,
        receiver,
        sharer_attention_mask=sharer_mask,
        receiver_attention_mask=receiver_mask,
    )
    zero.set_pool_diagnostics(False)
    diagnostics = zero.pool_diagnostics()
    assert set(diagnostics) == {
        "sharer_pool_entropy",
        "sharer_pool_slot_cosine",
        "receiver_pool_entropy",
        "receiver_pool_slot_cosine",
    }
    assert 0.0 <= diagnostics["sharer_pool_entropy"] <= 1.000001
    assert 0.0 <= diagnostics["receiver_pool_entropy"] <= 1.000001
    assert -1.000001 <= diagnostics["sharer_pool_slot_cosine"] <= 1.000001
    assert -1.000001 <= diagnostics["receiver_pool_slot_cosine"] <= 1.000001
    one_slot = XKVTranslator(**kwargs, pool_slots=1, zero_init=True).eval()
    one_slot.set_pool_diagnostics(True)
    one_slot.translate(
        sharer,
        receiver,
        sharer_attention_mask=sharer_mask,
        receiver_attention_mask=receiver_mask,
    )
    assert set(one_slot.pool_diagnostics()) == {
        "sharer_pool_entropy",
        "receiver_pool_entropy",
    }
    receiver_pooled = zero.receiver_pool(
        receiver,
        attention_mask=receiver_mask,
    )
    assert receiver_pooled.shape == (2, 2, 2 * 2 * 1 * 6)
    assert torch.isfinite(receiver_pooled).all()
    receiver_corrupt = [(key.clone(), value.clone()) for key, value in receiver]
    for key, value in receiver_corrupt:
        key[1, :, :2, :] = 10_000
        value[1, :, :2, :] = -10_000
    receiver_pooled_corrupt = zero.receiver_pool(
        receiver_corrupt,
        attention_mask=receiver_mask,
    )
    assert torch.allclose(receiver_pooled[1], receiver_pooled_corrupt[1])
    residual = zero.translate(
        sharer,
        receiver,
        sharer_attention_mask=sharer_mask,
        receiver_attention_mask=receiver_mask,
    )
    assert len(residual) == 2
    assert residual[0][0].shape == (2, 1, 5, 6)
    assert all(
        key.count_nonzero() == 0 and value.count_nonzero() == 0
        for key, value in residual
    )
    assert zero.layer_map is not None
    assert tuple(zero.layer_map.weight.shape) == (2, 3)
    print("heterogeneous shape/layer-map/zero-update checks OK")

    trained = XKVTranslator(**kwargs, zero_init=False).train()
    residual = trained.translate(
        sharer,
        receiver,
        sharer_attention_mask=sharer_mask,
        receiver_attention_mask=receiver_mask,
    )
    loss = sum(key.square().mean() + value.square().mean() for key, value in residual)
    loss.backward()
    missing = [
        name
        for name, parameter in trained.named_parameters()
        if parameter.requires_grad
        and (parameter.grad is None or not torch.isfinite(parameter.grad).all())
    ]
    assert not missing, missing
    assert residual[0][0][1, :, :2].count_nonzero() == 0
    assert not torch.equal(residual[0][0][0, :, 0], residual[0][0][0, :, 1])
    assert not torch.equal(residual[0][1][1, :, 2], residual[0][1][1, :, 4])
    print("finite gradients, position-specific updates, and padding masking OK")

    # Gemma-3 and Ministral use hybrid sliding/full-attention cache layers. The
    # translator uses the shortest recent suffix internally, then restores a
    # Receiver-shaped residual with zeros over each full layer's older prefix.
    hybrid_sharer = _variable_cache(2, 2, [5, 7, 5], 4)
    hybrid_receiver = _variable_cache(2, 1, [5, 7], 6)
    hybrid_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1]])
    trained.zero_grad(set_to_none=True)
    hybrid_residual = trained.translate(
        hybrid_sharer,
        hybrid_receiver,
        sharer_attention_mask=hybrid_mask,
        receiver_attention_mask=hybrid_mask,
        sharer_windowed_cache=True,
        receiver_windowed_cache=True,
    )
    assert hybrid_residual[0][0].shape == (2, 1, 5, 6)
    assert hybrid_residual[1][0].shape == (2, 1, 7, 6)
    assert hybrid_residual[1][0][:, :, :2].count_nonzero() == 0
    assert hybrid_residual[1][1][:, :, :2].count_nonzero() == 0
    hybrid_loss = sum(
        key.square().mean() + value.square().mean() for key, value in hybrid_residual
    )
    hybrid_loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trained.parameters()
        if parameter.requires_grad
    )
    try:
        trained.translate(hybrid_sharer, hybrid_receiver)
    except ValueError as exc:
        assert "hybrid full/sliding-attention cache" in str(exc)
    else:
        raise AssertionError("undeclared unequal cache widths were accepted")
    print("hybrid-cache suffix normalization checks OK")

    specs = SimpleNamespace(
        sharer_spec=SimpleNamespace(
            name="sharer", layers=3, kv_heads=2, head_dim=4, hidden=48
        ),
        receiver_spec=SimpleNamespace(
            name="receiver", layers=2, kv_heads=1, head_dim=6, hidden=64
        ),
    )
    dynamic = XKVTranslator.from_specs(
        specs.sharer_spec,
        specs.receiver_spec,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
    )
    assert dynamic.d_model == 8
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "translator.pt")
        save_translator(trained, path, specs)
        loaded = load_translator(
            path,
            specs,
            device="cpu",
            d_model=8,
            num_blocks=1,
            num_heads=2,
            dropout=0.0,
        )
        assert loaded.arch_config() == trained.arch_config()
        pre_ablation_path = str(Path(directory) / "pre_ablation_full.pt")
        pre_ablation = torch.load(path, map_location="cpu", weights_only=True)
        for field in ("variant", "receiver_pooling", "pool_spans"):
            pre_ablation["arch"].pop(field)
        torch.save(pre_ablation, pre_ablation_path)
        loaded_pre_ablation = load_translator(
            pre_ablation_path,
            specs,
            device="cpu",
            d_model=8,
            num_blocks=1,
            num_heads=2,
            dropout=0.0,
        )
        assert loaded_pre_ablation.arch_config() == trained.arch_config()
        ablation_path = str(Path(directory) / "lcfx_pooling.pt")
        save_translator(lcfx_pooling, ablation_path, specs)
        loaded_ablation = load_translator(
            ablation_path,
            specs,
            device="cpu",
            d_model=8,
            num_blocks=1,
            num_heads=2,
            dropout=0.0,
            variant="xkv-lcfx-pooling",
            pool_spans=3,
        )
        assert loaded_ablation.arch_config() == lcfx_pooling.arch_config()
        try:
            load_translator(
                ablation_path,
                specs,
                device="cpu",
                d_model=8,
                num_blocks=1,
                num_heads=2,
                dropout=0.0,
                variant="xkv",
            )
        except ValueError as exc:
            assert "arch.variant" in str(exc)
        else:
            raise AssertionError("cross-variant XKV checkpoint was accepted")
        legacy_path = str(Path(directory) / "legacy.pt")
        legacy = torch.load(path, map_location="cpu", weights_only=True)
        legacy["format"] = 9
        torch.save(legacy, legacy_path)
        try:
            load_translator(
                legacy_path,
                specs,
                device="cpu",
                d_model=8,
                num_blocks=1,
                num_heads=2,
                dropout=0.0,
            )
        except ValueError as exc:
            assert "format=9, expected 10" in str(exc)
        else:
            raise AssertionError("legacy asymmetric-pool checkpoint was accepted")
    print("stamped checkpoint round trip / legacy rejection OK")


if __name__ == "__main__":
    main()
