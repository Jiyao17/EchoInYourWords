"""Dev sanity tests for the cross-attention (v2) EIYW translator (no models needed).

Run: ./python/bin/python scripts/dev_test_eiyw_translator.py
Checks: Sharer LCF-X pool slot shapes (incl. the k_active prefix); translate
output shape (k*L_s tokens) / finiteness for a heterogeneous pair (mixed
paragraph/fixed boundaries, padded lengths); d_model='half' resolution; the
sinusoidal PE on the receiver memory (permuting the context embeddings must
change the output — without PE cross-attention is permutation-invariant);
the receiver_lengths key-padding mask (pad-region embeddings must NOT affect
the output); the adaptive token budget (active k scales with the batch's
longest Sharer length, capped at queries_per_layer); the gentle near-zero
(but nonzero) init; the embedding-RMS calibration of the output head
(embed_scale buffer x learnable out_gain, incl. state_dict round-trip); and
gradient flow into the pool queries, projections, embeddings, both attention
paths of every block, and out_gain.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from methods.eiyw.translator import EIYWTranslator, _LCFXPool, _resolve_d_model


class _Spec:
    def __init__(self, layers, kv_heads, head_dim, hidden, embed_rms=1.0):
        self.layers, self.kv_heads, self.head_dim, self.hidden = layers, kv_heads, head_dim, hidden
        self.embed_rms = embed_rms


def main() -> None:
    torch.manual_seed(0)

    # Shared fixtures: Sharer = 4 layers, H=2, D=8; Receiver hidden = 32.
    summary = [(torch.randn(2, 2, 30, 8), torch.randn(2, 2, 30, 8)) for _ in range(4)]
    recv_embeds = torch.randn(2, 21, 32)

    # --- Sharer pool: per-layer slots [B, k*L, 2*H*D] ---
    pool = _LCFXPool(layers=4, kv_heads=2, head_dim=8)
    slots = pool(
        summary,
        lengths=torch.tensor([30, 17]),
        num_spans=5,
        span_boundaries=[[0, 6, 14, 22], None],  # ex0 paragraphs, ex1 fixed chunks
    )
    assert slots.shape == (2, 4, 2 * 2 * 8), slots.shape
    assert pool.q_base.shape == (4, 2, 8) and pool.q_layer.shape == (4, 1, 2, 8)
    print("sharer pool slots OK:", tuple(slots.shape))

    # --- k>1 pool: k distinct cross-span queries per layer -> k*L slots (the
    # k slots of a layer are NOT copies), and k_active pools a prefix ---
    pool_k = _LCFXPool(layers=4, kv_heads=2, head_dim=8, queries_per_layer=3)
    assert pool_k.q_layer.shape == (4, 3, 2, 8)
    slots_k = pool_k(summary, num_spans=5)
    assert slots_k.shape == (2, 12, 2 * 2 * 8), slots_k.shape  # layer-major, query-minor
    assert not torch.allclose(slots_k[:, 0], slots_k[:, 1])  # layer 0, queries 0 vs 1
    slots_k2 = pool_k(summary, num_spans=5, k_active=2)
    assert slots_k2.shape == (2, 8, 2 * 2 * 8), slots_k2.shape
    # Prefix semantics: layer l's first two slots match the full-k pool's.
    assert torch.allclose(slots_k2[:, 0:2], slots_k[:, 0:2])
    assert torch.allclose(slots_k2[:, 2:4], slots_k[:, 3:5])
    print("k>1 pool + k_active prefix OK:", tuple(slots_k.shape), "->", tuple(slots_k2.shape))

    # --- d_model: fixed default 512 (receiver-independent); 'half'/'hidden'
    # remain as sweep options ---
    assert _resolve_d_model("", 4096) == 512  # default: NOT receiver-scaled
    assert _resolve_d_model(512, 2048) == 512
    assert _resolve_d_model("half", 1024) == 512
    assert _resolve_d_model("hidden", 2560) == 2560
    assert _resolve_d_model(768, 4096) == 768
    print("d_model resolution OK (default 512, receiver-independent)")

    # --- heterogeneous pair: output is [B, k*L_s, hidden_r] = [B, 4, 32] ---
    tr = EIYWTranslator.from_specs(
        _Spec(4, 2, 8, 32), _Spec(6, 4, 16, 32),
        d_model=64, queries_per_layer=1, num_blocks=2, num_heads=4,
        zero_init=True,
    )
    soft = tr.translate(
        summary,
        recv_embeds,
        sharer_lengths=torch.tensor([30, 17]),
        receiver_lengths=torch.tensor([21, 12]),
        num_spans=5,
        sharer_span_boundaries=[[0, 6, 14, 22], None],
    )
    assert soft.shape == (2, 4, 32), soft.shape
    assert tr.num_tokens == 4, tr.num_tokens
    assert torch.isfinite(soft).all()
    print("translate (heterogeneous, k=1) OK:", tuple(soft.shape))

    # --- sinusoidal PE: cross-attention alone is permutation-invariant over
    # the memory, so reversing the receiver context must change the output ---
    flipped = tr.translate(summary, recv_embeds.flip(dims=[1]), num_spans=5)
    base = tr.translate(summary, recv_embeds, num_spans=5)
    assert (base - flipped).abs().max() > 1e-6, "receiver order had no effect (PE missing?)"
    print("sinusoidal PE gives the memory order OK")

    # --- receiver_lengths mask: garbage in the pad region must not leak ---
    recv_dirty = recv_embeds.clone()
    recv_dirty[1, 12:] = 999.0  # row 1's real length is 12
    lengths = torch.tensor([21, 12])
    clean = tr.translate(summary, recv_embeds, receiver_lengths=lengths, num_spans=5)
    dirty = tr.translate(summary, recv_dirty, receiver_lengths=lengths, num_spans=5)
    assert torch.allclose(clean, dirty, atol=1e-6), "padded positions leaked into the output"
    print("receiver key-padding mask OK")

    # --- gentle init: near zero but NOT exactly zero (exact zero is a
    # numerical degeneracy for embedding-space tokens) ---
    assert soft.abs().max() > 0
    assert soft.abs().max() < 0.5, float(soft.abs().max())
    print(f"gentle init OK: max |soft| = {float(soft.abs().max()):.4f}")

    # --- adaptive token budget: active k = clamp(ceil(N_s/w), 1, k_max),
    # resolved per batch from the longest real Sharer length ---
    tr_ad = EIYWTranslator.from_specs(
        _Spec(4, 2, 8, 32), _Spec(6, 4, 16, 32),
        d_model=64, queries_per_layer=3, adaptive_k_tokens=8, num_blocks=1, num_heads=4,
    )
    assert tr_ad.active_queries(5) == 1 and tr_ad.active_queries(12) == 2
    assert tr_ad.active_queries(17) == 3 and tr_ad.active_queries(4000) == 3  # capped
    short = tr_ad.translate(summary, recv_embeds, sharer_lengths=torch.tensor([5, 3]), num_spans=5)
    assert short.shape == (2, 1 * 4, 32), short.shape
    mixed = tr_ad.translate(summary, recv_embeds, sharer_lengths=torch.tensor([5, 17]), num_spans=5)
    assert mixed.shape == (2, 3 * 4, 32), mixed.shape  # batch k from the longest row
    full = tr_ad.translate(summary, recv_embeds, num_spans=5)  # no lengths -> full N=30
    assert full.shape == (2, 3 * 4, 32), full.shape
    assert tr_ad.num_tokens == 12  # max budget
    print("adaptive k OK: tokens 4 (short) / 12 (mixed batch, capped)")

    # --- embedding-RMS calibration: output scales with the receiver's
    # embed_rms (buffer embed_scale x learnable out_gain, init 0.05) ---
    torch.manual_seed(1)
    tr_quiet = EIYWTranslator.from_specs(
        _Spec(4, 2, 8, 32), _Spec(6, 4, 16, 32, embed_rms=0.01),
        d_model=64, queries_per_layer=1, num_blocks=2, num_heads=4, zero_init=True,
    )
    torch.manual_seed(1)
    tr_loud = EIYWTranslator.from_specs(
        _Spec(4, 2, 8, 32), _Spec(6, 4, 16, 32, embed_rms=1.0),
        d_model=64, queries_per_layer=1, num_blocks=2, num_heads=4, zero_init=True,
    )
    assert float(tr_quiet.embed_scale) == 0.01 and float(tr_loud.embed_scale) == 1.0
    assert float(tr_quiet.out_gain) == 0.05
    s_quiet = tr_quiet.translate(summary, recv_embeds, num_spans=5)
    s_loud = tr_loud.translate(summary, recv_embeds, num_spans=5)
    ratio = float(s_loud.float().pow(2).mean().sqrt() / s_quiet.float().pow(2).mean().sqrt())
    assert abs(ratio - 100.0) < 1e-2, ratio  # same weights (same seed), 100x scale
    print(f"embed-RMS calibration OK: loud/quiet output RMS ratio = {ratio:.2f} (expect 100)")

    # --- embed_scale buffer survives a state_dict round-trip into a
    # default-built (embed_rms=1.0) translator, so load_translator needs no pair ---
    tr_default = EIYWTranslator.from_arch_config(tr_quiet.arch_config())
    tr_default.load_state_dict(tr_quiet.state_dict())
    assert float(tr_default.embed_scale) == 0.01
    print("embed_scale buffer round-trip OK")

    # --- gradient flow into every trainable stage (k=1 translator) ---
    loss = soft.float().square().sum()
    loss.backward()
    for name in [
        "sharer_pool.q_base", "sharer_pool.q_layer",
        "sharer_proj.weight", "layer_emb", "receiver_proj.weight",
        "blocks.0.self_attn.in_proj_weight", "blocks.0.cross_attn.in_proj_weight",
        "blocks.0.mlp.0.weight",
        "blocks.1.self_attn.in_proj_weight", "blocks.1.cross_attn.in_proj_weight",
        "out.weight", "out_gain",
    ]:
        param = dict(tr.named_parameters())[name]
        assert param.grad is not None and param.grad.abs().sum() > 0, name
    print("gradients reach pool / projections / embeddings / self+cross attn / out OK")

    # --- k>1: query_emb exists, gets gradient, and tokens = k*L_s ---
    tr_k = EIYWTranslator.from_specs(
        _Spec(4, 2, 8, 32), _Spec(6, 4, 16, 32),
        d_model=64, queries_per_layer=3, num_blocks=1, num_heads=4,
    )
    assert tr_k.num_tokens == 12 and tr_k.sharer_pool.q_layer.shape == (4, 3, 2, 8)
    soft_k = tr_k.translate(summary, recv_embeds, num_spans=5)
    assert soft_k.shape == (2, 12, 32), soft_k.shape
    assert tr_k.query_emb is not None and tr_k.query_emb.shape == (3, 64)
    soft_k.float().square().sum().backward()
    assert tr_k.query_emb.grad is not None and tr_k.query_emb.grad.abs().sum() > 0
    print("k>1 in-pool queries + query_emb OK:", tuple(soft_k.shape))

    # --- fixed budget: token count independent of BOTH context lengths ---
    short_summary = [(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8)) for _ in range(4)]
    short_recv = torch.randn(1, 2, 32)
    soft2 = tr.translate(short_summary, short_recv, num_spans=5)
    assert soft2.shape == (1, 4, 32)
    print("fixed token count for short contexts OK")

    # --- arch round-trip (incl. adaptive_k_tokens) ---
    cfg = tr_ad.arch_config()
    assert cfg["adaptive_k_tokens"] == 8
    tr2 = EIYWTranslator.from_arch_config(cfg)
    tr2.load_state_dict(tr_ad.state_dict())
    assert tr2.active_queries(12) == 2
    print("arch_config round-trip OK")
    print("ALL EIYW TRANSLATOR TESTS PASSED")


if __name__ == "__main__":
    main()
