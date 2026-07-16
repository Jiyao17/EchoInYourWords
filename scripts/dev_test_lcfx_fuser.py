"""Dev sanity tests for the paper-faithful LCF-X fuser (no models needed).

Run: ./python/bin/python scripts/dev_test_lcfx_fuser.py
Checks: span-start computation, paper parameter counts (57,344 pool; 693,010
per LCF-128 projector layer), align shapes (per-layer [B,H,1,D] summaries),
zero-init => zero residual under the hard eval gate, hard/soft gate modes,
the L_s != L_r layer_map bridge, and gradient flow into every trainable part.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from methods.lcfx.fuser import LCFXFuser, _LayerProjector, _gumbel_sigmoid, _span_starts


def main() -> None:
    # --- span starts helper ---
    assert _span_starts(10, 5, None) == [0, 2, 4, 6, 8]
    assert _span_starts(3, 5, None) == [0, 1, 2]  # fewer tokens than spans
    assert _span_starts(10, 5, [0, 4, 7]) == [0, 4, 7]  # explicit boundaries
    assert _span_starts(10, 5, [4, 7]) == [0, 4, 7]  # 0 anchored
    assert _span_starts(5, 5, [0, 4, 9, 12]) == [0, 4]  # clamped to n-1, deduped
    print("span starts OK")

    # --- symmetric pair (paper's LCF-X: Qwen3-0.6B pair: L=28, H=8, D=128) ---
    f = LCFXFuser(
        sharer_layers=28, sharer_kv_heads=8, sharer_head_dim=128,
        receiver_layers=28, receiver_kv_heads=8, receiver_head_dim=128,
        bottleneck_dim=128, zero_init=True,
    )
    pool = f.q_base.numel() + f.q_layer.numel()
    proj_per_layer = sum(p.numel() for p in f.projectors[0].parameters())
    print(f"pool params: {pool}  (paper E.3: 57,344)")
    print(f"projector total: {proj_per_layer * 28 / 1e6:.2f}M  (paper E.3: ~25.87M)")
    assert pool == 57344
    assert f.layer_map is None
    assert f.layer_gate.shape == (28, 2)

    # LCF-128 config (asymmetric shared-context pair: sharer H=2,D=64; joint 2304)
    lp = _LayerProjector(2, 64, 8, 128, 128, zero_init=False)
    n = sum(p.numel() for p in lp.parameters())
    print(f"LCF-128 projector/layer: {n:,}  (paper Table 5: 693K; total {n * 28 / 1e6:.1f}M vs 19.4M)")
    assert n == 693010

    # --- align shapes: batch 2, per-example lengths + boundaries ---
    torch.manual_seed(0)
    summary = [(torch.randn(2, 8, 40, 128), torch.randn(2, 8, 40, 128)) for _ in range(28)]
    lengths = torch.tensor([40, 25])
    bounds = [[0, 10, 20, 30], None]  # ex0: paragraph starts; ex1: fixed chunks
    aligned = f.align(summary, lengths=lengths, num_spans=5, span_boundaries=bounds)
    assert len(aligned) == 28
    assert aligned[0][0].shape == (2, 8, 1, 128), aligned[0][0].shape
    print("align (symmetric) OK:", tuple(aligned[0][0].shape))

    # --- zero-init => zero residual (with hard eval gate) ---
    f.eval()
    recv = [(torch.randn(2, 8, 33, 128), torch.randn(2, 8, 33, 128)) for _ in range(28)]
    res = f.project(aligned, recv)
    assert all(dk.abs().max() == 0 and dv.abs().max() == 0 for dk, dv in res)
    assert res[0][0].shape == (2, 8, 33, 128)
    print("zero-init residual == 0 OK (hard gate eval)")

    # --- gates: eval hard {0,1}; training soft ---
    logit = torch.tensor([1.0, -0.5])
    assert _gumbel_sigmoid(logit, training=False).tolist() == [1.0, 0.0]
    soft = _gumbel_sigmoid(logit, training=True, tau=0.7)
    assert ((soft > 0) & (soft < 1)).all()
    print("gate: hard at eval, Gumbel-sigmoid in training OK")

    # --- asymmetric pair: L_s=24 != L_r=28, different H/D -> layer_map bridges ---
    g = LCFXFuser(
        sharer_layers=24, sharer_kv_heads=2, sharer_head_dim=64,
        receiver_layers=28, receiver_kv_heads=8, receiver_head_dim=128,
        bottleneck_dim=128, zero_init=True,
    )
    assert g.layer_map is not None
    summary2 = [(torch.randn(1, 2, 30, 64), torch.randn(1, 2, 30, 64)) for _ in range(24)]
    aligned2 = g.align(summary2, num_spans=5)
    assert len(aligned2) == 28 and aligned2[0][0].shape == (1, 2, 1, 64)
    recv2 = [(torch.randn(1, 8, 12, 128), torch.randn(1, 8, 12, 128)) for _ in range(28)]
    g.eval()
    res2 = g.project(aligned2, recv2)
    assert res2[0][0].shape == (1, 8, 12, 128)
    assert all(dk.abs().max() == 0 for dk, _ in res2)
    print("asymmetric L_s!=L_r via layer_map OK; receiver-shaped zero residual OK")

    # --- gradients reach every trainable part (non-zero init, training mode) ---
    h = LCFXFuser(
        sharer_layers=24, sharer_kv_heads=2, sharer_head_dim=64,
        receiver_layers=28, receiver_kv_heads=8, receiver_head_dim=128,
        bottleneck_dim=128, zero_init=False,
    )
    h.train()
    res4 = h.project(h.align(summary2, num_spans=5), recv2)
    loss = sum(dk.float().square().sum() + dv.float().square().sum() for dk, dv in res4)
    loss.backward()
    assert h.q_base.grad is not None and h.q_base.grad.abs().sum() > 0
    assert h.q_layer.grad is not None and h.q_layer.grad.abs().sum() > 0
    assert h.layer_map.weight.grad is not None and h.layer_map.weight.grad.abs().sum() > 0
    assert h.layer_gate.grad is not None
    print("gradients reach q_base / q_layer / layer_map / layer_gate OK")
    print("ALL FUSER TESTS PASSED")


if __name__ == "__main__":
    main()
