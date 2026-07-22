"""Dev sanity tests for the paper-faithful LCF-X fuser (no models needed).

Run: ./python/bin/python scripts/dev_test_lcfx_fuser.py
Checks: span-start computation, the paper-stated d/2 latent split, paper
initialization/dropout, align shapes, optional zero-init plumbing mode,
hard/soft gate modes, the L_s != L_r layer_map bridge, exact 10% warmup, and
gradient flow into every trainable part.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from methods.lcfx.fuser import LCFXFuser, _LayerProjector, _gumbel_sigmoid, _span_starts
from methods.lcfx.spans import paragraph_span_starts
from methods.lcfx.training import _optimization_steps, _warmup_steps


def main() -> None:
    # --- span starts helper ---
    assert _span_starts(10, 5, None) == [0, 2, 4, 6, 8]
    assert _span_starts(3, 5, None) == [0, 1, 2]  # fewer tokens than spans
    assert _span_starts(10, 5, [0, 4, 7]) == [0, 4, 7]  # explicit boundaries
    assert _span_starts(10, 5, [4, 7]) == [0, 4, 7]  # 0 anchored
    assert _span_starts(5, 5, [0, 4, 9, 12]) == [0, 4]  # clamped to n-1, deduped

    class _CharTokenizer:
        def __call__(self, text: str, *, return_offsets_mapping: bool) -> dict:
            assert return_offsets_mapping
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    evidence = "p1\n\np2\n\np3\n\np4\n\np5"
    prompt = f"Information:\n{evidence}\n\nQuestion:\nq"
    natural = paragraph_span_starts(_CharTokenizer(), prompt, evidence)
    assert len(natural) == 5, natural  # prompt/question are not extra spans
    assert paragraph_span_starts(_CharTokenizer(), prompt, "") == [0]
    print("span starts OK")

    # --- symmetric pair (paper's LCF-X: Qwen3-0.6B pair: L=28, H=8, D=128) ---
    f = LCFXFuser(
        sharer_layers=28, sharer_kv_heads=8, sharer_head_dim=128,
        receiver_layers=28, receiver_kv_heads=8, receiver_head_dim=128,
        bottleneck_dim=128,
    )
    pool = f.q_base.numel() + f.q_layer.numel()
    proj_per_layer = sum(p.numel() for p in f.projectors[0].parameters())
    print(f"pool params: {pool}  (paper E.3: 57,344)")
    print(f"projector total: {proj_per_layer * 28 / 1e6:.2f}M  (paper-stated d/2 split)")
    assert pool == 57344
    assert f.layer_map is None
    assert f.layer_gate.shape == (28, 2)
    assert f.projectors[0].up_key.in_features == 64
    assert f.projectors[0].up_value.in_features == 64
    assert isinstance(f.projectors[0].mlp[2], torch.nn.Dropout)
    assert f.projectors[0].mlp[2].p == 0.1
    assert f.projectors[0].up_key.weight.abs().sum() > 0

    # The paper prose specifies d/2 inputs. Its 693K Table-5 count instead
    # implies full-d up-projections, so strict prose necessarily gives 561,936.
    lp = _LayerProjector(2, 64, 8, 128, 128, zero_init=False)
    n = sum(p.numel() for p in lp.parameters())
    print(f"LCF-128 projector/layer: {n:,}  (paper-stated d/2 split)")
    assert n == 561936

    try:
        _LayerProjector(2, 64, 8, 128, 127)
    except ValueError:
        pass
    else:
        raise AssertionError("odd bottleneck width must be rejected")

    # --- align shapes: batch 2, per-example lengths + boundaries ---
    torch.manual_seed(0)
    summary = [(torch.randn(2, 8, 40, 128), torch.randn(2, 8, 40, 128)) for _ in range(28)]
    lengths = torch.tensor([40, 25])
    bounds = [[0, 10, 20, 30], None]  # ex0: paragraph starts; ex1: fixed chunks
    aligned = f.align(summary, lengths=lengths, num_spans=5, span_boundaries=bounds)
    assert len(aligned) == 28
    assert aligned[0][0].shape == (2, 8, 1, 128), aligned[0][0].shape
    print("align (symmetric) OK:", tuple(aligned[0][0].shape))

    # --- paper init => active residual (with hard eval gate) ---
    f.eval()
    recv = [(torch.randn(2, 8, 33, 128), torch.randn(2, 8, 33, 128)) for _ in range(28)]
    res = f.project(aligned, recv)
    assert any(dk.abs().max() > 0 and dv.abs().max() > 0 for dk, dv in res)
    assert res[0][0].shape == (2, 8, 33, 128)
    print("paper Kaiming x0.06 residual active OK (hard gate eval)")

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

    # ROPES-like 10,671 examples at batch4/accum64 has 42 optimizer steps,
    # including the final partial accumulation; automatic warmup is 4 steps.
    total = _optimization_steps(10671, 4, 64, 1, None)
    assert total == 42
    assert _warmup_steps(total, None) == 4
    assert _warmup_steps(total, 7) == 7
    assert _optimization_steps(10671, 4, 64, 1, 5) == 5
    print("optimizer-step and 10% warmup resolution OK")

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
