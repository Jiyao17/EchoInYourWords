"""Small CPU checks for compact KVTok-v2 shapes, masking, and gradients."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.kvtok.training import load_translator, save_translator  # noqa: E402
from methods.kvtok.translator import KVTokTranslator  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    translator = KVTokTranslator(
        sharer_layers=3,
        sharer_kv_heads=2,
        sharer_head_dim=4,
        receiver_hidden=12,
        d_model=16,
        num_heads=4,
        num_spans=2,
        num_tokens=3,
    )

    summary = [
        (torch.randn(2, 2, 7, 4), torch.randn(2, 2, 7, 4)) for _ in range(3)
    ]
    receiver = torch.randn(2, 6, 12)
    lengths = torch.tensor([6, 4])
    output = translator(
        summary,
        receiver,
        sharer_lengths=torch.tensor([7, 5]),
        receiver_lengths=lengths,
    )
    assert output.shape == (2, 3, 12)
    assert torch.isfinite(output).all()

    # Masked Receiver padding cannot affect translated tokens.
    dirty = receiver.clone()
    dirty[1, 4:] = 1e4
    masked = translator(
        summary,
        dirty,
        sharer_lengths=torch.tensor([7, 5]),
        receiver_lengths=lengths,
    )
    assert torch.allclose(output[1], masked[1], atol=1e-6)

    output.square().mean().backward()
    for name, param in translator.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"

    # Regression guard: the old Cartesian dense projection was the source of
    # billions of parameters. This small fixture should remain comfortably tiny.
    assert translator.trainable_parameters() < 10_000
    assert not hasattr(translator, "receiver_pool")
    assert not hasattr(translator, "down")

    # A loaded checkpoint must agree with the requested Make/CLI width and
    # divisor-safe head count; otherwise eval metadata could mislabel a run.
    pair = SimpleNamespace(
        sharer_spec=SimpleNamespace(
            name="fixture/sharer", layers=3, kv_heads=2, head_dim=4
        ),
        receiver_spec=SimpleNamespace(name="fixture/receiver", hidden=12),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = str(Path(tmpdir) / "translator.pt")
        save_translator(translator, checkpoint, pair)
        loaded = load_translator(
            checkpoint,
            pair,
            num_spans=2,
            num_tokens=3,
            d_model=16,
            num_heads=4,
        )
        assert loaded.arch_config() == translator.arch_config()
        try:
            load_translator(
                checkpoint,
                pair,
                num_spans=2,
                num_tokens=3,
                d_model=32,
                num_heads=4,
            )
        except ValueError as exc:
            assert "arch.d_model" in str(exc)
        else:
            raise AssertionError("d_model mismatch was not rejected")

    print(
        "compact KVTok-v2 OK:",
        translator.trainable_parameters(),
        "params; output",
        tuple(output.shape),
    )


if __name__ == "__main__":
    main()
