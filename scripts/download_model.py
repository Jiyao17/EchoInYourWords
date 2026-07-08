"""Download a model into the project HF cache (``data/hf_cache``).

The model analogue of ``scripts/prepare_*`` for datasets: ``make model MODEL=...``
fetches the weights once so later runs (which default to ``HF_HUB_OFFLINE=1``)
find them locally. Idempotent — the Hub cache skips files already present.

Gated repos (e.g. ``meta-llama/*``, ``google/*``) need an accepted license + an
exported ``HF_TOKEN`` (read automatically here), but none of the project's
known-good ids are gated — ``unsloth/*`` mirrors cover Llama/Gemma without a
token. Only the files transformers actually loads are fetched — raw
``original/`` checkpoints, ``*.pth`` and ``*.gguf`` are skipped.

    python scripts/download_model.py --model unsloth/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._hf_cache import configure_hf_cache  # noqa: E402

configure_hf_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="HF model id, e.g. unsloth/gemma-3-4b-it"
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=args.model,
        ignore_patterns=["original/*", "*.pth", "*.gguf"],
    )
    print(f"Downloaded {args.model} -> {path}")


if __name__ == "__main__":
    main()
