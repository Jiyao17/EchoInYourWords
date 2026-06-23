"""Point the Hugging Face cache at the project's ``data/hf_cache`` directory.

This follows HuggingFace's native single-root layout: setting ``HF_HOME`` makes
HF derive every sub-cache from it —

    data/hf_cache/
      hub/        all Hub downloads, models and datasets together, told apart by
                  their ``models--`` / ``datasets--`` name prefix
      datasets/   built (arrow) datasets
      xet/        dedup blob cache
      modules/    dataset loading-script modules

Call :func:`configure_hf_cache` at the top of a script, *before* ``datasets`` /
``transformers`` are imported. An existing ``HF_HOME`` is respected.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = REPO_ROOT / "data" / "hf_cache"


def configure_hf_cache() -> None:
    """Route all HF caches under ``data/hf_cache`` unless ``HF_HOME`` is set."""

    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
