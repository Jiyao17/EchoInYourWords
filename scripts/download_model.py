"""Download a model into the project HF cache (``data/hf_cache``).

The model analogue of ``scripts/prepare_*`` for datasets: ``make model MODEL=...``
fetches the weights once so later runs (which default to ``HF_HUB_OFFLINE=1``)
find them locally. Idempotent — the Hub cache skips files already present.

Gated repos (e.g. ``meta-llama/*``, ``google/*``) need an accepted license + an
exported ``HF_TOKEN`` (read automatically here), but none of the project's
known-good ids are gated — ``unsloth/*`` mirrors cover Llama/Gemma without a
token. Only the files transformers actually loads are fetched — duplicate
``consolidated.safetensors``, raw ``original/`` checkpoints, ``*.pth`` and
``*.gguf`` are skipped.

    python scripts/download_model.py --model unsloth/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._hf_cache import configure_hf_cache  # noqa: E402

configure_hf_cache()


RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
MODELSCOPE_MIRRORS = {
    "mistralai/Ministral-8B-Instruct-2410": (
        "AI-ModelScope/Ministral-8B-Instruct-2410"
    ),
}


def _exception_chain(error: BaseException):
    """Yield an exception and its wrapped causes without looping forever."""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_retryable_download_error(error: BaseException) -> bool:
    """Return true for transient Hub HTTP failures and network interruptions."""

    from requests.exceptions import ConnectionError, RequestException, Timeout

    for item in _exception_chain(error):
        if isinstance(item, (ConnectionError, Timeout)):
            return True
        if isinstance(item, RequestException):
            response = getattr(item, "response", None)
            status = getattr(response, "status_code", None)
            if status is None or status in RETRYABLE_HTTP_STATUS:
                return True
    return False


def download_with_retries(
    model: str,
    *,
    max_attempts: int,
    retry_delay: float,
    snapshot_download_fn: Callable[..., str],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    """Download a snapshot, retrying only failures likely to be transient."""

    for attempt in range(1, max_attempts + 1):
        try:
            return snapshot_download_fn(
                repo_id=model,
                # Ministral publishes both one consolidated 16 GB checkpoint
                # and four Transformers shards containing the same weights.
                # AutoModel uses the shards/index, so never fetch the duplicate.
                ignore_patterns=[
                    "consolidated.safetensors",
                    "original/*",
                    "*.pth",
                    "*.gguf",
                ],
            )
        except Exception as error:
            if attempt == max_attempts or not _is_retryable_download_error(error):
                raise
            delay = min(retry_delay * (2 ** (attempt - 1)), 30.0)
            print(
                f"Download attempt {attempt}/{max_attempts} for {model} failed "
                f"with a transient {type(error).__name__}; retrying in "
                f"{delay:g}s...",
                file=sys.stderr,
                flush=True,
            )
            sleep_fn(delay)

    raise AssertionError("unreachable")


def _validate_transformers_snapshot(snapshot_dir: Path) -> None:
    """Ensure a mirrored snapshot contains all files needed by Transformers."""

    required = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    missing = sorted(name for name in required if not (snapshot_dir / name).is_file())
    if missing:
        raise RuntimeError(
            f"mirrored snapshot is missing required files: {', '.join(missing)}"
        )

    with (snapshot_dir / "model.safetensors.index.json").open(
        encoding="utf-8"
    ) as handle:
        index = json.load(handle)
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        raise RuntimeError("mirrored safetensor index contains no weight shards")
    missing_shards = [name for name in shards if not (snapshot_dir / name).is_file()]
    if missing_shards:
        raise RuntimeError(
            f"mirrored snapshot is missing weight shards: {', '.join(missing_shards)}"
        )


def download_from_modelscope(
    model: str,
    mirror: str,
    *,
    snapshot_download_fn: Callable[..., str] | None = None,
    hf_home: Path | None = None,
) -> str:
    """Install a ModelScope mirror in the native Hugging Face cache layout."""

    if snapshot_download_fn is None:
        try:
            from modelscope import snapshot_download as snapshot_download_fn
        except ImportError as error:
            raise RuntimeError(
                "Hugging Face is unavailable and the ModelScope fallback requires "
                "the `modelscope` package"
            ) from error

    hf_home = hf_home or Path(os.environ["HF_HOME"])
    repo_cache = hf_home / "hub" / f"models--{model.replace('/', '--')}"
    # A deterministic 40-hex snapshot id satisfies Transformers' normal cache
    # path parsing while making the non-HF source explicit in the hash input.
    snapshot_id = hashlib.sha1(
        f"modelscope:{mirror}:master".encode("utf-8")
    ).hexdigest()
    snapshot_dir = repo_cache / "snapshots" / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    resolved = Path(
        snapshot_download_fn(
            model_id=mirror,
            revision="master",
            local_dir=str(snapshot_dir),
            ignore_patterns=[
                "consolidated.safetensors",
                "original/*",
                "*.pth",
                "*.gguf",
            ],
        )
    )
    _validate_transformers_snapshot(resolved)

    refs_dir = repo_cache / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    ref_tmp = refs_dir / "main.tmp"
    ref_tmp.write_text(snapshot_id, encoding="utf-8")
    ref_tmp.replace(refs_dir / "main")
    return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="HF model id, e.g. unsloth/gemma-3-4b-it"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=int(os.environ.get("HF_MODEL_DOWNLOAD_ATTEMPTS", "3")),
        help="Maximum Hugging Face attempts before a configured mirror fallback (default: 3)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=float(os.environ.get("HF_MODEL_DOWNLOAD_RETRY_DELAY", "2")),
        help="Initial retry delay in seconds; exponential backoff caps at 30s",
    )
    args = parser.parse_args()

    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    if args.retry_delay < 0:
        parser.error("--retry-delay must be non-negative")

    from huggingface_hub import snapshot_download

    try:
        path = download_with_retries(
            args.model,
            max_attempts=args.max_attempts,
            retry_delay=args.retry_delay,
            snapshot_download_fn=snapshot_download,
        )
    except Exception as error:
        mirror = MODELSCOPE_MIRRORS.get(args.model)
        if mirror is None or not _is_retryable_download_error(error):
            raise
        print(
            f"Hugging Face remained unavailable for {args.model}; falling back "
            f"to ModelScope mirror {mirror}...",
            file=sys.stderr,
            flush=True,
        )
        path = download_from_modelscope(args.model, mirror)
    print(f"Downloaded {args.model} -> {path}")


if __name__ == "__main__":
    main()
