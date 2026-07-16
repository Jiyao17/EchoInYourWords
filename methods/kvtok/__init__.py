"""KVTok: compact Sharer Stage-1 pooling with Receiver embedding conditioning.

The Sharer's cache is pooled with LCF-X's Fig. 3 **Stage 1 only** (within-span
attention). A factorized feature/token-axis linear down-projection produces a
fixed number of slots. The Receiver contributes processed context embeddings
only (no initial Receiver forward/cache); cross-attention and an EIYW-v2-style
head emit soft tokens prepended to the Receiver's input embeddings.
"""

from functools import partial

from methods.common.io import write_run_outputs as _write_run_outputs
from methods.common.prompts import build_receiver_prompt, build_sharer_encode_text

from .model import KVTokPair, ModelSpec, check_pair_compatible, load_kvtok_pair
from .runner import KVTokResult, run_kvtok_example
from .schema import KVTokExample
from .training import (
    evaluate,
    example_loss,
    load_translator,
    save_translator,
    train_translator,
)
from .translator import KVTokTranslator

# Standard writer with this method's default name (a "method" in metadata still
# overrides it).
write_run_outputs = partial(_write_run_outputs, method="kvtok")

__all__ = [
    "KVTokExample",
    "KVTokTranslator",
    "KVTokPair",
    "KVTokResult",
    "ModelSpec",
    "build_receiver_prompt",
    "build_sharer_encode_text",
    "check_pair_compatible",
    "load_kvtok_pair",
    "run_kvtok_example",
    "evaluate",
    "example_loss",
    "load_translator",
    "save_translator",
    "train_translator",
    "write_run_outputs",
]
