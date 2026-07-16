"""LCF-X: cross-context, position-free latent cache communication."""

from functools import partial

from methods.common.io import write_run_outputs as _write_run_outputs
from methods.common.prompts import build_receiver_prompt, build_sharer_encode_text

from .fuser import LCFXFuser
from .model import LCFXPair, ModelSpec, check_pair_compatible, load_lcfx_pair
from .runner import LCFXResult, run_lcfx_example
from .schema import LCFXExample
from .spans import paragraph_span_starts
from .training import evaluate, example_loss, load_fuser, save_fuser, train_fuser

# Standard writer with this method's default name (a "method" in metadata still
# overrides it).
write_run_outputs = partial(_write_run_outputs, method="lcf_x")

__all__ = [
    "LCFXExample",
    "LCFXFuser",
    "LCFXPair",
    "LCFXResult",
    "ModelSpec",
    "build_receiver_prompt",
    "build_sharer_encode_text",
    "check_pair_compatible",
    "load_lcfx_pair",
    "run_lcfx_example",
    "evaluate",
    "example_loss",
    "load_fuser",
    "paragraph_span_starts",
    "save_fuser",
    "train_fuser",
    "write_run_outputs",
]
