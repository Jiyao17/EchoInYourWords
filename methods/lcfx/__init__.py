"""LCF-X: cross-context, position-free latent cache communication."""

from .fuser import LCFXFuser
from .io import write_run_outputs
from .model import LCFXPair, ModelSpec, check_pair_compatible, load_lcfx_pair
from .prompts import build_receiver_prompt, build_sharer_encode_text
from .runner import LCFXResult, run_lcfx_example, run_receiver_only_example
from .schema import LCFXExample
from .training import evaluate, example_loss, load_fuser, save_fuser, train_fuser

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
    "run_receiver_only_example",
    "evaluate",
    "example_loss",
    "load_fuser",
    "save_fuser",
    "train_fuser",
    "write_run_outputs",
]
