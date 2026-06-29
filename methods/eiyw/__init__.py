"""EIYW: receiver-anchored communication via soft pseudo-tokens.

The Sharer's context is translated into K soft tokens — conditioned on the
Receiver's own question — that are prepended to the Receiver's input embeddings.
The Receiver attends to them natively; only the translator is trained.
"""

from .io import write_run_outputs
from .model import EIYWPair, ModelSpec, check_pair_compatible, load_eiyw_pair
from .prompts import build_receiver_prompt, build_sharer_encode_text
from .runner import EIYWResult, run_eiyw_example, run_receiver_only_example
from .schema import EIYWExample
from .training import (
    evaluate,
    example_loss,
    load_translator,
    save_translator,
    train_translator,
)
from .translator import EIYWTranslator

__all__ = [
    "EIYWExample",
    "EIYWTranslator",
    "EIYWPair",
    "EIYWResult",
    "ModelSpec",
    "build_receiver_prompt",
    "build_sharer_encode_text",
    "check_pair_compatible",
    "load_eiyw_pair",
    "run_eiyw_example",
    "run_receiver_only_example",
    "evaluate",
    "example_loss",
    "load_translator",
    "save_translator",
    "train_translator",
    "write_run_outputs",
]
