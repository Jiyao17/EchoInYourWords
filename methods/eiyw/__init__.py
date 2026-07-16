"""EIYW: receiver-anchored communication via soft pseudo-tokens.

The Sharer's context is pooled per layer with the paper-exact LCF-X pool (up
to ``k`` position-free ``[B,H,1,D]`` slots per layer, adaptive in the context
length when ``adaptive_k_tokens`` is set), projected to ``d_model`` (a fixed,
receiver-independent width, default 512) and tagged with layer/query
embeddings. The
Receiver's own context enters as raw input *embeddings* (a table lookup — no
Receiver forward pass) plus sinusoidal PE; ``num_blocks`` decoder-shaped
blocks (self-attn over the slots, cross-attn into the Receiver memory, MLP)
refine the slots into ``k·L_s`` soft tokens, and an output head maps them
into the Receiver's embedding space (see ``translator.py``), prepended to the
Receiver's input embeddings. The Receiver attends to them natively; only the
translator is trained.
"""

from functools import partial

from methods.common.io import write_run_outputs as _write_run_outputs
from methods.common.prompts import build_receiver_prompt, build_sharer_encode_text

from .model import EIYWPair, ModelSpec, check_pair_compatible, load_eiyw_pair
from .runner import EIYWResult, run_eiyw_example
from .schema import EIYWExample
from .spans import paragraph_span_starts
from .training import (
    evaluate,
    example_loss,
    load_translator,
    save_translator,
    train_translator,
)
from .translator import EIYWTranslator

# Standard writer with this method's default name (a "method" in metadata still
# overrides it).
write_run_outputs = partial(_write_run_outputs, method="eiyw")

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
    "evaluate",
    "example_loss",
    "load_translator",
    "paragraph_span_starts",
    "save_translator",
    "train_translator",
    "write_run_outputs",
]
