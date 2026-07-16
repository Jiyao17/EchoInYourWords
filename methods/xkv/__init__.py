"""XKV: dual-cache attention translation into Receiver KV updates."""

from functools import partial

from methods.common.io import write_run_outputs as _write_run_outputs
from methods.common.prompts import build_receiver_prompt, build_sharer_encode_text

from .model import ModelSpec, XKVPair, check_pair_compatible, load_xkv_pair
from .runner import XKVResult, run_xkv_example
from .schema import XKVExample
from .training import (
    batched_example_loss,
    evaluate,
    evaluate_em_f1,
    example_loss,
    load_translator,
    save_translator,
    train_translator,
)
from .translator import AttentionPool, XKVTranslator

write_run_outputs = partial(_write_run_outputs, method="xkv")

__all__ = [
    "ModelSpec",
    "AttentionPool",
    "XKVExample",
    "XKVPair",
    "XKVResult",
    "XKVTranslator",
    "batched_example_loss",
    "build_receiver_prompt",
    "build_sharer_encode_text",
    "check_pair_compatible",
    "evaluate",
    "evaluate_em_f1",
    "example_loss",
    "load_translator",
    "load_xkv_pair",
    "run_xkv_example",
    "save_translator",
    "train_translator",
    "write_run_outputs",
]
