"""Frozen Hugging Face Sharer/Receiver backend for XKV."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from methods.common.decoding import resolve_stop_ids
from methods.lcfx.spans import (
    paragraph_span_starts,
    paragraph_span_starts_from_offsets,
)

from .cache import add_residual_inplace, extract_kv


MINISTRAL_8B = "mistralai/Ministral-8B-Instruct-2410"


def _offline_enabled() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _validate_ministral_offline_cache(name: str) -> None:
    """Fail clearly when an interrupted download left a partial snapshot."""

    if name != MINISTRAL_8B or not _offline_enabled():
        return

    from huggingface_hub import try_to_load_from_cache

    hf_home = os.environ.get("HF_HOME")
    cache_dir = str(Path(hf_home) / "hub") if hf_home else None

    def cached(filename: str) -> str | None:
        path = try_to_load_from_cache(name, filename, cache_dir=cache_dir)
        return path if isinstance(path, str) else None

    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    missing = [filename for filename in required if cached(filename) is None]
    index_path = cached("model.safetensors.index.json")
    if index_path is None:
        missing.append("model.safetensors.index.json")
    else:
        with Path(index_path).open(encoding="utf-8") as handle:
            index = json.load(handle)
        for shard in sorted(set(index.get("weight_map", {}).values())):
            if cached(shard) is None:
                missing.append(shard)

    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            f"incomplete offline Ministral cache; missing: {missing_text}. "
            "Run `HF_MODEL_DOWNLOAD_ATTEMPTS=1 make model "
            "MODEL=mistralai/Ministral-8B-Instruct-2410` and retry."
        )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: int
    kv_heads: int
    head_dim: int
    n_heads: int
    hidden: int
    is_gemma3: bool = False
    has_hybrid_cache: bool = False


def _has_hybrid_cache(config: Any) -> bool:
    """Whether layers can expose different full/sliding cache widths."""

    model_type = getattr(config, "model_type", "")
    if model_type in {"gemma3", "gemma3_text"}:
        return True

    layer_types = getattr(config, "layer_types", None)
    if layer_types:
        normalized = {str(layer_type).lower() for layer_type in layer_types}
        if any("full" in layer_type for layer_type in normalized) and any(
            "sliding" in layer_type for layer_type in normalized
        ):
            return True

    sliding_window = getattr(config, "sliding_window", None)
    return (
        isinstance(sliding_window, (list, tuple))
        and any(window is None for window in sliding_window)
        and any(window is not None for window in sliding_window)
    )


def _spec_from_model(name: str, model: Any) -> ModelSpec:
    config = model.config
    if not hasattr(config, "num_attention_heads") and hasattr(config, "text_config"):
        config = config.text_config
    n_heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or n_heads
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // n_heads)
    is_gemma3 = getattr(config, "model_type", "") in {"gemma3", "gemma3_text"}
    return ModelSpec(
        name=name,
        layers=config.num_hidden_layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        n_heads=n_heads,
        hidden=config.hidden_size,
        is_gemma3=is_gemma3,
        has_hybrid_cache=_has_hybrid_cache(config),
    )


def _resolve_dtype(dtype: str | None, torch_mod: Any) -> Any:
    if dtype is None:
        return None
    mapping = {
        "float16": torch_mod.float16,
        "fp16": torch_mod.float16,
        "half": torch_mod.float16,
        "bfloat16": torch_mod.bfloat16,
        "bf16": torch_mod.bfloat16,
        "float32": torch_mod.float32,
        "fp32": torch_mod.float32,
        "float": torch_mod.float32,
        "auto": "auto",
    }
    name = str(dtype).strip().lower()
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype!r}")
    return mapping[name]


class XKVPair:
    def __init__(
        self,
        sharer: Any,
        receiver: Any,
        sharer_tok: Any,
        receiver_tok: Any,
        device: Any,
        sharer_spec: ModelSpec,
        receiver_spec: ModelSpec,
    ) -> None:
        self.sharer = sharer
        self.receiver = receiver
        self.sharer_tok = sharer_tok
        self.receiver_tok = receiver_tok
        self.device = device
        self.sharer_spec = sharer_spec
        self.receiver_spec = receiver_spec

    def _encode_plain(self, tokenizer: Any, text: str) -> Any:
        return tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)

    def _encode_chat(self, tokenizer: Any, text: str) -> Any:
        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": text}]
            kwargs = dict(
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            try:
                encoded = tokenizer.apply_chat_template(
                    messages, enable_thinking=False, **kwargs
                )
            except TypeError:
                encoded = tokenizer.apply_chat_template(messages, **kwargs)
            return encoded["input_ids"].to(self.device)
        return self._encode_plain(tokenizer, text)

    def receiver_span_starts(
        self,
        text: str,
        context: str,
        *,
        encoded_ids: "Any | None" = None,
    ) -> list[int]:
        """Natural LCF-X paragraph starts in Receiver-prefix token coordinates."""

        tokenizer = self.receiver_tok
        if not getattr(tokenizer, "chat_template", None):
            return paragraph_span_starts(tokenizer, text, context)

        messages = [{"role": "user", "content": text}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            rendered = tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
        offset_encoding = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offset_ids = offset_encoding["input_ids"]
        expected = (
            encoded_ids
            if encoded_ids is not None
            else self._encode_chat(tokenizer, text)[0]
        )
        expected_ids = expected.detach().cpu().tolist()
        if offset_ids != expected_ids:
            raise ValueError(
                "Receiver chat rendering did not reproduce apply_chat_template "
                "token ids; cannot compute exact LCF-X paragraph boundaries"
            )
        return paragraph_span_starts_from_offsets(
            rendered,
            offset_encoding["offset_mapping"],
            context,
        )

    def sharer_summary(self, text: str) -> tuple[list[tuple[Any, Any]], int]:
        import torch

        ids = self._encode_plain(self.sharer_tok, text)
        with torch.no_grad():
            output = self.sharer(input_ids=ids, use_cache=True)
        return extract_kv(output.past_key_values), int(ids.shape[1])

    def decode_receiver(
        self,
        text: str,
        residual_fn: "Any | None" = None,
        max_new_tokens: int = 64,
    ) -> tuple[str, int, int, "float | None", "float | None"]:
        """Prefill ``prompt[:-1]``, inject XKV once, then greedily decode."""

        import torch
        from time import perf_counter

        started = perf_counter()
        ids = self._encode_chat(self.receiver_tok, text)
        input_tokens = int(ids.shape[1])
        stop_ids = resolve_stop_ids(self.receiver_tok, self.receiver)
        first_token_ms: float | None = None
        last_token_ms: float | None = None

        with torch.no_grad():
            if ids.shape[1] >= 2:
                prefix, current = ids[:, :-1], ids[:, -1:]
                output = self.receiver(input_ids=prefix, use_cache=True)
                cache = output.past_key_values
                if residual_fn is not None:
                    add_residual_inplace(cache, residual_fn(cache))
            else:
                current = ids
                cache = self.receiver(
                    input_ids=ids[:, :0], use_cache=True
                ).past_key_values

            generated: list[int] = []
            for _ in range(max_new_tokens):
                step = self.receiver(
                    input_ids=current, past_key_values=cache, use_cache=True
                )
                cache = step.past_key_values
                next_id = int(step.logits[:, -1, :].argmax(dim=-1))
                if next_id in stop_ids:
                    break
                generated.append(next_id)
                last_token_ms = (perf_counter() - started) * 1000.0
                if first_token_ms is None:
                    first_token_ms = last_token_ms
                current = torch.tensor([[next_id]], device=self.device)

        decoded = self.receiver_tok.decode(generated, skip_special_tokens=True)
        return (
            decoded.strip(),
            input_tokens,
            len(generated),
            first_token_ms,
            last_token_ms,
        )


def check_pair_compatible(pair: XKVPair) -> dict[str, Any]:
    sharer, receiver = pair.sharer_spec, pair.receiver_spec
    return {
        "sharer": {
            "layers": sharer.layers,
            "kv_heads": sharer.kv_heads,
            "head_dim": sharer.head_dim,
            "gemma3_hybrid_cache": sharer.is_gemma3,
            "hybrid_cache": sharer.has_hybrid_cache,
        },
        "receiver": {
            "layers": receiver.layers,
            "kv_heads": receiver.kv_heads,
            "head_dim": receiver.head_dim,
            "gemma3_hybrid_cache": receiver.is_gemma3,
            "hybrid_cache": receiver.has_hybrid_cache,
        },
        "same_layers": sharer.layers == receiver.layers,
        "same_kv_heads": sharer.kv_heads == receiver.kv_heads,
        "same_head_dim": sharer.head_dim == receiver.head_dim,
    }


def load_xkv_pair(
    sharer_model: str,
    receiver_model: str | None = None,
    *,
    device: str | None = None,
    dtype: str | None = "auto",
    trust_remote_code: bool = False,
) -> XKVPair:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    receiver_model = receiver_model or sharer_model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    resolved_dtype = _resolve_dtype(dtype, torch)

    def _load(name: str) -> tuple[Any, Any]:
        _validate_ministral_offline_cache(name)
        tokenizer_kwargs = dict(
            trust_remote_code=trust_remote_code,
            clean_up_tokenization_spaces=False,
            use_fast=True,
        )
        if name == MINISTRAL_8B:
            # The repo labels its JSON/Tekken tokenizer as LlamaTokenizer.
            # Forcing the fast path avoids a nonexistent SentencePiece model;
            # legacy=False also suppresses the irrelevant Llama warning.
            tokenizer_kwargs["legacy"] = False
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            **tokenizer_kwargs,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=resolved_dtype, trust_remote_code=trust_remote_code
        )
        model.to(torch_device)
        model.eval()
        return model, tokenizer

    sharer, sharer_tok = _load(sharer_model)
    if receiver_model == sharer_model:
        receiver, receiver_tok = sharer, sharer_tok
    else:
        receiver, receiver_tok = _load(receiver_model)
    return XKVPair(
        sharer=sharer,
        receiver=receiver,
        sharer_tok=sharer_tok,
        receiver_tok=receiver_tok,
        device=torch_device,
        sharer_spec=_spec_from_model(sharer_model, sharer),
        receiver_spec=_spec_from_model(receiver_model, receiver),
    )


__all__ = ["ModelSpec", "XKVPair", "check_pair_compatible", "load_xkv_pair"]
