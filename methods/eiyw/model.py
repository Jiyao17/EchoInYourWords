"""Hugging Face model backend for EIYW.

Loads a frozen Sharer + Receiver pair and exposes the primitives the runner
needs: encode the Sharer context into a raw KV cache, compute the Receiver
question *anchor*, and greedily decode the Receiver with optional soft pseudo-
tokens prepended to its input. This is the only EIYW module that loads models;
``torch``/``transformers`` are imported lazily.

Decoding differs from LCF-X: EIYW prepends the translated soft tokens to the
Receiver's input *embeddings*, so prefill uses ``inputs_embeds`` (LCF-X prefilled
``input_ids`` and injected a KV residual). With no soft tokens this reduces to
ordinary greedy decoding (the Receiver answering from its own context alone).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from methods.common.decoding import resolve_stop_ids

from .cache import extract_kv


@dataclass(frozen=True)
class ModelSpec:
    """Cache-relevant shape metadata for one model."""

    name: str
    layers: int
    kv_heads: int
    head_dim: int
    n_heads: int
    hidden: int


def _spec_from_model(name: str, model: Any) -> ModelSpec:
    cfg = model.config
    n_heads = cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", None) or n_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_heads)
    return ModelSpec(
        name=name,
        layers=cfg.num_hidden_layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        n_heads=n_heads,
        hidden=cfg.hidden_size,
    )


def _resolve_dtype(dtype: str | None, torch_mod: Any) -> Any:
    if dtype is None:
        return None
    name = str(dtype).strip().lower()
    mapping = {
        "float16": torch_mod.float16, "fp16": torch_mod.float16, "half": torch_mod.float16,
        "bfloat16": torch_mod.bfloat16, "bf16": torch_mod.bfloat16,
        "float32": torch_mod.float32, "fp32": torch_mod.float32, "float": torch_mod.float32,
        "auto": "auto",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype!r}")
    return mapping[name]


class EIYWPair:
    """A frozen Sharer + Receiver pair with EIYW soft-token primitives."""

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

    # -- encoding -----------------------------------------------------------
    def _encode_plain(self, tok: Any, text: str) -> Any:
        enc = tok(text, return_tensors="pt")
        return enc["input_ids"].to(self.device)

    def _encode_chat(self, tok: Any, text: str) -> Any:
        if getattr(tok, "chat_template", None):
            messages = [{"role": "user", "content": text}]
            kwargs = dict(
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            # Disable "thinking" so the Receiver emits a short answer directly.
            # Qwen3 defaults to thinking-on, which fills the token budget with
            # <think> traces and never reaches the answer; non-thinking
            # tokenizers (e.g. Qwen2.5) reject the kwarg, so fall back.
            try:
                enc = tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
            except TypeError:
                enc = tok.apply_chat_template(messages, **kwargs)
            return enc["input_ids"].to(self.device)
        return self._encode_plain(tok, text)

    def _receiver_embed(self) -> Any:
        return self.receiver.get_input_embeddings()

    # -- sharer side --------------------------------------------------------
    def sharer_summary(self, text: str) -> tuple[list[tuple[Any, Any]], int]:
        """Encode Sharer text and return its raw per-layer KV cache.

        Returns ``(summary, input_tokens)`` where ``summary`` is per layer
        ``(K, V)`` of shape ``[B, H_s, N_S, D_s]``. The translator pools these
        into K receiver-anchored soft tokens.
        """

        import torch

        ids = self._encode_plain(self.sharer_tok, text)
        with torch.no_grad():
            out = self.sharer(input_ids=ids, use_cache=True)
        return extract_kv(out.past_key_values), int(ids.shape[1])

    # -- receiver side ------------------------------------------------------
    def receiver_anchor(self, text: str) -> Any:
        """Mean-pooled embedding of the Receiver prompt — the receiver-anchoring signal.

        Returns ``[1, hidden_r]``. This is what the translator's query bank is
        conditioned on, so *what the Sharer communicates depends on the Receiver's
        own task representation*. Uses the frozen embedding table (no grad).
        """

        import torch

        ids = self._encode_chat(self.receiver_tok, text)
        with torch.no_grad():
            emb = self._receiver_embed()(ids)  # [1, T, H]
        return emb.mean(dim=1)  # [1, H]

    def decode_receiver(
        self,
        text: str,
        soft_tokens: "Any | None" = None,
        max_new_tokens: int = 64,
    ) -> tuple[str, int, int]:
        """Greedy-decode the Receiver, optionally prepending soft tokens to its input.

        ``soft_tokens`` is ``[1, K, hidden_r]`` (or ``None`` for the Receiver-only
        floor). Prefill uses ``inputs_embeds`` = ``[soft_tokens ; embed(prompt)]``;
        subsequent steps feed new ``input_ids`` against the growing cache.

        Returns ``(output_text, input_tokens, output_tokens)``.
        """

        import torch

        ids = self._encode_chat(self.receiver_tok, text)
        input_tokens = int(ids.shape[1])
        stop_ids = resolve_stop_ids(self.receiver_tok, self.receiver)
        embed = self._receiver_embed()

        with torch.no_grad():
            prompt_embeds = embed(ids)  # [1, T, H]
            if soft_tokens is not None and soft_tokens.shape[1] > 0:
                st = soft_tokens.to(prompt_embeds.dtype)
                prefill = torch.cat([st, prompt_embeds], dim=1)
            else:
                prefill = prompt_embeds
            out = self.receiver(inputs_embeds=prefill, use_cache=True)
            cache = out.past_key_values

            generated: list[int] = []
            next_id = int(out.logits[:, -1, :].argmax(dim=-1))
            for _ in range(max_new_tokens):
                if next_id in stop_ids:
                    break
                generated.append(next_id)
                cur = torch.tensor([[next_id]], device=self.device)
                step = self.receiver(input_ids=cur, past_key_values=cache, use_cache=True)
                cache = step.past_key_values
                next_id = int(step.logits[:, -1, :].argmax(dim=-1))

        text_out = self.receiver_tok.decode(generated, skip_special_tokens=True)
        return text_out.strip(), input_tokens, len(generated)


def check_pair_compatible(pair: EIYWPair) -> dict[str, Any]:
    """Report the dimension bridge the translator must span (records, never requires)."""

    s, r = pair.sharer_spec, pair.receiver_spec
    return {
        "sharer": {"layers": s.layers, "kv_heads": s.kv_heads, "head_dim": s.head_dim},
        "receiver": {"layers": r.layers, "hidden": r.hidden},
        "same_kv_heads": s.kv_heads == r.kv_heads,
        "same_head_dim": s.head_dim == r.head_dim,
    }


def load_eiyw_pair(
    sharer_model: str,
    receiver_model: str | None = None,
    *,
    device: str | None = None,
    dtype: str | None = "auto",
    trust_remote_code: bool = False,
) -> EIYWPair:
    """Load frozen Sharer + Receiver models and return an :class:`EIYWPair`."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    receiver_model = receiver_model or sharer_model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    resolved_dtype = _resolve_dtype(dtype, torch)

    def _load(name: str) -> tuple[Any, Any]:
        # clean_up_tokenization_spaces is a WordPiece-era step that corrupts BPE
        # output (strips spaces before punctuation); disable it for these BPE models.
        tok = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=trust_remote_code,
            clean_up_tokenization_spaces=False,
        )
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=resolved_dtype, trust_remote_code=trust_remote_code
        )
        model.to(torch_device)
        model.eval()
        return model, tok

    sharer, sharer_tok = _load(sharer_model)
    if receiver_model == sharer_model:
        receiver, receiver_tok = sharer, sharer_tok
    else:
        receiver, receiver_tok = _load(receiver_model)

    return EIYWPair(
        sharer=sharer,
        receiver=receiver,
        sharer_tok=sharer_tok,
        receiver_tok=receiver_tok,
        device=torch_device,
        sharer_spec=_spec_from_model(sharer_model, sharer),
        receiver_spec=_spec_from_model(receiver_model, receiver),
    )
