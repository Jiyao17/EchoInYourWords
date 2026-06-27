"""Hugging Face model backend for LCF-X.

Loads a frozen Sharer + Receiver pair and exposes the primitives the runner
needs: encode the Sharer context into a pooled latent summary, and greedily
decode the Receiver with an optional fused-cache residual. This is the only
LCF-X module that loads models; ``torch``/``transformers`` are imported lazily.

Decoding uses an explicit greedy loop so we can inject the residual into the KV
cache: we prefill all-but-the-last prompt token, then compute the residual *from
that prefilled Receiver cache* (the LCF projector takes both the aligned-Sharer
and Receiver caches), add it, and feed the last token to continue. With a zero
residual this reduces to ordinary greedy decoding, which is what makes the
zero-init fuser reproduce the Receiver-only run exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cache import add_residual_inplace, pool_summary


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


class LCFXPair:
    """A frozen Sharer + Receiver pair with LCF-X cache primitives."""

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
        import torch

        enc = tok(text, return_tensors="pt")
        return enc["input_ids"].to(self.device)

    def _encode_chat(self, tok: Any, text: str) -> Any:
        if getattr(tok, "chat_template", None):
            enc = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            return enc["input_ids"].to(self.device)
        return self._encode_plain(tok, text)

    # -- sharer side --------------------------------------------------------
    def sharer_summary(
        self, text: str, num_spans: int = 1
    ) -> tuple[list[tuple[Any, Any]], int]:
        """Encode Sharer text and span-pool its cache into resampler tokens.

        Returns ``(summary, input_tokens)`` where ``summary`` is per layer
        ``(K, V)`` of shape ``[B, H_s, P, D_s]`` (the fuser's attention resampler
        turns these into a Receiver-shaped residual).
        """

        import torch

        ids = self._encode_plain(self.sharer_tok, text)
        with torch.no_grad():
            out = self.sharer(input_ids=ids, use_cache=True)
        return pool_summary(out.past_key_values, num_spans=num_spans), int(ids.shape[1])

    # -- receiver side ------------------------------------------------------
    def decode_receiver(
        self,
        text: str,
        residual_fn: "Any | None" = None,
        max_new_tokens: int = 64,
    ) -> tuple[str, int, int]:
        """Greedy-decode the Receiver, optionally fusing a residual into its cache.

        ``residual_fn`` is called with the prefilled Receiver cache (a
        ``DynamicCache``) and must return per-layer ``(dK, dV)`` residuals to add
        — so the projector can read the Receiver cache, per the LCF design. ``None``
        runs the Receiver alone.

        Returns ``(output_text, input_tokens, output_tokens)``.
        """

        import torch

        ids = self._encode_chat(self.receiver_tok, text)
        input_tokens = int(ids.shape[1])
        eos_id = self.receiver_tok.eos_token_id

        with torch.no_grad():
            if ids.shape[1] >= 2:
                prefix, last = ids[:, :-1], ids[:, -1:]
                out = self.receiver(input_ids=prefix, use_cache=True)
                cache = out.past_key_values
                if residual_fn is not None:
                    add_residual_inplace(cache, residual_fn(cache))
            else:
                # Degenerate 1-token prompt: nothing to prefill/fuse.
                last = ids
                cache = self.receiver(input_ids=ids[:, :0], use_cache=True).past_key_values

            generated: list[int] = []
            cur = last
            for _ in range(max_new_tokens):
                step = self.receiver(input_ids=cur, past_key_values=cache, use_cache=True)
                cache = step.past_key_values
                next_id = int(step.logits[:, -1, :].argmax(dim=-1))
                if eos_id is not None and next_id == eos_id:
                    break
                generated.append(next_id)
                cur = torch.tensor([[next_id]], device=self.device)

        text_out = self.receiver_tok.decode(generated, skip_special_tokens=True)
        return text_out.strip(), input_tokens, len(generated)


def check_pair_compatible(pair: LCFXPair) -> dict[str, Any]:
    """Report the dimension bridge the fuser must span (records, never requires)."""

    s, r = pair.sharer_spec, pair.receiver_spec
    return {
        "sharer": {"layers": s.layers, "kv_heads": s.kv_heads, "head_dim": s.head_dim},
        "receiver": {"layers": r.layers, "kv_heads": r.kv_heads, "head_dim": r.head_dim},
        "same_layers": s.layers == r.layers,
        "same_kv_heads": s.kv_heads == r.kv_heads,
        "same_head_dim": s.head_dim == r.head_dim,
    }


def load_lcfx_pair(
    sharer_model: str,
    receiver_model: str | None = None,
    *,
    device: str | None = None,
    dtype: str | None = "auto",
    trust_remote_code: bool = False,
) -> LCFXPair:
    """Load frozen Sharer + Receiver models and return an :class:`LCFXPair`."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    receiver_model = receiver_model or sharer_model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    resolved_dtype = _resolve_dtype(dtype, torch)

    def _load(name: str) -> tuple[Any, Any]:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
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

    return LCFXPair(
        sharer=sharer,
        receiver=receiver,
        sharer_tok=sharer_tok,
        receiver_tok=receiver_tok,
        device=torch_device,
        sharer_spec=_spec_from_model(sharer_model, sharer),
        receiver_spec=_spec_from_model(receiver_model, receiver),
    )
