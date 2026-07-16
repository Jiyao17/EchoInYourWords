"""Hugging Face model backend for EIYW.

Loads a frozen Sharer + Receiver pair and exposes the primitives the runner
needs: encode the Sharer's own context (full forward, raw KV cache), look up
the Receiver's own context *embeddings* (table read only — v2 needs no
Receiver forward pass for its context), and greedily decode the Receiver with
an optional soft token prepended to its input. This is the only EIYW module
that loads models; ``torch``/``transformers`` are imported lazily.

Decoding prepends the translated token to the Receiver's input *embeddings*,
so prefill uses ``inputs_embeds`` (not ``input_ids`` + a KV residual, unlike
LCF-X). With no soft token this reduces to ordinary greedy decoding (the
Receiver answering from its own context alone).
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
    # RMS of the model's real input embeddings, measured through the embedding
    # *module* (not the raw weight table) so families that rescale inside the
    # module -- Gemma multiplies by sqrt(hidden) there -- report what the
    # decoder layers actually see. The translator calibrates its soft-token
    # magnitude against this: embedding scales differ ~100x across families
    # (Llama-3.2 ~0.01 raw vs Gemma-3 ~1.0 post-scale), so any fixed init is
    # simultaneously too loud for one receiver and too quiet for another.
    embed_rms: float = 1.0


def _embed_rms(model: Any) -> float:
    """RMS of the input-embedding module's output over a vocabulary sample."""

    import torch

    embed = model.get_input_embeddings()
    weight = embed.weight
    with torch.no_grad():
        ids = torch.arange(min(4096, weight.shape[0]), device=weight.device)
        return float(embed(ids).float().pow(2).mean().sqrt())


def _spec_from_model(name: str, model: Any) -> ModelSpec:
    cfg = model.config
    if not hasattr(cfg, "num_attention_heads") and hasattr(cfg, "text_config"):
        # Composite configs (e.g. Gemma3Config) nest the decoder dims under
        # text_config instead of exposing them on the top-level config.
        cfg = cfg.text_config
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
        embed_rms=_embed_rms(model),
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
        ``(K, V)`` of shape ``[B, H_s, N_S, D_s]``. The translator's pool turns
        this into a single position-free Sharer summary.
        """

        import torch

        ids = self._encode_plain(self.sharer_tok, text)
        with torch.no_grad():
            out = self.sharer(input_ids=ids, use_cache=True)
        return extract_kv(out.past_key_values), int(ids.shape[1])

    # -- receiver side --------------------------------------------------------
    def receiver_context_embeds(self, text: str) -> tuple[Any, int]:
        """Raw input embeddings of the Receiver's own context: ``[1, N_r, hidden]``.

        A pure embedding-table lookup (plain encode, ``no_grad``) — no decoder
        forward pass, unlike the Sharer side. The translator cross-attends its
        pooled Sharer slots into these embeddings (adding its own sinusoidal
        PE; raw embeddings carry no order). Build the text with
        ``methods.common.prompts.build_receiver_encode_text``; this is separate
        from :meth:`decode_receiver`'s chat-templated generation prompt.
        """

        import torch

        ids = self._encode_plain(self.receiver_tok, text)
        with torch.no_grad():
            embeds = self._receiver_embed()(ids)  # [1, N_r, hidden]
        return embeds, int(ids.shape[1])

    def decode_receiver(
        self,
        text: str,
        soft_token: "Any | None" = None,
        max_new_tokens: int = 64,
    ) -> tuple[str, int, int, "float | None", "float | None"]:
        """Greedy-decode the Receiver, optionally prepending soft tokens to its input.

        ``soft_token`` is ``[1, T, hidden_r]`` (``T`` = the translator's
        ``k·L_s`` token count, or ``None``/empty for the Receiver-only floor).
        Prefill uses ``inputs_embeds`` = ``[soft_token ; embed(prompt)]``;
        subsequent steps feed new ``input_ids`` against the growing cache.

        Returns ``(output_text, input_tokens, output_tokens, first_token_ms,
        last_token_ms)``. The token timestamps are milliseconds elapsed since this
        call started (tokenize + prefill included) to the first/last *generated*
        token; both are ``None`` if no tokens were generated.
        """

        import torch
        from time import perf_counter

        decode_start = perf_counter()
        ids = self._encode_chat(self.receiver_tok, text)
        input_tokens = int(ids.shape[1])
        stop_ids = resolve_stop_ids(self.receiver_tok, self.receiver)
        embed = self._receiver_embed()

        first_token_ms: float | None = None
        last_token_ms: float | None = None

        with torch.no_grad():
            prompt_embeds = embed(ids)  # [1, T, H]
            if soft_token is not None and soft_token.shape[1] > 0:
                st = soft_token.to(prompt_embeds.dtype)
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
                last_token_ms = (perf_counter() - decode_start) * 1000.0
                if first_token_ms is None:
                    first_token_ms = last_token_ms
                cur = torch.tensor([[next_id]], device=self.device)
                step = self.receiver(input_ids=cur, past_key_values=cache, use_cache=True)
                cache = step.past_key_values
                next_id = int(step.logits[:, -1, :].argmax(dim=-1))

        text_out = self.receiver_tok.decode(generated, skip_special_tokens=True)
        return text_out.strip(), input_tokens, len(generated), first_token_ms, last_token_ms


def check_pair_compatible(pair: EIYWPair) -> dict[str, Any]:
    """Report the dimension bridge the translator must span (records, never requires)."""

    s, r = pair.sharer_spec, pair.receiver_spec
    return {
        "sharer": {"layers": s.layers, "kv_heads": s.kv_heads, "head_dim": s.head_dim},
        "receiver": {"layers": r.layers, "kv_heads": r.kv_heads, "head_dim": r.head_dim},
        "same_layers": s.layers == r.layers,
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
