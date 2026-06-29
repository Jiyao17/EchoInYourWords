"""Hugging Face model backend for the text-to-text baseline.

This module turns a frozen Hugging Face causal language model into a
``TextGenerator`` callable (``Callable[[str], GenerationOutput]``) that the
runner can use. It is the only part of the package that depends on
``torch``/``transformers``; imports are deferred so ``import methods.t2t`` works
without those libraries installed.
"""

from __future__ import annotations

from typing import Any

from .runner import GenerationOutput, TextGenerator


def _resolve_dtype(dtype: str | None, torch_mod: Any) -> Any:
    """Map a dtype name to a torch dtype, defaulting to fp16 on CUDA."""

    if dtype is None:
        return None
    name = str(dtype).strip().lower()
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
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype!r}")
    return mapping[name]


def load_generator(
    model_name: str,
    *,
    max_new_tokens: int = 256,
    deterministic: bool = True,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str | None = None,
    dtype: str | None = "auto",
    trust_remote_code: bool = False,
    use_chat_template: bool = True,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
) -> TextGenerator:
    """Load a frozen causal LM and return a ``TextGenerator``.

    The returned callable maps a prompt string to a :class:`GenerationOutput`
    carrying the generated text plus input/output token counts.

    Args:
        model_name: Hugging Face model id or local path.
        max_new_tokens: Maximum number of tokens to generate per call.
        deterministic: If ``True`` use greedy decoding (``do_sample=False``).
        temperature: Sampling temperature (ignored when ``deterministic``).
        top_p: Nucleus sampling parameter (ignored when ``deterministic``).
        device: Target device (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``).
            Defaults to CUDA when available, else CPU.
        dtype: Weight dtype name or ``"auto"``. Defaults to ``"auto"``.
        trust_remote_code: Forwarded to the HF loaders.
        use_chat_template: If ``True`` and the tokenizer defines a chat
            template, wrap the prompt as a single user turn. This matches how
            instruction-tuned models are meant to be used.
        system_prompt: Optional system message used with the chat template.
        enable_thinking: Forwarded to chat templates that support it
            (e.g. Qwen3); disabled by default to keep outputs concise.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    # clean_up_tokenization_spaces is a WordPiece-era step that corrupts BPE output
    # (strips spaces before punctuation); disable it for these BPE models.
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_dtype = _resolve_dtype(dtype, torch)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolved_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(torch_device)
    model.eval()
    # We always cap generation with max_new_tokens; drop the model's default
    # max_length (often 131072) so transformers doesn't warn about both being set.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None

    has_chat_template = use_chat_template and getattr(
        tokenizer, "chat_template", None
    )

    def _to_device(encoded: Any) -> dict[str, Any]:
        # ``encoded`` may be a Mapping/BatchEncoding (input_ids + attention_mask)
        # or a bare tensor depending on the tokenizer / transformers version.
        if hasattr(encoded, "keys"):
            return {
                key: value.to(torch_device)
                for key, value in encoded.items()
                if hasattr(value, "to")
            }
        return {"input_ids": encoded.to(torch_device)}

    def _encode(prompt: str) -> dict[str, Any]:
        if has_chat_template:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            template_kwargs: dict[str, Any] = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "return_dict": True,
            }
            try:
                encoded = tokenizer.apply_chat_template(
                    messages, enable_thinking=enable_thinking, **template_kwargs
                )
            except TypeError:
                # Tokenizer template does not accept enable_thinking.
                encoded = tokenizer.apply_chat_template(messages, **template_kwargs)
            return _to_device(encoded)
        return _to_device(tokenizer(prompt, return_tensors="pt"))

    def generate(prompt: str) -> GenerationOutput:
        inputs = _encode(prompt)
        input_len = int(inputs["input_ids"].shape[1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": not deterministic,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if not deterministic:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[0, input_len:]
        output_len = int(generated_ids.shape[0])
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return GenerationOutput(
            text=text,
            input_tokens=input_len,
            output_tokens=output_len,
        )

    return generate
