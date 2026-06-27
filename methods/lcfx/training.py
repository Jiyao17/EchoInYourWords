"""LCF-X fuser training: frozen base models, gradients flow only into the fuser.

Objective: teacher-forced next-token cross-entropy on the gold answer tokens,
with the Sharer's aligned summary fused into the Receiver's prompt cache.

Gradient flow (the important part): the base models are frozen
(``requires_grad_(False)``), but the Receiver forward over the answer tokens is
**grad-enabled** so gradients flow through the Receiver's attention over the fused
cache, into the injected residual, and back into the fuser. Only the prompt
prefill is run under ``no_grad`` — the Receiver's prompt KV is just a (detached)
input to the projector. This mirrors the inference decode path (prefill
``prompt[:-1]`` → inject residual → feed the last prompt token + answer), so
training and evaluation share the same fusion.

The loop is a complete trainer: materialized data, per-epoch shuffling, periodic
validation, best-checkpointing by eval loss, gradient clipping, and a seed.
"""

from __future__ import annotations

import os
import random
from typing import Any, Iterable

from .cache import add_residual_inplace, extract_kv
from .fuser import LCFXFuser
from .model import LCFXPair
from .prompts import build_receiver_prompt, build_sharer_encode_text
from .schema import LCFXExample


def _freeze(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()


def _as_example(raw: Any) -> LCFXExample:
    return raw if isinstance(raw, LCFXExample) else LCFXExample.from_mapping(raw)


def example_loss(
    pair: LCFXPair, fuser: LCFXFuser, example: LCFXExample, num_spans: int
) -> "Any | None":
    """Teacher-forced next-token CE for one example (None if it can't be scored).

    Caller controls grad: wrap in ``torch.no_grad()`` for evaluation; call plainly
    for training (the prompt prefill is always ``no_grad`` internally).
    """

    import torch
    from torch.nn import functional as F

    if not example.answer:
        return None

    summary, _ = pair.sharer_summary(
        build_sharer_encode_text(example), num_spans=num_spans
    )
    aligned = fuser.align(summary)

    prompt_ids = pair._encode_chat(pair.receiver_tok, build_receiver_prompt(example))
    answer_ids = pair.receiver_tok(
        example.answer, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(pair.device)
    if prompt_ids.shape[1] < 2 or answer_ids.shape[1] == 0:
        return None

    # Prefill prompt[:-1] (no grad): Receiver KV is a detached projector input.
    with torch.no_grad():
        cache = pair.receiver(input_ids=prompt_ids[:, :-1], use_cache=True).past_key_values

    residual = fuser.project(aligned, extract_kv(cache))
    add_residual_inplace(cache, residual)

    # Teacher-forced answer forward: [p_last, a_0, ..., a_{T-1}] over the fused cache.
    decode_input = torch.cat([prompt_ids[:, -1:], answer_ids], dim=1)
    logits = pair.receiver(
        input_ids=decode_input, past_key_values=cache, use_cache=True
    ).logits[:, : answer_ids.shape[1], :]

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), answer_ids.reshape(-1)
    )


def evaluate(
    pair: LCFXPair,
    fuser: LCFXFuser,
    eval_data: list[Any],
    *,
    num_spans: int = 1,
    max_examples: int | None = None,
) -> float:
    """Mean teacher-forced loss over an eval set (lower is better)."""

    import torch

    was_training = fuser.training
    fuser.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, raw in enumerate(eval_data):
            if max_examples is not None and i >= max_examples:
                break
            loss = example_loss(pair, fuser, _as_example(raw), num_spans)
            if loss is not None:
                total += float(loss.item())
                count += 1
    if was_training:
        fuser.train()
    return total / count if count else float("nan")


def _save(fuser: LCFXFuser, path: str) -> None:
    import torch

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(fuser.state_dict(), path)


def train_fuser(
    pair: LCFXPair,
    fuser: LCFXFuser,
    train_data: Iterable[Any],
    *,
    eval_data: Iterable[Any] | None = None,
    num_spans: int = 1,
    lr: float = 1e-4,
    epochs: int = 1,
    max_steps: int | None = None,
    grad_clip: float = 1.0,
    log_every: int = 10,
    eval_every: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
    checkpoint_path: str | None = None,
) -> LCFXFuser:
    """Train only the fuser with frozen Sharer/Receiver. Returns the trained fuser.

    With ``eval_data``, the best checkpoint (lowest eval loss) is saved to
    ``checkpoint_path``; otherwise the final fuser is saved there.
    """

    import torch

    _freeze(pair.sharer)
    _freeze(pair.receiver)
    fuser.train()
    torch.manual_seed(seed)
    rng = random.Random(seed)

    train_list = list(train_data)
    eval_list = list(eval_data) if eval_data is not None else None
    if not train_list:
        raise ValueError("train_data is empty")

    optimizer = torch.optim.Adam(
        [p for p in fuser.parameters() if p.requires_grad], lr=lr
    )

    print(
        f"Training fuser: {len(train_list)} train"
        + (f" / {len(eval_list)} eval" if eval_list else "")
        + f" | {fuser.trainable_parameters()} params | epochs={epochs} lr={lr}"
    )

    global_step = 0
    best_eval = float("inf")
    saved_best = False
    running, running_count = 0.0, 0
    stop = False

    def _maybe_eval_and_save() -> None:
        nonlocal best_eval, saved_best
        if not eval_list:
            return
        eval_loss = evaluate(pair, fuser, eval_list, num_spans=num_spans)
        marker = ""
        if eval_loss < best_eval:
            best_eval = eval_loss
            if checkpoint_path:
                _save(fuser, checkpoint_path)
                saved_best = True
                marker = "  (new best -> saved)"
        print(f"  [eval] step {global_step} | eval_loss {eval_loss:.4f}{marker}", flush=True)

    for epoch in range(max(1, epochs)):
        order = list(range(len(train_list)))
        if shuffle:
            rng.shuffle(order)
        for idx in order:
            loss = example_loss(pair, fuser, _as_example(train_list[idx]), num_spans)
            if loss is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(fuser.parameters(), grad_clip)
            optimizer.step()

            global_step += 1
            running += float(loss.item())
            running_count += 1
            if log_every and global_step % log_every == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} step {global_step} "
                    f"| train_loss {running / running_count:.4f}",
                    flush=True,
                )
                running, running_count = 0.0, 0
            if eval_every and global_step % eval_every == 0:
                _maybe_eval_and_save()
            if max_steps is not None and global_step >= max_steps:
                stop = True
                break
        # End-of-epoch validation.
        _maybe_eval_and_save()
        if stop:
            break

    # Persist: best already saved when eval was used; otherwise save the final fuser.
    if checkpoint_path and not saved_best:
        _save(fuser, checkpoint_path)
        print(f"Saved fuser checkpoint -> {checkpoint_path}")
    elif checkpoint_path:
        print(f"Best fuser checkpoint (eval_loss {best_eval:.4f}) -> {checkpoint_path}")

    fuser.eval()
    return fuser
