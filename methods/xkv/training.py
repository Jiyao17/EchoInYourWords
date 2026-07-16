"""Training, evaluation, and checkpointing for XKV."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Iterable

from methods.common.decoding import resolve_answer_stop_id
from methods.common.progress import ProgressLogger
from methods.common.prompts import build_receiver_prompt, build_sharer_encode_text

from .cache import add_residual_inplace, extract_kv
from .model import XKVPair
from .runner import run_xkv_example
from .schema import XKVExample
from .translator import XKVTranslator, _resolve_d_model, _safe_num_heads

CHECKPOINT_FORMAT = 9


def _freeze(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def _as_example(raw: Any) -> XKVExample:
    return raw if isinstance(raw, XKVExample) else XKVExample.from_mapping(raw)


def _pad(sequences: list, pad_id: int, side: str) -> tuple:
    import torch

    width = max(int(sequence.numel()) for sequence in sequences)
    ids = torch.full(
        (len(sequences), width),
        pad_id,
        dtype=torch.long,
        device=sequences[0].device,
    )
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        length = int(sequence.numel())
        if side == "left":
            ids[index, width - length :] = sequence
            mask[index, width - length :] = 1
        else:
            ids[index, :length] = sequence
            mask[index, :length] = 1
    return ids, mask


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def batched_example_loss(
    pair: XKVPair,
    translator: XKVTranslator,
    examples: list,
    *,
    reduction: str = "mean",
) -> "Any | None":
    """Teacher-forced CE through an XKV-updated Receiver prefix cache."""

    import torch
    from torch.nn import functional as F

    stop_id = resolve_answer_stop_id(pair.receiver_tok, pair.receiver)
    rows = []
    for raw in examples:
        example = _as_example(raw)
        if not example.answer:
            continue
        receiver_prompt = build_receiver_prompt(example)
        prompt = pair._encode_chat(pair.receiver_tok, receiver_prompt)[0]
        answer = pair.receiver_tok(
            example.answer, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].to(pair.device)
        if stop_id is not None:
            answer = torch.cat(
                [
                    answer,
                    torch.tensor([stop_id], dtype=answer.dtype, device=pair.device),
                ]
            )
        sharer_text = build_sharer_encode_text(example)
        sharer = pair.sharer_tok(sharer_text, return_tensors="pt")["input_ids"][0].to(
            pair.device
        )
        if prompt.numel() < 2 or answer.numel() < 1 or sharer.numel() < 1:
            continue
        rows.append((prompt, answer, sharer))
    if not rows:
        return None

    prompts, answers, sharers = zip(*rows)
    sharer_side = "left" if pair.sharer_spec.is_gemma3 else "right"
    sharer_ids, sharer_mask = _pad(
        list(sharers), pair.sharer_tok.pad_token_id, sharer_side
    )
    sharer_positions = (sharer_mask.cumsum(dim=-1) - 1).clamp(min=0)
    with torch.no_grad():
        sharer_dynamic_cache = pair.sharer(
            input_ids=sharer_ids,
            attention_mask=sharer_mask,
            position_ids=sharer_positions,
            use_cache=True,
        ).past_key_values

    # Left-padding keeps every held-out final prompt token in the last column.
    prompt_ids, prompt_mask = _pad(
        list(prompts), pair.receiver_tok.pad_token_id, "left"
    )
    prefix_mask = prompt_mask[:, :-1]
    prefix_positions = (prefix_mask.cumsum(dim=-1) - 1).clamp(min=0)
    with torch.no_grad():
        receiver_dynamic_cache = pair.receiver(
            input_ids=prompt_ids[:, :-1],
            attention_mask=prefix_mask,
            position_ids=prefix_positions,
            use_cache=True,
        ).past_key_values

    residual = translator.translate(
        extract_kv(sharer_dynamic_cache),
        extract_kv(receiver_dynamic_cache),
        sharer_attention_mask=sharer_mask,
        receiver_attention_mask=prefix_mask,
        sharer_windowed_cache=pair.sharer_spec.is_gemma3,
        receiver_windowed_cache=pair.receiver_spec.is_gemma3,
    )
    add_residual_inplace(receiver_dynamic_cache, residual)

    answer_ids, answer_mask = _pad(
        list(answers), pair.receiver_tok.pad_token_id, "right"
    )
    decode_input = torch.cat([prompt_ids[:, -1:], answer_ids], dim=1)
    full_mask = torch.cat([prompt_mask, answer_mask], dim=1)
    position_ids = (full_mask.cumsum(dim=-1) - 1).clamp(min=0)
    position_ids = position_ids[:, -decode_input.shape[1] :]
    logits = pair.receiver(
        input_ids=decode_input,
        attention_mask=full_mask,
        position_ids=position_ids,
        past_key_values=receiver_dynamic_cache,
        use_cache=True,
    ).logits[:, : answer_ids.shape[1], :]

    targets = answer_ids.masked_fill(answer_mask == 0, -100)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(targets.shape)
    per_example = token_loss.sum(dim=1) / answer_mask.sum(dim=1).clamp(min=1)
    return per_example if reduction == "none" else per_example.mean()


def example_loss(
    pair: XKVPair,
    translator: XKVTranslator,
    example: XKVExample,
) -> "Any | None":
    return batched_example_loss(
        pair,
        translator,
        [example],
        reduction="mean",
    )


def evaluate(
    pair: XKVPair,
    translator: XKVTranslator,
    eval_data: list[Any],
    *,
    batch_size: int = 16,
    max_examples: int | None = None,
) -> float:
    import torch

    items = list(eval_data)
    if max_examples is not None:
        items = items[:max_examples]
    was_training = translator.training
    translator.eval()
    total = count = 0
    with torch.no_grad():
        for chunk in _chunks(items, batch_size):
            losses = batched_example_loss(
                pair,
                translator,
                chunk,
                reduction="none",
            )
            if losses is not None:
                total += float(losses.sum().item())
                count += int(losses.numel())
    if was_training:
        translator.train()
    return total / count if count else float("nan")


def evaluate_em_f1(
    pair: XKVPair,
    translator: XKVTranslator,
    eval_data: list[Any],
    *,
    max_new_tokens: int = 64,
    max_examples: int | None = None,
    brief_path: "str | Any | None" = None,
) -> dict[str, float]:
    import torch

    from methods.common.io import write_predictions_brief
    from methods.common.scoring import score_free_form

    items = [_as_example(item) for item in eval_data]
    if max_examples is not None:
        items = items[:max_examples]
    was_training = translator.training
    translator.eval()
    results = []
    with torch.no_grad():
        for example in items:
            if example.answer:
                results.append(
                    run_xkv_example(
                        example,
                        pair,
                        translator,
                        max_new_tokens=max_new_tokens,
                        score_fn=score_free_form,
                    )
                )
    if brief_path is not None:
        write_predictions_brief(results, brief_path)
    if was_training:
        translator.train()
    count = len(results)
    return {
        "em": sum(float(result.scores.get("em", 0.0)) for result in results) / count
        if count
        else float("nan"),
        "f1": sum(float(result.scores.get("f1", 0.0)) for result in results) / count
        if count
        else float("nan"),
        "n": count,
    }


def _pair_arch(pair: XKVPair) -> dict:
    sharer, receiver = pair.sharer_spec, pair.receiver_spec
    return {
        "sharer_layers": sharer.layers,
        "sharer_kv_heads": sharer.kv_heads,
        "sharer_head_dim": sharer.head_dim,
        "receiver_layers": receiver.layers,
        "receiver_kv_heads": receiver.kv_heads,
        "receiver_head_dim": receiver.head_dim,
    }


def save_translator(
    translator: XKVTranslator,
    path: str,
    pair: XKVPair,
) -> None:
    import torch

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "state_dict": translator.state_dict(),
            "arch": translator.arch_config(),
            "sharer_model": pair.sharer_spec.name,
            "receiver_model": pair.receiver_spec.name,
        },
        path,
    )


def load_translator(
    path: str,
    pair: XKVPair,
    *,
    device: Any = None,
    allow_model_mismatch: bool = False,
    d_model: "str | int" = "eighth",
    num_blocks: int = 1,
    num_heads: int = 8,
    dropout: float = 0.1,
    pool_slots: int = 2,
) -> XKVTranslator:
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"XKV checkpoint {path} is not a stamped checkpoint")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"XKV checkpoint {path} format={checkpoint.get('format')}, "
            f"expected {CHECKPOINT_FORMAT}"
        )
    resolved_d_model = _resolve_d_model(d_model, pair.receiver_spec.hidden)
    expected = {
        **_pair_arch(pair),
        "d_model": resolved_d_model,
        "num_blocks": int(num_blocks),
        "num_heads": _safe_num_heads(resolved_d_model, int(num_heads)),
        "dropout": float(dropout),
        "pool_slots": int(pool_slots),
        "pooling_mode": "direct_attention_k",
        "output_mode": "position_cross_attention",
    }
    mismatches: list[str] = []
    if checkpoint.get("sharer_model") != pair.sharer_spec.name:
        mismatches.append(
            f"sharer_model={checkpoint.get('sharer_model')!r} != {pair.sharer_spec.name!r}"
        )
    if checkpoint.get("receiver_model") != pair.receiver_spec.name:
        mismatches.append(
            f"receiver_model={checkpoint.get('receiver_model')!r} != {pair.receiver_spec.name!r}"
        )
    architecture = checkpoint.get("arch", {})
    for key, wanted in expected.items():
        if architecture.get(key) != wanted:
            mismatches.append(f"arch.{key}={architecture.get(key)!r} != {wanted!r}")
    if mismatches:
        message = f"XKV checkpoint {path} mismatch:\n  - " + "\n  - ".join(mismatches)
        if not allow_model_mismatch:
            raise ValueError(
                message + "\nPass --allow-model-mismatch to force loading."
            )
        print(f"[load_translator] WARNING (forced): {message}")
    translator = XKVTranslator.from_arch_config(architecture, zero_init=False)
    translator.load_state_dict(checkpoint["state_dict"])
    return translator


def train_translator(
    pair: XKVPair,
    translator: XKVTranslator,
    train_data: Iterable[Any],
    *,
    eval_data: Iterable[Any] | None = None,
    lr: float = 1e-4,
    epochs: int = 1,
    batch_size: int = 16,
    grad_accum: int = 1,
    warmup_steps: int = 0,
    max_steps: int | None = None,
    grad_clip: float = 1.0,
    log_every: int = 10,
    eval_every: int | None = None,
    eval_max_examples: int | None = None,
    save_every: int | None = 50,
    eval_em_examples: int = 0,
    shuffle: bool = True,
    seed: int = 0,
    checkpoint_path: str | None = None,
    log_path: str | None = None,
) -> XKVTranslator:
    import torch

    _freeze(pair.sharer)
    _freeze(pair.receiver)
    translator.train()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    training = list(train_data)
    evaluation = list(eval_data) if eval_data is not None else None
    if not training:
        raise ValueError("train_data is empty")
    grad_accum = max(1, int(grad_accum))
    warmup = max(0, int(warmup_steps))
    steps_per_epoch = max(1, -(-len(training) // (batch_size * grad_accum)))
    total_steps = (
        max_steps if max_steps is not None else max(1, epochs * steps_per_epoch)
    )

    if log_path is None and checkpoint_path:
        log_path = str(Path(checkpoint_path).with_name("training_log.jsonl"))
    logger = ProgressLogger(
        log_path,
        config={
            "method": "xkv",
            "sharer_model": pair.sharer_spec.name,
            "receiver_model": pair.receiver_spec.name,
            "trainable_params": translator.trainable_parameters(),
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch": batch_size * grad_accum,
            "warmup": warmup,
            "d_model": translator.d_model,
            "num_blocks": translator.num_blocks,
            "num_heads": translator.num_heads,
            "pool_slots": translator.pool_slots,
            "pool_k": translator.pool_slots,
            "pooling_mode": translator.pooling_mode,
            "output_mode": translator.output_mode,
            "checkpoint_path": checkpoint_path,
            "total_steps": total_steps,
        },
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in translator.parameters() if parameter.requires_grad],
        lr=lr,
    )

    def lr_scale(step: int) -> float:
        return (step + 1) / warmup if warmup and step < warmup else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    print(
        f"Training XKV: {len(training)} train"
        + (f" / {len(evaluation)} eval" if evaluation else "")
        + f" | {translator.trainable_parameters()} params | d={translator.d_model}"
        f" n={translator.num_blocks} | epochs={epochs} lr={lr}"
        f" | batch={batch_size} x grad_accum={grad_accum}"
    )

    optimizer.zero_grad()
    opt_step = micro_step = 0
    best_eval = float("inf")
    saved_best = False
    running_total = 0.0
    running_count = 0
    stop = False

    def maybe_evaluate() -> None:
        nonlocal best_eval, saved_best
        if not evaluation:
            return
        eval_loss = evaluate(
            pair,
            translator,
            evaluation,
            batch_size=batch_size,
            max_examples=eval_max_examples,
        )
        em = f1 = None
        if eval_em_examples > 0:
            brief = (
                str(Path(checkpoint_path).with_name("eval_predictions_brief.tsv"))
                if checkpoint_path
                else None
            )
            metrics = evaluate_em_f1(
                pair,
                translator,
                evaluation,
                max_examples=eval_em_examples,
                brief_path=brief,
            )
            em, f1 = metrics["em"], metrics["f1"]
        is_best = eval_loss < best_eval
        marker = ""
        if is_best:
            best_eval = eval_loss
            if checkpoint_path:
                save_translator(translator, checkpoint_path, pair)
                saved_best = True
                marker = " (new best -> saved)"
        metric_text = f" | em {em:.4f} f1 {f1:.4f}" if em is not None else ""
        print(
            f"  [eval] opt_step {opt_step} | eval_loss {eval_loss:.4f}"
            f"{metric_text}{marker}",
            flush=True,
        )
        logger.log(
            "eval", opt_step=opt_step, eval_loss=eval_loss, em=em, f1=f1, best=is_best
        )

    for epoch in range(max(1, epochs)):
        order = list(range(len(training)))
        if shuffle:
            rng.shuffle(order)
        for indices in _chunks(order, batch_size):
            will_log = bool(
                log_every
                and (micro_step + 1) % grad_accum == 0
                and (opt_step + 1) % log_every == 0
            )
            translator.set_pool_diagnostics(will_log)
            loss = batched_example_loss(
                pair,
                translator,
                [training[index] for index in indices],
                reduction="mean",
            )
            translator.set_pool_diagnostics(False)
            if loss is None:
                continue
            (loss / grad_accum).backward()
            micro_step += 1
            running_total += float(loss.item())
            running_count += 1
            if micro_step % grad_accum:
                continue
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(translator.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            opt_step += 1
            if log_every and opt_step % log_every == 0:
                train_loss = running_total / max(1, running_count)
                current_lr = scheduler.get_last_lr()[0]
                pool_diagnostics = translator.pool_diagnostics()
                pool_text = ""
                if pool_diagnostics:
                    pool_text = (
                        " | pool H(s/r) "
                        f"{pool_diagnostics['sharer_pool_entropy']:.4f}/"
                        f"{pool_diagnostics['receiver_pool_entropy']:.4f}"
                        " cos(s/r) "
                        f"{pool_diagnostics['sharer_pool_slot_cosine']:.4f}/"
                        f"{pool_diagnostics['receiver_pool_slot_cosine']:.4f}"
                        " scale(s/r) "
                        f"{pool_diagnostics['sharer_pool_scale']:.3f}/"
                        f"{pool_diagnostics['receiver_pool_scale']:.3f}"
                    )
                print(
                    f"epoch {epoch + 1}/{epochs} opt_step {opt_step}"
                    f" | train_loss {train_loss:.4f} | lr {current_lr:.2e}"
                    f"{pool_text}",
                    flush=True,
                )
                logger.log(
                    "train",
                    epoch=epoch + 1,
                    opt_step=opt_step,
                    train_loss=train_loss,
                    lr=current_lr,
                    **pool_diagnostics,
                )
                running_total = 0.0
                running_count = 0
            if eval_every and opt_step % eval_every == 0:
                maybe_evaluate()
            if checkpoint_path and save_every and opt_step % save_every == 0:
                save_translator(
                    translator,
                    checkpoint_path + ".latest",
                    pair,
                )
            if max_steps is not None and opt_step >= max_steps:
                stop = True
                break
        maybe_evaluate()
        if stop:
            break

    if checkpoint_path and not saved_best:
        save_translator(translator, checkpoint_path, pair)
        print(f"Saved XKV checkpoint -> {checkpoint_path}")
    elif checkpoint_path:
        print(f"Best XKV checkpoint (eval_loss {best_eval:.4f}) -> {checkpoint_path}")
    logger.log("end", opt_step=opt_step, best_eval=best_eval if evaluation else None)
    if log_path:
        print(f"Training log -> {log_path}")
    logger.close()
    translator.eval()
    return translator


__all__ = [
    "batched_example_loss",
    "evaluate",
    "evaluate_em_f1",
    "example_loss",
    "load_translator",
    "save_translator",
    "train_translator",
]
