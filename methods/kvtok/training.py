"""KVTok translator training: frozen base models, gradients flow only into the translator.

Objective: teacher-forced next-token cross-entropy on the gold answer tokens,
with the translated soft tokens (``num_tokens`` of them, see ``translator.py``)
prepended to the Receiver's input.

Gradient flow: the base models are frozen (``requires_grad_(False)``), but
because the soft tokens sit at the *front* of the sequence, every prompt/answer
activation depends on them. So the Receiver forward over
``[soft ; prompt ; answer]`` is **grad-enabled end-to-end**, and gradients flow
through the (frozen) Receiver into the soft tokens and back into the translator
-- this mirrors the inference decode path exactly, and is identical to EIYW's
recipe. The Sharer's context encoding is computed under ``no_grad``. The
Receiver's own context contributes only an embedding-table lookup (no initial
Receiver forward or Receiver KV cache); only the *generation* forward (which
the soft tokens lead) carries gradient.

Because the generation prompt prefill is grad-enabled (more memory than a
no_grad-prefill design), gradient checkpointing on the frozen Receiver is
enabled when available.

The loop is a complete trainer: materialized data, per-epoch shuffling, periodic
validation, best-checkpointing by eval loss, gradient clipping, and a seed.

Unlike EIYW (where ``num_spans`` is a free per-call runtime knob), KVTok's
``num_spans``/``num_tokens`` are baked into the translator's weight shapes at
construction time (see ``translator.KVTokTranslator.arch_config``), so they
are never threaded through the loss/eval functions here -- the translator
passed in already carries them.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Iterable

from methods.common.decoding import resolve_answer_stop_id
from methods.common.progress import ProgressLogger
from methods.common.prompts import (
    build_receiver_encode_text,
    build_receiver_prompt,
    build_sharer_encode_text,
)

from .cache import extract_kv
from .model import KVTokPair
from .schema import KVTokExample
from .translator import KVTokTranslator, _safe_num_heads


def _freeze(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()


def _as_example(raw: Any) -> KVTokExample:
    return raw if isinstance(raw, KVTokExample) else KVTokExample.from_mapping(raw)


def batched_example_loss(
    pair: KVTokPair,
    translator: KVTokTranslator,
    examples: list,
    *,
    reduction: str = "mean",
) -> "Any | None":
    """Teacher-forced loss for a batch (the same fusion as inference).

    Each example's full sequence ``[soft ; prompt ; answer ; stop]`` (see the
    stop-token note below) is built with no internal padding, then the batch
    is **right-padded** for one grad-enabled Receiver forward. Right-padding
    is required here: the whole forward carries gradient -- left-padded
    leading tokens would attend to nothing (softmax over all ``-inf`` -> NaN),
    whereas trailing pad tokens always attend causally to real tokens. Each
    example's answer logits are gathered at its own offset. ``reduction='none'``
    returns a per-example loss vector ``[B]`` (used for the batched==unbatched
    check).
    """

    import torch
    from torch.nn import functional as F

    # Append a resolved stop-token id to each example's answer *before*
    # embedding it, so it's fed as a real input token and supervised as a
    # real target -- see prompt/EOS_STOP_TOKEN_FIX.md and the matching fix in
    # methods/eiyw/training.py.
    stop_id = resolve_answer_stop_id(pair.receiver_tok, pair.receiver)

    exs = [_as_example(e) for e in examples]
    rows = []
    for ex in exs:
        if not ex.answer:
            continue
        prompt = pair._encode_chat(pair.receiver_tok, build_receiver_prompt(ex))[0]
        answer = pair.receiver_tok(
            ex.answer, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].to(pair.device)
        if stop_id is not None:
            stop_tensor = torch.tensor([stop_id], dtype=answer.dtype, device=pair.device)
            answer = torch.cat([answer, stop_tensor], dim=0)
        sharer = pair.sharer_tok(
            build_sharer_encode_text(ex), return_tensors="pt"
        )["input_ids"][0].to(pair.device)
        receiver_ctx = pair.receiver_tok(
            build_receiver_encode_text(ex), return_tensors="pt"
        )["input_ids"][0].to(pair.device)
        if prompt.numel() >= 2 and answer.numel() >= 1:
            rows.append((prompt, answer, sharer, receiver_ctx))
    if not rows:
        return None

    prompts, answers, sharers, receiver_ctxs = zip(*rows)
    s_pad = pair.sharer_tok.pad_token_id
    r_pad = pair.receiver_tok.pad_token_id
    embed = pair._receiver_embed()

    # Sharer: batched forward; the translator's Stage-1 pool reduces each
    # example's real tokens (via lengths) -> a fixed-size Sharer summary.
    s_length = max(int(s.numel()) for s in sharers)
    sharer_ids = torch.full((len(sharers), s_length), s_pad, dtype=torch.long, device=pair.device)
    sharer_mask = torch.zeros((len(sharers), s_length), dtype=torch.long, device=pair.device)
    for i, seq in enumerate(sharers):  # right-pad (pooled over real tokens only)
        n = int(seq.numel())
        sharer_ids[i, :n] = seq
        sharer_mask[i, :n] = 1
    with torch.no_grad():
        sharer_cache = pair.sharer(
            input_ids=sharer_ids, attention_mask=sharer_mask, use_cache=True
        ).past_key_values

    # Receiver context IDs are right-padded for an embedding lookup only. The
    # Receiver LM must not run or create a cache before generation prefill.
    r_length = max(int(s.numel()) for s in receiver_ctxs)
    receiver_ctx_ids = torch.full((len(receiver_ctxs), r_length), r_pad, dtype=torch.long, device=pair.device)
    receiver_ctx_mask = torch.zeros((len(receiver_ctxs), r_length), dtype=torch.long, device=pair.device)
    for i, seq in enumerate(receiver_ctxs):
        n = int(seq.numel())
        receiver_ctx_ids[i, :n] = seq
        receiver_ctx_mask[i, :n] = 1
    with torch.no_grad():
        prompt_embeds = [embed(p.unsqueeze(0)) for p in prompts]  # each [1, P_i, H]
        answer_embeds = [embed(a.unsqueeze(0)) for a in answers]  # each [1, A_i, H]
        receiver_ctx_embeds = embed(receiver_ctx_ids)  # [B, N_r(max), H]

    # Translator (grad-enabled) -> num_tokens soft tokens [B, num_tokens, H] per example.
    soft = translator.translate(
        extract_kv(sharer_cache),
        receiver_ctx_embeds,
        sharer_lengths=sharer_mask.sum(dim=1),
        receiver_lengths=receiver_ctx_mask.sum(dim=1),
    ).to(prompt_embeds[0].dtype)
    k = soft.shape[1]
    batch = soft.shape[0]

    # Build [soft_i ; prompt_i ; answer_i] per example, then RIGHT-pad the batch.
    seqs = [
        torch.cat([soft[i : i + 1], prompt_embeds[i], answer_embeds[i]], dim=1)[0]
        for i in range(batch)
    ]  # each [k + P_i + A_i, H]
    lengths = [int(s.shape[0]) for s in seqs]
    l_max = max(lengths)
    hidden = soft.shape[-1]
    full_embeds = soft.new_zeros((batch, l_max, hidden))
    full_mask = torch.zeros((batch, l_max), dtype=torch.long, device=pair.device)
    for i, s in enumerate(seqs):
        full_embeds[i, : lengths[i]] = s
        full_mask[i, : lengths[i]] = 1

    out = pair.receiver(
        inputs_embeds=full_embeds, attention_mask=full_mask, use_cache=False
    )

    # Gather each example's answer logits at its own offset: the logit predicting
    # answer token j sits at index (k + P_i + j - 1).
    losses = []
    for i in range(batch):
        p_len = int(prompts[i].numel())
        a_len = int(answers[i].numel())
        start = k + p_len - 1
        logits_i = out.logits[i, start : start + a_len, :].float()
        losses.append(F.cross_entropy(logits_i, answers[i]))
    per_example = torch.stack(losses)
    return per_example if reduction == "none" else per_example.mean()


def example_loss(
    pair: KVTokPair, translator: KVTokTranslator, example: KVTokExample
) -> "Any | None":
    """Teacher-forced next-token CE for one example (delegates to the batched path).

    Caller controls grad: wrap in ``torch.no_grad()`` for evaluation; call plainly
    for training.
    """

    return batched_example_loss(pair, translator, [example], reduction="mean")


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def evaluate(
    pair: KVTokPair,
    translator: KVTokTranslator,
    eval_data: list[Any],
    *,
    batch_size: int = 16,
    max_examples: int | None = None,
) -> float:
    """Mean per-example teacher-forced loss over an eval set (lower is better)."""

    import torch

    items = list(eval_data)
    if max_examples is not None:
        items = items[:max_examples]

    was_training = translator.training
    translator.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for chunk in _chunks(items, batch_size):
            per_example = batched_example_loss(pair, translator, chunk, reduction="none")
            if per_example is not None:
                total += float(per_example.sum().item())
                count += int(per_example.numel())
    if was_training:
        translator.train()
    return total / count if count else float("nan")


def evaluate_em_f1(
    pair: KVTokPair,
    translator: KVTokTranslator,
    eval_data: list[Any],
    *,
    max_new_tokens: int = 64,
    max_examples: int | None = None,
    brief_path: "str | Any | None" = None,
) -> dict[str, float]:
    """Generation-based EM/F1 over the eval set (the metric the paper reports).

    Unlike :func:`evaluate` (teacher-forced loss only), this runs **real greedy
    decoding** with the soft tokens prepended, so it reflects EM/F1 -- and surfaces
    failures the loss hides (e.g. the model not stopping after the answer).
    Returns ``{"em", "f1", "n"}``. Slower than the loss eval, so cap it with
    ``max_examples``. With ``brief_path`` set, the per-example
    answer/prediction/EM/F1 are written there as a TSV (a training-time snapshot,
    same format as the inference ``predictions_brief.tsv``).
    """

    import torch

    from methods.common.io import write_predictions_brief
    from methods.common.scoring import score_free_form

    from .runner import run_kvtok_example

    items = [_as_example(e) for e in eval_data]
    if max_examples is not None:
        items = items[:max_examples]

    was_training = translator.training
    translator.eval()
    results = []
    em_sum = f1_sum = 0.0
    n = 0
    with torch.no_grad():
        for ex in items:
            if not ex.answer:
                continue
            result = run_kvtok_example(
                ex, pair, translator,
                max_new_tokens=max_new_tokens, score_fn=score_free_form,
            )
            results.append(result)
            em_sum += float(result.scores.get("em", 0.0))
            f1_sum += float(result.scores.get("f1", 0.0))
            n += 1
    if brief_path is not None:
        write_predictions_brief(results, brief_path)
    if was_training:
        translator.train()
    return {
        "em": em_sum / n if n else float("nan"),
        "f1": f1_sum / n if n else float("nan"),
        "n": n,
    }


CHECKPOINT_FORMAT = 2


def _pair_arch(pair: KVTokPair) -> dict:
    """Model-shape architecture the translator must have for this Sharer/Receiver pair."""

    s, r = pair.sharer_spec, pair.receiver_spec
    return {
        "sharer_layers": s.layers,
        "sharer_kv_heads": s.kv_heads,
        "sharer_head_dim": s.head_dim,
        "receiver_hidden": r.hidden,
    }


def save_translator(translator: KVTokTranslator, path: str, pair: KVTokPair) -> None:
    """Save the translator stamped with the Sharer/Receiver it was trained for.

    The payload is primitive-only (str/int + the tensor state_dict) so it loads
    cleanly under ``torch.load(..., weights_only=True)``. ``arch`` is
    ``translator.arch_config()``, which (unlike EIYW) already includes
    ``num_spans``/``num_tokens`` since both determine this translator's weight
    shapes. ``load_translator`` validates the stamped model names + architecture
    against the pair (and requested ``num_spans``/``num_tokens``) at load time.
    """

    import torch

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "state_dict": translator.state_dict(),
        "arch": translator.arch_config(),
        "sharer_model": pair.sharer_spec.name,
        "receiver_model": pair.receiver_spec.name,
    }
    torch.save(payload, path)


def load_translator(
    path: str,
    pair: KVTokPair,
    *,
    num_spans: int,
    num_tokens: int,
    d_model: int = 512,
    num_heads: int = 8,
    device: Any = None,
    allow_model_mismatch: bool = False,
) -> KVTokTranslator:
    """Load a trained translator, validating it matches the Sharer/Receiver ``pair``
    and the requested translator shape.

    Checkpoints carry the model names and full architecture (including
    ``num_spans``/``num_tokens``/``d_model``/``num_heads``). By default any
    mismatch raises ``ValueError`` -- pass ``allow_model_mismatch=True`` to
    downgrade to a warning. ``num_heads`` is compared after applying the same
    divisor-safe resolution used at construction time.
    """

    import torch

    ckpt = torch.load(path, map_location=device)
    if not (isinstance(ckpt, dict) and "state_dict" in ckpt):
        raise ValueError(f"{path}: not a KVTok checkpoint (missing 'state_dict')")

    expected_arch = {
        "architecture_version": 2,
        **_pair_arch(pair),
        "num_spans": num_spans,
        "num_tokens": num_tokens,
        "d_model": int(d_model),
        "num_heads": _safe_num_heads(int(d_model), int(num_heads)),
    }
    ckpt_arch = dict(ckpt.get("arch", {}))
    mismatches: list[str] = []
    if ckpt.get("sharer_model") != pair.sharer_spec.name:
        mismatches.append(
            f"sharer_model: checkpoint={ckpt.get('sharer_model')!r} != pair={pair.sharer_spec.name!r}"
        )
    if ckpt.get("receiver_model") != pair.receiver_spec.name:
        mismatches.append(
            f"receiver_model: checkpoint={ckpt.get('receiver_model')!r} != pair={pair.receiver_spec.name!r}"
        )
    for key, want in expected_arch.items():
        if ckpt_arch.get(key) != want:
            mismatches.append(f"arch.{key}: checkpoint={ckpt_arch.get(key)} != expected={want}")

    if mismatches:
        detail = "\n  - ".join(mismatches)
        message = (
            f"Translator checkpoint {path} was trained for a different setup:\n  - {detail}"
        )
        if not allow_model_mismatch:
            raise ValueError(
                message
                + "\nPass --allow-model-mismatch (or allow_model_mismatch=True) to force loading."
            )
        print(f"[load_translator] WARNING (forced): {message}")

    translator = KVTokTranslator.from_arch_config(ckpt_arch, zero_init=False)
    translator.load_state_dict(ckpt["state_dict"])
    return translator


def train_translator(
    pair: KVTokPair,
    translator: KVTokTranslator,
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
) -> KVTokTranslator:
    """Train only the translator with frozen Sharer/Receiver. Returns it trained.

    Optimization (mirrors the LCF-X/EIYW recipe): AdamW, linear ``warmup_steps``,
    gradient accumulation (``grad_accum`` micro-batches per optimizer step). All of
    ``log_every`` / ``eval_every`` / ``save_every`` / ``max_steps`` count optimizer
    steps. With ``eval_data`` the best (lowest eval loss) translator is saved to
    ``checkpoint_path``; otherwise the final one. A crash-safety snapshot is
    written to ``<checkpoint_path>.latest`` every ``save_every`` steps.

    Progress logging: the full run (train loss / lr per logged step, eval loss,
    EM/F1, best markers) is persisted to ``log_path`` as JSONL — defaults to
    ``training_log.jsonl`` beside ``checkpoint_path`` — so it survives the session.
    ``eval_em_examples > 0`` adds a generation-based EM/F1 to each eval (the metric
    the paper reports; the teacher-forced loss alone hides issues like the model
    not stopping).
    """

    import torch

    _freeze(pair.sharer)
    _freeze(pair.receiver)
    # Grad-enabled generation prefill makes this heavier than a no_grad-prefill
    # design; gradient checkpointing on the frozen Receiver trades compute for memory.
    try:
        pair.receiver.gradient_checkpointing_enable()
    except Exception:  # pragma: no cover - not all models support it
        pass
    translator.train()
    torch.manual_seed(seed)
    rng = random.Random(seed)

    train_list = list(train_data)
    eval_list = list(eval_data) if eval_data is not None else None
    if not train_list:
        raise ValueError("train_data is empty")

    if log_path is None and checkpoint_path:
        log_path = str(Path(checkpoint_path).with_name("training_log.jsonl"))
    logger = ProgressLogger(
        log_path,
        config={
            "method": "kvtok",
            "architecture_version": translator.arch_config()["architecture_version"],
            "sharer_model": pair.sharer_spec.name,
            "receiver_model": pair.receiver_spec.name,
            "trainable_params": translator.trainable_parameters(),
            "num_soft_tokens": translator.num_tokens,
            "num_spans": translator.num_spans,
            "d_model": translator.d_model,
            "num_heads": translator.num_heads,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch": batch_size * grad_accum,
            "warmup": max(0, int(warmup_steps)),
            "eval_em_examples": eval_em_examples,
            "checkpoint_path": checkpoint_path,
        },
    )

    grad_accum = max(1, int(grad_accum))
    warmup = max(0, int(warmup_steps))
    optimizer = torch.optim.AdamW(
        [p for p in translator.parameters() if p.requires_grad], lr=lr
    )

    def _lr_scale(step: int) -> float:
        return (step + 1) / warmup if warmup and step < warmup else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_scale)

    print(
        f"Training translator: {len(train_list)} train"
        + (f" / {len(eval_list)} eval" if eval_list else "")
        + f" | {translator.trainable_parameters()} params | num_tokens={translator.num_tokens}"
        f" spans={translator.num_spans}"
        f" | epochs={epochs} lr={lr}"
        f" | batch={batch_size} x grad_accum={grad_accum}"
        f" (effective batch {batch_size * grad_accum}) warmup={warmup}"
    )

    opt_step = 0
    micro = 0
    best_eval = float("inf")
    saved_best = False
    running, running_count = 0.0, 0
    stop = False

    def _maybe_eval_and_save() -> None:
        nonlocal best_eval, saved_best
        if not eval_list:
            return
        eval_loss = evaluate(
            pair,
            translator,
            eval_list,
            batch_size=batch_size,
            max_examples=eval_max_examples,
        )
        em = f1 = None
        if eval_em_examples > 0:
            brief_path = (
                str(Path(checkpoint_path).with_name("eval_predictions_brief.tsv"))
                if checkpoint_path else None
            )
            scores = evaluate_em_f1(
                pair, translator, eval_list,
                max_examples=eval_em_examples, brief_path=brief_path,
            )
            em, f1 = scores["em"], scores["f1"]
        is_best = eval_loss < best_eval
        marker = ""
        if is_best:
            best_eval = eval_loss
            if checkpoint_path:
                save_translator(translator, checkpoint_path, pair)
                saved_best = True
                marker = "  (new best -> saved)"
        em_str = f" | em {em:.4f} f1 {f1:.4f}" if em is not None else ""
        print(
            f"  [eval] opt_step {opt_step} | eval_loss {eval_loss:.4f}{em_str}{marker}",
            flush=True,
        )
        logger.log(
            "eval", opt_step=opt_step, eval_loss=eval_loss, em=em, f1=f1, best=is_best
        )

    optimizer.zero_grad()
    for epoch in range(max(1, epochs)):
        order = list(range(len(train_list)))
        if shuffle:
            rng.shuffle(order)
        for chunk in _chunks(order, batch_size):
            batch = [train_list[i] for i in chunk]
            loss = batched_example_loss(pair, translator, batch, reduction="mean")
            if loss is None:
                continue
            (loss / grad_accum).backward()
            micro += 1
            running += float(loss.item())
            running_count += 1
            if micro % grad_accum != 0:
                continue

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(translator.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            opt_step += 1

            if log_every and opt_step % log_every == 0:
                train_loss = running / running_count
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"epoch {epoch + 1}/{epochs} opt_step {opt_step} "
                    f"| train_loss {train_loss:.4f} "
                    f"| lr {cur_lr:.2e}",
                    flush=True,
                )
                logger.log(
                    "train", epoch=epoch + 1, opt_step=opt_step,
                    train_loss=train_loss, lr=cur_lr,
                )
                running, running_count = 0.0, 0
            if eval_every and opt_step % eval_every == 0:
                _maybe_eval_and_save()
            if checkpoint_path and save_every and opt_step % save_every == 0:
                save_translator(translator, checkpoint_path + ".latest", pair)
            if max_steps is not None and opt_step >= max_steps:
                stop = True
                break
        _maybe_eval_and_save()
        if stop:
            break

    if checkpoint_path and not saved_best:
        save_translator(translator, checkpoint_path, pair)
        print(f"Saved translator checkpoint -> {checkpoint_path}")
    elif checkpoint_path:
        print(f"Best translator checkpoint (eval_loss {best_eval:.4f}) -> {checkpoint_path}")

    logger.log(
        "end", opt_step=opt_step,
        best_eval=(best_eval if eval_list else None),
    )
    if log_path:
        print(f"Training log -> {log_path}")
    logger.close()

    translator.eval()
    return translator
