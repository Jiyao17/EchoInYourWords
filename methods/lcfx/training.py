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
validation, final-checkpoint persistence, gradient clipping, and a seed.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Iterable

from methods.common.decoding import resolve_answer_stop_id
from methods.common.progress import ProgressLogger
from methods.common.prompts import build_receiver_prompt, build_sharer_encode_text

from .cache import add_residual_inplace, extract_kv
from .fuser import LCFXFuser
from .model import LCFXPair
from .schema import LCFXExample
from .spans import paragraph_span_starts


def _freeze(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()


def _as_example(raw: Any) -> LCFXExample:
    return raw if isinstance(raw, LCFXExample) else LCFXExample.from_mapping(raw)


def _boundaries(
    pair: LCFXPair,
    text: str,
    span_scheme: str,
    context: str | None = None,
) -> "list[int] | None":
    """Span starts for one sharer text: paragraph blocks (paper's training
    scheme) or None (fixed near-equal chunks of ``num_spans``)."""

    if span_scheme == "paragraphs":
        return paragraph_span_starts(pair.sharer_tok, text, context)
    return None


def example_loss(
    pair: LCFXPair,
    fuser: LCFXFuser,
    example: LCFXExample,
    num_spans: int,
    span_scheme: str = "paragraphs",
) -> "Any | None":
    """Teacher-forced next-token CE for one example (None if it can't be scored).

    Caller controls grad: wrap in ``torch.no_grad()`` for evaluation; call plainly
    for training (the prompt prefill is always ``no_grad`` internally).
    """

    import torch
    from torch.nn import functional as F

    if not example.answer:
        return None

    sharer_text = build_sharer_encode_text(example)
    summary, _ = pair.sharer_summary(sharer_text)
    aligned = fuser.align(
        summary,
        num_spans=num_spans,
        span_boundaries=[
            _boundaries(pair, sharer_text, span_scheme, example.sharer_context)
        ],
    )

    prompt_ids = pair._encode_chat(pair.receiver_tok, build_receiver_prompt(example))
    answer_ids = pair.receiver_tok(
        example.answer, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(pair.device)
    if prompt_ids.shape[1] < 2 or answer_ids.shape[1] == 0:
        return None

    # Append the stop token to the *target*, not the input: the forward below
    # already produces one logit position past the last answer token (it
    # predicts "what comes next" from the last real token, same as any other
    # position) -- we just weren't using it. Supervising it as `stop_id`
    # teaches the fuser to end the turn instead of running on forever; without
    # this the model never receives gradient on the answer->EOS transition, so
    # inference (which stops only on an actually-generated stop token, see
    # ``resolve_stop_ids``) runs to ``max_new_tokens`` instead of stopping
    # cleanly after the answer (EM craters, F1 only partially drops -- see
    # ``prompt/EOS_STOP_TOKEN_FIX.md``).
    stop_id = resolve_answer_stop_id(pair.receiver_tok, pair.receiver)
    target_ids = answer_ids
    if stop_id is not None:
        stop_tensor = torch.full((1, 1), stop_id, dtype=answer_ids.dtype, device=pair.device)
        target_ids = torch.cat([answer_ids, stop_tensor], dim=1)

    # Prefill prompt[:-1] (no grad): Receiver KV is a detached projector input.
    with torch.no_grad():
        cache = pair.receiver(input_ids=prompt_ids[:, :-1], use_cache=True).past_key_values

    residual = fuser.project(aligned, extract_kv(cache))
    add_residual_inplace(cache, residual)

    # Teacher-forced answer forward: [p_last, a_0, ..., a_{T-1}] over the fused
    # cache. Deliberately NOT feeding stop_id as input -- the position after
    # a_{T-1} already yields the "what comes next" logit we target above.
    decode_input = torch.cat([prompt_ids[:, -1:], answer_ids], dim=1)
    logits = pair.receiver(
        input_ids=decode_input, past_key_values=cache, use_cache=True
    ).logits[:, : target_ids.shape[1], :]

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), target_ids.reshape(-1)
    )


def _pad(seqs: list, pad_id: int, side: str) -> tuple:
    """Pad a list of 1-D token tensors to a batch; returns (ids, attention_mask)."""

    import torch

    length = max(int(s.numel()) for s in seqs)
    device = seqs[0].device
    ids = torch.full((len(seqs), length), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), length), dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        n = int(seq.numel())
        if side == "left":
            ids[i, length - n :] = seq
            mask[i, length - n :] = 1
        else:
            ids[i, :n] = seq
            mask[i, :n] = 1
    return ids, mask


def batched_example_loss(
    pair: LCFXPair,
    fuser: LCFXFuser,
    examples: list,
    num_spans: int,
    *,
    span_scheme: str = "paragraphs",
    reduction: str = "mean",
) -> "Any | None":
    """Teacher-forced loss for a batch of examples (the same fusion as the single path).

    Sharer prompts are right-padded (pooled over real tokens only); Receiver
    prompts are left-padded so the last real token shares a column; answers
    (each with a stop token appended -- see below) are right-padded and masked
    out of the loss. ``reduction='none'`` returns a per-example loss vector
    ``[B]`` (used for the batched==unbatched check).
    """

    import torch
    from torch.nn import functional as F

    # Same stop-token supervision as `example_loss` (see its comment), appended
    # per-example *before* padding: `_pad` then naturally places it right after
    # each example's own real answer tokens, and everything downstream (the
    # decode-input length, the logits slice, the mask-based targets) already
    # generalizes to the wider tensors with no further changes needed.
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
        sharer_text = build_sharer_encode_text(ex)
        sharer = pair.sharer_tok(sharer_text, return_tensors="pt")["input_ids"][0].to(
            pair.device
        )
        if prompt.numel() >= 2 and answer.numel() >= 1:
            rows.append(
                (
                    prompt,
                    answer,
                    sharer,
                    _boundaries(
                        pair, sharer_text, span_scheme, ex.sharer_context
                    ),
                )
            )
    if not rows:
        return None

    prompts, answers, sharers, boundaries = zip(*rows)
    r_pad = pair.receiver_tok.pad_token_id
    s_pad = pair.sharer_tok.pad_token_id

    # Sharer: batched forward; the fuser's per-layer 2-stage pooling reduces over
    # each example's real tokens (via lengths) -> per-layer Sharer summaries.
    sharer_ids, sharer_mask = _pad(list(sharers), s_pad, "right")
    with torch.no_grad():
        sharer_cache = pair.sharer(
            input_ids=sharer_ids, attention_mask=sharer_mask, use_cache=True
        ).past_key_values
    aligned = fuser.align(
        extract_kv(sharer_cache),
        lengths=sharer_mask.sum(dim=1),
        num_spans=num_spans,
        span_boundaries=list(boundaries),
    )

    # Receiver: left-pad prompts so prompt[:, -1] is the last real token for all.
    # With left-padding the real tokens must get positions 0,1,2,... (not shifted
    # by the pad offset), so pass explicit position_ids from the mask.
    prompt_ids, prompt_mask = _pad(list(prompts), r_pad, "left")
    prefix_mask = prompt_mask[:, :-1]
    prefix_pos = (prefix_mask.cumsum(dim=-1) - 1).clamp(min=0)
    with torch.no_grad():
        cache = pair.receiver(
            input_ids=prompt_ids[:, :-1],
            attention_mask=prefix_mask,
            position_ids=prefix_pos,
            use_cache=True,
        ).past_key_values
    residual = fuser.project(aligned, extract_kv(cache))
    add_residual_inplace(cache, residual)

    # Teacher-forced answer forward: [p_last, a_0, ...] over the fused cache.
    answer_ids, answer_mask = _pad(list(answers), r_pad, "right")
    decode_input = torch.cat([prompt_ids[:, -1:], answer_ids], dim=1)
    full_mask = torch.cat([prompt_mask, answer_mask], dim=1)  # past(P-1)+new(1+A) = P+A
    position_ids = (full_mask.cumsum(dim=-1) - 1).clamp(min=0)[:, -decode_input.shape[1] :]
    logits = pair.receiver(
        input_ids=decode_input,
        attention_mask=full_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    ).logits[:, : answer_ids.shape[1], :]

    targets = answer_ids.masked_fill(answer_mask == 0, -100)
    token_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(targets.shape)
    per_example = token_ce.sum(dim=1) / answer_mask.sum(dim=1).clamp(min=1)
    return per_example if reduction == "none" else per_example.mean()


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _optimization_steps(
    num_examples: int,
    batch_size: int,
    grad_accum: int,
    epochs: int,
    max_steps: int | None,
) -> int:
    """Resolved optimizer-step budget, including a final partial accumulation."""

    micro_batches = max(1, -(-num_examples // max(1, batch_size)))
    per_epoch = max(1, -(-micro_batches // max(1, grad_accum)))
    available = max(1, max(1, epochs) * per_epoch)
    return min(available, max_steps) if max_steps is not None else available


def _warmup_steps(total_steps: int, requested: int | None) -> int:
    """Explicit warmup or 10% of the resolved run, rounded to a whole step."""

    if requested is not None:
        return max(0, int(requested))
    return max(1, int(0.10 * total_steps + 0.5))


def evaluate(
    pair: LCFXPair,
    fuser: LCFXFuser,
    eval_data: list[Any],
    *,
    num_spans: int = 5,
    span_scheme: str = "paragraphs",
    batch_size: int = 16,
    max_examples: int | None = None,
) -> float:
    """Mean per-example teacher-forced loss over an eval set (lower is better)."""

    import torch

    items = list(eval_data)
    if max_examples is not None:
        items = items[:max_examples]

    was_training = fuser.training
    fuser.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for chunk in _chunks(items, batch_size):
            per_example = batched_example_loss(
                pair, fuser, chunk, num_spans, span_scheme=span_scheme, reduction="none"
            )
            if per_example is not None:
                total += float(per_example.sum().item())
                count += int(per_example.numel())
    if was_training:
        fuser.train()
    return total / count if count else float("nan")


def evaluate_em_f1(
    pair: LCFXPair,
    fuser: LCFXFuser,
    eval_data: list[Any],
    *,
    num_spans: int = 5,
    span_scheme: str = "paragraphs",
    max_new_tokens: int = 64,
    max_examples: int | None = None,
    brief_path: "str | Any | None" = None,
) -> dict[str, float]:
    """Generation-based EM/F1 over the eval set (the metric the paper reports).

    Unlike :func:`evaluate` (teacher-forced loss only), this runs **real greedy
    decoding** through the full fusion path, so it reflects EM/F1 — and surfaces
    failures the loss hides (e.g. the model not stopping after the answer).
    Returns ``{"em", "f1", "n"}``. Slower than the loss eval, so cap it with
    ``max_examples``. With ``brief_path`` set, the per-example
    answer/prediction/EM/F1 are written there as a TSV (a training-time snapshot,
    same format as the inference ``predictions_brief.tsv``).
    """

    import torch

    from methods.common.io import write_predictions_brief
    from methods.common.scoring import score_free_form

    from .runner import run_lcfx_example

    items = [_as_example(e) for e in eval_data]
    if max_examples is not None:
        items = items[:max_examples]

    was_training = fuser.training
    fuser.eval()
    results = []
    em_sum = f1_sum = 0.0
    n = 0
    with torch.no_grad():
        for ex in items:
            if not ex.answer:
                continue
            result = run_lcfx_example(
                ex, pair, fuser, num_spans=num_spans, span_scheme=span_scheme,
                max_new_tokens=max_new_tokens, score_fn=score_free_form,
            )
            results.append(result)
            em_sum += float(result.scores.get("em", 0.0))
            f1_sum += float(result.scores.get("f1", 0.0))
            n += 1
    if brief_path is not None:
        write_predictions_brief(results, brief_path)
    if was_training:
        fuser.train()
    return {
        "em": em_sum / n if n else float("nan"),
        "f1": f1_sum / n if n else float("nan"),
        "n": n,
    }


# Format 4: paper-stated z -> (z_K, z_V) split, d/2-input K/V up-projections,
# and dropout stamped in the architecture. Format 3 used full-d up-projections
# to match the paper's inconsistent parameter table and is shape-incompatible.
CHECKPOINT_FORMAT = 4


def _pair_arch(pair: LCFXPair) -> dict:
    """Architecture the fuser must have for this Sharer/Receiver pair (sans bottleneck)."""

    s, r = pair.sharer_spec, pair.receiver_spec
    return {
        "sharer_layers": s.layers,
        "sharer_kv_heads": s.kv_heads,
        "sharer_head_dim": s.head_dim,
        "receiver_layers": r.layers,
        "receiver_kv_heads": r.kv_heads,
        "receiver_head_dim": r.head_dim,
    }


def save_fuser(
    fuser: LCFXFuser,
    path: str,
    pair: LCFXPair,
    num_spans: int | None = None,
    span_scheme: str | None = None,
) -> None:
    """Save the fuser stamped with the Sharer/Receiver it was trained for.

    The payload is primitive-only (str/int + the tensor state_dict) so it loads
    cleanly under ``torch.load(..., weights_only=True)``. ``load_fuser`` validates
    the stamped model names + architecture against the pair used at eval time.
    """

    import torch

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "state_dict": fuser.state_dict(),
        "arch": fuser.arch_config(),
        "sharer_model": pair.sharer_spec.name,
        "receiver_model": pair.receiver_spec.name,
        "num_spans": num_spans,
        "span_scheme": span_scheme,
    }
    if hasattr(fuser, "training_seed"):
        payload["training_seed"] = fuser.training_seed
    if hasattr(fuser, "training_history"):
        payload["training_history"] = fuser.training_history
    torch.save(payload, path)


def load_fuser(
    path: str,
    pair: LCFXPair,
    *,
    device: Any = None,
    allow_model_mismatch: bool = False,
    bottleneck_dim: int = 128,
    dropout: float = 0.1,
) -> LCFXFuser:
    """Load a trained fuser, validating it matches the Sharer/Receiver ``pair``.

    New-format checkpoints (written by :func:`save_fuser`) carry the model names and
    architecture. By default any mismatch against ``pair`` raises ``ValueError`` --
    pass ``allow_model_mismatch=True`` to downgrade to a warning. Checkpoints from
    before the paper-stated latent-split rewrite (format < 4, including legacy
    bare ``state_dict`` files) have incompatible shapes and are rejected.
    """

    import torch

    ckpt = torch.load(path, map_location=device)

    if not (isinstance(ckpt, dict) and "state_dict" in ckpt):
        raise ValueError(
            f"Fuser checkpoint {path} is a legacy bare state_dict from before the "
            "paper-stated latent-split rewrite and cannot be loaded. Retrain with "
            "scripts/train_lcfx_fuser.py."
        )
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"Fuser checkpoint {path} has format {ckpt.get('format')}, expected "
            f"{CHECKPOINT_FORMAT}. Earlier formats use incompatible projector "
            "shapes. Retrain with scripts/train_lcfx_fuser.py."
        )

    expected_arch = {
        **_pair_arch(pair),
        "bottleneck_dim": bottleneck_dim,
        "dropout": float(dropout),
    }
    mismatches: list[str] = []
    if ckpt.get("sharer_model") != pair.sharer_spec.name:
        mismatches.append(
            f"sharer_model: checkpoint={ckpt.get('sharer_model')!r} != pair={pair.sharer_spec.name!r}"
        )
    if ckpt.get("receiver_model") != pair.receiver_spec.name:
        mismatches.append(
            f"receiver_model: checkpoint={ckpt.get('receiver_model')!r} != pair={pair.receiver_spec.name!r}"
        )
    ckpt_arch = ckpt.get("arch", {})
    for key, want in expected_arch.items():
        if ckpt_arch.get(key) != want:
            mismatches.append(f"arch.{key}: checkpoint={ckpt_arch.get(key)} != pair={want}")

    if mismatches:
        detail = "\n  - ".join(mismatches)
        message = (
            f"Fuser checkpoint {path} was trained for a different setup:\n  - {detail}"
        )
        if not allow_model_mismatch:
            raise ValueError(
                message
                + "\nPass --allow-model-mismatch (or allow_model_mismatch=True) to force loading."
            )
        print(f"[load_fuser] WARNING (forced): {message}")

    fuser = LCFXFuser.from_arch_config(ckpt_arch, zero_init=False)
    fuser.load_state_dict(ckpt["state_dict"])
    if "training_seed" in ckpt:
        fuser.training_seed = int(ckpt["training_seed"])
    if "training_history" in ckpt:
        fuser.training_history = ckpt["training_history"]
    return fuser


def train_fuser(
    pair: LCFXPair,
    fuser: LCFXFuser,
    train_data: Iterable[Any],
    *,
    eval_data: Iterable[Any] | None = None,
    num_spans: int = 5,
    span_scheme: str = "paragraphs",
    lr: float = 1e-4,
    epochs: int = 1,
    batch_size: int = 16,
    grad_accum: int = 1,
    warmup_steps: int | None = None,
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
) -> LCFXFuser:
    """Train only the fuser with frozen Sharer/Receiver. Returns the trained fuser.

    Optimization (matches the LCF-X paper's recipe):
    - **AdamW**, ``lr`` with a linear warmup over 10% of the resolved optimizer
      steps unless ``warmup_steps`` is explicitly provided.
    - **Gradient accumulation**: ``grad_accum`` micro-batches of ``batch_size``
      examples per optimizer step, so the effective batch size is
      ``batch_size * grad_accum``. All of
      ``log_every`` / ``eval_every`` / ``save_every`` / ``max_steps`` count
      **optimizer steps**, not micro-batches.

    Checkpointing:
    - The final checkpoint is saved to ``checkpoint_path``, matching Appendix
      E.2 of the paper.
    - Every ``save_every`` optimizer steps a crash-safety snapshot is written to
      ``<checkpoint_path>.latest``.

    Progress logging:
    - The full run (train loss / lr per logged step, eval loss, EM/F1, best
      markers) is persisted to ``log_path`` as JSONL — defaults to
      ``training_log.jsonl`` beside ``checkpoint_path`` — so it survives the
      session. ``eval_em_examples > 0`` adds a generation-based EM/F1 to each eval
      (the metric the paper reports; the teacher-forced loss alone hides issues
      like the model not stopping).
    """

    import torch

    _freeze(pair.sharer)
    _freeze(pair.receiver)
    fuser.train()
    torch.manual_seed(seed)
    fuser.training_seed = int(seed)
    rng = random.Random(seed)

    train_list = list(train_data)
    eval_list = list(eval_data) if eval_data is not None else None
    if not train_list:
        raise ValueError("train_data is empty")

    grad_accum = max(1, int(grad_accum))
    total_steps = _optimization_steps(
        len(train_list), batch_size, grad_accum, epochs, max_steps
    )
    warmup = _warmup_steps(total_steps, warmup_steps)

    def _gate_tau(step: int) -> float:
        if total_steps <= 1:
            return 0.5
        return 1.0 - 0.5 * min(1.0, step / (total_steps - 1))

    if log_path is None and checkpoint_path:
        log_path = str(Path(checkpoint_path).with_name("training_log.jsonl"))
    logger = ProgressLogger(
        log_path,
        config={
            "method": "lcf_x",
            "seed": int(seed),
            "sharer_model": pair.sharer_spec.name,
            "receiver_model": pair.receiver_spec.name,
            "trainable_params": fuser.trainable_parameters(),
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch": batch_size * grad_accum,
            "warmup": warmup,
            "num_spans": num_spans,
            "span_scheme": span_scheme,
            "bottleneck_dim": fuser.bottleneck_dim,
            "dropout": fuser.dropout,
            "eval_em_examples": eval_em_examples,
            "checkpoint_path": checkpoint_path,
            "training_history": getattr(fuser, "training_history", None),
            "total_steps": total_steps,
        },
    )

    trainable = [p for p in fuser.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    def _lr_scale(step: int) -> float:
        return (step + 1) / warmup if warmup and step < warmup else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_scale)

    print(
        f"Training fuser: {len(train_list)} train"
        + (f" / {len(eval_list)} eval" if eval_list else "")
        + f" | {fuser.trainable_parameters()} params | epochs={epochs} lr={lr}"
        f" | batch={batch_size} x grad_accum={grad_accum}"
        f" (effective batch {batch_size * grad_accum}) warmup={warmup}"
    )

    opt_step = 0
    accumulated = 0
    best_eval = float("inf")
    running, running_count = 0.0, 0
    stop = False

    def _maybe_evaluate() -> None:
        nonlocal best_eval
        if not eval_list:
            return
        eval_loss = evaluate(
            pair,
            fuser,
            eval_list,
            num_spans=num_spans,
            span_scheme=span_scheme,
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
                pair, fuser, eval_list, num_spans=num_spans, span_scheme=span_scheme,
                max_examples=eval_em_examples, brief_path=brief_path,
            )
            em, f1 = scores["em"], scores["f1"]
        is_best = eval_loss < best_eval
        if is_best:
            best_eval = eval_loss
        em_str = f" | em {em:.4f} f1 {f1:.4f}" if em is not None else ""
        marker = " | best" if is_best else ""
        print(
            f"  [eval] opt_step {opt_step} | eval_loss {eval_loss:.4f}{em_str}{marker}",
            flush=True,
        )
        logger.log(
            "eval", opt_step=opt_step, eval_loss=eval_loss, em=em, f1=f1, best=is_best
        )

    def _finish_optimizer_step(accum_count: int, epoch: int) -> None:
        """Apply one full or final-partial accumulated update."""

        nonlocal opt_step, running, running_count, stop
        if accum_count < grad_accum:
            # Each micro-loss was divided by grad_accum. Renormalize a final
            # partial group so its gradient is still the mean of that group.
            correction = grad_accum / accum_count
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
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
                f"| lr {cur_lr:.2e} | gate_tau {fuser.gate_tau:.3f}",
                flush=True,
            )
            logger.log(
                "train", epoch=epoch + 1, opt_step=opt_step,
                train_loss=train_loss, lr=cur_lr, gate_tau=fuser.gate_tau,
            )
            running, running_count = 0.0, 0
        if eval_every and opt_step % eval_every == 0:
            _maybe_evaluate()
        if checkpoint_path and save_every and opt_step % save_every == 0:
            save_fuser(
                fuser, checkpoint_path + ".latest", pair, num_spans, span_scheme
            )
        if opt_step >= total_steps:
            stop = True

    optimizer.zero_grad()
    for epoch in range(max(1, epochs)):
        order = list(range(len(train_list)))
        if shuffle:
            rng.shuffle(order)
        for chunk in _chunks(order, batch_size):
            batch = [train_list[i] for i in chunk]
            fuser.gate_tau = _gate_tau(opt_step)
            loss = batched_example_loss(
                pair, fuser, batch, num_spans, span_scheme=span_scheme, reduction="mean"
            )
            if loss is None:
                continue
            (loss / grad_accum).backward()
            accumulated += 1
            running += float(loss.item())
            running_count += 1
            if accumulated < grad_accum:
                continue
            _finish_optimizer_step(accumulated, epoch)
            accumulated = 0
            if stop:
                break
        if accumulated and not stop:
            _finish_optimizer_step(accumulated, epoch)
            accumulated = 0
        # End-of-epoch validation.
        _maybe_evaluate()
        if stop:
            break

    # Appendix E.2 evaluates the final one-epoch checkpoint.
    if checkpoint_path:
        save_fuser(fuser, checkpoint_path, pair, num_spans, span_scheme)
        print(f"Saved fuser checkpoint -> {checkpoint_path}")

    logger.log(
        "end", opt_step=opt_step,
        best_eval=(best_eval if eval_list else None),
    )
    if log_path:
        print(f"Training log -> {log_path}")
    logger.close()

    fuser.eval()
    return fuser
