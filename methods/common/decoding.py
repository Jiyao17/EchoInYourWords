"""Shared greedy-decoding helpers for the method receivers.

The method wrappers (LCF-X, EIYW) hand-roll a greedy loop instead of calling
``model.generate`` — they must inject a KV residual / soft-token prefill that the
HF ``generate`` API does not expose cleanly. That means they also have to
replicate ``generate``'s stop condition themselves.

The subtlety this addresses: a chat model's turn terminator is not always the
tokenizer's ``eos_token_id``. Qwen ends a chat turn with ``<|im_end|>`` while
listing *both* ``<|im_end|>`` and ``<|endoftext|>`` in
``generation_config.eos_token_id``. A loop that checks only ``tok.eos_token_id``
happens to work for Qwen (that field is ``<|im_end|>``) but is fragile for
heterogeneous receivers whose chat terminator lives only in the generation
config. Collecting the full stop set — as ``generate`` does — makes the loops
stop on any configured terminator.
"""

from __future__ import annotations

from typing import Any


def resolve_stop_ids(tokenizer: Any, model: Any | None = None) -> set[int]:
    """Return every token id that should terminate greedy decoding.

    Unions the tokenizer's ``eos_token_id`` with the model's
    ``generation_config.eos_token_id`` (either may be an int or a list), matching
    what ``model.generate`` treats as a stop token.
    """

    stop: set[int] = set()

    def _add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _add(item)
        else:
            stop.add(int(value))

    _add(getattr(tokenizer, "eos_token_id", None))
    gen_config = getattr(model, "generation_config", None)
    if gen_config is not None:
        _add(getattr(gen_config, "eos_token_id", None))

    return stop


def resolve_answer_stop_id(tokenizer: Any, model: Any | None = None) -> int | None:
    """Pick one stop-token id to teach as "what comes after the answer".

    Training supervises a single next-token target per position, so unlike
    :func:`resolve_stop_ids` (which returns every id ``generate``-style
    decoding should treat as a stop), this picks exactly one: the
    tokenizer's own ``eos_token_id`` when set -- already the correct
    chat-turn terminator for every receiver this project uses (see the note
    above on Qwen) -- else an arbitrary member of the full resolved stop
    set, so a receiver whose tokenizer lacks ``eos_token_id`` still gets
    *some* terminator supervision instead of none. Different receivers can
    and do have different ids here; always resolve per-receiver, never
    hardcode one.
    """

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    stop_ids = resolve_stop_ids(tokenizer, model)
    return next(iter(stop_ids), None)
