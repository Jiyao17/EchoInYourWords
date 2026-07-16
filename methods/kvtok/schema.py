"""Data structures for KVTok.

KVTok consumes the shared cross-method data contract; ``KVTokExample`` is an
alias of ``methods.common.schema.Example`` (kept for readable, method-local
naming). Same contract as EIYW: in the cross-context setting the Sharer and
Receiver contexts differ (disjoint evidence partitions).
"""

from __future__ import annotations

from methods.common.schema import Example as KVTokExample

__all__ = ["KVTokExample"]
