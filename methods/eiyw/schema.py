"""Data structures for EIYW (the receiver-anchored method).

EIYW consumes the same normalized example as every other method, so we reuse the
shared ``T2TExample`` (the project's data contract) under an EIYW-friendly name
rather than duplicate it. In the cross-context setting the Sharer and Receiver
contexts differ (disjoint evidence partitions).
"""

from __future__ import annotations

from methods.t2t.schema import T2TExample as EIYWExample

__all__ = ["EIYWExample"]
