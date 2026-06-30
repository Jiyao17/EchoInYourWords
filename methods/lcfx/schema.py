"""Data structures for LCF-X.

LCF-X consumes the shared cross-method data contract; ``LCFXExample`` is an alias
of ``methods.common.schema.Example`` (kept for readable, method-local naming). In
the cross-context setting the Sharer and Receiver contexts differ (disjoint
evidence partitions).
"""

from __future__ import annotations

from methods.common.schema import Example as LCFXExample

__all__ = ["LCFXExample"]
