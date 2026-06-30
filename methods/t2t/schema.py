"""Data structures for the text-to-text baseline.

T2T consumes the shared cross-method data contract; ``T2TExample`` is an alias of
``methods.common.schema.Example`` (kept for readable, method-local naming).
"""

from __future__ import annotations

from methods.common.schema import Example as T2TExample

__all__ = ["T2TExample"]
