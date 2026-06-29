"""Default Sharer/Receiver models, shared by every method/baseline entry point.

The project always treats the Sharer and the Receiver as two *independent*
(potentially heterogeneous) models -- every run takes both a ``--sharer-model``
and a ``--receiver-model``. When a run does not specify them, it falls back to
this symmetric Qwen pair; specifying only one still yields a well-defined pair
(the other stays at its Qwen default rather than mirroring the one you set).

Override per run via the scripts' ``--sharer-model`` / ``--receiver-model`` flags
or the Makefile's ``SHARER`` / ``RECEIVER`` variables.
"""

DEFAULT_SHARER = "Qwen/Qwen3-0.6B"
DEFAULT_RECEIVER = "Qwen/Qwen3-0.6B"
