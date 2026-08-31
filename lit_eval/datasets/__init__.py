"""Registered literature mechanisms.

Adding a mechanism is intentionally one module plus one import here. The
shared builder, validation, replicate layout and on-disk format do not need a
dataset-specific branch.
"""

from lit_eval.datasets.sverdrup_appb import SverdrupAppB  # noqa: F401

__all__ = ["SverdrupAppB"]
