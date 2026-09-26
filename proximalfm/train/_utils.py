"""Generic training-loop infrastructure used by ``proximalfm.train._run.Trainer``.

Nothing here depends on ``Trainer`` or the model; each helper is a standalone
utility (seeding, OOM handling, checkpoint transfer rules, timing).
"""

from __future__ import annotations

import os
import random
import timeit

import numpy as np
import torch
import yaml


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed NumPy (and torch/random) independently in each dataloader worker.

    PyTorch reseeds ``torch`` and ``random`` per worker but NOT NumPy, and the SCM
    prior draws dimensions / hyperparameters via ``np.random`` in many places. Left
    unseeded, workers can produce correlated (partially duplicated) datasets. Derive
    every generator's seed from ``torch.initial_seed()``, which PyTorch already makes
    unique per worker, so this stays reproducible and decorrelated.
    """
    worker_seed = torch.initial_seed() % (2**31 - 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def is_oom_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is an out-of-memory error from any accelerator.

    Covers CUDA (``torch.cuda.OutOfMemoryError``, a ``RuntimeError`` subclass) as
    well as MPS / generic backends, which surface OOM as a plain ``RuntimeError``
    whose message mentions running out of memory. Note that *asynchronous* MPS
    command-buffer failures (kIOGPUCommandBufferCallbackErrorOutOfMemory) may not
    be raised at the originating op and can leave the context unusable; those are
    not reliably recoverable even when caught here.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "insufficient memory" in msg
            or "mps backend out of memory" in msg
        )
    return False


def empty_accelerator_cache(device: str) -> None:
    """Free cached allocator memory on the active accelerator (CUDA or MPS)."""
    device = str(device)
    if "cuda" in device:
        torch.cuda.empty_cache()
    elif "mps" in device and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def get_parameter_state(param_name: str, rules: dict) -> str:
    """Finds the longest matching prefix in the rules dict for a given parameter name."""
    best_match = "default"
    longest_prefix = 0
    for prefix, state in rules.items():
        if prefix == "default":
            continue
        if param_name.startswith(prefix) and len(prefix) > longest_prefix:
            best_match = prefix
            longest_prefix = len(prefix)
    return rules.get(best_match, rules.get("default", "scratch"))


def load_transfer_rules(path: str) -> dict:
    """Load the transfer/freeze rules yaml.

    Resolved relative to the CWD first, then to the repository root, so that a run
    launched from a subdirectory picks up the same file rather than silently failing
    or using a different one.
    """
    candidates = [path]
    if not os.path.isabs(path):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        candidates.append(os.path.join(repo_root, path))

    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, "r") as f:
                rules_config = yaml.safe_load(f) or {}
            return rules_config.get("transfer_rules", {})

    raise FileNotFoundError(
        f"Transfer rules file not found: tried {', '.join(candidates)}"
    )


class Timer:
    """Context manager for timing code execution."""

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False  # Don't suppress exceptions
