from __future__ import annotations

import os
import random
import timeit
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import yaml


class UnsupportedLossTypeError(ValueError):
    pass


def str2bool(value):
    return value.lower() == "true"


def nullable_int(value):
    if value is None or str(value).lower() == "none":
        return None
    return int(value)


def train_size_type(value):
    val = float(value)
    if 0 < val < 1:
        return val
    elif val.is_integer():
        return int(val)
    else:
        raise argparse.ArgumentTypeError(
            "Train size must be an integer or a float between 0 and 1."
        )


def f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().double()


def check_dist_shapes(probs: torch.Tensor, bin_edges: torch.Tensor) -> None:
    if bin_edges.ndim != 1:
        raise ValueError(f"bin_edges must be 1-D, got shape {tuple(bin_edges.shape)}")
    if probs.shape[-1] + 1 != bin_edges.shape[0]:
        raise ValueError(
            f"probs has {probs.shape[-1]} bins but bin_edges has "
            f"{bin_edges.shape[0]} edges (expected {probs.shape[-1] + 1})"
        )


def prepare_dist(
    probs: torch.Tensor, bin_edges: torch.Tensor, eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    check_dist_shapes(probs, bin_edges)
    p = probs.float()
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    edges = bin_edges.to(device=p.device, dtype=torch.float32)
    e_lo, e_hi = edges[:-1], edges[1:]
    return p, e_lo, e_hi, e_hi - e_lo


def require_single_cate_head(model) -> None:
    from proximalfm._model.proximalfm import CATE_SINGLE_LOSS_TYPES

    loss_type = getattr(model, "loss_type", None)
    if loss_type in CATE_SINGLE_LOSS_TYPES:
        return
    if loss_type in ("bins_po", "hl"):
        raise UnsupportedLossTypeError(
            f"loss_type={loss_type!r} predicts two per-arm histograms, not a "
            "distribution over tau."
        )
    raise UnsupportedLossTypeError(
        f"loss_type={loss_type!r} is not supported for distributional metrics."
    )


def seed_dataloader_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**31 - 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def is_oom_error(exc: BaseException) -> bool:
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
    device = str(device)
    if "cuda" in device:
        torch.cuda.empty_cache()
    elif "mps" in device and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def get_parameter_state(param_name: str, rules: dict) -> str:
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
    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False
