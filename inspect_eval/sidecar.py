"""Sidecar identity for replicate-bearing ``inspect_eval`` sets."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from synt_eval.realistic_sidecar import DEFAULT_N_GRID, episode_records

INSPECT_META_VERSION = 1
SET_KIND = "inspect"
META_NAME = "frozen_meta.pt"


class InspectSetConflictError(RuntimeError):
    """Raised when a build would overwrite a different named set."""


@dataclass
class InspectBuildConfig:
    n_datasets: int
    seq_len: int
    prop_train: float
    mc_cate: int
    seed: int
    overlap_eps: float = 0.02
    extra_observation_noise: float = 0.0
    shuffle: bool = True
    mechanism: str = "linear_gaussian"
    mechanism_version: int = 1
    gen_device: str = "cpu"

    def n_grid(self, n_max: int) -> List[int]:
        grid = [n for n in DEFAULT_N_GRID if n <= n_max]
        if not grid or grid[-1] != n_max:
            grid.append(n_max)
        return grid

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "set_kind": SET_KIND}

    def content_digest(self) -> str:
        payload = {"format_version": INSPECT_META_VERSION, **self.to_dict()}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _git_rev() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def inspect_content_digest(meta: Dict[str, Any]) -> Optional[str]:
    g = meta.get("global", {})
    if g.get("content_digest") is not None:
        return g["content_digest"]
    fields = set(InspectBuildConfig.__dataclass_fields__)
    try:
        return InspectBuildConfig(
            **{k: v for k, v in g.items() if k in fields}
        ).content_digest()
    except TypeError:
        return None


def check_no_silent_mutation(
    out: Path, cfg: InspectBuildConfig, force: bool = False
) -> None:
    path = Path(out) / META_NAME
    if force or not path.exists():
        return
    previous = torch.load(path, weights_only=False)
    old = inspect_content_digest(previous)
    new = cfg.content_digest()
    if old != new:
        raise InspectSetConflictError(
            f"{out} already holds inspect set {old}; this build would write {new}. "
            "Choose a new --name or pass --force."
        )


def write_sidecar(
    out: Path,
    cfg: InspectBuildConfig,
    records: Sequence[Dict[str, Any]],
    n_max: int,
    datasets_per_file: int,
    cli_command: Optional[str] = None,
) -> str:
    out = Path(out)
    digest = cfg.content_digest()
    meta = {
        "format_version": INSPECT_META_VERSION,
        "global": {
            **cfg.to_dict(),
            "n_max": int(n_max),
            "n_q": int(cfg.seq_len - n_max),
            "n_grid": cfg.n_grid(n_max),
            "content_digest": digest,
            "datasets_per_file": int(datasets_per_file),
            "cli_command": cli_command,
            "git_rev": _git_rev(),
        },
        "episodes": list(records),
    }
    tmp = out / f"{META_NAME}.tmp"
    torch.save(meta, tmp)
    tmp.replace(out / META_NAME)
    return digest


__all__ = [
    "InspectBuildConfig",
    "InspectSetConflictError",
    "check_no_silent_mutation",
    "episode_records",
    "inspect_content_digest",
    "write_sidecar",
]
