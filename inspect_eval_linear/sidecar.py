"""Frozen sidecar schema for linear-Gaussian inspection sets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional, Sequence

import torch

from .generator import LinearGeneratorConfig, proxy_reliability

META_VERSION = 1
SET_KIND = "inspect_linear"
META_NAME = "frozen_meta.pt"
Y_STD_FLOOR = 1e-3
DEFAULT_N_GRID = (32, 64, 128, 256, 512, 1024, 2048)


class LinearSetConflictError(RuntimeError):
    """Raised before replacing a named set with different content."""


@dataclass(frozen=True)
class LinearBuildConfig:
    n_datasets: int
    seq_len: int
    prop_train: float
    mc_cate: int
    seed: int
    proxy_noise_stds: list[float]
    shuffle: bool = True
    gen_device: str = "cpu"
    beta_ux: float = 0.5
    sigma_u: float = 1.0
    proxy_loading: float = 1.0
    alpha_0: float = 0.0
    alpha_x: float = 0.25
    alpha_u: float = 1.0
    sigma_a: float = 1.0
    beta_0: float = 0.0
    beta_x: float = 0.5
    beta_u: float = 1.0
    sigma_y: float = 1.0
    tau_0: float = 1.0
    tau_x: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        values["proxy_noise_stds"] = [float(v) for v in self.proxy_noise_stds]
        values["set_kind"] = SET_KIND
        return values

    def content_digest(self) -> str:
        payload = {"format_version": META_VERSION, **self.to_dict()}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def generator_config(self, mc_mem_budget: int) -> LinearGeneratorConfig:
        names = set(LinearGeneratorConfig.__dataclass_fields__)
        values = {k: v for k, v in asdict(self).items() if k in names}
        values["mc_mem_budget"] = mc_mem_budget
        return LinearGeneratorConfig(**values)


def _git_rev() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent.parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def frozen_scale(y_obs: torch.Tensor, n_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    context = y_obs.reshape(-1)[:n_max].float()
    return context.mean(), context.std(unbiased=False).clamp_min(Y_STD_FLOOR)


def episode_records(
    pool: Sequence[torch.Tensor],
    fields: Sequence[str],
    noise_stds: torch.Tensor,
    seeds: torch.Tensor,
    n_max: int,
    cfg: LinearBuildConfig,
) -> list[dict[str, Any]]:
    p = dict(zip(fields, pool))
    records = []
    coefficients = {
        name: getattr(cfg, name)
        for name in (
            "beta_ux",
            "sigma_u",
            "proxy_loading",
            "alpha_0",
            "alpha_x",
            "alpha_u",
            "sigma_a",
            "beta_0",
            "beta_x",
            "beta_u",
            "sigma_y",
            "tau_0",
            "tau_x",
        )
    }
    for i in range(len(seeds)):
        y_mean, y_std = frozen_scale(p["Y_obs"][i], n_max)
        tau_disk = (p["MU1"][i, n_max:] - p["MU0"][i, n_max:]).reshape(-1).float()
        ladder = {
            name: p[name][i, n_max:].reshape(-1).float()
            for name in (
                "Q_naive_X",
                "Q_adj_W",
                "Q_adj_Z",
                "Q_adj_WZ",
                "Q_true",
            )
        }
        if not torch.allclose(ladder["Q_true"], tau_disk, atol=1e-5, rtol=1e-5):
            raise ValueError(f"episode {i}: Q_true does not match MU1-MU0")
        noise = float(noise_stds[i])
        reliability = proxy_reliability(
            noise, sigma_u=cfg.sigma_u, proxy_loading=cfg.proxy_loading
        )
        s2 = p["S2"][i, n_max:].reshape(-1).float()
        records.append(
            {
                "index": i,
                "seed": int(seeds[i]),
                "proxy_noise_std": noise,
                "proxy_reliability": reliability,
                "coefficients": coefficients.copy(),
                "sampled_hp": {
                    "proxy_noise_std": noise,
                    "proxy_reliability": reliability,
                },
                "k_eff": int(p["K_eff"][i]),
                "query_index": torch.arange(n_max, cfg.seq_len, dtype=torch.long),
                "y_context_mean": y_mean.cpu(),
                "y_context_std": y_std.cpu(),
                "tau_star": ladder["Q_true"].cpu(),
                "tau_star_std": (ladder["Q_true"] / y_std).cpu(),
                "s2": s2.cpu(),
                "s2_std": (s2 / y_std.square()).cpu(),
                **{name: value.cpu() for name, value in ladder.items()},
                **{
                    f"{name}_std": (value / y_std).cpu()
                    for name, value in ladder.items()
                },
                "d_U": 1,
                "d_X": 1,
                "d_W": 1,
                "d_Z": 1,
            }
        )
    return records


def inspect_linear_content_digest(meta: Dict[str, Any]) -> Optional[str]:
    global_meta = meta.get("global", {})
    stored = global_meta.get("content_digest")
    if stored is not None:
        return stored
    names = set(LinearBuildConfig.__dataclass_fields__)
    try:
        return LinearBuildConfig(
            **{k: v for k, v in global_meta.items() if k in names}
        ).content_digest()
    except TypeError:
        return None


def check_no_silent_mutation(
    out: Path, cfg: LinearBuildConfig, *, force: bool = False
) -> None:
    path = Path(out) / META_NAME
    if force or not path.exists():
        return
    previous = torch.load(path, weights_only=False)
    old = inspect_linear_content_digest(previous)
    new = cfg.content_digest()
    if old != new:
        raise LinearSetConflictError(
            f"{out} contains content_digest={old}; this build would write {new}. "
            "Choose a new --name or pass --force."
        )


def write_sidecar(
    out: Path,
    cfg: LinearBuildConfig,
    records: Sequence[dict[str, Any]],
    n_max: int,
    datasets_per_file: int,
    *,
    cli_command: str,
) -> str:
    out = Path(out)
    digest = cfg.content_digest()
    grid = [n for n in DEFAULT_N_GRID if n <= n_max]
    if not grid or grid[-1] != n_max:
        grid.append(n_max)
    meta = {
        "format_version": META_VERSION,
        "global": {
            **cfg.to_dict(),
            "n_max": n_max,
            "n_q": cfg.seq_len - n_max,
            "n_grid": grid,
            "content_digest": digest,
            "datasets_per_file": datasets_per_file,
            "cli_command": cli_command,
            "git_rev": _git_rev(),
        },
        "episodes": list(records),
    }
    tmp = out / f"{META_NAME}.tmp"
    torch.save(meta, tmp)
    tmp.replace(out / META_NAME)
    return digest


def load_linear_meta(out: Path) -> Dict[str, Any]:
    return torch.load(Path(out) / META_NAME, weights_only=False)
