"""M1 -- build a frozen high-``K`` evaluation set from the SCM prior.

``POSTERIOR_EVAL_NOTES.md`` sections 6 and 11. The set is the substrate every
later posterior metric sits on, so three properties matter more than speed:

*Determinism.* Every episode is a pure function of its own seed. The per-episode
seeding recipe is the one proven in ``analysis_prior/generate.py``: seed the
three RNGs, then build a fresh ``PriorDataset`` with ``n_jobs=1`` (``get_batch``
dispatches through joblib otherwise, which destroys reproducibility).
Parallelism moves up one level, across episodes.

*A realised ``K`` equal to the requested one.* ``SCMPrior._cap_mc_samples``
silently caps ``mc_cate`` to fit ``prior_mc_mem_budget``, and its warning is
gated by a module global that fires **once per process** -- so a 400-episode
build would warn once and be quiet for the other 399. Every episode is checked
against ``K_eff`` and the build fails loudly instead.

*Affordable replicates.* The generator materialises
``N = T_ctx + T_q * K`` rows. With ``mc_targets_on_context=True`` (the training
default) ``T_ctx = 0`` and ``N = T*K``, hopeless at large ``K``. This builder
sets ``mc_targets_on_context=False`` with a fixed ``train_size = n_max``, so only
the ``n_q`` query rows replicate: ``N = n_max + n_q*K``. ``MU0``/``MU1``/``S2``
come back ``NaN`` on the context prefix, which is the existing convention.

Output is the standard on-disk ``val/`` format (written through
``proximalfm.prior._genload.write_batch_file``, the single implementation), so
the set is a drop-in ``--val_dir``, plus a ``frozen_meta.pt`` sidecar holding
what the format cannot carry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from proximalfm.prior._dataset import PriorDataset
from proximalfm.prior._genload import write_batch_file
from proximalfm.prior._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP

# Bump when a change would make an existing frozen_meta.pt *wrong* rather than
# merely differently labelled.
FROZEN_META_VERSION = 1

DEFAULT_N_GRID = (32, 64, 128, 256, 512, 1024)

# The trainer's context-standardisation floor (proximalfm/train/_run.py). The
# frozen comparison space must use this, not predict()'s looser 1e-5 clamp.
Y_STD_FLOOR = 1e-3


def episode_seed(base_seed: int, index: int) -> int:
    """Deterministic seed for episode ``index``.

    Hashes ``(base_seed, index)`` rather than adding them, so nearby master
    seeds do not produce overlapping episode streams and appending episodes
    leaves the existing ones bit-identical.
    """
    payload = f"{base_seed}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31 - 1)


@dataclass
class BuildConfig:
    """Everything that determines the content of a frozen set."""

    n_datasets: int
    n_max: int
    n_q: int
    mc_cate: int
    seed: int
    prior_mc_mem_budget: int
    min_dim_U: int = 3
    max_dim_U: int = 10
    min_dim_W: int = 3
    max_dim_W: int = 10
    min_dim_Z: int = 3
    max_dim_Z: int = 10
    min_dim_X: int = 5
    max_dim_X: int = 20
    prior_type: str = "mlp_scm"

    @property
    def seq_len(self) -> int:
        return self.n_max + self.n_q

    def n_grid(self) -> List[int]:
        """The nested context prefixes of section 4, clipped to ``n_max``.

        Prefixes are pure metadata: prefix ``n`` is the first ``n`` context rows
        of an already-generated episode, so the sweep needs no regeneration.
        """
        grid = [n for n in DEFAULT_N_GRID if n <= self.n_max]
        if grid and grid[-1] != self.n_max:
            grid.append(self.n_max)
        return grid

    def to_dict(self) -> Dict[str, Any]:
        d = dict(vars(self))
        d["seq_len"] = self.seq_len
        d["n_grid"] = self.n_grid()
        return d


class MonteCarloBudgetError(RuntimeError):
    """Raised when ``prior_mc_mem_budget`` capped ``K`` below the request."""


def _build_prior(cfg: BuildConfig) -> PriorDataset:
    """One independent SCM per episode, at fixed geometry."""
    fixed_hp = {
        **DEFAULT_FIXED_HP,
        "prior_mc_mem_budget": cfg.prior_mc_mem_budget,
        # Populates PriorDataset.last_diagnostics with the per-episode scalar
        # hyperparameters used for the section-4 stratifications. Collection
        # runs under fork_rng after the accepting draw, so the generated data is
        # bit-identical with or without it.
        "collect_mechanism_diagnostics": True,
    }
    return PriorDataset(
        batch_size=1,
        # No hyperparameter sharing: one independent SCM per episode.
        batch_size_per_gp=1,
        batch_size_per_subgp=1,
        min_dim_U=cfg.min_dim_U,
        max_dim_U=cfg.max_dim_U,
        min_dim_W=cfg.min_dim_W,
        max_dim_W=cfg.max_dim_W,
        min_dim_Z=cfg.min_dim_Z,
        max_dim_Z=cfg.max_dim_Z,
        min_dim_X=cfg.min_dim_X,
        max_dim_X=cfg.max_dim_X,
        # min_seq_len=None makes the generator use max_seq_len exactly, so every
        # episode shares T -- which is what lets episodes generated by separate
        # calls be stacked into one batch file.
        min_seq_len=None,
        max_seq_len=cfg.seq_len,
        log_seq_len=False,
        seq_len_per_gp=False,
        # sample_train_size draws from a half-open randint, so [n_max, n_max+1)
        # pins train_size to exactly n_max while satisfying the min < max check.
        min_train_size=cfg.n_max,
        max_train_size=cfg.n_max + 1,
        replay_small=False,
        prior_type=cfg.prior_type,
        mc_cate=cfg.mc_cate,
        # Replicate the query rows only. This is what makes a large K affordable.
        mc_targets_on_context=False,
        scm_fixed_hp=fixed_hp,
        scm_sampled_hp=DEFAULT_SAMPLED_HP,
        # Nested parallelism would reintroduce the RNG problem; parallelism
        # happens across episodes instead.
        n_jobs=1,
        device="cpu",
    )


def _frozen_scale(y_obs: torch.Tensor, n_max: int) -> tuple:
    """The comparison scale, computed once at ``n_max`` and never again.

    Predictions live in standardised-``y`` space with context-only statistics,
    and that scale moves with ``n``. Freezing it at ``n_max`` is what makes the
    ``n``-sweep and cross-checkpoint deltas comparable (section 2, nuisance 3).

    Deliberately the *trainer's* formula -- biased ``std`` with the ``1e-3``
    floor -- not ``ProximalFM.predict``'s ``1e-5`` clamp.
    """
    ctx = y_obs.reshape(-1)[:n_max].float()
    mean = ctx.mean()
    std = ctx.std(unbiased=False).clamp(min=Y_STD_FLOOR)
    return mean, std


def generate_episode(cfg: BuildConfig, index: int) -> Dict[str, Any]:
    """Generate episode ``index``: batch tensors plus its sidecar record."""
    seed = episode_seed(cfg.seed, index)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    prior = _build_prior(cfg)
    batch = prior.get_batch()
    (
        U,
        X,
        W,
        Z,
        A,
        A_probs,
        Y_obs,
        Y0,
        Y1,
        MU0,
        MU1,
        d_U,
        d_X,
        d_W,
        d_Z,
        seq_lens,
        train_sizes,
        S2,
        K_eff,
    ) = batch

    k_eff = int(K_eff[0])
    if k_eff != cfg.mc_cate:
        raise MonteCarloBudgetError(
            f"episode {index}: prior_mc_mem_budget={cfg.prior_mc_mem_budget:,} "
            f"capped mc_cate from {cfg.mc_cate} to {k_eff}. The frozen set "
            "must have a uniform, known K -- a silently capped episode carries "
            "more target noise than the sidecar claims. Either raise "
            "--prior_mc_mem_budget or lower --n_q (memory scales as "
            f"n_max + n_q*K = {cfg.n_max + cfg.n_q * cfg.mc_cate:,} rows)."
        )
    if int(train_sizes[0]) != cfg.n_max:
        raise RuntimeError(
            f"episode {index}: train_size={int(train_sizes[0])} != n_max={cfg.n_max}"
        )

    tensors = {
        "U": U[0],
        "X": X[0],
        "W": W[0],
        "Z": Z[0],
        "A": A[0],
        "A_probs": A_probs[0],
        "Y_obs": Y_obs[0],
        "Y0": Y0[0],
        "Y1": Y1[0],
        "MU0": MU0[0],
        "MU1": MU1[0],
        "d_U": d_U[0],
        "d_X": d_X[0],
        "d_W": d_W[0],
        "d_Z": d_Z[0],
        "seq_lens": seq_lens[0],
        "train_sizes": train_sizes[0],
        "S2": S2[0],
        "K_eff": K_eff[0],
    }

    y_mean, y_std = _frozen_scale(tensors["Y_obs"], cfg.n_max)
    mu0_q = tensors["MU0"].reshape(-1)[cfg.n_max :].float()
    mu1_q = tensors["MU1"].reshape(-1)[cfg.n_max :].float()
    s2_q = tensors["S2"].reshape(-1)[cfg.n_max :].float()
    tau_star = mu1_q - mu0_q

    diagnostics = (prior.last_diagnostics or [None])[0] or {}
    mechanisms = diagnostics.get("mechanisms", {})

    record = {
        "index": index,
        "seed": seed,
        "k_eff": k_eff,
        "query_index": torch.arange(cfg.n_max, cfg.seq_len, dtype=torch.long),
        # The frozen comparison space.
        "y_context_mean": y_mean,
        "y_context_std": y_std,
        # Raw-y targets, and the same in the head's units. s2 is divided by the
        # variance, matching how the noise-aware loss rescales it -- section 2.2
        # item 2, the likeliest bug in the whole plan, pre-computed here once.
        "tau_star": tau_star,
        "tau_star_std": tau_star / y_std,
        "s2": s2_q,
        "s2_std": s2_q / (y_std**2),
        # Stratification inputs (section 4).
        "sampled_hp": diagnostics.get("params", {}),
        "gamma_hetero": diagnostics.get("gamma_hetero"),
        "a_logit_mean": diagnostics.get(
            "a_logit_mean", diagnostics.get("A_logit_mean")
        ),
        "a_logit_std": diagnostics.get("a_logit_std", diagnostics.get("A_logit_std")),
        "mechanisms": mechanisms,
        "d_U": int(tensors["d_U"]),
        "d_X": int(tensors["d_X"]),
        "d_W": int(tensors["d_W"]),
        "d_Z": int(tensors["d_Z"]),
    }
    return {"tensors": tensors, "record": record}


def _write_group(save_dir: Path, batch_idx: int, group: Sequence[Dict[str, Any]]):
    """Stack a group of episodes into one ``batch_*.pt``.

    Safe to stack episodes from separate ``get_batch`` calls: with
    ``seq_len_per_gp=False`` the feature blocks are padded to the configured
    ``max_dim_*``, which is constant across episodes, and every episode shares
    ``T`` -- both assumptions ``LoadPriorDataset`` makes when it reconstructs a
    file at ``seq_lens[0]``.
    """
    stacked = {
        key: torch.stack([ep["tensors"][key] for ep in group])
        for key in group[0]["tensors"]
    }
    write_batch_file(save_dir, batch_idx, seq_len_per_gp=False, **stacked)


def _write_metadata(save_dir: Path, cfg: BuildConfig):
    """The usual ``metadata.json``, so the directory reads like any ``val/``."""
    metadata = {
        "prior_type": cfg.prior_type,
        "batch_size": 1,
        "batch_size_per_gp": 1,
        "min_seq_len": None,
        "max_seq_len": cfg.seq_len,
        "log_seq_len": False,
        "seq_len_per_gp": False,
        "min_dim_U": cfg.min_dim_U,
        "max_dim_U": cfg.max_dim_U,
        "min_dim_W": cfg.min_dim_W,
        "max_dim_W": cfg.max_dim_W,
        "min_dim_Z": cfg.min_dim_Z,
        "max_dim_Z": cfg.max_dim_Z,
        "min_dim_X": cfg.min_dim_X,
        "max_dim_X": cfg.max_dim_X,
        "min_train_size": cfg.n_max,
        "max_train_size": cfg.n_max + 1,
        "replay_small": False,
        "mc_cate": cfg.mc_cate,
        "mc_targets_on_context": False,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


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


def build(
    cfg: BuildConfig,
    out: Path,
    datasets_per_file: int = 16,
    n_jobs: int = 1,
    cli_command: Optional[str] = None,
    progress: bool = True,
) -> Path:
    """Generate the frozen set into ``out``; returns the ``val/`` directory."""
    out = Path(out)
    val_dir = out / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(val_dir, cfg)

    indices = range(cfg.n_datasets)
    if n_jobs == 1:
        iterator = tqdm(indices, desc="episodes", disable=not progress)
        episodes = [generate_episode(cfg, i) for i in iterator]
    else:
        # Each episode is fully determined by its own seed, so parallelising
        # across episodes preserves bit-identity. Each worker holds the whole
        # N-row replicate block, hence the conservative default of 1.
        episodes = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(generate_episode)(cfg, i)
            for i in tqdm(indices, desc="episodes", disable=not progress)
        )

    for batch_idx, start in enumerate(range(0, len(episodes), datasets_per_file)):
        _write_group(val_dir, batch_idx, episodes[start : start + datasets_per_file])

    meta = {
        "format_version": FROZEN_META_VERSION,
        "global": {
            **cfg.to_dict(),
            "datasets_per_file": datasets_per_file,
            "cli_command": cli_command,
            "git_rev": _git_rev(),
        },
        "episodes": [ep["record"] for ep in episodes],
    }
    meta_tmp = out / "frozen_meta.pt.tmp"
    torch.save(meta, meta_tmp)
    meta_tmp.replace(out / "frozen_meta.pt")

    return val_dir


def load_frozen_meta(out: Path) -> Dict[str, Any]:
    """Read the sidecar of a frozen set built by :func:`build`."""
    return torch.load(Path(out) / "frozen_meta.pt", weights_only=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m posterior_eval.build",
        description="Build a frozen high-K posterior-evaluation set from the prior.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--n_datasets", type=int, default=400)
    p.add_argument("--n_max", type=int, default=1024, help="context rows per episode")
    p.add_argument(
        "--n_q",
        type=int,
        default=256,
        help="query rows per episode; memory is n_max + n_q*mc_cate rows",
    )
    p.add_argument(
        "--mc_cate",
        type=int,
        default=2000,
        help="K, the Monte Carlo replicate count; trades directly against --n_q for RAM",
    )
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument(
        "--prior_mc_mem_budget",
        type=int,
        default=2_000_000_000,
        help=(
            "element budget for the replicate block; the DEFAULT_FIXED_HP value "
            "of 500M caps the widest episodes near K=1900 at n_q=256"
        ),
    )
    p.add_argument("--datasets_per_file", type=int, default=16)
    p.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="episodes in parallel; each worker holds the full replicate block",
    )
    p.add_argument("--min_dim_U", type=int, default=3)
    p.add_argument("--max_dim_U", type=int, default=10)
    p.add_argument("--min_dim_W", type=int, default=3)
    p.add_argument("--max_dim_W", type=int, default=10)
    p.add_argument("--min_dim_Z", type=int, default=3)
    p.add_argument("--max_dim_Z", type=int, default=10)
    p.add_argument("--min_dim_X", type=int, default=5)
    p.add_argument("--max_dim_X", type=int, default=20)
    p.add_argument(
        "--report",
        action="store_true",
        help="print the s/sigma_post table after building (section 2.1)",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="checkpoint for the sigma_post ~ CATE RMSE stand-in in --report",
    )
    p.add_argument("--device", type=str, default="cpu", help="device for --report")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = BuildConfig(
        n_datasets=args.n_datasets,
        n_max=args.n_max,
        n_q=args.n_q,
        mc_cate=args.mc_cate,
        seed=args.seed,
        prior_mc_mem_budget=args.prior_mc_mem_budget,
        min_dim_U=args.min_dim_U,
        max_dim_U=args.max_dim_U,
        min_dim_W=args.min_dim_W,
        max_dim_W=args.max_dim_W,
        min_dim_Z=args.min_dim_Z,
        max_dim_Z=args.max_dim_Z,
        min_dim_X=args.min_dim_X,
        max_dim_X=args.max_dim_X,
    )
    cli_command = shlex.join(
        [sys.executable, "-m", "posterior_eval.build", *(argv or sys.argv[1:])]
    )

    val_dir = build(
        cfg,
        args.out,
        datasets_per_file=args.datasets_per_file,
        n_jobs=args.n_jobs,
        cli_command=cli_command,
    )
    print(
        f"Frozen set written to {val_dir} ({cfg.n_datasets} episodes, K={cfg.mc_cate})"
    )
    print(f"Sidecar: {Path(args.out) / 'frozen_meta.pt'}")

    if args.report:
        from posterior_eval.report import report

        report(args.out, ckpt=args.ckpt, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
