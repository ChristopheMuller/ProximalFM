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
any consumer of that format reads it directly, plus a ``frozen_meta.pt``
sidecar holding what the format cannot carry.
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
# merely differently labelled. Part of ``content_digest``, so bumping it makes
# every existing set read as a different set rather than silently comparable.
FROZEN_META_VERSION = 1

DEFAULT_N_GRID = (32, 64, 128, 256, 512, 1024)

#: ``BuildConfig`` fields excluded from :meth:`BuildConfig.content_digest`.
#: ``n_datasets`` is how many episodes were asked for, not what an episode looks
#: like: ``episode_seed`` hashes ``(seed, index)``, so raising it leaves episodes
#: ``0..N-1`` bit-identical and a 1000-episode set is a prefix of a 1200-episode
#: one. Every other field changes episode content and so must move the digest.
_NON_GENERATIVE_FIELDS = frozenset({"n_datasets"})

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
    min_dim_U: int = 1
    max_dim_U: int = 10
    min_dim_W: int = 1
    max_dim_W: int = 10
    min_dim_Z: int = 1
    max_dim_Z: int = 10
    min_dim_X: int = 1
    max_dim_X: int = 10
    prior_type: str = "mlp_scm"
    # Generation device. The mechanisms are pure torch and device-agnostic (only
    # the scalar hyperparameter draws go through numpy), so "cuda:0" moves the
    # N-row replicate block onto GPU memory -- which is the whole constraint on
    # K. Determinism is per-device: a set built on cuda is reproducible on the
    # same GPU + torch build, but is NOT bit-identical to a cpu build. The
    # device is recorded in frozen_meta.pt for exactly this reason.
    gen_device: str = "cpu"

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

    def generative_dict(self) -> Dict[str, Any]:
        """The fields that determine what an episode *contains*.

        Excludes ``n_datasets`` -- see :data:`_NON_GENERATIVE_FIELDS`.
        """
        return {
            k: v for k, v in self.to_dict().items() if k not in _NON_GENERATIVE_FIELDS
        }

    def content_digest(self) -> str:
        """Short stable hash naming this set by its content, not its directory.

        A directory name is chosen by a human and can be reused; this cannot.
        Two things depend on it:

        - :func:`build` refuses to write into an existing set whose digest
          differs, so a named set cannot be silently mutated in place.
        - the trainer logs it beside ``--prior_eval_dir`` (see
          ``proximalfm/train/_run.py``), so two curves that both claim
          ``frozen_prod_v1`` are visibly different sets rather than an
          unexplained regression.

        Built from canonical JSON rather than :func:`hash`, which is salted per
        process for strings and so is not stable across runs. ``gen_device``
        participates: determinism is per-device, so a ``cuda:0`` set is not
        bit-identical to a cpu rebuild of the same config and must not share an
        identity with it.
        """
        payload = {"format_version": FROZEN_META_VERSION, **self.generative_dict()}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


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
        # episode shares the context size -- which is what lets episodes
        # generated by separate calls be stacked into one batch file. The
        # generator composes seq_len = n_ctx + n_query itself, matching
        # BuildConfig.seq_len.
        min_seq_len=None,
        max_seq_len=cfg.n_max,
        log_seq_len=False,
        seq_len_per_gp=False,
        n_query=cfg.n_q,
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
        device=cfg.gen_device,
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


def _to_host(obj: Any) -> Any:
    """Recursively move tensors in a diagnostics payload to CPU.

    The sidecar is written with ``torch.save`` and read on machines that need
    not have a GPU, so a stray cuda tensor in ``last_diagnostics`` would make
    ``frozen_meta.pt`` unloadable on cpu-only hosts.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_host(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_host(v) for v in obj)
    return obj


def generate_episode(cfg: BuildConfig, index: int) -> Dict[str, Any]:
    """Generate episode ``index``: batch tensors plus its sidecar record."""
    seed = episode_seed(cfg.seed, index)
    random.seed(seed)
    np.random.seed(seed)
    # Seeds cpu and every cuda device, so the per-episode determinism holds on
    # whichever device cfg.gen_device names -- within that device, not across.
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

    # Brought back to host immediately: only the transient N-row replicate block
    # needs to live on the accelerator, and everything downstream (stacking,
    # write_batch_file, the sidecar) is CPU-side and must stay picklable.
    tensors = {
        "U": U[0].cpu(),
        "X": X[0].cpu(),
        "W": W[0].cpu(),
        "Z": Z[0].cpu(),
        "A": A[0].cpu(),
        "A_probs": A_probs[0].cpu(),
        "Y_obs": Y_obs[0].cpu(),
        "Y0": Y0[0].cpu(),
        "Y1": Y1[0].cpu(),
        "MU0": MU0[0].cpu(),
        "MU1": MU1[0].cpu(),
        "d_U": d_U[0].cpu(),
        "d_X": d_X[0].cpu(),
        "d_W": d_W[0].cpu(),
        "d_Z": d_Z[0].cpu(),
        "seq_lens": seq_lens[0].cpu(),
        "train_sizes": train_sizes[0].cpu(),
        "S2": S2[0].cpu(),
        "K_eff": K_eff[0].cpu(),
    }
    diagnostics = _to_host((prior.last_diagnostics or [None])[0] or {})
    mechanisms = diagnostics.get("mechanisms", {})

    # Drop every device-resident reference before the next episode allocates its
    # own replicate block; on cuda the caching allocator otherwise holds the peak
    # of episode i while episode i+1 asks for its own.
    del U, X, W, Z, A, A_probs, Y_obs, Y0, Y1, MU0, MU1, S2, batch, prior
    if cfg.gen_device.startswith("cuda"):
        torch.cuda.empty_cache()

    y_mean, y_std = _frozen_scale(tensors["Y_obs"], cfg.n_max)
    mu0_q = tensors["MU0"].reshape(-1)[cfg.n_max :].float()
    mu1_q = tensors["MU1"].reshape(-1)[cfg.n_max :].float()
    s2_q = tensors["S2"].reshape(-1)[cfg.n_max :].float()
    tau_star = mu1_q - mu0_q

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
        "n_query": cfg.n_q,
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


class FrozenSetConflictError(RuntimeError):
    """Raised when a build would overwrite an existing set with different content."""


def _check_no_silent_mutation(
    out: Path, cfg: BuildConfig, datasets_per_file: int
) -> None:
    """Refuse to rewrite an existing set in place with different content.

    The failure this prevents: rebuilding a *named* directory (say
    ``frozen_prod_v1``) with a different ``n_q``/``K``/seed instead of choosing a
    new name. Runs before and after log the same ``--prior_eval_dir`` string, so
    their curves look comparable and are not -- and the old set, hours of GPU
    time, is gone. Detection after the fact is impossible: the rewritten sidecar
    is perfectly self-consistent. Only refusing up front works.

    Appending is *not* a conflict: ``n_datasets`` is outside the digest, so
    raising it re-emits episodes ``0..N-1`` bit-identically and extends the set.
    Lowering it is refused, because the surplus ``batch_*.pt`` files would stay
    on disk and be loaded as episodes the sidecar does not describe.
    """
    meta_path = out / "frozen_meta.pt"
    if not meta_path.exists():
        return

    prev = load_frozen_meta(out)
    prev_g = prev.get("global", {})
    # Recomputes for sets built before the field existed, so those are compared
    # properly rather than blanket-refused.
    prev_digest = frozen_content_digest(prev)
    new_digest = cfg.content_digest()

    if prev_digest is None:
        raise FrozenSetConflictError(
            f"{out} already holds a frozen set whose config cannot be read back "
            "into a BuildConfig, so this build cannot prove it would produce "
            "the same episodes. Build into a new --out, or pass --force."
        )
    if prev_digest != new_digest:
        raise FrozenSetConflictError(
            f"{out} already holds a *different* frozen set:\n"
            f"  on disk:   content_digest={prev_digest}  "
            f"n_q={prev_g.get('n_q')} K={prev_g.get('mc_cate')} "
            f"n_max={prev_g.get('n_max')} seed={prev_g.get('seed')} "
            f"gen_device={prev_g.get('gen_device')}\n"
            f"  requested: content_digest={new_digest}  "
            f"n_q={cfg.n_q} K={cfg.mc_cate} n_max={cfg.n_max} seed={cfg.seed} "
            f"gen_device={cfg.gen_device}\n"
            "Overwriting would make every run that logged this path "
            "incomparable, with nothing on disk to reveal it. Choose a new "
            "--out, or pass --force if the old set is genuinely disposable."
        )
    if cfg.n_datasets < prev_g.get("n_datasets", 0):
        raise FrozenSetConflictError(
            f"{out} holds {prev_g['n_datasets']} episodes and this build "
            f"requests {cfg.n_datasets}. The surplus batch files would remain "
            "on disk and be read as episodes the sidecar does not describe. "
            "Pass --force to rebuild from scratch, or just score a prefix with "
            "--prior_eval_max_episodes."
        )
    if datasets_per_file != prev_g.get("datasets_per_file", datasets_per_file):
        raise FrozenSetConflictError(
            f"{out} was packed at datasets_per_file="
            f"{prev_g['datasets_per_file']}, this build uses {datasets_per_file}. "
            "Re-packing leaves stale batch files. Pass --force to rebuild."
        )


def build(
    cfg: BuildConfig,
    out: Path,
    datasets_per_file: int = 16,
    n_jobs: int = 1,
    cli_command: Optional[str] = None,
    progress: bool = True,
    force: bool = False,
) -> Path:
    """Generate the frozen set into ``out``; returns the ``val/`` directory."""
    out = Path(out)
    if not force:
        _check_no_silent_mutation(out, cfg, datasets_per_file)
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
            "content_digest": cfg.content_digest(),
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


def frozen_content_digest(meta: Dict[str, Any]) -> Optional[str]:
    """The content digest of the set described by ``meta``.

    Prefers the stored value, and *recomputes* it from the sidecar's own config
    when absent -- which is what sets built before the field existed carry. The
    stored field is therefore a cache, not the source of truth, so a legacy set
    still has a stable identity and never needed rewriting to acquire one.

    Returns ``None`` only when the global block cannot be turned back into a
    :class:`BuildConfig` at all (a field renamed since). That is "unknown", not
    "different": callers should degrade to the directory name, not treat it as
    a mismatch.

    Realistic replicate sets (``synt_eval``) and literature sets (``lit_eval``)
    share this sidecar schema so the evaluator needs no branch, but their
    generative config is a different dataclass; ``set_kind`` dispatches the
    recompute to it. Without that, the ``BuildConfig(**subset)`` below raises
    ``TypeError`` on the missing required fields and a perfectly good set
    reports "no stable identity".
    """
    g = meta.get("global", {})
    stored = g.get("content_digest")
    if stored is not None:
        return stored
    if g.get("set_kind") == "realistic":
        try:
            from synt_eval.realistic_sidecar import realistic_content_digest
        except ImportError:
            # synt_eval/ is resolved from the repo root, so this can miss when
            # running from elsewhere. "Unknown", not "different" -- the caller
            # degrades to the directory name.
            return None
        return realistic_content_digest(meta)
    if g.get("set_kind") == "literature":
        try:
            from lit_eval.sidecar import lit_content_digest
        except ImportError:
            return None
        return lit_content_digest(meta)
    fields = set(BuildConfig.__dataclass_fields__)
    try:
        return BuildConfig(
            **{k: v for k, v in g.items() if k in fields}
        ).content_digest()
    except TypeError:
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m prior_eval.build",
        description="Build a frozen high-K posterior-evaluation set from the prior.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Set identifier; the build lands in <out_root>/<name>/ "
        "(e.g. frozen_prod_v1). Either --name or --out is required.",
    )
    p.add_argument(
        "--out_root",
        type=Path,
        default=Path("data/prior_eval"),
        help="Root directory under which --name is written.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Explicit output directory, overriding --out_root/--name.",
    )
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
    p.add_argument(
        "--gen_device",
        type=str,
        default="cpu",
        help=(
            "device the generator allocates the n_max + n_q*K replicate block "
            "on, e.g. cuda:0. Reproducible per-device, not across devices"
        ),
    )
    p.add_argument("--min_dim_U", type=int, default=1)
    p.add_argument("--max_dim_U", type=int, default=10)
    p.add_argument("--min_dim_W", type=int, default=1)
    p.add_argument("--max_dim_W", type=int, default=10)
    p.add_argument("--min_dim_Z", type=int, default=1)
    p.add_argument("--max_dim_Z", type=int, default=10)
    p.add_argument("--min_dim_X", type=int, default=1)
    p.add_argument("--max_dim_X", type=int, default=10)
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "overwrite an existing set at --out even if its content_digest "
            "differs. Without this, rebuilding a named set with a different "
            "config is refused, because runs that logged the old path would "
            "silently become incomparable"
        ),
    )
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.out is None:
        if args.name is None:
            parser.error("one of --name (with --out_root) or --out is required")
        args.out = args.out_root / args.name
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
        gen_device=args.gen_device,
    )
    if args.n_jobs != 1 and cfg.gen_device != "cpu":
        raise SystemExit(
            "--n_jobs > 1 with --gen_device cuda would give every loky worker "
            "its own CUDA context and its own copy of the replicate block. "
            "Keep --n_jobs 1 on GPU; the GPU is already the parallelism."
        )
    cli_command = shlex.join(
        [sys.executable, "-m", "prior_eval.build", *(argv or sys.argv[1:])]
    )

    try:
        val_dir = build(
            cfg,
            args.out,
            datasets_per_file=args.datasets_per_file,
            n_jobs=args.n_jobs,
            cli_command=cli_command,
            force=args.force,
        )
    except FrozenSetConflictError as exc:
        raise SystemExit(str(exc))
    print(
        f"Frozen set written to {val_dir} ({cfg.n_datasets} episodes, K={cfg.mc_cate})"
    )
    print(f"Sidecar: {Path(args.out) / 'frozen_meta.pt'}")
    print(f"content_digest: {cfg.content_digest()}")

    if args.report:
        from prior_eval.report import report

        report(args.out, ckpt=args.ckpt, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
