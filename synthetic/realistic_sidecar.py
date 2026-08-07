"""The ``frozen_meta.pt`` sidecar for replicate-bearing realistic sets (ROADMAP B3).

``val/`` files carry the episodes; they cannot carry the two things a
distributional score needs:

* the **frozen comparison scale**. Predictions come back in standardised-``y``
  space with context-only statistics, and that scale moves with ``n``. Scoring
  in each episode's own moving scale manufactures trends, so the scale is
  computed once at ``n_max`` and stored.
* ``tau_star`` and ``s2`` **in the head's units**, so no consumer has to
  re-derive the ``y_std`` vs ``y_std**2`` distinction. That is the one place a
  scale bug is invisible: both forms look plausible and only one is right.

The schema is deliberately identical to ``posterior_eval``'s, so
``proximalfm.train._dist_eval.score_frozen_set`` scores a realistic set with no
realistic-specific branch. It reads exactly four things --
``meta["global"]["n_max"]``, ``meta["episodes"]``, and per record
``y_context_std`` / ``tau_star_std`` / ``s2_std`` -- and everything else here
exists for ``verify``, stratification and provenance.

**A sidecar is written only when ``mc_cate > 1``.** Its absence is the gate that
makes a non-replicate set fail loudly instead of scoring a posterior against
sampled ITEs, which would not error and would be minimised by an over-wide
posterior.

One deliberate difference from the prior sets: ``n_datasets`` **is** a
generative field here and does participate in the digest. The prior builder can
append because its per-episode seed is index-local; this builder shuffles the
whole pool with a single permutation whose length is ``n_datasets``, so adding
episodes reorders every existing one. Appending is therefore not a supported
operation on a realistic set, and the digest says so rather than letting a
reorder pass as an extension.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

Tensor = torch.Tensor

# Bumping this re-identifies every realistic set.
REALISTIC_META_VERSION = 1

SET_KIND = "realistic"

# The *trainer's* standardisation floor, deliberately not ProximalFM.predict's
# 1e-5. Kept in step with posterior_eval.build_prior_frozen_set.Y_STD_FLOOR.
Y_STD_FLOOR = 1e-3

META_NAME = "frozen_meta.pt"

# Nested context prefixes for the n-sweep; clipped to n_max. Same ladder the
# prior builder uses, so contraction curves are read on the same axis.
DEFAULT_N_GRID = (32, 64, 128, 256, 512, 1024, 2048)


class RealisticSetConflictError(RuntimeError):
    """Raised when a build would overwrite an existing set with different content."""


@dataclass
class RealisticBuildConfig:
    """Everything that determines the content of a realistic replicate set.

    Every field is generative -- including ``n_datasets``, see the module
    docstring. ``gen_device`` participates because determinism is per-device.
    """

    n_datasets: int
    seq_len: int
    prop_train: float
    mc_cate: int
    seed: int
    mechanisms: List[int]
    min_dim_U: int = 1
    max_dim_U: int = 10
    min_dim_X: int = 1
    max_dim_X: int = 10
    min_dim_W: int = 1
    max_dim_W: int = 10
    min_dim_Z: int = 1
    max_dim_Z: int = 10
    dim_U_distribution: str = "triangular"
    shuffle: bool = True
    gen_device: str = "cpu"

    def n_grid(self, n_max: int) -> List[int]:
        """Nested context prefixes for the ``n``-sweep, always closed at ``n_max``.

        Advisory only -- prefixes are pure row selection at score time, so a
        consumer may extend or trim this. Mirrors the prior builder's grid.
        """
        grid = [n for n in DEFAULT_N_GRID if n <= n_max]
        if not grid or grid[-1] != n_max:
            grid.append(n_max)
        return grid

    def to_dict(self) -> Dict[str, Any]:
        d = dict(asdict(self))
        d["mechanisms"] = list(self.mechanisms)
        d["set_kind"] = SET_KIND
        return d

    def content_digest(self) -> str:
        """Stable 16-hex identity of this set's generative configuration.

        Canonical JSON rather than ``hash()``, which is PYTHONHASHSEED-salted
        and so differs between processes.
        """
        payload = {"format_version": REALISTIC_META_VERSION, **self.to_dict()}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


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


def frozen_scale(y_obs: Tensor, n_max: int):
    """Comparison scale from the first ``n_max`` outcomes of one episode.

    Byte-for-byte the prior builder's ``_frozen_scale``: biased ``std`` with the
    trainer's ``1e-3`` floor. The mean is stored but never used at score time --
    ``tau`` is a difference, so no mean shift is needed.
    """
    ctx = y_obs.reshape(-1)[:n_max].float()
    return ctx.mean(), ctx.std(unbiased=False).clamp(min=Y_STD_FLOOR)


def episode_records(
    pool: Sequence[Tensor],
    fields: Sequence[str],
    mech_ids: Tensor,
    seeds: Tensor,
    n_max: int,
) -> List[Dict[str, Any]]:
    """Per-episode sidecar records, positionally aligned with ``val/`` order.

    ``pool`` is the *shuffled* stack, so record ``i`` describes the ``i``-th
    episode ``LoadPriorDataset`` will yield. That alignment is the invariant
    every downstream tool relies on and only ``verify``'s sidecar-vs-disk check
    can catch a violation of.
    """
    p = dict(zip(fields, pool))
    n_sets = int(p["X"].shape[0])
    records: List[Dict[str, Any]] = []
    for i in range(n_sets):
        y_mean, y_std = frozen_scale(p["Y_obs"][i], n_max)
        mu0 = p["MU0"][i].reshape(-1)[n_max:].float()
        mu1 = p["MU1"][i].reshape(-1)[n_max:].float()
        s2 = p["S2"][i].reshape(-1)[n_max:].float()
        tau_star = mu1 - mu0
        seq_len = int(p["seq_lens"][i])
        records.append(
            {
                "index": i,
                "seed": int(seeds[i]),
                "mechanism_id": int(mech_ids[i]),
                "k_eff": int(p["K_eff"][i]),
                "query_index": torch.arange(n_max, seq_len, dtype=torch.long),
                # The frozen comparison space.
                "y_context_mean": y_mean.cpu(),
                "y_context_std": y_std.cpu(),
                # Raw-y targets, and the same in the head's units. s2 is divided
                # by the *variance*, matching how the noise-aware loss rescales
                # it -- the likeliest bug in the whole file, done once here.
                "tau_star": tau_star.cpu(),
                "tau_star_std": (tau_star / y_std).cpu(),
                "s2": s2.cpu(),
                "s2_std": (s2 / (y_std**2)).cpu(),
                "d_U": int(p["d_U"][i]),
                "d_X": int(p["d_X"][i]),
                "d_W": int(p["d_W"][i]),
                "d_Z": int(p["d_Z"][i]),
            }
        )
    return records


def check_no_silent_mutation(
    out: Path, cfg: RealisticBuildConfig, force: bool = False
) -> None:
    """Refuse to rewrite an existing set in place with different content.

    The failure this prevents: rebuilding a *named* directory with a different
    ``mc_cate``/seed instead of choosing a new name. Runs before and after log
    the same path, so their curves look comparable and are not -- and detection
    after the fact is impossible, because the rewritten sidecar is perfectly
    self-consistent. Only refusing up front works.
    """
    meta_path = Path(out) / META_NAME
    if force or not meta_path.exists():
        return
    prev = torch.load(meta_path, weights_only=False)
    prev_g = prev.get("global", {})
    prev_digest = realistic_content_digest(prev)
    new_digest = cfg.content_digest()
    if prev_digest == new_digest:
        return
    raise RealisticSetConflictError(
        f"{out} already holds a realistic set with content_digest="
        f"{prev_digest} (n_datasets={prev_g.get('n_datasets')}, "
        f"mc_cate={prev_g.get('mc_cate')}, seed={prev_g.get('seed')}); this "
        f"build would write {new_digest} (n_datasets={cfg.n_datasets}, "
        f"mc_cate={cfg.mc_cate}, seed={cfg.seed}). Choose a new --name, or "
        "pass --force to discard the existing set."
    )


def write_sidecar(
    out: Path,
    cfg: RealisticBuildConfig,
    records: Sequence[Dict[str, Any]],
    n_max: int,
    datasets_per_file: int,
    cli_command: Optional[str] = None,
) -> str:
    """Write ``frozen_meta.pt`` atomically. Returns the content digest."""
    out = Path(out)
    digest = cfg.content_digest()
    meta = {
        "format_version": REALISTIC_META_VERSION,
        "global": {
            **cfg.to_dict(),
            # n_max is what score_frozen_set reads to split context from query.
            "n_max": int(n_max),
            "n_q": int(cfg.seq_len - n_max),
            "n_grid": cfg.n_grid(int(n_max)),
            "content_digest": digest,
            "datasets_per_file": int(datasets_per_file),
            "cli_command": cli_command,
            "git_rev": _git_rev(),
        },
        "episodes": list(records),
    }
    tmp = out / (META_NAME + ".tmp")
    torch.save(meta, tmp)
    tmp.replace(out / META_NAME)
    return digest


def load_realistic_meta(out: Path) -> Dict[str, Any]:
    return torch.load(Path(out) / META_NAME, weights_only=False)


def realistic_content_digest(meta: Dict[str, Any]) -> Optional[str]:
    """Identity of a realistic set: stored value, else recomputed, else None.

    The stored field is a *cache*. Recomputing from the config is what lets a
    sidecar written before the field existed still have an identity, and what
    lets ``verify`` catch a stored value that disagrees with the config beside
    it. ``None`` means "unknown", never "different".
    """
    g = meta.get("global", {})
    stored = g.get("content_digest")
    if stored is not None:
        return stored
    names = set(RealisticBuildConfig.__dataclass_fields__)
    try:
        return RealisticBuildConfig(
            **{k: v for k, v in g.items() if k in names}
        ).content_digest()
    except TypeError:
        return None
