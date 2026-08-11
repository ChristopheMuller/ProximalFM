"""The ``frozen_meta.pt`` sidecar for replicate-bearing literature sets.

Schema-identical to ``prior_eval``'s and ``synt_eval``'s, deliberately:
``proximalfm.train._dist_eval.score_frozen_set`` reads exactly four things --
``meta["global"]["n_max"]``, ``meta["episodes"]``, and per record
``y_context_std`` / ``tau_star_std`` / ``s2_std`` -- so a literature set scores
with no literature-specific branch. Everything else here exists for
``prior_eval.verify``, stratification and provenance.

Two things the ``val/`` files cannot carry, and which are the reason this file
exists at all:

* the **frozen comparison scale**. Predictions come back in standardised-``y``
  space computed from the context rows, and that scale moves with ``n``.
  Scoring in each episode's own moving scale manufactures trends, so the scale
  is computed once at ``n_max`` and stored.
* ``tau_star`` and ``s2`` **in the head's units**, so no consumer re-derives the
  ``y_std`` vs ``y_std**2`` distinction -- the one place a scale bug is
  invisible, because both forms look plausible and only one is right.

``frozen_scale`` and the ``1e-3`` floor are imported from ``synt_eval`` rather
than restated: it is the trainer's standardisation convention, not this
package's, and a second copy of it is how the two surfaces would drift into
scoring on different scales without anything failing.

**A sidecar is written only when ``mc_cate > 1``.** Its absence is the gate that
makes a non-replicate set fail loudly instead of scoring a posterior against
sampled ITEs -- which would not error, and would be minimised by an over-wide
posterior.

As with realistic sets, ``n_datasets`` **is** generative and participates in the
digest: episodes are seeded by index here, but the set's identity should still
refuse to call a longer build the same set, since nothing downstream is prepared
to append to a literature set.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from synt_eval.realistic_sidecar import Y_STD_FLOOR, frozen_scale  # noqa: F401

Tensor = torch.Tensor

# Bumping this re-identifies every literature set.
LIT_META_VERSION = 1

SET_KIND = "literature"

META_NAME = "frozen_meta.pt"

# Nested context prefixes for the n-sweep; clipped to n_max. Same ladder the
# prior and realistic builders use, so contraction curves share an axis.
DEFAULT_N_GRID = (32, 64, 128, 256, 512, 1024, 2048)


class LitSetConflictError(RuntimeError):
    """Raised when a build would overwrite an existing set with different content."""


@dataclass
class LitBuildConfig:
    """Everything that determines the content of a literature replicate set.

    ``options`` is the adapter's resolved option dict -- it is part of the
    identity precisely so a knob that changes the data cannot be changed without
    the digest noticing. ``gen_device`` participates because determinism is
    per-device.
    """

    dataset: str
    n_datasets: int
    seq_len: int
    prop_train: float
    mc_cate: int
    seed: int
    options: Dict[str, Any] = field(default_factory=dict)
    gen_device: str = "cpu"

    def n_grid(self, n_max: int) -> List[int]:
        """Nested context prefixes for the ``n``-sweep, always closed at ``n_max``.

        Advisory only -- prefixes are pure row selection at score time, so a
        consumer may extend or trim this.
        """
        grid = [n for n in DEFAULT_N_GRID if n <= n_max]
        if not grid or grid[-1] != n_max:
            grid.append(n_max)
        return grid

    def to_dict(self) -> Dict[str, Any]:
        d = dict(asdict(self))
        d["options"] = {k: v for k, v in sorted(self.options.items())}
        d["set_kind"] = SET_KIND
        return d

    def content_digest(self) -> str:
        """Stable 16-hex identity of this set's generative configuration.

        Canonical JSON rather than ``hash()``, which is PYTHONHASHSEED-salted
        and so differs between processes.
        """
        payload = {"format_version": LIT_META_VERSION, **self.to_dict()}
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        )
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


def episode_records(
    pool: Sequence[Tensor],
    fields: Sequence[str],
    infos: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    dataset: str,
    n_max: int,
) -> List[Dict[str, Any]]:
    """Per-episode sidecar records, positionally aligned with ``val/`` order.

    Record ``i`` must describe the ``i``-th episode ``LoadPriorDataset`` yields.
    That alignment is the invariant every downstream tool relies on and only
    ``prior_eval.verify``'s sidecar-vs-disk check can catch a violation of.
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
        info = infos[i]
        exact = info.get("tau_star_exact")
        records.append(
            {
                "index": i,
                "seed": int(seeds[i]),
                "dataset": dataset,
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
                # Closed-form target where the DGP has one; None otherwise. Not
                # a scoring target -- it is what lets a test check the Monte
                # Carlo estimate against the truth it is estimating.
                "tau_star_exact": (exact[n_max:].cpu() if exact is not None else None),
                # Scalar knobs -> the cut axes prior_eval.strata reads.
                "sampled_hp": dict(info.get("sampled_hp") or {}),
                "d_U": int(p["d_U"][i]),
                "d_X": int(p["d_X"][i]),
                "d_W": int(p["d_W"][i]),
                "d_Z": int(p["d_Z"][i]),
            }
        )
    return records


def check_no_silent_mutation(
    out: Path, cfg: LitBuildConfig, force: bool = False
) -> None:
    """Refuse to rewrite an existing set in place with different content.

    The failure this prevents: rebuilding a *named* directory with a different
    option/seed instead of choosing a new name. Runs before and after log the
    same path, so their curves look comparable and are not -- and detection
    after the fact is impossible, because the rewritten sidecar is perfectly
    self-consistent. Only refusing up front works.
    """
    meta_path = Path(out) / META_NAME
    if force or not meta_path.exists():
        return
    prev = torch.load(meta_path, weights_only=False)
    prev_g = prev.get("global", {})
    prev_digest = lit_content_digest(prev)
    new_digest = cfg.content_digest()
    if prev_digest == new_digest:
        return
    raise LitSetConflictError(
        f"{out} already holds a set with content_digest={prev_digest} "
        f"(dataset={prev_g.get('dataset')}, n_datasets={prev_g.get('n_datasets')}, "
        f"mc_cate={prev_g.get('mc_cate')}, seed={prev_g.get('seed')}); this build "
        f"would write {new_digest} (dataset={cfg.dataset}, "
        f"n_datasets={cfg.n_datasets}, mc_cate={cfg.mc_cate}, seed={cfg.seed}). "
        "Choose a new --name, or pass --force to discard the existing set."
    )


def write_sidecar(
    out: Path,
    cfg: LitBuildConfig,
    records: Sequence[Dict[str, Any]],
    n_max: int,
    datasets_per_file: int,
    citation: str = "",
    cli_command: Optional[str] = None,
) -> str:
    """Write ``frozen_meta.pt`` atomically. Returns the content digest."""
    out = Path(out)
    digest = cfg.content_digest()
    meta = {
        "format_version": LIT_META_VERSION,
        "global": {
            **cfg.to_dict(),
            # n_max is what score_frozen_set reads to split context from query.
            "n_max": int(n_max),
            "n_q": int(cfg.seq_len - n_max),
            "n_grid": cfg.n_grid(int(n_max)),
            "content_digest": digest,
            "datasets_per_file": int(datasets_per_file),
            "citation": citation,
            "cli_command": cli_command,
            "git_rev": _git_rev(),
        },
        "episodes": list(records),
    }
    tmp = out / (META_NAME + ".tmp")
    torch.save(meta, tmp)
    tmp.replace(out / META_NAME)
    return digest


def load_lit_meta(out: Path) -> Dict[str, Any]:
    return torch.load(Path(out) / META_NAME, weights_only=False)


def lit_content_digest(meta: Dict[str, Any]) -> Optional[str]:
    """Identity of a literature set: stored value, else recomputed, else None.

    The stored field is a *cache*. Recomputing from the config is what lets a
    sidecar written before the field existed still have an identity, and what
    lets ``verify`` catch a stored value that disagrees with the config beside
    it. ``None`` means "unknown", never "different".
    """
    g = meta.get("global", {})
    stored = g.get("content_digest")
    if stored is not None:
        return stored
    names = set(LitBuildConfig.__dataclass_fields__)
    try:
        return LitBuildConfig(
            **{k: v for k, v in g.items() if k in names}
        ).content_digest()
    except TypeError:
        return None
