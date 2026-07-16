"""Reproducible description of a prior configuration.

A :class:`PriorSpec` is the single source of truth for "which datasets are we
analysing". It is JSON-serialisable, hashes to a stable cache key, and is stored
alongside every cache file so that any figure can be traced back to the exact
configuration that produced it.

Design choices that differ from the training-time prior, and why:

- ``batch_size_per_gp = 1``: at training time datasets are generated in groups
  that share hyperparameters and mechanisms. For an audit we want every dataset
  to be an *independent* draw from the prior, otherwise the "distribution across
  datasets" in every figure would be artificially clumped.
- fixed ``seq_len``: sequence length is constant across datasets, so a cache is a
  plain dense array rather than a nested tensor. Sweeps over ``seq_len`` are done
  by comparing several specs, which keeps each cache self-describing.
- ``replay_small`` / ``log_seq_len`` disabled: both randomise ``seq_len``, which
  is incompatible with the two points above.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

from proximalfm.prior._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP

# Bump when a change to generation makes existing caches invalid (i.e. when the
# same spec would now produce different data). This is part of the cache key, so
# bumping it forces regeneration rather than silently mixing old and new data.
CACHE_FORMAT_VERSION = 1

#: Spec fields that do not affect the content of any individual dataset, and so
#: are excluded from the per-dataset seed. ``name`` is a label; ``n_datasets`` is
#: how many draws to take, not what a draw looks like.
_NON_GENERATIVE_FIELDS = frozenset({"name", "n_datasets"})


@dataclass(frozen=True)
class PriorSpec:
    """A reproducible specification of a set of prior datasets.

    Parameters
    ----------
    name : str
        Human-readable label, used for figure titles and filenames. Part of the
        cache key, so two specs differing only by name get separate caches.

    n_datasets : int
        Number of independent SCMs to draw. Each is generated from its own seed,
        derived deterministically from ``seed`` and the dataset index.

    seq_len : int
        Rows per dataset. Constant across datasets (see module docstring).

    min_dim_U, max_dim_U : int
        Range for the latent confounder dimension.

    min_dim_W, max_dim_W : int
        Range for the outcome-inducing proxy dimension. The generator raises the
        lower bound to ``dim_U`` to enforce the completeness heuristic.

    min_dim_Z, max_dim_Z : int
        Range for the treatment-inducing proxy dimension. Same completeness
        clamp as ``W``.

    min_dim_X, max_dim_X : int
        Range for the observed baseline covariate dimension.

    prior_type : str
        ``"mlp_scm"``, ``"tree_scm"`` or ``"mix_scm"``.

    mcmc_cate : int
        Monte Carlo replicates ``K`` per row used to build the MU0/MU1 targets.

    seed : int
        Master seed. Dataset ``i`` is generated from a seed derived from
        ``(seed, i)``, so generation is reproducible under any ``n_jobs`` and
        adding datasets never perturbs the existing ones.

    fixed_hp_overrides : dict
        Overrides merged over ``DEFAULT_FIXED_HP``. Used for ablations, e.g.
        ``{"cat_prob": 0.0}`` to disable feature categorization.

    sampled_hp_overrides : dict
        Overrides merged over ``DEFAULT_SAMPLED_HP``. Each value is a
        distribution config, so pinning a hyperparameter to a constant is done by
        replacing it with a degenerate distribution, e.g.
        ``{"confounding_duplication_of_U_in_Y": {"distribution": "uniform",
        "min": 0.0, "max": 0.0}}`` to switch off confounding duplication.

        Note that ``DEFAULT_SAMPLED_HP`` contains non-JSON-serialisable entries
        (activation classes), so only the *overrides* participate in the cache
        key, not the resolved config. Overrides must be JSON-serialisable.
    """

    # Defaults mirror the training runs in runs.txt (check4 onward and the
    # mc_run_* series): dims 1-10 for every block, seq_len 2048, MLP SCM. The
    # MC-CATE runs used mcmc_cate 100-200; 100 matches the most recent (mc_run_4).
    name: str = "default"
    n_datasets: int = 64
    seq_len: int = 2048

    min_dim_U: int = 1
    max_dim_U: int = 10
    min_dim_W: int = 1
    max_dim_W: int = 10
    min_dim_Z: int = 1
    max_dim_Z: int = 10
    min_dim_X: int = 1
    max_dim_X: int = 10

    prior_type: str = "mlp_scm"
    mcmc_cate: int = 100
    seed: int = 0

    fixed_hp_overrides: Dict[str, Any] = field(default_factory=dict)
    sampled_hp_overrides: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The name becomes part of a filename stem; a dot or separator would
        # corrupt the suffix handling in the cache layer.
        if not self.name or not all(c.isalnum() or c in "_-" for c in self.name):
            raise ValueError(
                f"name must be non-empty and contain only alphanumerics, '_' or "
                f"'-'; got {self.name!r}"
            )
        if self.n_datasets < 1:
            raise ValueError("n_datasets must be >= 1")
        if self.seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if self.mcmc_cate < 1:
            raise ValueError("mcmc_cate must be >= 1")
        if self.prior_type not in ("mlp_scm", "tree_scm", "mix_scm"):
            raise ValueError(f"Unknown prior_type: {self.prior_type!r}")
        for lo, hi, label in (
            (self.min_dim_U, self.max_dim_U, "U"),
            (self.min_dim_W, self.max_dim_W, "W"),
            (self.min_dim_Z, self.max_dim_Z, "Z"),
            (self.min_dim_X, self.max_dim_X, "X"),
        ):
            if lo < 1:
                raise ValueError(f"min_dim_{label} must be >= 1")
            if lo > hi:
                raise ValueError(f"min_dim_{label} must be <= max_dim_{label}")

        # Overrides land in the cache key, so they must survive a JSON round trip.
        for attr in ("fixed_hp_overrides", "sampled_hp_overrides"):
            try:
                json.dumps(getattr(self, attr), sort_keys=True)
            except TypeError as exc:
                raise ValueError(
                    f"{attr} must be JSON-serialisable to be reproducible; got "
                    f"{getattr(self, attr)!r}"
                ) from exc

    def to_dict(self) -> Dict[str, Any]:
        """Return the spec as a plain JSON-serialisable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PriorSpec":
        """Rebuild a spec from :meth:`to_dict` output.

        Unknown keys are rejected rather than ignored: silently dropping a key
        would produce a spec that hashes differently from the one that was saved.
        """
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown PriorSpec fields: {sorted(unknown)}")
        return cls(**data)

    @staticmethod
    def _hash(payload: Dict[str, Any]) -> str:
        """Stable short hash of a JSON-serialisable payload.

        Stable across processes and Python versions: built from canonical JSON
        rather than :func:`hash`, which is salted per process for strings.
        """
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def digest(self) -> str:
        """Stable short hash of the whole spec, used as the cache key."""
        return self._hash({"version": CACHE_FORMAT_VERSION, **self.to_dict()})

    def generative_digest(self) -> str:
        """Hash of the fields that determine the *content* of a dataset.

        Excludes ``name`` (a cosmetic label) and ``n_datasets`` (how many draws
        we want, not what each draw looks like). Per-dataset seeds derive from
        this rather than from :meth:`digest` so that growing ``n_datasets`` or
        renaming a spec extends or relabels the sample instead of resampling it.
        """
        payload = {
            k: v for k, v in self.to_dict().items() if k not in _NON_GENERATIVE_FIELDS
        }
        return self._hash({"version": CACHE_FORMAT_VERSION, **payload})

    @property
    def cache_key(self) -> str:
        """Filename stem for this spec's cache entry."""
        return f"{self.name}-{self.digest()}"

    def resolved_hp(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return ``(fixed_hp, sampled_hp)`` with overrides applied.

        The defaults are deep-copied so that a spec can never mutate the
        module-level config shared with training.
        """
        fixed = {
            **copy.deepcopy(DEFAULT_FIXED_HP),
            **copy.deepcopy(self.fixed_hp_overrides),
        }
        sampled = {
            **copy.deepcopy(DEFAULT_SAMPLED_HP),
            **copy.deepcopy(self.sampled_hp_overrides),
        }
        return fixed, sampled

    def dataset_seed(self, index: int) -> int:
        """Deterministic seed for dataset ``index``.

        Derived by hashing ``(seed, index)`` rather than using ``seed + index``,
        so that nearby master seeds do not produce overlapping dataset streams.
        Adding datasets to a spec leaves existing ones bit-identical.
        """
        if not 0 <= index < self.n_datasets:
            raise IndexError(
                f"index {index} out of range for n_datasets={self.n_datasets}"
            )
        payload = f"{self.generative_digest()}:{index}".encode("utf-8")
        # 2**31 - 1 keeps the value valid for numpy's legacy seeding.
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (
            2**31 - 1
        )


def pinned(value: float) -> Dict[str, Any]:
    """Distribution config that always returns ``value``.

    Convenience for ablations: ``sampled_hp_overrides={"propensity_shift":
    pinned(0.0)}`` pins a sampled hyperparameter to a constant while keeping the
    override JSON-serialisable.
    """
    return {"distribution": "uniform", "min": float(value), "max": float(value)}


#: Named specs used by the appendix. Kept here so that a figure can be
#: reproduced from ``--spec <name>`` alone, and so ablations sit next to the
#: baseline they are compared against.
PRESETS: Dict[str, PriorSpec] = {
    "default": PriorSpec(name="default"),
    "smoke": PriorSpec(
        name="smoke",
        n_datasets=4,
        seq_len=100,
        mcmc_cate=8,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=3,
        min_dim_Z=2,
        max_dim_Z=3,
        min_dim_X=2,
        max_dim_X=4,
    ),
    # Ablation: no confounding duplication (see appendix A.5.2). Isolates how
    # much of the prior's confounding strength comes from f_u_augment.
    "no_u_augment": PriorSpec(
        name="no_u_augment",
        sampled_hp_overrides={
            "confounding_duplication_of_U_in_Y": pinned(0.0),
            "confounding_duplication_of_U_in_A": pinned(0.0),
        },
    ),
    # Ablation: no feature categorization (see appendix A.3).
    "no_categorization": PriorSpec(
        name="no_categorization",
        fixed_hp_overrides={"cat_prob": 0.0},
    ),
}
