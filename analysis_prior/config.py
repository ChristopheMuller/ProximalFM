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
import dataclasses
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
        ``{"confounding_dim_control": False}`` to disable U-input subsampling.

    sampled_hp_overrides : dict
        Overrides merged over ``DEFAULT_SAMPLED_HP``. Each value is a
        distribution config, so pinning a hyperparameter to a constant is done by
        replacing it with a degenerate distribution, e.g.
        ``{"confounding_share_of_U_in_Y": {"distribution": "uniform",
        "min": 0.75, "max": 0.75}}`` to pin the targeted U input share.

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

    def generative_dict(self) -> Dict[str, Any]:
        """The spec fields that determine the *content* of each dataset.

        Excludes ``name`` (a cosmetic label) and ``n_datasets`` (how many draws
        we want, not what each draw looks like).
        """
        return {
            k: v for k, v in self.to_dict().items() if k not in _NON_GENERATIVE_FIELDS
        }

    def generative_digest(self) -> str:
        """Hash of the fields that determine the *content* of a dataset.

        Per-dataset seeds derive from this rather than from :meth:`digest`, so
        growing ``n_datasets`` or renaming a spec extends or relabels the sample
        instead of resampling it.
        """
        return self._hash({"version": CACHE_FORMAT_VERSION, **self.generative_dict()})

    def same_data_as(self, other: "PriorSpec") -> bool:
        """Whether ``other`` generates the same datasets, ignoring name/count."""
        return self.generative_dict() == other.generative_dict()

    @property
    def cache_key(self) -> str:
        """Filename stem for this spec's cache entry.

        Content-addressed: the key is the *generative* digest, so two specs that
        differ only by ``name`` or ``n_datasets`` share one cache. Renaming a
        spec never invalidates data, and a cache of 256 datasets can serve a
        request for 64 (a prefix) or be extended to 512 -- dataset ``i``'s
        content does not depend on how many were asked for (see
        :meth:`dataset_seed`).
        """
        return self.generative_digest()

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
    "general_ite": PriorSpec(
        name="general_ite",
        n_datasets=1024,
        seq_len=2048,
        mcmc_cate=1,
        min_dim_U=1,
        max_dim_U=10,
        min_dim_W=1,
        max_dim_W=10,
        min_dim_Z=1,
        max_dim_Z=10,
        min_dim_X=1,
        max_dim_X=10,
    ),
    "mc_targets": PriorSpec(
        name="mc_targets",
        n_datasets=128,
        seq_len=256,
        mcmc_cate=100,
        min_dim_U=1,
        max_dim_U=10,
        min_dim_W=1,
        max_dim_W=10,
        min_dim_Z=1,
        max_dim_Z=10,
        min_dim_X=1,
        max_dim_X=10,
    ),
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
    # Ablation: confounding-dim control off, so f_A / f_Y see raw full U with
    # no subsampling or augmentation. Isolates how much of the prior's
    # confounding strength comes from the share-targeting mechanism.
    "no_u_augment": PriorSpec(
        name="no_u_augment",
        fixed_hp_overrides={"confounding_dim_control": False},
    ),
    # Ablation: no feature categorization (see appendix A.3). Pins the
    # per-column categorisation probability to zero; proxy-completeness slots
    # remain governed by proxy_cat_cover_prob, matching the prior behaviour.
    "no_categorization": PriorSpec(
        name="no_categorization",
        sampled_hp_overrides={"cat_col_prob": pinned(0.0)},
    ),
}


@dataclass(frozen=True)
class SpecFamily:
    """A base spec plus named interventions on it, for comparison studies.

    A family answers "what does changing one knob do to the prior?": each
    intervention is a small dict of :class:`PriorSpec` field overrides applied
    on top of ``base``. The comparison axis lives here in config -- inspectable
    and reusable -- rather than being hardcoded inside any analysis; a dimension
    ablation, a ``mcmc_cate`` sweep and a ``seq_len`` sweep are all the same
    object with different interventions.

    Merge semantics: within an intervention, the dict-valued fields
    ``fixed_hp_overrides`` and ``sampled_hp_overrides`` are **deep-merged** over
    the base's (so intervening on ``dim_U_distribution`` atop a base that pins
    ``max_categories`` keeps both); every other field **replaces** the base value.
    Consequence: an intervention cannot *remove* a base override, only add or
    change keys.

    The baseline is not implicit: include it as an explicit identity entry
    (``{}``), which also fixes its label and the plotting order (dict order is
    preserved by :meth:`resolve`).

    Parameters
    ----------
    name : str
        Family label; names the output directory (``results/<name>/...``).

    base : PriorSpec
        The spec every intervention starts from.

    interventions : dict
        ``label -> {PriorSpec field: value}``. ``{}`` is the identity. ``name``
        may not be intervened on (resolve derives variant names).
    """

    name: str
    base: PriorSpec
    interventions: Dict[str, Dict[str, Any]]

    def __post_init__(self) -> None:
        if not self.name or not all(c.isalnum() or c in "_-" for c in self.name):
            raise ValueError(
                f"family name must be non-empty and contain only alphanumerics, "
                f"'_' or '-'; got {self.name!r}"
            )
        if not self.interventions:
            raise ValueError("a SpecFamily needs at least one intervention")
        for label, overrides in self.interventions.items():
            if not label or not all(c.isalnum() or c in "_-" for c in label):
                raise ValueError(
                    f"intervention label must be non-empty and contain only "
                    f"alphanumerics, '_' or '-'; got {label!r}"
                )
            if "name" in overrides:
                raise ValueError(
                    f"intervention {label!r} sets 'name'; variant names are "
                    "derived from the base name and the label"
                )
            unknown = set(overrides) - set(PriorSpec.__dataclass_fields__)
            if unknown:
                raise ValueError(
                    f"intervention {label!r} has unknown PriorSpec fields: "
                    f"{sorted(unknown)}"
                )
        # Eagerly resolve so an invalid override value (e.g. seq_len=1) fails at
        # construction time, not mid-run after other variants were generated.
        self.resolve()

    def resolve(self) -> Dict[str, PriorSpec]:
        """Materialise ``label -> PriorSpec``, preserving intervention order."""
        variants = {}
        for label, overrides in self.interventions.items():
            fields = dict(overrides)
            merged: Dict[str, Any] = {}
            for key in ("fixed_hp_overrides", "sampled_hp_overrides"):
                if key in fields:
                    merged[key] = {**getattr(self.base, key), **fields.pop(key)}
            variants[label] = dataclasses.replace(
                self.base,
                name=f"{self.base.name}_{label}",
                **merged,
                **fields,
            )
        return variants

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable provenance: the resolved spec of every variant."""
        return {
            "name": self.name,
            "base": self.base.to_dict(),
            "variants": {
                label: spec.to_dict() for label, spec in self.resolve().items()
            },
        }


#: Named comparison studies. Referenced from the CLI via ``--family <name>``, so
#: a comparison figure is reproducible from the family name alone.
FAMILIES: Dict[str, SpecFamily] = {
    # Justifies the two prior modifications (triangular d_U, completeness
    # enforcement) by contrast with the prior as it was before each.
    "dim_ablation": SpecFamily(
        name="dim_ablation",
        base=PRESETS["general_ite"],
        interventions={
            "triangular": {},
            "uniform": {"fixed_hp_overrides": {"dim_U_distribution": "uniform"}},
            "unenforced": {
                "fixed_hp_overrides": {"enforce_completeness_heuristic": False}
            },
        },
    ),
    "cat_ablation": SpecFamily(
        name="cat_ablation",
        base=PRESETS["general_ite"],
        interventions={
            "full": {},
            "ablation": {
                "fixed_hp_overrides": {"enforce_categorical_completeness": False}
            },
            "baseline": {
                "fixed_hp_overrides": {
                    "corrupt_U": False,
                    "enforce_categorical_completeness": False,
                }
            },
        },
    ),
    # Tiny end-to-end family for tests and dry runs.
    "smoke_family": SpecFamily(
        name="smoke_family",
        base=PRESETS["smoke"],
        interventions={
            "triangular": {},
            "uniform": {"fixed_hp_overrides": {"dim_U_distribution": "uniform"}},
        },
    ),
}
