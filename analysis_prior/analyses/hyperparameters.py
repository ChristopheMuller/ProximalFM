"""Hyperparameter tables for the prior (appendix B.2).

Two paper tables, both derived from a single :class:`PriorSpec` with no data
generation required:

- **sampled** hyperparameters (``DEFAULT_SAMPLED_HP`` + the spec's overrides):
  drawn once per dataset from a meta-distribution. Table columns describe the
  sampling law, grouped by role (mechanism shape / exogenous noise / causal
  knobs).
- **fixed** hyperparameters (``DEFAULT_FIXED_HP`` + overrides): constants shared
  by every dataset, including the three flags introduced for the appendix
  (``enforce_completeness_heuristic``, ``completeness_max_attempts``,
  ``dim_U_distribution``).

The table builders are pure functions of a spec, so they can be tested on any
spec without touching a cache; the :class:`Analysis` wrapper only supplies
``cache.spec`` and writes the ``.tex``/``.csv`` pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..config import PriorSpec
from ..generate import PriorCache
from ..plotting import save_table
from .base import Analysis, AnalysisArtifacts, register

# ---------------------------------------------------------------------------
# Grouping and human-readable notes
# ---------------------------------------------------------------------------

#: Sampled hyperparameters, grouped by role and ordered for the table. Keys not
#: listed here still appear (under "Other"), so a new HP is never silently
#: dropped from the table.
SAMPLED_GROUPS: List[Tuple[str, List[str]]] = [
    (
        "Mechanism shape",
        [
            "num_layers",
            "hidden_dim",
            "mlp_activations",
            "init_std",
            "noise_std",
            "block_wise_dropout",
            "mlp_dropout_prob",
        ],
    ),
    (
        "Exogenous noise",
        ["sampling", "pre_sample_cause_stats", "pre_sample_noise_std"],
    ),
    (
        "Causal knobs",
        [
            "propensity_shift",
            "propensity_scale",
            "confounding_duplication_of_U_in_Y",
            "confounding_duplication_of_U_in_A",
        ],
    ),
]

#: Fixed hyperparameters, grouped and ordered. TreeSCM-only knobs (mixture
#: weights + tree-specific parameters, see :class:`proximalfm.prior._dataset`)
#: are dropped from the table entirely by :func:`build_fixed_table`, not just
#: left ungrouped, since this paper's spec never exercises the tree-SCM branch.
FIXED_GROUPS: List[Tuple[str, List[str]]] = [
    ("Feature corruption", ["cat_prob", "max_categories", "corrupt_U"]),
    ("Positivity", ["propensity_epsilon"]),
    ("Monte Carlo", ["prior_mc_mem_budget"]),
    (
        "Completeness \\& dims",
        [
            "enforce_completeness_heuristic",
            "completeness_max_attempts",
            "dim_U_distribution",
        ],
    ),
]

#: Fixed hyperparameters tied to the TreeSCM mechanism family (see
#: :class:`proximalfm.prior._dataset.SCMPrior`). Excluded from
#: :func:`build_fixed_table` rather than merely left out of ``FIXED_GROUPS``,
#: so they don't resurface as an "Other" group.
TREE_SCM_FIXED_HP = {
    "mix_probs",
    "tree_model",
    "tree_depth_lambda",
    "tree_n_estimators_lambda",
}

#: One-line gloss per hyperparameter. Kept deliberately terse (fits a table
#: cell) and free of LaTeX specials so it can be printed without escaping.
NOTES: Dict[str, str] = {
    # sampled -- mechanism shape
    "num_layers": "mechanism MLP depth",
    "hidden_dim": "mechanism MLP width",
    "mlp_activations": "activation, drawn from a random library",
    "init_std": "weight-initialisation scale",
    "noise_std": "per-layer additive noise scale",
    "block_wise_dropout": "block-sparse vs.\\ dense weight init",
    "mlp_dropout_prob": "dropout rate (dense init only)",
    # sampled -- exogenous noise
    "sampling": "family of the root cause variables",
    "pre_sample_cause_stats": "pre-sample per-cause mean/std (normal sampling)",
    "pre_sample_noise_std": "pre-sample the layer noise scale",
    # sampled -- causal knobs
    "propensity_shift": "treatment-logit bias (marginal treated fraction)",
    "propensity_scale": "treatment-logit gain (overlap vs.\\ confounding)",
    "confounding_duplication_of_U_in_Y": "share of $U$ re-encoded into $f_Y$",
    "confounding_duplication_of_U_in_A": "share of $U$ re-encoded into $f_A$",
    # fixed
    "mix_probs": "MLP vs.\\ tree SCM mixture weights",
    "tree_model": "tree SCM family",
    "tree_depth_lambda": "tree-depth exponential rate",
    "tree_n_estimators_lambda": "tree-count exponential rate",
    "cat_prob": "per-column categorisation probability",
    "max_categories": "cap on categories per column",
    "corrupt_U": "categorise $U$ as well as the proxies",
    "enforce_categorical_completeness": "accept/reject on each proxy dominating $U$'s category profile",
    "proxy_cat_cover_prob": "chance a proxy mirrors a $U$ category (vs.\\ a continuous wildcard)",
    "propensity_epsilon": "positivity clamp on the propensity",
    "prior_mc_mem_budget": "memory cap on the MC-CATE pass",
    "enforce_completeness_heuristic": "accept/reject on $\\min(d_W, d_Z) \\ge d_U$",
    "completeness_max_attempts": "resample budget before giving up",
    "dim_U_distribution": "sampling law for $d_U$",
}


# ---------------------------------------------------------------------------
# Value / distribution formatting (LaTeX-ready)
# ---------------------------------------------------------------------------


def _num(x: Any) -> str:
    """Compact number for a math cell: drops trailing zeros, keeps ``inf``."""
    if isinstance(x, bool):  # guard: bool is an int subclass
        return "True" if x else "False"
    if isinstance(x, float) and math.isinf(x):
        return "\\infty"
    return f"{x:g}"


def _tt(text: str) -> str:
    r"""``\texttt`` with underscores escaped, for a config key or string value.

    Also inserts ``\allowbreak`` after each underscore. Typewriter identifiers
    like ``confounding_duplication_of_U_in_Y`` are otherwise a single
    unbreakable box: fine in an ``l`` column, but once the cell lives in a
    wrapped ``tabularx`` ``X`` column (see :func:`save_table`'s
    ``wrap_columns``) a long name would still force the column wide unless it
    has explicit break points, since automatic hyphenation does not apply
    inside ``\texttt``.
    """
    return "\\texttt{" + str(text).replace("_", "\\_\\allowbreak ") + "}"


def _choice_value(v: Any) -> str:
    """Render one categorical option (string/bool) for a choice-set."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return _tt(v)
    return _num(v)


def describe_sampled(cfg: Dict[str, Any]) -> str:
    """Human-readable LaTeX description of a sampled-HP distribution config.

    Covers exactly the distribution families used in ``DEFAULT_SAMPLED_HP``
    (see :mod:`proximalfm.prior._hp_sampling`). Unknown families fall back to
    the raw distribution name so the table still renders rather than raising.
    """
    dist = cfg["distribution"]

    if dist == "uniform":
        return f"$\\mathrm{{Unif}}[{_num(cfg['min'])}, {_num(cfg['max'])}]$"

    if dist == "beta":
        return f"$\\mathrm{{Beta}}({_num(cfg['a'])}, {_num(cfg['b'])})$"

    if dist == "meta_beta":
        # value = scale * Beta(b, k) with b, k ~ Unif[min, max] per dataset.
        return (
            f"scaled Beta on $[0, {_num(cfg['scale'])}]$, "
            f"shapes $\\sim\\mathrm{{Unif}}[{_num(cfg['min'])}, {_num(cfg['max'])}]$"
        )

    if dist == "meta_trunc_norm_log_scaled":
        lower = cfg.get("lower_bound", 0.0)
        kind = "int" if cfg.get("round", False) else "real"
        return (
            "log-scaled trunc.\\ normal, "
            f"mean $\\sim\\log\\mathrm{{Unif}}[{_num(cfg['min_mean'])}, "
            f"{_num(cfg['max_mean'])}]$, {kind} $\\ge {_num(lower)}$"
        )

    if dist == "meta_choice":
        options = ", ".join(_choice_value(v) for v in cfg["choice_values"])
        return "categorical $\\{" + options + "\\}$ (random weights)"

    if dist == "meta_choice_mixed":
        n = len(cfg["choice_values"])
        return f"categorical over {n} random activation functions"

    return _tt(dist)


def format_fixed(value: Any) -> str:
    """LaTeX description of a fixed-HP value (scalar, bool, string, or tuple)."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float) and math.isinf(value):
        return "$\\infty$"
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(_num(v) for v in value) + ")"
    if isinstance(value, str):
        return _tt(value)
    if isinstance(value, int) and abs(value) >= 10_000:
        # Group digits so a memory budget like 100000000 stays legible.
        return "$" + f"{value:,}".replace(",", "{,}") + "$"
    return f"${_num(value)}$"


# ---------------------------------------------------------------------------
# Table construction (pure functions of a spec)
# ---------------------------------------------------------------------------


def _grouped_rows(
    values: Dict[str, Any],
    groups: List[Tuple[str, List[str]]],
    describe,
    value_header: str,
) -> pd.DataFrame:
    """Assemble a grouped table from a resolved-HP dict.

    ``describe`` maps a value to its display string. Keys present in ``values``
    but absent from ``groups`` are collected into a trailing "Other" group so
    the table can never silently omit a hyperparameter. The group label is
    printed once per group (blank on subsequent rows) for a clean float.
    """
    listed = {key for _, keys in groups for key in keys}
    extra = [key for key in values if key not in listed]
    ordered = list(groups) + ([("Other", extra)] if extra else [])

    rows = []
    for group, keys in ordered:
        first = True
        for key in keys:
            if key not in values:
                continue
            rows.append(
                {
                    "group": group if first else "",
                    "hyperparameter": _tt(key),
                    value_header: describe(values[key]),
                    "description": NOTES.get(key, ""),
                }
            )
            first = False
    return pd.DataFrame(
        rows, columns=["group", "hyperparameter", value_header, "description"]
    )


def build_sampled_table(spec: PriorSpec) -> pd.DataFrame:
    """Table of the per-dataset *sampled* hyperparameters for ``spec``."""
    _, sampled = spec.resolved_hp()
    return _grouped_rows(sampled, SAMPLED_GROUPS, describe_sampled, "sampling law")


def build_fixed_table(spec: PriorSpec) -> pd.DataFrame:
    """Table of the *fixed* hyperparameters for ``spec``.

    TreeSCM-only knobs (:data:`TREE_SCM_FIXED_HP`) are dropped unconditionally:
    this paper's spec never exercises the tree-SCM branch, so they'd otherwise
    show up as noise (an "Other" group, since they're absent from
    ``FIXED_GROUPS``).
    """
    fixed, _ = spec.resolved_hp()
    fixed = {k: v for k, v in fixed.items() if k not in TREE_SCM_FIXED_HP}
    return _grouped_rows(fixed, FIXED_GROUPS, format_fixed, "value")


# ---------------------------------------------------------------------------
# Analysis wrapper
# ---------------------------------------------------------------------------


@dataclass
class HyperparameterTables:
    """The two hyperparameter tables plus the spec name they describe."""

    sampled: pd.DataFrame
    fixed: pd.DataFrame
    spec_name: str


@register
class HyperparametersAnalysis(Analysis):
    name = "hyperparameters"
    description = "Sampled and fixed prior hyperparameters as paper tables."

    def compute(self, cache: PriorCache) -> HyperparameterTables:
        # Only the spec is needed: hyperparameters are a property of the
        # configuration, not of any realised dataset.
        spec = cache.spec
        return HyperparameterTables(
            sampled=build_sampled_table(spec),
            fixed=build_fixed_table(spec),
            spec_name=spec.name,
        )

    def render(self, results: HyperparameterTables, outdir: Path) -> AnalysisArtifacts:
        outdir = Path(outdir)

        tables = save_table(
            results.sampled,
            outdir,
            "sampled_hyperparameters",
            caption=(
                "Per-dataset sampled hyperparameters of the prior. Each is drawn "
                "once per dataset from the listed meta-distribution; "
                "log-scaled truncated normals draw their centre log-uniformly "
                "over the stated mean range and add the lower bound as an offset."
            ),
            label="tab:prior_sampled_hp",
            escape=False,
            wrap_columns=["hyperparameter", "sampling law", "description"],
            small=True,
            centering=True,
        )
        tables += save_table(
            results.fixed,
            outdir,
            "fixed_hyperparameters",
            caption=(
                "Fixed hyperparameters of the prior, held constant across every "
                "dataset. The last group collects the flags introduced for this "
                "appendix (completeness enforcement and the $d_U$ sampling law)."
            ),
            label="tab:prior_fixed_hp",
            escape=False,
            wrap_columns=["hyperparameter", "description"],
            small=True,
            centering=True,
        )

        summary = {
            "spec_name": results.spec_name,
            "n_sampled_hp": int(len(results.sampled)),
            "n_fixed_hp": int(len(results.fixed)),
        }
        return AnalysisArtifacts(summary=summary, figures=[], tables=tables)
