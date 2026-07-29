"""Hyperparameter tables for the prior (appendix B.2).

Three paper tables, all derived from a single :class:`PriorSpec` with no data
generation required:

- **structural** parameters: the :class:`PriorSpec` fields themselves (block
  dimension ranges, ``seq_len``, ``mcmc_cate``, ``prior_type``). These live on
  the spec rather than in the hyperparameter dicts only because the training
  CLI passes them separately, but they are the most load-bearing knobs in the
  prior -- $d^{\\max}$ is what makes the dimension laws of B.2.2 concrete -- so
  the appendix would not be self-contained without them.
- **sampled** hyperparameters (``DEFAULT_SAMPLED_HP`` + the spec's overrides):
  drawn once per dataset from a meta-distribution. Table columns describe the
  sampling law, grouped by role (mechanism shape / exogenous noise / causal
  knobs).
- **fixed** hyperparameters (``DEFAULT_FIXED_HP`` + overrides): constants shared
  by every dataset.

Not everything in ``DEFAULT_FIXED_HP`` reaches the paper: see
:data:`REMOVED_FIXED_HP` for what is dropped and why.

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
        "Mechanism",
        [
            "num_layers",
            "hidden_dim",
            "mlp_activations",
            "init_std",
            "block_wise_dropout",
            "mlp_dropout_prob",
        ],
    ),
    (
        "Noise",
        [
            "sampling",
            "pre_sample_cause_stats",
            "noise_share_of_input_in_U",
            "noise_share_of_input_in_W",
            "noise_share_of_input_in_Z",
            "noise_share_of_input_in_A",
            "noise_share_of_input_in_Y",
        ],
    ),
    (
        "Causal",
        [
            "propensity_shift",
            "propensity_scale",
            "confounding_share_of_U_in_Y",
            "confounding_share_of_U_in_A",
            "direct_edge_share_of_W_in_Y",
            "direct_edge_share_of_Z_in_A",
            "covariate_share_of_X_in_U",
            "covariate_share_of_X_in_W",
            "covariate_share_of_X_in_Z",
            "covariate_share_of_X_in_A",
            "covariate_share_of_X_in_Y",
        ],
    ),
    ("Feature corruption", ["cat_col_prob"]),
]

FIXED_GROUPS: List[Tuple[str, List[str]]] = [
    ("Mechanism", ["noise_std"]),
    ("Feature corruption", ["max_categories", "corrupt_U", "proxy_cat_cover_prob"]),
    ("Positivity", ["propensity_epsilon"]),
    ("Completeness", ["completeness_max_attempts"]),
]

#: Fixed hyperparameters kept out of the paper table entirely. Dropped in
#: :func:`build_fixed_table` rather than merely left out of ``FIXED_GROUPS``,
#: so they don't resurface as an "Other" group. Three reasons to be here:
#:
#: - **Dead branch.** The tree-SCM knobs (``mix_probs``, ``tree_*``): this
#:   paper's prior is ``mlp_scm`` throughout, so they configure code that never
#:   runs (see :class:`proximalfm.prior._dataset.SCMPrior`).
#: - **Always-on switches.** ``enforce_*``, ``*_subsampling``,
#:   ``confounding_dim_control``, ``dim_U_distribution``: booleans (or a single
#:   named law) whose shipped value *is* the behaviour the appendix prose
#:   describes at length. A row reading ``True`` next to "accept/reject on
#:   $\\min(d_W,d_Z)\\ge d_U$" adds nothing to the prose and misleadingly
#:   advertises a degree of freedom the prior does not exercise. Where the
#:   value does change what a reader sees, it still shows up -- via
#:   :func:`build_structural_table`, whose dimension laws are written out of
#:   ``dim_U_distribution`` and ``enforce_completeness_heuristic``.
#: - **No statistical content.** ``prior_mc_mem_budget`` caps chunk size on the
#:   MC-CATE pass; it cannot change the distribution being sampled.
REMOVED_FIXED_HP = {
    # dead branch: tree SCM
    "mix_probs",
    "tree_model",
    "tree_depth_lambda",
    "tree_n_estimators_lambda",
    # always-on switches, described in prose instead
    "enforce_categorical_completeness",
    "enforce_completeness_heuristic",
    "confounding_dim_control",
    "proxy_edge_subsampling",
    "covariate_edge_subsampling",
    "dim_U_distribution",
    # implementation-only
    "prior_mc_mem_budget",
}

#: Structural parameters, grouped and ordered for the table. Unlike the two
#: hyperparameter dicts these are :class:`PriorSpec` *fields*, so the keys here
#: are the display names built by :func:`build_structural_table`, not config
#: keys resolved from a dict.
STRUCTURAL_GROUPS: List[Tuple[str, List[str]]] = [
    ("Block dimensions", ["dim_X", "dim_U", "dim_W", "dim_Z"]),
    ("Dimension bounds", ["max_dim_X", "max_dim_U", "max_dim_W", "max_dim_Z"]),
    ("Dataset", ["seq_len", "mcmc_cate", "prior_type"]),
]

#: One-line gloss per hyperparameter. Kept deliberately terse (fits a table
#: cell) and free of LaTeX specials so it can be printed without escaping.
NOTES: Dict[str, str] = {
    # sampled -- mechanism shape
    "num_layers": "mechanism MLP depth",
    "hidden_dim": "mechanism MLP width",
    "mlp_activations": "activation, drawn from a random library",
    "init_std": "weight-initialisation scale",
    "block_wise_dropout": "block-sparse vs.\\ dense weight init",
    "mlp_dropout_prob": "dropout rate (dense init only)",
    # sampled -- exogenous noise
    "sampling": "family of the root cause variables",
    "pre_sample_cause_stats": "pre-sample per-cause mean/std (normal sampling)",
    "noise_share_of_input_in_U": "target noise share of $f_U$ inputs ($\\varepsilon_U$ width)",
    "noise_share_of_input_in_W": "target noise share of $f_W$ inputs ($\\varepsilon_W$ width)",
    "noise_share_of_input_in_Z": "target noise share of $f_Z$ inputs ($\\varepsilon_Z$ width)",
    "noise_share_of_input_in_A": "target noise share of $f_A$ inputs ($\\varepsilon_A$ width)",
    "noise_share_of_input_in_Y": "target noise share of $f_Y$ inputs ($\\varepsilon_Y$ width)",
    # sampled -- causal knobs
    "propensity_shift": "treatment-logit bias (marginal treated fraction)",
    "propensity_scale": "treatment-logit gain (overlap vs.\\ confounding)",
    "confounding_share_of_U_in_Y": "target share of $f_Y$ inputs drawn from $U$",
    "confounding_share_of_U_in_A": "target share of $f_A$ inputs drawn from $U$",
    "direct_edge_share_of_W_in_Y": "per-column keep probability of the direct $W \\to Y$ edge",
    "direct_edge_share_of_Z_in_A": "per-column keep probability of the direct $Z \\to A$ edge",
    "covariate_share_of_X_in_U": "per-column keep probability of the direct $X \\to U$ edge",
    "covariate_share_of_X_in_W": "per-column keep probability of the direct $X \\to W$ edge",
    "covariate_share_of_X_in_Z": "per-column keep probability of the direct $X \\to Z$ edge",
    "covariate_share_of_X_in_A": "per-column keep probability of the direct $X \\to A$ edge",
    "covariate_share_of_X_in_Y": "per-column keep probability of the direct $X \\to Y$ edge",
    "cat_col_prob": "per-column categorisation probability",
    # structural (PriorSpec fields, not config keys)
    "dim_X": "observed baseline covariate dimension",
    "dim_U": "latent confounder dimension",
    "dim_W": "outcome-inducing proxy dimension",
    "dim_Z": "treatment-inducing proxy dimension",
    "max_dim_X": "covariate-dimension ceiling",
    "max_dim_U": "confounder-dimension ceiling ($d^{\\max}$ of \\Cref{app:prior_dimensions})",
    "max_dim_W": "outcome-proxy dimension ceiling",
    "max_dim_Z": "treatment-proxy dimension ceiling",
    "seq_len": "rows per dataset",
    "mcmc_cate": "MC replicates $K$ per row behind the $\\mu_0/\\mu_1$ targets",
    "prior_type": "mechanism family",
    # fixed
    "mix_probs": "MLP vs.\\ tree SCM mixture weights",
    "tree_model": "tree SCM family",
    "tree_depth_lambda": "tree-depth exponential rate",
    "tree_n_estimators_lambda": "tree-count exponential rate",
    "max_categories": "cap on categories per column",
    "corrupt_U": "categorise $U$ as well as the proxies",
    "enforce_categorical_completeness": "accept/reject on each proxy dominating $U$'s category profile",
    "proxy_cat_cover_prob": "chance a proxy mirrors a $U$ category (vs.\\ a continuous wildcard)",
    "propensity_epsilon": "positivity clamp on the propensity",
    "prior_mc_mem_budget": "memory cap on the MC-CATE pass",
    "enforce_completeness_heuristic": "accept/reject on $\\min(d_W, d_Z) \\ge d_U$",
    "completeness_max_attempts": "resample budget before giving up",
    "dim_U_distribution": "sampling law for $d_U$",
    "confounding_dim_control": "subsample/augment $U$'s inputs to $f_Y$, $f_A$ to a target share",
    "proxy_edge_subsampling": "make the direct $W \\to Y$ / $Z \\to A$ edges per-column optional",
    "covariate_edge_subsampling": "make the $X \\to \\{U,W,Z,A,Y\\}$ edges per-column optional",
    "noise_std": "tiny fixed per-layer dither so continuous columns realise distinct values",
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
    like ``confounding_share_of_U_in_Y`` are otherwise a single
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


def _ttm(name: str) -> str:
    r"""``\texttt`` for a config key used *inside* a math cell.

    Unlike :func:`_tt` this omits the ``\allowbreak`` hints: these names sit
    inside a ``$\{\dots\}$`` support where a line break would split the set
    notation, and they are short enough not to need one.
    """
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def _int_range(lo: int, hi: str) -> str:
    """LaTeX for the support ``{lo, ..., hi}``.

    ``hi`` is a pre-rendered upper bound -- for the structural table it is the
    *name* of the bound (``max_dim_X``) rather than its value, which is listed
    on its own row, so the law and the number it depends on stay visibly tied.
    """
    return f"\\{{{_num(lo)}, \\dots, {hi}\\}}"


def _proxy_law(lo: int, hi: str, clamped: bool) -> str:
    """Sampling law of a proxy dimension, given the completeness clamp.

    With the heuristic on, the generator raises the lower bound to $d_U$, so the
    law is conditional on the confounder dimension rather than plain uniform.
    """
    if not clamped:
        return f"$\\mathrm{{Unif}}{_int_range(lo, hi)}$"
    bound = "d_U" if lo <= 1 else f"\\max({_num(lo)}, d_U)"
    return f"$\\mathrm{{Unif}}\\{{{bound}, \\dots, {hi}\\}}$"


def build_structural_table(spec: PriorSpec) -> pd.DataFrame:
    """Table of the *structural* parameters of ``spec``.

    These are :class:`PriorSpec` fields rather than entries of the two
    hyperparameter dicts, so the rows are built explicitly. Two of them read
    the resolved fixed HP: ``dim_U_distribution`` and
    ``enforce_completeness_heuristic`` decide which sampling law the dimension
    rows report, which is how those (otherwise dropped, see
    :data:`REMOVED_FIXED_HP`) flags still reach the reader.

    ``name``, ``n_datasets`` and ``seed`` are omitted: they say how many draws
    an audit asked for and under which seed, not what the prior is.

    The dimension laws are written in terms of the ``max_dim_*`` bounds, which
    get a row of their own: the ceiling is the quantity the rest of the appendix
    argues about (it is the $d^{\\max}$ of the marginal derivations), so it
    should be readable as a number without being buried inside four separate
    supports. The ``min_dim_*`` bounds stay inlined in the laws, where they are
    already visible.
    """
    fixed, _ = spec.resolved_hp()

    up_U, up_W = _ttm("max_dim_U"), _ttm("max_dim_W")
    up_Z, up_X = _ttm("max_dim_Z"), _ttm("max_dim_X")

    dU_law = (
        f"triangular on ${_int_range(spec.min_dim_U, up_U)}$"
        if fixed.get("dim_U_distribution") == "triangular"
        else f"$\\mathrm{{Unif}}{_int_range(spec.min_dim_U, up_U)}$"
    )
    clamped = bool(fixed.get("enforce_completeness_heuristic"))

    values = {
        "dim_X": f"$\\mathrm{{Unif}}{_int_range(spec.min_dim_X, up_X)}$",
        "dim_U": dU_law,
        "dim_W": _proxy_law(spec.min_dim_W, up_W, clamped),
        "dim_Z": _proxy_law(spec.min_dim_Z, up_Z, clamped),
        "max_dim_X": f"${_num(spec.max_dim_X)}$",
        "max_dim_U": f"${_num(spec.max_dim_U)}$",
        "max_dim_W": f"${_num(spec.max_dim_W)}$",
        "max_dim_Z": f"${_num(spec.max_dim_Z)}$",
        "seq_len": f"${_num(spec.seq_len)}$",
        "mcmc_cate": f"${_num(spec.mcmc_cate)}$",
        "prior_type": _tt(spec.prior_type),
    }
    # Values are pre-rendered, so the describer is the identity.
    return _grouped_rows(values, STRUCTURAL_GROUPS, lambda v: v, "value")


def build_sampled_table(spec: PriorSpec) -> pd.DataFrame:
    """Table of the per-dataset *sampled* hyperparameters for ``spec``."""
    _, sampled = spec.resolved_hp()
    return _grouped_rows(sampled, SAMPLED_GROUPS, describe_sampled, "sampling law")


def build_fixed_table(spec: PriorSpec) -> pd.DataFrame:
    """Table of the *fixed* hyperparameters for ``spec``.

    Keys in :data:`REMOVED_FIXED_HP` are dropped unconditionally, so what
    survives is the set of constants a reader could not recover from the
    appendix prose: numeric values that pin down a behaviour described only
    qualitatively there.
    """
    fixed, _ = spec.resolved_hp()
    fixed = {k: v for k, v in fixed.items() if k not in REMOVED_FIXED_HP}
    return _grouped_rows(fixed, FIXED_GROUPS, format_fixed, "value")


# ---------------------------------------------------------------------------
# Analysis wrapper
# ---------------------------------------------------------------------------


@dataclass
class HyperparameterTables:
    """The three parameter tables plus the spec name they describe."""

    structural: pd.DataFrame
    sampled: pd.DataFrame
    fixed: pd.DataFrame
    spec_name: str


@register
class HyperparametersAnalysis(Analysis):
    name = "hyperparameters"
    description = "Structural, sampled and fixed prior parameters as paper tables."

    def compute(self, cache: PriorCache) -> HyperparameterTables:
        # Only the spec is needed: hyperparameters are a property of the
        # configuration, not of any realised dataset.
        spec = cache.spec
        return HyperparameterTables(
            structural=build_structural_table(spec),
            sampled=build_sampled_table(spec),
            fixed=build_fixed_table(spec),
            spec_name=spec.name,
        )

    def render(self, results: HyperparameterTables, outdir: Path) -> AnalysisArtifacts:
        outdir = Path(outdir)

        tables = save_table(
            results.structural,
            outdir,
            "structural_hyperparameters",
            caption=(
                "Structural parameters of the prior: the block dimensions and "
                "the dataset shape. Each dimension is drawn per dataset from "
                "the law shown, stated in terms of the ceilings listed below "
                "it. The proxy dimensions are drawn conditionally on $d_U$ so "
                "that the completeness heuristic of "
                "\\Cref{app:prior_procedure} holds by construction; the "
                "resulting marginals are derived in "
                "\\Cref{app:prior_dimensions}."
            ),
            label="tab:prior_structural_hp",
            escape=False,
            wrap_columns=["hyperparameter", "value", "description"],
            small=True,
            centering=True,
            group_column="group",
        )
        tables += save_table(
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
            group_column="group",
        )
        tables += save_table(
            results.fixed,
            outdir,
            "fixed_hyperparameters",
            caption=(
                "Fixed hyperparameters of the prior, held constant across every "
                "dataset. Listed are the constants that pin down behaviour the "
                "surrounding text describes only qualitatively; switches whose "
                "shipped value is that behaviour itself are stated in the text "
                "instead."
            ),
            label="tab:prior_fixed_hp",
            escape=False,
            wrap_columns=["hyperparameter", "description"],
            small=True,
            centering=True,
            group_column="group",
        )

        summary = {
            "spec_name": results.spec_name,
            "n_structural": int(len(results.structural)),
            "n_sampled_hp": int(len(results.sampled)),
            "n_fixed_hp": int(len(results.fixed)),
        }
        return AnalysisArtifacts(summary=summary, figures=[], tables=tables)
