"""Tests for the hyperparameter tables (appendix B.2).

The table builders are pure functions of a :class:`PriorSpec`, so they are
tested directly without generating any data. Formatting helpers are pinned on
representative inputs.
"""

import math

import pytest

from analysis_prior.analyses.hyperparameters import (
    FIXED_GROUPS,
    NOTES,
    SAMPLED_GROUPS,
    build_fixed_table,
    build_sampled_table,
    describe_sampled,
    format_fixed,
)
from analysis_prior.config import PriorSpec
from proximalfm.prior._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


def _tt(key: str) -> str:
    """Build the ``\\texttt{...}`` cell a bare key renders to (mirrors the
    module's escaping + \\allowbreak insertion), so tests can address rows by
    key without hard-coding the break points."""
    return "\\texttt{" + key.replace("_", "\\_\\allowbreak ") + "}"


def _bare_key(cell: str) -> str:
    """Invert ``_tt``: recover the bare key from a rendered cell."""
    return (
        cell.replace("\\texttt{", "")
        .replace("}", "")
        .replace("\\allowbreak ", "")
        .replace("\\_", "_")
    )


# --------------------------------------------------------------------------
# Distribution / value formatting
# --------------------------------------------------------------------------


def test_describe_uniform():
    assert (
        describe_sampled({"distribution": "uniform", "min": -2.5, "max": 2.5})
        == "$\\mathrm{Unif}[-2.5, 2.5]$"
    )


def test_describe_beta():
    assert (
        describe_sampled({"distribution": "beta", "a": 1.0, "b": 1.0})
        == "$\\mathrm{Beta}(1, 1)$"
    )


def test_describe_meta_trunc_norm_reports_kind_and_bound():
    out = describe_sampled(
        {
            "distribution": "meta_trunc_norm_log_scaled",
            "min_mean": 1,
            "max_mean": 6,
            "round": True,
            "lower_bound": 2,
        }
    )
    assert "int $\\ge 2$" in out
    assert "[1, 6]" in out


def test_describe_meta_trunc_norm_continuous_is_real():
    out = describe_sampled(
        {
            "distribution": "meta_trunc_norm_log_scaled",
            "min_mean": 0.01,
            "max_mean": 10.0,
            "round": False,
            "lower_bound": 0.0,
        }
    )
    assert "real $\\ge 0$" in out


def test_describe_meta_choice_lists_options():
    out = describe_sampled(
        {"distribution": "meta_choice", "choice_values": ["normal", "mixed"]}
    )
    assert "\\texttt{normal}" in out and "\\texttt{mixed}" in out


def test_describe_meta_choice_mixed_counts_without_dumping_values():
    # choice_values here are non-serialisable activation classes; the describer
    # must only count them, never render them.
    out = describe_sampled(
        {"distribution": "meta_choice_mixed", "choice_values": [object(), object()]}
    )
    assert out == "categorical over 2 random activation functions"


def test_format_fixed_bool_inf_tuple_string_and_bigint():
    assert format_fixed(True) == "True"
    assert format_fixed(float("inf")) == "$\\infty$"
    assert format_fixed((0.7, 0.3)) == "(0.7, 0.3)"
    assert format_fixed("triangular") == "\\texttt{triangular}"
    assert format_fixed(100_000_000) == "$100{,}000{,}000$"
    assert format_fixed(0.001) == "$0.001$"


# --------------------------------------------------------------------------
# Table construction
# --------------------------------------------------------------------------


def test_sampled_table_covers_every_default_sampled_hp():
    table = build_sampled_table(PriorSpec(name="default"))
    # The hyperparameter column is \texttt-wrapped; recover the bare keys.
    listed = {_bare_key(cell) for cell in table["hyperparameter"]}
    assert listed == set(DEFAULT_SAMPLED_HP)


def test_fixed_table_covers_every_default_fixed_hp():
    table = build_fixed_table(PriorSpec(name="default"))
    listed = {_bare_key(cell) for cell in table["hyperparameter"]}
    assert listed == set(DEFAULT_FIXED_HP)


def test_group_label_printed_once_per_group():
    table = build_sampled_table(PriorSpec(name="default"))
    non_empty = [g for g in table["group"] if g]
    # Each configured group heads exactly one row.
    assert non_empty == [name for name, _ in SAMPLED_GROUPS]


def test_overrides_flow_into_the_tables():
    spec = PriorSpec(
        name="ablation",
        fixed_hp_overrides={"dim_U_distribution": "uniform", "cat_prob": 0.0},
    )
    fixed = build_fixed_table(spec).set_index("hyperparameter")
    assert fixed.loc[_tt("dim_U_distribution"), "value"] == "\\texttt{uniform}"
    assert fixed.loc[_tt("cat_prob"), "value"] == "$0$"


def test_unlisted_hp_lands_in_an_other_group():
    # A hyperparameter with no assigned group must still appear, not vanish.
    spec = PriorSpec(name="x", fixed_hp_overrides={"brand_new_flag": 1})
    table = build_fixed_table(spec)
    keys = table["hyperparameter"].tolist()
    assert _tt("brand_new_flag") in keys
    assert "Other" in table["group"].tolist()


def test_every_default_key_has_a_note():
    for key in list(DEFAULT_SAMPLED_HP) + list(DEFAULT_FIXED_HP):
        assert key in NOTES, f"missing note for {key}"


def test_tables_are_not_empty_and_have_expected_columns():
    sampled = build_sampled_table(PriorSpec(name="default"))
    fixed = build_fixed_table(PriorSpec(name="default"))
    assert list(sampled.columns) == [
        "group",
        "hyperparameter",
        "sampling law",
        "description",
    ]
    assert list(fixed.columns) == ["group", "hyperparameter", "value", "description"]
    assert len(sampled) == len(DEFAULT_SAMPLED_HP)
    assert len(fixed) == len(DEFAULT_FIXED_HP)
