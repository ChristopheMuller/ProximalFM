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
    REMOVED_FIXED_HP,
    SAMPLED_GROUPS,
    STRUCTURAL_GROUPS,
    build_fixed_table,
    build_sampled_table,
    build_structural_table,
    describe_sampled,
    format_fixed,
)
from analysis_prior.config import PRESETS, PriorSpec
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


def test_fixed_table_covers_every_default_fixed_hp_except_removed():
    table = build_fixed_table(PriorSpec(name="default"))
    listed = {_bare_key(cell) for cell in table["hyperparameter"]}
    assert listed == set(DEFAULT_FIXED_HP) - REMOVED_FIXED_HP
    assert listed.isdisjoint(REMOVED_FIXED_HP)


def test_removed_fixed_hp_are_all_real_config_keys():
    # A typo here would silently fail to drop the key it meant to.
    assert REMOVED_FIXED_HP <= set(DEFAULT_FIXED_HP)


def test_fixed_table_has_no_other_group():
    # Every surviving key must have an explicit home in FIXED_GROUPS; an
    # "Other" group means the grouping has drifted from the config.
    table = build_fixed_table(PriorSpec(name="default"))
    assert "Other" not in table["group"].tolist()


def test_group_label_printed_once_per_group():
    table = build_sampled_table(PriorSpec(name="default"))
    non_empty = [g for g in table["group"] if g]
    # Each configured group heads exactly one row.
    assert non_empty == [name for name, _ in SAMPLED_GROUPS]


def test_overrides_flow_into_the_tables():
    spec = PriorSpec(
        name="ablation",
        fixed_hp_overrides={"max_categories": 7, "proxy_cat_cover_prob": 0.0},
    )
    fixed = build_fixed_table(spec).set_index("hyperparameter")
    assert fixed.loc[_tt("max_categories"), "value"] == "$7$"
    assert fixed.loc[_tt("proxy_cat_cover_prob"), "value"] == "$0$"


# --------------------------------------------------------------------------
# Structural table
# --------------------------------------------------------------------------


def _bound(name: str) -> str:
    """The in-math rendering of a ``max_dim_*`` name (no \\allowbreak)."""
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def test_structural_table_reports_spec_fields():
    spec = PriorSpec(name="s", seq_len=2048, mc_cate=100, max_dim_X=10)
    table = build_structural_table(spec).set_index("hyperparameter")
    assert table.loc[_tt("seq_len"), "value"] == "$2048$"
    assert table.loc[_tt("mc_cate"), "value"] == "$100$"


def test_production_structural_table_reports_the_training_row_layout():
    table = build_structural_table(PRESETS["pretrain_prod"]).set_index("hyperparameter")
    assert _tt("seq_len") not in table.index
    assert table.loc[_tt("dim_X"), "value"].startswith("$\\mathrm{Unif}\\{0")
    assert (
        table.loc[_tt("context_rows"), "value"]
        == "$\\mathrm{Unif}\\{64, \\dots, 2047\\}$"
    )
    assert table.loc[_tt("n_query"), "value"] == "$128$"
    assert table.loc[_tt("mc_cate"), "value"] == "$250$"
    assert table.loc[_tt("mc_target_rows"), "value"] == "query rows only"


def test_structural_laws_name_their_ceiling_and_the_ceiling_has_a_row():
    # The law references the bound by name; the number lives on its own row.
    spec = PriorSpec(name="s", max_dim_X=7)
    table = build_structural_table(spec).set_index("hyperparameter")
    assert (
        table.loc[_tt("dim_X"), "value"]
        == "$\\mathrm{Unif}\\{1, \\dots, " + _bound("max_dim_X") + "\\}$"
    )
    assert table.loc[_tt("max_dim_X"), "value"] == "$7$"


def test_structural_bound_rows_track_the_spec():
    spec = PriorSpec(name="s", max_dim_U=3, max_dim_W=4, max_dim_Z=5, max_dim_X=6)
    table = build_structural_table(spec).set_index("hyperparameter")
    assert table.loc[_tt("max_dim_U"), "value"] == "$3$"
    assert table.loc[_tt("max_dim_W"), "value"] == "$4$"
    assert table.loc[_tt("max_dim_Z"), "value"] == "$5$"
    assert table.loc[_tt("max_dim_X"), "value"] == "$6$"


def test_structural_table_writes_the_dim_U_law_from_the_fixed_flag():
    triangular = build_structural_table(PriorSpec(name="a")).set_index("hyperparameter")
    assert "triangular" in triangular.loc[_tt("dim_U"), "value"]

    uniform = build_structural_table(
        PriorSpec(name="b", fixed_hp_overrides={"dim_U_distribution": "uniform"})
    ).set_index("hyperparameter")
    assert (
        uniform.loc[_tt("dim_U"), "value"]
        == "$\\mathrm{Unif}\\{1, \\dots, " + _bound("max_dim_U") + "\\}$"
    )


def test_structural_table_clamps_proxy_dims_to_dU_only_when_enforced():
    clamped = build_structural_table(PriorSpec(name="a")).set_index("hyperparameter")
    assert (
        clamped.loc[_tt("dim_W"), "value"]
        == "$\\mathrm{Unif}\\{d_U, \\dots, " + _bound("max_dim_W") + "\\}$"
    )

    free = build_structural_table(
        PriorSpec(
            name="b", fixed_hp_overrides={"enforce_completeness_heuristic": False}
        )
    ).set_index("hyperparameter")
    assert (
        free.loc[_tt("dim_Z"), "value"]
        == "$\\mathrm{Unif}\\{1, \\dots, " + _bound("max_dim_Z") + "\\}$"
    )


def test_structural_table_uses_max_when_the_min_dim_binds():
    spec = PriorSpec(name="s", min_dim_W=2, max_dim_W=5)
    table = build_structural_table(spec).set_index("hyperparameter")
    assert (
        table.loc[_tt("dim_W"), "value"]
        == "$\\mathrm{Unif}\\{\\max(2, d_U), \\dots, " + _bound("max_dim_W") + "\\}$"
    )


def test_structural_table_groups_and_omits_bookkeeping_fields():
    table = build_structural_table(PriorSpec(name="s", n_datasets=17, seed=3))
    keys = {_bare_key(cell) for cell in table["hyperparameter"]}
    assert keys == {
        "dim_X",
        "dim_U",
        "dim_W",
        "dim_Z",
        "max_dim_X",
        "max_dim_U",
        "max_dim_W",
        "max_dim_Z",
        "seq_len",
        "mc_cate",
    }
    assert keys.isdisjoint({"name", "n_datasets", "seed"})
    non_empty = [g for g in table["group"] if g]
    assert non_empty == [name for name, _ in STRUCTURAL_GROUPS]


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
    assert len(fixed) == len(DEFAULT_FIXED_HP) - len(REMOVED_FIXED_HP)
