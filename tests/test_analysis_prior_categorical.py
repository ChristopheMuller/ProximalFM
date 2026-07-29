"""Tests for the categorical-completeness analysis.

The profile reconstruction and the dominance/survival math are tested exactly on
hand-built inputs; the full comparison analysis is exercised once end-to-end on a
tiny three-arm family.
"""

import numpy as np
import pytest

from proximalfm.prior._reg2cls import BlockProfile

from analysis_prior import generate as gen
from analysis_prior.analyses.categorical import (
    CategoricalCompletenessAnalysis,
    binned_category_marginals,
    categorical_survival_stats,
    category_count_distribution,
    composition_by_demand,
    proxy_stats_by_u_feature,
    reconstruct_profile,
)
from analysis_prior.config import PriorSpec, SpecFamily


# --------------------------------------------------------------------------
# reconstruct_profile
# --------------------------------------------------------------------------


def test_reconstruct_profile_separates_continuous_from_categorical():
    # Col 0: many distinct values -> continuous. Col 1: 3 distinct -> categorical.
    block = np.column_stack(
        [
            np.arange(20, dtype=float),  # 20 distinct
            np.array([0, 1, 2] * 6 + [0, 1], dtype=float),  # 3 distinct
        ]
    )
    profile = reconstruct_profile(block, max_categories=5)
    assert profile == BlockProfile(continuous=1, categorical=(3,))


def test_reconstruct_profile_cap_pushes_high_cardinality_to_continuous():
    # A 6-distinct column is categorical at cap 10 but continuous at cap 5.
    block = np.array([[float(i % 6)] for i in range(30)])
    assert reconstruct_profile(block, max_categories=10) == BlockProfile(0, (6,))
    assert reconstruct_profile(block, max_categories=5) == BlockProfile(1, ())


def test_reconstruct_profile_sorts_categories_descending():
    block = np.column_stack(
        [
            np.array([0, 1] * 15, dtype=float),  # 2 distinct
            np.array([0, 1, 2, 3] * 7 + [0, 1], dtype=float),  # 4 distinct
        ]
    )
    profile = reconstruct_profile(block, max_categories=10)
    assert profile == BlockProfile(continuous=0, categorical=(4, 2))


def test_reconstruct_profile_empty_block():
    profile = reconstruct_profile(np.empty((10, 0)), max_categories=5)
    assert profile == BlockProfile(continuous=0, categorical=())


# --------------------------------------------------------------------------
# categorical_survival_stats
# --------------------------------------------------------------------------


def test_survival_all_hold_when_proxies_dominate():
    u = [BlockProfile(2, (3,))]
    w = [BlockProfile(3, ())]  # 1 continuous wildcard covers the categorical(3)
    z = [BlockProfile(2, (4,))]  # exact continuous + categorical(4) >= (3)
    stats = categorical_survival_stats(u, w, z)
    assert stats["holds"].tolist() == [True]
    assert stats["overall_fraction"] == 1.0


def test_survival_detects_cardinality_violation():
    # Matching column counts but insufficient cardinality: categorical(2) < (3).
    u = [BlockProfile(2, (3,))]
    w = [BlockProfile(2, (2,))]  # continuous ok, but 2 < 3 -> fails
    z = [BlockProfile(3, ())]  # fine on its own
    stats = categorical_survival_stats(u, w, z)
    assert stats["holds"].tolist() == [False]  # both must dominate; W fails


def test_survival_requires_both_proxies():
    u = [BlockProfile(1, ())]
    w = [BlockProfile(1, ())]  # dominates
    z = [BlockProfile(0, (2,))]  # 0 continuous < 1 -> fails continuous slot
    stats = categorical_survival_stats(u, w, z)
    assert stats["holds"].tolist() == [False]


def test_survival_breakdown_by_n_cat_U():
    u = [BlockProfile(1, (3,)), BlockProfile(1, (3,)), BlockProfile(1, ())]
    w = [BlockProfile(2, ()), BlockProfile(1, (2,)), BlockProfile(1, ())]
    z = [BlockProfile(2, ()), BlockProfile(2, ()), BlockProfile(1, ())]
    stats = categorical_survival_stats(u, w, z)
    table = stats["by_n_cat_U"].set_index("n_cat_U")
    # n_cat_U == 1: first holds (wildcard covers 3), second fails (2 < 3).
    assert table.loc[1, "n_datasets"] == 2
    assert table.loc[1, "n_hold"] == 1
    assert table.loc[1, "fraction_hold"] == 0.5
    # n_cat_U == 0: continuous U, dimension-style check holds.
    assert table.loc[0, "fraction_hold"] == 1.0


def test_survival_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        categorical_survival_stats([BlockProfile(1, ())], [], [BlockProfile(1, ())])


def test_survival_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        categorical_survival_stats([], [], [])


# --------------------------------------------------------------------------
# category_count_distribution
# --------------------------------------------------------------------------


def test_category_count_distribution_pools_columns():
    profiles = [BlockProfile(2, (3, 2)), BlockProfile(1, (3,))]
    dist = category_count_distribution(profiles, max_categories=10)
    assert dist["continuous"] == 3  # 2 + 1
    assert dist[3] == 2
    assert dist[2] == 1


# --------------------------------------------------------------------------
# binned_category_marginals
# --------------------------------------------------------------------------


def test_binned_category_marginals_proportions_sum_to_one():
    dist = {"continuous": 6, 2: 2, 3: 1, 15: 1}  # 10 columns total
    rows = binned_category_marginals(dist, [(2, 2), (3, 10), (11, 50)])
    labels = [r[0] for r in rows]
    props = dict(rows)
    assert labels == ["cont.", "2", "3–10", "11–50"]
    assert props["cont."] == 0.6
    assert props["2"] == 0.2
    assert props["3–10"] == 0.1  # only the k=3 column
    assert props["11–50"] == 0.1  # the k=15 column
    assert sum(props.values()) == pytest.approx(1.0)


def test_binned_category_marginals_empty_block():
    rows = binned_category_marginals({}, [(2, 2)])
    assert dict(rows) == {"cont.": 0.0, "2": 0.0}


# --------------------------------------------------------------------------
# composition_by_demand
# --------------------------------------------------------------------------


def test_composition_by_demand_aggregates_by_u_cat_count():
    # Two datasets with |cats(U)|=1, one with |cats(U)|=0.
    u = [BlockProfile(2, (3,)), BlockProfile(2, (4,)), BlockProfile(3, ())]
    p = [BlockProfile(4, (5,)), BlockProfile(2, (5, 6)), BlockProfile(5, ())]
    df = composition_by_demand(u, p).set_index("n_cat_U")
    # |cats(U)|=1 bin: proxies have 1 and 2 categorical cols -> mean 1.5.
    assert df.loc[1, "n_datasets"] == 2
    assert df.loc[1, "mean_n_cat_P"] == 1.5
    # fractions categorical: 1/5 and 2/4 -> mean 0.35.
    assert df.loc[1, "mean_frac_cat_P"] == pytest.approx(0.35)
    # |cats(U)|=0 bin: single dataset, sem 0.
    assert df.loc[0, "n_datasets"] == 1
    assert df.loc[0, "mean_n_cat_P"] == 0.0
    assert df.loc[0, "sem_n_cat_P"] == 0.0


def test_composition_by_demand_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        composition_by_demand([BlockProfile(1, ())], [])


# --------------------------------------------------------------------------
# proxy_stats_by_u_feature
# --------------------------------------------------------------------------


def test_proxy_stats_by_u_feature_emits_one_row_per_u_column():
    # U: 2 continuous + 1 categorical(3). Proxy: 3 continuous + 1 categorical(5).
    u = [BlockProfile(2, (3,))]
    p = [BlockProfile(3, (5,))]
    df = proxy_stats_by_u_feature(u, p)
    # One row per U column: 2 continuous (-1) + 1 categorical (3).
    assert sorted(df["u_card"].tolist()) == [-1, -1, 3]
    # Proxy block: 3/4 columns continuous, mean categorical cardinality 5.
    assert (df["prop_num"] == 0.75).all()
    assert (df["mean_card"] == 5.0).all()


def test_proxy_stats_by_u_feature_nan_when_proxy_all_continuous():
    df = proxy_stats_by_u_feature([BlockProfile(0, (2,))], [BlockProfile(4, ())])
    assert df["prop_num"].tolist() == [1.0]
    assert df["mean_card"].isna().all()


def test_proxy_stats_by_u_feature_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        proxy_stats_by_u_feature([BlockProfile(1, ())], [])


# --------------------------------------------------------------------------
# End-to-end on a tiny three-arm family
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_family_caches():
    """default / no_enforcement, mirroring the shipped cat_ablation family."""
    base = PriorSpec(
        name="cattest",
        n_datasets=8,
        seq_len=80,
        mcmc_cate=1,
        min_dim_U=1,
        max_dim_U=3,
        min_dim_W=1,
        max_dim_W=3,
        min_dim_Z=1,
        max_dim_Z=3,
        min_dim_X=1,
        max_dim_X=3,
        seed=0,
    )
    family = SpecFamily(
        name="cat_family",
        base=base,
        interventions={
            "default": {},
            "no_enforcement": {
                "fixed_hp_overrides": {"enforce_categorical_completeness": False}
            },
        },
    )
    return {
        label: gen.generate(spec, n_jobs=1, progress=False)
        for label, spec in family.resolve().items()
    }


def test_compute_reads_flags_from_specs(tiny_family_caches):
    results = CategoricalCompletenessAnalysis().compute(tiny_family_caches)

    assert list(results.variants) == ["default", "no_enforcement"]
    assert results.baseline == "default"
    assert (
        results.variants["default"].corrupt_U and results.variants["default"].enforced
    )
    assert results.variants["no_enforcement"].corrupt_U
    assert not results.variants["no_enforcement"].enforced
    for var in results.variants.values():
        assert set(var.profiles) == {"U", "W", "Z"}
        assert var.n_datasets == 8


def test_enforcement_improves_dominance(tiny_family_caches):
    # Enforcement (default) must dominate the unenforced arm.
    results = CategoricalCompletenessAnalysis().compute(tiny_family_caches)
    default = results.variants["default"].survival["overall_fraction"]
    no_enforcement = results.variants["no_enforcement"].survival["overall_fraction"]
    assert default == 1.0
    assert default > no_enforcement


def test_run_writes_survival_table(tiny_family_caches, tmp_path):
    artifacts = CategoricalCompletenessAnalysis().run(tiny_family_caches, tmp_path)

    names = {p.name for p in artifacts.tables}
    assert "categorical_completeness_survival.tex" in names
    assert "categorical_completeness_survival.csv" in names
    for path in artifacts.tables:
        assert path.exists()

    tex = (tmp_path / "categorical_completeness_survival.tex").read_text()
    assert "\\label{tab:prior_categorical_completeness_survival}" in tex
    assert "\\caption" in tex


def test_run_writes_category_figures(tiny_family_caches, tmp_path):
    artifacts = CategoricalCompletenessAnalysis().run(tiny_family_caches, tmp_path)

    names = {p.name for p in artifacts.figures}
    for stem in (
        "category_count_marginals",
        "proxy_enforcement_vs_u_demand",
    ):
        assert f"{stem}.pdf" in names
        assert f"{stem}.png" in names
    # The joint-count heatmap was removed; make sure it does not come back.
    assert "category_profile_joint_heatmaps.pdf" not in names
    # The per-U-column and per-dataset views were merged into the single
    # three-panel `proxy_enforcement_vs_u_demand`; the split figures are gone.
    assert "proxy_stats_by_u_cardinality.pdf" not in names
    assert "proxy_composition_vs_u_demand.pdf" not in names
    for path in artifacts.figures:
        assert path.exists()


def test_summary_is_per_arm(tiny_family_caches, tmp_path):
    summary = (
        CategoricalCompletenessAnalysis().run(tiny_family_caches, tmp_path).summary
    )
    assert summary["baseline_label"] == "default"
    for label in ("default", "no_enforcement"):
        stats = summary[label]
        assert stats["n_datasets"] == 8
        assert 0.0 <= stats["dominance_fraction"] <= 1.0
        assert stats["violation_fraction"] == pytest.approx(
            1.0 - stats["dominance_fraction"]
        )
    assert summary["default"]["dominance_fraction"] == 1.0
