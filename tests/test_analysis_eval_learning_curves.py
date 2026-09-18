from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_eval.design import (
    COLORS,
    FULL_PAGE_WIDTH_IN,
    MARGINALIZED_LINESTYLE,
    METHOD_STYLES,
    RELEVANT_METHODS,
    method_style,
    ordered_methods,
    relevant_methods,
)
from analysis_eval.error_decomposition import (
    build_parser as build_decomposition_parser,
    make_error_decomposition_figure,
)
from analysis_eval.learning_curves import build_parser, make_learning_curve_figure
from analysis_eval.learning_curves_error_decomposition import (
    _load_results,
    build_parser as build_combined_parser,
    make_combined_figure,
)
from analysis_eval.speed_rank import make_speed_rank_figure
from analysis_eval.semisynt import (
    complete_learning_curve_methods,
    load_cate_results,
    summarize_error_decomposition,
    summarize_learning_curves,
    summarize_speed_rank,
)


CASES = [
    f"{family}_conf_{confounding}_proxy_{proxy}"
    for family in ("linear", "nonlinear")
    for confounding in ("low", "high")
    for proxy in ("low", "high")
]


def _matrix(tmp_path: Path) -> Path:
    root = tmp_path / "matrix"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"cases": [{"case": case} for case in CASES]})
    )
    for case_idx, case in enumerate(CASES):
        out = root / case / "bench" / "results"
        out.mkdir(parents=True)
        rows = []
        for method_idx, method in enumerate(("S-OLS", "ProximalFM")):
            for n_ctx in (128, 512):
                for episode in range(4):
                    rows.append(
                        {
                            "method": method,
                            "episode": episode,
                            "n_ctx": n_ctx,
                            "target": "cate",
                            "estimand": "cate(x)",
                            "is_reference": False,
                            "failed": False,
                            "time": 0.01 * (method_idx + 1) * (case_idx + 1),
                            "time_post_tune": (
                                0.01 * (method_idx + 1) * (case_idx + 1)
                                if method == "S-OLS"
                                else 0.005 * (case_idx + 1)
                            ),
                            "ate_bias_norm": 0.1 + episode / 100,
                            "shape_r2": 0.5 + n_ctx / 10000,
                        }
                    )
                    rows[-1]["npehe"] = (
                        rows[-1]["ate_bias_norm"] ** 2 + 1 - rows[-1]["shape_r2"]
                    ) ** 0.5
        pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    return root


def _dimension_matrix(tmp_path: Path) -> Path:
    """Wrap a legacy fixture in the layout emitted by the current builder."""
    legacy = _matrix(tmp_path)
    root = tmp_path / "dimension-matrix"
    root.mkdir()
    dimension_root = root / "d-5"
    legacy.rename(dimension_root)
    (dimension_root / "manifest.json").unlink()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dimensions": [2, 5],
                "cases": [{"dimension": 5, "case": case} for case in CASES],
            }
        )
    )
    return root


def test_loader_and_summary_cover_factorial_grid(tmp_path):
    frame = load_cate_results(_matrix(tmp_path))
    assert frame["case"].nunique() == 8
    assert set(frame["family"]) == {"linear", "nonlinear"}
    summary = summarize_learning_curves(frame)
    assert len(summary) == 8 * 2 * 2
    assert set(summary["n"]) == {4}
    complete, incomplete = complete_learning_curve_methods(frame)
    assert set(complete) == {"S-OLS", "ProximalFM"}
    assert incomplete == []


def test_single_case_loader_does_not_require_other_benchmark_results(tmp_path):
    root = _matrix(tmp_path)
    missing_case = CASES[0]
    (root / missing_case / "bench" / "results" / "results.csv").unlink()

    frame = load_cate_results(
        root,
        selected_case="nonlinear_conf_high_proxy_high",
    )

    assert set(frame["case"]) == {"nonlinear_conf_high_proxy_high"}


def test_loader_resolves_the_default_dimension_from_a_build_root(tmp_path):
    root = _dimension_matrix(tmp_path)

    frame = load_cate_results(root)

    assert frame["case"].nunique() == 8


def test_loader_accepts_dimension_selection_and_a_direct_dimension_root(tmp_path):
    root = _dimension_matrix(tmp_path)

    explicit = load_cate_results(root, dimension=5)
    direct = load_cate_results(root / "d-5")

    pd.testing.assert_frame_equal(explicit, direct)


def test_loader_rejects_an_unavailable_dimension_with_available_values(tmp_path):
    root = _dimension_matrix(tmp_path)

    try:
        load_cate_results(root, dimension=3)
    except ValueError as error:
        assert "available dimensions: 5" in str(error)
    else:
        raise AssertionError("an unavailable matrix dimension must fail")


def test_combined_loader_resolves_a_dimension_matrix(tmp_path):
    root = _dimension_matrix(tmp_path)

    frame = _load_results(root, selected_case=None, dimension=5)

    assert frame["case"].nunique() == 8


def test_figure_has_eight_axes_and_stable_styles(tmp_path):
    summary = summarize_learning_curves(load_cate_results(_matrix(tmp_path)))
    methods = ordered_methods(set(summary["method"]))
    fig = make_learning_curve_figure(summary, methods=methods)
    assert len(fig.axes) == 8
    assert fig.axes[0].get_ylim() == (0.0, 1.2)
    assert fig.axes[0].get_title() == "Low confounding\nLow proxy reliability"
    assert all(not ax.get_title() for ax in fig.axes[4:])
    assert fig.get_size_inches().tolist() == [FULL_PAGE_WIDTH_IN, 4.55]
    assert not fig.axes[0].collections
    assert method_style("ProximalFM").color == "#D55E00"
    assert method_style("ProximalFM").linewidth > method_style("S-OLS").linewidth


def test_method_styles_share_regressor_colours_and_learner_linestyles():
    for regressor in ("OLS", "RF", "TabICL"):
        styles = [method_style(f"{learner}-{regressor}") for learner in "STX"]
        assert len({style.color for style in styles}) == 1

    assert method_style("S-OLS").linestyle == "--"
    assert method_style("S-RF").linestyle == "--"
    assert method_style("S-TabICL").linestyle == "--"
    assert method_style("T-OLS").linestyle == method_style("T-RF").linestyle
    assert method_style("T-RF").linestyle == method_style("T-TabICL").linestyle
    assert method_style("X-OLS").linestyle == method_style("X-RF").linestyle
    assert method_style("X-RF").linestyle == method_style("X-TabICL").linestyle

    plearner = {
        method_style(name).color
        for name in ("P-Kernel-RF", "P-Kernel-OLS", "P-Kernel-ATE")
    }
    pmmr = {
        method_style(name).color for name in ("PMMR-X-Wind", "PMMR-X-Wcond", "PMMR-ATE")
    }
    assert len(plearner) == len(pmmr) == 1
    assert plearner.isdisjoint(pmmr)

    nmmr = {
        method_style(name).color for name in ("NMMR-V-fixed-Wind", "NMMR-V-fixed-Wcond")
    }
    tabicl = {method_style(f"{learner}-TabICL").color for learner in "STX"}
    assert len(nmmr) == len(tabicl) == 1
    assert nmmr.isdisjoint(tabicl)


def test_every_method_has_a_unique_visual_identity_and_label():
    identities = [style.identity() for style in METHOD_STYLES.values()]
    labels = [style.label for style in METHOD_STYLES.values()]

    assert len(identities) == len(set(identities))
    assert len(labels) == len(set(labels))


def test_marginalized_methods_keep_parent_colour_but_use_distinct_format():
    parent_pairs = (
        ("S-RF", "S-RF-WZ-Kernel-128"),
        ("S-RF", "S-RF-WZ-kNN-128"),
        ("S-TabICL", "S-TabICL-WZ-kNN-16"),
        ("CausalPFN", "CausalPFN-WZ-kNN-16"),
    )
    for parent_name, marginalized_name in parent_pairs:
        parent = method_style(parent_name)
        marginalized = method_style(marginalized_name)
        assert marginalized.color == parent.color
        assert marginalized.linestyle == MARGINALIZED_LINESTYLE
        assert marginalized.identity() != parent.identity()


def test_figure_can_render_one_case_as_a_single_subplot(tmp_path):
    summary = summarize_learning_curves(load_cate_results(_matrix(tmp_path)))
    fig = make_learning_curve_figure(
        summary,
        methods=["S-OLS", "ProximalFM"],
        case="linear_conf_low_proxy_low",
    )

    assert len(fig.axes) == 1
    assert fig.legends[0]._ncols == 2
    assert fig.axes[0].get_title() == "Low confounding\nLow proxy reliability"
    assert fig.axes[0].get_xlabel() == "Context size"
    assert fig.axes[0].get_ylabel() == "Linear\nCATE nPEHE"


def test_learning_curve_default_uses_compact_method_set(tmp_path):
    summary = summarize_learning_curves(load_cate_results(_matrix(tmp_path)))
    relevant = summary[summary["method"].eq("S-OLS")].copy()
    relevant["method"] = "T-OLS"
    summary = pd.concat([summary, relevant], ignore_index=True)
    fig = make_learning_curve_figure(summary)
    legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert legend_labels == ["T-learner (ridge)", "ProximalFM"]


def test_learning_curve_keeps_curated_method_with_incomplete_coverage(tmp_path):
    summary = summarize_learning_curves(load_cate_results(_matrix(tmp_path)))
    plearner = summary[summary["method"].eq("S-OLS")].copy()
    plearner["method"] = "P-Kernel-RF"
    missing_case = CASES[0]
    plearner = plearner[~plearner["case"].eq(missing_case)]
    summary = pd.concat([summary, plearner], ignore_index=True)

    fig = make_learning_curve_figure(summary)

    legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert "P-learner (RF)" in legend_labels
    assert len(fig.axes[0].lines) == 3  # two available methods plus chance line
    assert np.isnan(fig.axes[0].lines[0].get_ydata()).all()
    assert np.isfinite(fig.axes[1].lines[0].get_ydata()).all()


def test_learning_curve_cli_options_control_axis_and_interval(tmp_path):
    args = build_parser().parse_args([])
    assert args.y_max == 1.2
    assert args.confidence_interval is False
    assert args.case is None
    assert args.dim is None
    assert build_parser().parse_args(["--dim", "5"]).dim == 5
    assert (
        build_parser().parse_args(["--config", "linear_conf_low_proxy_low"]).case
        == "linear_conf_low_proxy_low"
    )
    args = build_parser().parse_args(["--y-max", "2.5", "--confidence-interval", "ste"])
    assert args.y_max == 2.5
    assert args.confidence_interval == "ste"

    summary = summarize_learning_curves(load_cate_results(_matrix(tmp_path)))
    assert {"std", "ste"}.issubset(summary.columns)
    fig = make_learning_curve_figure(
        summary,
        y_max=2.5,
        confidence_interval="ste",
    )
    assert fig.axes[0].get_ylim() == (0.0, 2.5)
    assert fig.axes[0].collections


def test_unknown_method_has_actionable_error():
    try:
        method_style("new-method")
    except KeyError as error:
        assert "METHOD_STYLES" in str(error)
    else:
        raise AssertionError("unknown methods must not receive order-dependent styles")


def test_compact_method_selection_is_curated_not_performance_based():
    available = ["unknown-extra", "ProximalFM", "T-OLS", "CausalPFN"]
    assert relevant_methods(available) == ["T-OLS", "CausalPFN", "ProximalFM"]
    assert set(RELEVANT_METHODS).issubset(METHOD_STYLES)


def test_mean_error_decomposition_preserves_identity(tmp_path):
    frame = load_cate_results(_matrix(tmp_path))
    summary = summarize_error_decomposition(
        frame, ["S-OLS", "ProximalFM"], aggregation="mean"
    )
    assert summary["identity_gap"].abs().max() < 1e-10


def test_error_decomposition_figure_has_eight_axes(tmp_path):
    frame = load_cate_results(_matrix(tmp_path))
    summary = summarize_error_decomposition(
        frame, ["S-OLS", "ProximalFM"], aggregation="median"
    )
    fig = make_error_decomposition_figure(summary, methods=["S-OLS", "ProximalFM"])
    assert len(fig.axes) == 8


def test_error_decomposition_can_render_one_case_as_a_single_subplot(tmp_path):
    frame = load_cate_results(_matrix(tmp_path))
    summary = summarize_error_decomposition(
        frame, ["S-OLS", "ProximalFM"], aggregation="median"
    )
    fig = make_error_decomposition_figure(
        summary,
        methods=["S-OLS", "ProximalFM"],
        case="linear_conf_low_proxy_low",
    )

    assert len(fig.axes) == 1
    assert fig.legends[0]._ncols == 2
    assert fig.axes[0].get_title() == "Low confounding\nLow proxy reliability"
    assert fig.axes[0].get_xlabel() == r"Level error: normalized ATE bias$^2$"
    assert fig.axes[0].get_ylabel() == "Linear\nHeterogeneity error"


def test_error_decomposition_legend_wraps_long_method_lists(tmp_path):
    frame = load_cate_results(_matrix(tmp_path))
    summary = summarize_error_decomposition(
        frame, ["S-OLS", "ProximalFM"], aggregation="median"
    )
    methods = ["S-OLS", "S-RF", "T-OLS", "T-RF", "ProximalFM"]
    copies = [summary]
    for method in methods[1:-1]:
        copy = summary[summary["method"].eq("S-OLS")].copy()
        copy["method"] = method
        copies.append(copy)
    expanded = pd.concat(copies, ignore_index=True)

    fig = make_error_decomposition_figure(expanded, methods=methods)

    assert fig.legends[0]._ncols == 4


def test_error_decomposition_uses_panel_specific_limits(tmp_path):
    frame = load_cate_results(_matrix(tmp_path))
    summary = summarize_error_decomposition(
        frame, ["S-OLS", "ProximalFM"], aggregation="median"
    )
    summary.loc[summary["family"].eq("nonlinear"), "level_error"] *= 10
    summary.loc[summary["family"].eq("nonlinear"), "shape_error"] *= 2

    fig = make_error_decomposition_figure(summary, methods=["S-OLS", "ProximalFM"])

    assert fig.axes[0].get_xlim() != fig.axes[4].get_xlim()
    assert fig.axes[0].get_ylim() != fig.axes[4].get_ylim()


def test_error_decomposition_fixed_limits_clip_trajectory_data(tmp_path):
    frame = load_cate_results(_matrix(tmp_path))
    summary = summarize_error_decomposition(
        frame, ["S-OLS", "ProximalFM"], aggregation="median"
    )
    summary.loc[
        summary["method"].eq("ProximalFM") & summary["n_ctx"].eq(128),
        "level_error",
    ] = 0.01
    summary.loc[
        summary["method"].eq("ProximalFM") & summary["n_ctx"].eq(512),
        "level_error",
    ] = 0.04
    fig = make_error_decomposition_figure(
        summary,
        methods=["S-OLS", "ProximalFM"],
        xlim=(0.0, 0.03),
        ylim=(0.0, 1.0),
    )

    assert fig.axes[0].get_xlim() == (0.0, 0.03)
    assert fig.axes[0].get_ylim() == (0.0, 1.0)
    for line in fig.axes[0].lines:
        values = line.get_xydata()
        finite = values[np.isfinite(values).all(axis=1)]
        assert (finite[:, 0] >= 0.0).all()
        assert (finite[:, 0] <= 0.03).all()
        assert (finite[:, 1] >= 0.0).all()
        assert (finite[:, 1] <= 1.0).all()
    assert any(
        np.isclose(line.get_xydata()[:, 0], 0.03, equal_nan=False).any()
        for line in fig.axes[0].lines
    )


def test_error_decomposition_cli_accepts_fixed_limits():
    args = build_decomposition_parser().parse_args(
        ["--xlim", "0", "0.5", "--ylim", "0", "1.25"]
    )
    assert args.xlim == [0.0, 0.5]
    assert args.ylim == [0.0, 1.25]
    assert args.case is None
    assert (
        build_decomposition_parser()
        .parse_args(["--config", "linear_conf_low_proxy_low"])
        .case
        == "linear_conf_low_proxy_low"
    )


def test_combined_figure_renders_selected_case_as_two_panels(tmp_path):
    frame = load_cate_results(_matrix(tmp_path))
    methods = ["S-OLS", "ProximalFM"]
    learning = summarize_learning_curves(frame, methods)
    decomposition = summarize_error_decomposition(frame, methods)

    fig = make_combined_figure(
        learning,
        decomposition,
        methods=methods,
        case="nonlinear_conf_high_proxy_low",
        y_max=2.0,
    )

    assert len(fig.axes) == 2
    assert fig.axes[0].get_ylim() == (0.0, 2.0)
    assert fig.axes[0].get_title(loc="left").startswith("(a)")
    assert fig.axes[1].get_title(loc="left").startswith("(b)")
    assert len(fig.legends) == 1
    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        "S-learner (ridge)",
        "ProximalFM",
    ]


def test_combined_loader_accepts_standalone_root_without_cases(tmp_path):
    matrix = _matrix(tmp_path)
    root = tmp_path / "standalone"
    results_dir = root / "bench" / "results"
    results_dir.mkdir(parents=True)
    source = matrix / CASES[0] / "bench" / "results" / "results.csv"
    (results_dir / "results.csv").write_text(source.read_text())
    (root / "manifest.json").write_text(json.dumps({"name": "standalone"}))

    frame = _load_results(root, selected_case=None)
    assert frame["case"].unique().tolist() == ["standalone"]
    assert set(frame["family"]) == {"synthetic"}
    assert set(frame["confounding"]) == {"mixed"}
    assert set(frame["proxy"]) == {"mixed"}

    methods = ["S-OLS", "ProximalFM"]
    fig = make_combined_figure(
        summarize_learning_curves(frame, methods),
        summarize_error_decomposition(frame, methods),
        methods=methods,
    )
    assert len(fig.axes) == 2


def test_combined_cli_accepts_optional_case():
    assert build_combined_parser().parse_args([]).case is None
    args = build_combined_parser().parse_args(
        ["--case", "nonlinear_conf_high_proxy_low"]
    )
    assert args.case == "nonlinear_conf_high_proxy_low"
    assert args.y_max == 1.2


def test_speed_rank_summary_separates_confounding_levels(tmp_path):
    frame = load_cate_results(_matrix(tmp_path))
    summary = summarize_speed_rank(frame, ["S-OLS", "ProximalFM"], context_size=512)

    assert set(summary["method"]) == {"S-OLS", "ProximalFM"}
    assert set(summary["confounding"]) == {"low", "high"}
    assert set(summary["n_blocks"]) == {4 * 4}
    assert len(summary) == 2 * 2
    assert (summary["rank_q25"] <= summary["rank_median"]).all()
    assert (summary["rank_median"] <= summary["rank_q75"]).all()
    prox = summary[summary["method"].eq("ProximalFM")]
    assert (prox["time_post_tune_mean"] < prox["time_mean"]).all()

    pooled = summarize_speed_rank(
        frame, ["S-OLS", "ProximalFM"], context_size=512, by_confounding=False
    )
    assert set(pooled["confounding"]) == {"pooled"}
    assert set(pooled["n_blocks"]) == {8 * 4}


def test_speed_rank_figure_has_one_panel_per_confounding_level(tmp_path):
    frame = load_cate_results(_matrix(tmp_path))
    summary = summarize_speed_rank(frame, ["S-OLS", "ProximalFM"])
    fig = make_speed_rank_figure(summary)

    assert len(fig.axes) == 2
    assert {ax.get_title() for ax in fig.axes} == {
        "Low confounding",
        "High confounding",
    }
    assert all(ax.get_xscale() == "log" for ax in fig.axes)
    assert all(ax.yaxis_inverted() for ax in fig.axes)
    assert fig.get_size_inches()[1] == 3.5


def test_speed_rank_only_highlights_and_labels_relevant_methods(tmp_path):
    frame = load_cate_results(_matrix(tmp_path))
    summary = summarize_speed_rank(frame, ["S-OLS", "ProximalFM"])
    fig = make_speed_rank_figure(summary)

    for ax in fig.axes:
        assert [text.get_text() for text in ax.texts] == ["ProximalFM"]
        line_colors = {line.get_color() for line in ax.lines}
        assert COLORS["light_grey"] in line_colors
        assert method_style("ProximalFM").color in line_colors
