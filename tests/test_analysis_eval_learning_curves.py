from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_eval.design import method_style, ordered_methods
from analysis_eval.error_decomposition import (
    build_parser as build_decomposition_parser,
    make_error_decomposition_figure,
)
from analysis_eval.learning_curves import build_parser, make_learning_curve_figure
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


def test_figure_has_eight_axes_and_stable_styles(tmp_path):
    summary = summarize_learning_curves(load_cate_results(_matrix(tmp_path)))
    methods = ordered_methods(set(summary["method"]))
    fig = make_learning_curve_figure(summary, methods=methods)
    assert len(fig.axes) == 8
    assert fig.axes[0].get_ylim() == (0.0, 1.2)
    assert not fig.axes[0].collections
    assert method_style("ProximalFM").color == "#D55E00"
    assert method_style("ProximalFM").linewidth > method_style("S-OLS").linewidth


def test_learning_curve_cli_options_control_axis_and_interval(tmp_path):
    args = build_parser().parse_args([])
    assert args.y_max == 1.2
    assert args.confidence_interval is False
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


def test_speed_rank_summary_pairs_all_eight_cells(tmp_path):
    frame = load_cate_results(_matrix(tmp_path))
    summary = summarize_speed_rank(frame, ["S-OLS", "ProximalFM"], context_size=512)

    assert set(summary["method"]) == {"S-OLS", "ProximalFM"}
    assert set(summary["n_blocks"]) == {8 * 4}
    assert (summary["rank_q25"] <= summary["rank_median"]).all()
    assert (summary["rank_median"] <= summary["rank_q75"]).all()
    prox = summary.set_index("method").loc["ProximalFM"]
    assert prox["time_post_tune_mean"] < prox["time_mean"]


def test_speed_rank_figure_is_single_panel_with_log_time(tmp_path):
    frame = load_cate_results(_matrix(tmp_path))
    summary = summarize_speed_rank(frame, ["S-OLS", "ProximalFM"])
    fig = make_speed_rank_figure(summary)

    assert len(fig.axes) == 1
    assert fig.axes[0].get_xscale() == "log"
    assert fig.axes[0].yaxis_inverted()
