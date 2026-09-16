from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis_eval.causal_chambers_rank import (
    build_parser,
    build_rank_grid,
    make_rank_grid_figure,
)


def _condition(family: str, confounding: str, proxy: str) -> str:
    return (
        "out_binaryA_counterfactuals_"
        f"{confounding}_confounding_{family}_proxy_{proxy}_smoothA_a350"
    )


def _results() -> pd.DataFrame:
    rows = []
    methods = ("S-TabICL", "P2SLS", "ProximalFM-ATE")
    for family in ("linear", "non_linear"):
        for confounding in ("strong", "weak"):
            for proxy in ("low", "mid", "high"):
                condition = _condition(family, confounding, proxy)
                for method, error in zip(methods, (2.0, 3.0, 1.0)):
                    rows.append(
                        {
                            "condition": condition,
                            "permutation": 0,
                            "n_ctx": 1000,
                            "method": method,
                            "error": error,
                        }
                    )
    return pd.DataFrame(rows)


def test_rank_grid_orders_columns_and_rows_by_mean_rank():
    grid = build_rank_grid(_results())

    assert [condition.family for condition in grid.conditions] == [
        "linear",
        "linear",
        "linear",
        "linear",
        "linear",
        "linear",
        "non_linear",
        "non_linear",
        "non_linear",
        "non_linear",
        "non_linear",
        "non_linear",
    ]
    assert [condition.confounding for condition in grid.conditions] == [
        "strong",
        "strong",
        "strong",
        "weak",
        "weak",
        "weak",
        "strong",
        "strong",
        "strong",
        "weak",
        "weak",
        "weak",
    ]
    assert [condition.proxy for condition in grid.conditions] == [
        "low",
        "mid",
        "high",
    ] * 4
    assert list(grid.ranks.index) == ["ProximalFM-ATE", "S-TabICL", "P2SLS"]
    assert grid.ranks.iloc[0].tolist() == [1.0] * 12
    assert grid.mean_rank.tolist() == [1.0, 2.0, 3.0]


def test_rank_grid_figure_has_hierarchical_header_and_annotations():
    figure = make_rank_grid_figure(build_rank_grid(_results()))

    assert len(figure.axes) == 3  # header, heatmap, colorbar
    heatmap = figure.axes[1]
    assert len(heatmap.images) == 1
    assert len(heatmap.texts) == 36
    assert heatmap.get_xlabel() == "Rank at context size 1,000 (1 = best)"


def test_rank_grid_cli_defaults():
    args = build_parser().parse_args([])

    assert args.context_size == 1000
    assert args.name == "causal_chambers_rank_grid"
    assert args.methods is None

    args = build_parser().parse_args(["--context-size", "512", "--methods", "P2SLS"])
    assert args.context_size == 512
    assert args.methods == ["P2SLS"]
