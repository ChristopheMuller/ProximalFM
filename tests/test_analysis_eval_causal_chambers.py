from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis_eval.causal_chambers import (
    build_parser,
    load_rmse_summary,
    make_causal_chambers_figure,
)


def _summary() -> pd.DataFrame:
    rows = []
    for method, multiplier in (("S-TabICL", 1.0), ("ProximalFM-ATE", 0.5)):
        for n_ctx in (128, 256, 512):
            rows.append({"method": method, "n_ctx": n_ctx, "rmse": multiplier / n_ctx})
    return pd.DataFrame(rows)


def test_loader_validates_and_normalizes_summary(tmp_path: Path):
    path = tmp_path / "summary.csv"
    _summary().to_csv(path, index=False)

    loaded = load_rmse_summary(path)

    assert loaded["n_ctx"].tolist() == [128, 256, 512, 128, 256, 512]
    assert loaded["rmse"].dtype.kind == "f"


def test_figure_draws_one_complete_line_per_method():
    fig = make_causal_chambers_figure(_summary())

    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 2
    assert fig.axes[0].get_xlabel() == "Context size"
    assert fig.axes[0].get_ylabel() == "ATE RMSE"
    assert fig.axes[0].get_xscale() == "log"
    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        "S-learner (TabICL)",
        "ProximalFM (ATE)",
    ]


def test_figure_rejects_incomplete_method_curve():
    summary = _summary().query("~(method == 'S-TabICL' and n_ctx == 256)")

    with pytest.raises(ValueError, match="every selected method"):
        make_causal_chambers_figure(summary)


def test_cli_defaults_and_method_subset():
    args = build_parser().parse_args([])

    assert args.name == "causal_chambers_rmse"
    assert args.methods is None
    assert args.summary_csv.name == "summary.csv"

    subset = build_parser().parse_args(["--methods", "S-TabICL"])
    assert subset.methods == ["S-TabICL"]
