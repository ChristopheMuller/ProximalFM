from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis_eval.causal_chambers import (
    build_parser,
    load_results,
    make_causal_chambers_figure,
)


def _condition(family: str, confounding: str, proxy: str) -> str:
    return f"{confounding}_confounding_{family}_proxy_{proxy}_smoothA_a350"


def _results() -> pd.DataFrame:
    rows = []
    for family in ("linear", "non_linear"):
        for confounding in ("strong", "weak"):
            for proxy in ("low", "mid", "high"):
                for method, multiplier in (("X-TabICL", 1.0), ("ProximalFM-ATE", 2.0)):
                    for n_ctx in (128, 256, 512):
                        rows.append(
                            {
                                "condition": _condition(family, confounding, proxy),
                                "permutation": 0,
                                "n_ctx": n_ctx,
                                "method": method,
                                "abs_error": multiplier / n_ctx,
                            }
                        )
    return pd.DataFrame(rows)


def test_loader_validates_and_normalizes_results(tmp_path: Path):
    path = tmp_path / "results.csv"
    _results().to_csv(path, index=False)

    loaded = load_results(path)

    assert loaded["n_ctx"].dtype.kind == "i"
    assert loaded["abs_error"].dtype.kind == "f"
    assert len(loaded) == 72


def test_figure_draws_one_complete_curve_per_method_and_condition():
    figure = make_causal_chambers_figure(
        _results(), methods=["X-TabICL", "ProximalFM-ATE"]
    )

    assert len(figure.axes) == 12
    assert all(len(axis.lines) == 2 for axis in figure.axes)
    assert all(axis.get_xscale() == "log" for axis in figure.axes)
    assert all(axis.get_yscale() == "log" for axis in figure.axes)
    assert figure.axes[-1].get_xlabel() == "Context size"
    assert "Absolute ATE error (sensor counts)" in [
        text.get_text() for text in figure.texts
    ]
    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        "X-learner (TabICL)",
        "ProximalFM",
    ]


def test_figure_rejects_incomplete_method_curve():
    condition = _condition("linear", "strong", "low")
    results = _results().query(
        "~(method == 'X-TabICL' and n_ctx == 256 and condition == @condition)"
    )

    with pytest.raises(ValueError, match="every selected method"):
        make_causal_chambers_figure(results, methods=["X-TabICL", "ProximalFM-ATE"])


def test_cli_defaults_and_method_subset():
    args = build_parser().parse_args([])

    assert args.name == "causal_chambers_absolute_error"
    assert args.methods is None
    assert args.results_csv.name == "results.csv"
    assert args.out_dir.name == "causal_chambers"

    subset = build_parser().parse_args(["--methods", "X-TabICL"])
    assert subset.methods == ["X-TabICL"]
