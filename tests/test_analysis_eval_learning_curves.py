from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis_eval.design import method_style, ordered_methods
from analysis_eval.learning_curves import make_learning_curve_figure
from analysis_eval.semisynt import (
    complete_learning_curve_methods,
    load_cate_results,
    summarize_learning_curves,
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
                            "npehe": (
                                1 + case_idx / 10 + method_idx / 10 - n_ctx / 10000
                            ),
                        }
                    )
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
    assert method_style("ProximalFM").color == "#D55E00"
    assert method_style("ProximalFM").linewidth > method_style("S-OLS").linewidth


def test_unknown_method_has_actionable_error():
    try:
        method_style("new-method")
    except KeyError as error:
        assert "METHOD_STYLES" in str(error)
    else:
        raise AssertionError("unknown methods must not receive order-dependent styles")
