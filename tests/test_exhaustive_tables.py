"""Checks for the shared and selection table renderers."""

import json

import pandas as pd

from analysis.analysis_eval import exhaustive_tables
from analysis.analysis_eval.semisynt import MatrixCell


def test_load_results_can_select_one_case(tmp_path, monkeypatch):
    case_dir = tmp_path / "nonlinear_conf_high_proxy_low"
    (case_dir / "manifest.json").parent.mkdir(parents=True)
    (case_dir / "manifest.json").write_text(
        json.dumps({"episodes": [{"source_file": "source.pt", "replicate": 0}]})
    )
    shared_path = case_dir / "bench" / "results" / "results.csv"
    shared_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "method": ["S-OLS"],
            "episode": [0],
            "n_ctx": [128],
            "target": ["cate"],
            "estimand": ["cate(x)"],
            "is_reference": [False],
            "failed": [False],
            "npehe": [0.5],
        }
    ).to_csv(shared_path, index=False)

    cell = MatrixCell(
        "nonlinear_conf_high_proxy_low", "nonlinear", "high", "low", shared_path
    )
    monkeypatch.setattr(exhaustive_tables, "discover_cells", lambda *a, **k: [cell])

    selected = exhaustive_tables.load_results(tmp_path, selected_case=cell.case)
    assert set(selected["n_ctx"]) == {128}
    assert set(selected["case"]) == {cell.case}


def test_extended_table_formats_context_with_thousands_separator():
    summary = pd.DataFrame(
        {
            "case": ["nonlinear_conf_high_proxy_low"],
            "n_ctx": [8192],
            "method": ["S-OLS"],
            "mean_npehe": [0.5],
            "n_sources": [12],
            "best": [True],
            "not_significantly_worse": [True],
        }
    )

    latex = exhaustive_tables.render_extended_table(summary, alpha=0.05, replicates=3)
    assert "$8,192$" in latex
    assert "$8192$" not in latex
