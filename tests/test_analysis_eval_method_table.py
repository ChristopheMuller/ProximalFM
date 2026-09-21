import csv
from pathlib import Path

import pytest

from analysis_eval.method_table import (
    DEFAULT_METADATA,
    build_rows,
    load_metadata,
    main,
    render_latex,
)


def _config(methods):
    return {"data": {}, "run": {}, "methods": methods}


def test_build_rows_keeps_only_cate_x_outputs_and_derives_properties():
    metadata = load_metadata(DEFAULT_METADATA)
    config = _config(
        [
            {
                "name": "S-RF",
                "cls": "S_Learner",
                "base": "rf",
                "use_W": False,
                "use_Z": False,
            },
            {
                "name": "KPV",
                "cls": "KernelPV",
                "tune": {"enabled": True},
                "outputs": [
                    {"name": "KPV-XW", "strategy": "paired"},
                    {"name": "KPV-X", "strategy": "conditional"},
                    {"name": "KPV-ATE", "strategy": "marginal"},
                ],
            },
        ]
    )

    rows = build_rows(config, metadata)

    assert [row.method for row in rows] == ["S-RF", "KPV-X"]
    assert rows[0].fit_inputs == "X, A, Y"
    assert rows[0].method_label == "S-learner"
    assert rows[0].estimand == "cate(x)"
    assert rows[0].estimator == "random forest"
    assert rows[0].proximal is False
    assert rows[1].estimand == "cate(x)"
    assert rows[1].predict_inputs == "X"
    assert rows[1].tuned is True


def test_oracle_and_ate_outputs_are_excluded():
    metadata = load_metadata(DEFAULT_METADATA)
    rows = build_rows(
        _config(
            [
                {
                    "name": "P",
                    "cls": "PLearner",
                    "bridge": "kernel",
                    "is_oracle": True,
                    "outputs": [
                        {"name": "P-RF", "base": "rf"},
                        {"name": "P-ATE", "kind": "ate"},
                    ],
                }
            ]
        ),
        metadata,
    )

    assert rows == []


def test_finetuned_proximalfm_is_excluded_from_method_tables():
    metadata = load_metadata(DEFAULT_METADATA)
    rows = build_rows(
        _config(
            [
                {"name": "ProximalFM", "cls": "ProximalFM"},
                {"name": "ProximalFM-FT", "cls": "ProximalFM"},
            ]
        ),
        metadata,
    )

    assert [row.method for row in rows] == ["ProximalFM"]


def test_missing_class_metadata_fails_loudly():
    metadata = load_metadata(DEFAULT_METADATA)
    with pytest.raises(KeyError, match="NoSuchMethod"):
        build_rows(_config([{"name": "new", "cls": "NoSuchMethod"}]), metadata)


def test_latex_escapes_names_and_uses_bibtex_keys():
    metadata = load_metadata(DEFAULT_METADATA)
    rows = build_rows(_config([{"name": "KPV_special", "cls": "KernelPV"}]), metadata)
    latex = render_latex(rows)
    assert "KPV\\_special" in latex
    assert "\\citep{mastouriProximalCausalLearning2021}" in latex
    assert "Method name & Method label & Variant & Citation" in latex
    assert "Family" not in latex
    assert "Fit inputs" not in latex
    assert "Tuned" not in latex
    assert "\\begin{table}[t]" in latex
    assert "\\end{table}" in latex
    assert "\\linewidth" in latex
    assert "\\scriptsize" not in latex


def test_cli_writes_matching_tex_and_csv(tmp_path: Path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        """\
run: {}
methods:
  - name: DoPFN
    cls: DoPFN
    use_W: false
    use_Z: false
""",
        encoding="utf-8",
    )
    out = tmp_path / "table.tex"

    main(["--config", str(config_path), "--out", str(out)])

    assert out.exists()
    with out.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 1
    assert records[0]["method"] == "DoPFN"
    assert records[0]["method_label"] == "Do-PFN"
    assert records[0]["estimand"] == "cate(x)"
    assert records[0]["citation_keys"] == "robertsonDoPFNInContextLearning2025"
