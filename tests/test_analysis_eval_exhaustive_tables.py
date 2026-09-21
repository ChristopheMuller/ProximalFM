from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_eval.exhaustive_tables import (
    EXCLUDED_METHODS,
    aggregate_replicates,
    exact_sign_flip_pvalue,
    holm_adjust,
    render_tables,
    summarize,
)
from analysis_eval.semisynt import CELL_ORDER


def test_replicates_are_averaged_within_source_before_summary():
    rows = []
    for source, values in {"a.csv": (1.0, 2.0, 3.0), "b.csv": (4.0, 5.0, 6.0)}.items():
        for replicate, value in enumerate(values, start=1):
            rows.append(
                {
                    "case": "linear_conf_low_proxy_low",
                    "family": "linear",
                    "confounding": "low",
                    "proxy": "low",
                    "method": "ProximalFM",
                    "n_ctx": 128,
                    "source_file": source,
                    "replicate": replicate,
                    "npehe": value,
                }
            )

    result = aggregate_replicates(pd.DataFrame(rows))

    assert result.set_index("source_file")["source_mean"].to_dict() == {
        "a.csv": 2.0,
        "b.csv": 5.0,
    }
    assert result["complete"].all()
    assert set(result["n_replicates"]) == {3}


def test_incomplete_or_nonfinite_replicate_groups_are_not_averaged():
    frame = pd.DataFrame(
        {
            "case": ["linear_conf_low_proxy_low"] * 3,
            "family": ["linear"] * 3,
            "confounding": ["low"] * 3,
            "proxy": ["low"] * 3,
            "method": ["ProximalFM"] * 3,
            "n_ctx": [128] * 3,
            "source_file": ["a.csv"] * 3,
            "replicate": [1, 2, 3],
            "npehe": [1.0, np.nan, 3.0],
        }
    )

    result = aggregate_replicates(frame)

    assert np.isnan(result.loc[0, "source_mean"])
    assert not result.loc[0, "complete"]
    assert result.loc[0, "n_replicates"] == 2


def test_exact_sign_flip_and_holm_correction():
    assert EXCLUDED_METHODS == {"ProximalFM-FT"}
    assert exact_sign_flip_pvalue([1.0] * 5) == 1 / 32
    assert exact_sign_flip_pvalue([0.0] * 5) == 1.0
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}


def _source_means() -> pd.DataFrame:
    rows = []
    methods = {"ProximalFM": 0.5, "S-OLS": 1.5, "S-RF": 0.5}
    for family, confounding, proxy in CELL_ORDER:
        case = f"{family}_conf_{confounding}_proxy_{proxy}"
        for method, value in methods.items():
            for source in range(6):
                rows.append(
                    {
                        "case": case,
                        "family": family,
                        "confounding": confounding,
                        "proxy": proxy,
                        "method": method,
                        "n_ctx": 128,
                        "source_file": f"source-{source}",
                        "source_mean": value,
                        "n_replicates": 3,
                        "complete": True,
                    }
                )
    return pd.DataFrame(rows)


def test_summary_and_latex_mark_best_and_paired_comparison_set():
    summary = summarize(_source_means(), contexts=[128], alpha=0.05)
    first = summary[summary["case"].eq("linear_conf_low_proxy_low")].set_index("method")

    # The tie is broken deterministically by method name; the other tied method
    # remains in the paired failure-to-reject set. The clearly worse method does not.
    assert first.loc["ProximalFM", "best"]
    assert first.loc["S-RF", "not_significantly_worse"]
    assert not first.loc["S-OLS", "not_significantly_worse"]

    latex = render_tables(summary, contexts=[128], alpha=0.05)
    assert latex.count("\\begin{table*}") == 1
    assert "\\textbf{0.500}" in latex
    assert "\\underline{0.500}" in latex
    assert "\\\\textbf" not in latex
    assert "$n_{\\mathrm{ctx}}=128$" in latex
