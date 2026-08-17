"""The architecture table is derived, so pin what it derives from.

The hand-written table it replaces was wrong in the two ways a derived one can
still go wrong: reporting a flag the run does not use (it said 4 attention heads
where the trainer defaults to 8), and reporting a parameter count that no longer
follows from the configuration (28.7M, from a 1000-bin head the run replaced
with a 2000-bin one). These tests pin both -- that every row tracks the config,
and that the counts come from the model that config builds.
"""

from __future__ import annotations

import pytest

from analysis_training.arch_table import (
    DEFAULT_SCRIPT,
    MODEL_KEYS,
    STAGES,
    build_rows,
    count_parameters,
    power,
    ssmax,
)
from analysis_training.hparam_table import render_latex
from analysis_training.run_config import parse_script


@pytest.fixture(scope="module")
def run():
    return parse_script(DEFAULT_SCRIPT)


def _values(rows):
    return {row.label: row.value for row in rows}


def test_rows_report_the_trainer_defaults_the_run_used(run):
    """The script passes no architecture flag, so every row is a default."""
    c = run.config
    assert not {key for key in MODEL_KEYS if key in run.explicit_flags} - {
        "loss_type",
        "nbins",
        "cate_vmin",
        "cate_vmax",
    }

    values = _values(build_rows(run))
    assert values["Embedding dimension $E$"] == str(c.embed_dim)
    assert values["Blocks"] == str(c.icl_num_blocks)  # last group wins the key
    assert values["Inducing points"] == str(c.col_num_inds)
    assert values["Model dimension $E \\cdot n_{\\mathrm{cls}}$"] == str(
        c.embed_dim * c.row_num_cls
    )

    heads = [row.value for row in build_rows(run) if row.label == "Attention heads"]
    assert heads == [str(c.col_nhead), str(c.row_nhead), str(c.icl_nhead)]


def test_head_reports_the_cate_grid_under_the_cate_loss(run):
    """A single-head loss scores tau on its own grid, not on [vmin, vmax]."""
    c = run.config
    assert c.loss_type == "noise_aware"

    values = _values(build_rows(run))
    assert values["Bins $B$"] == "2\\,000"
    assert (
        values["Grid $[v_{\\min}, v_{\\max}]$"]
        == f"$[{c.cate_vmin:g}, {c.cate_vmax:g}]$"
    )


def test_grid_follows_the_loss_type(run):
    """Switching to a two-head loss switches the reported grid."""
    import copy

    config = copy.deepcopy(run.config)
    config.loss_type = "bins_po"
    other = type(run)(script=run.script, config=config, argv=run.argv, env=run.env)

    values = _values(build_rows(other))
    assert values["Grid $[v_{\\min}, v_{\\max}]$"] == (
        f"$[{config.vmin:g}, {config.vmax:g}]$"
    )


def test_parameter_counts_are_the_model_the_script_configures(run):
    counts = count_parameters(run.config)

    assert counts["total"] == sum(counts[attr] for _, attr in STAGES)
    assert 0 < counts["head"] < counts["icl_predictor"]
    # The stale hand-written total is not reachable from this config.
    assert counts["total"] > 29e6


def test_head_size_tracks_nbins(run):
    """The count is derived: halving the bins shrinks the head, not a constant."""
    import copy

    config = copy.deepcopy(run.config)
    config.nbins = run.config.nbins // 2

    assert count_parameters(config)["head"] < count_parameters(run.config)["head"]


def test_parameter_rows_are_optional(run):
    with_params = build_rows(run, count_parameters(run.config))
    without = build_rows(run, None)

    assert len(with_params) == len(without) + len(STAGES) + 1
    assert not any(row.group == "Parameters" for row in without)


def test_power_and_ssmax_do_not_prettify_beyond_the_value():
    assert power(100000) == "$10^{5}$"
    assert power(10000.0) == "$10^{4}$"
    assert power(50000) == "50\\,000"  # not rounded into a power of ten
    assert ssmax("qassmax-mlp-elementwise") == "query-aware, elementwise"
    assert ssmax(False) == "none"
    assert "\\texttt{" in ssmax("something-new")  # unknown reported verbatim


def test_production_table_renders(run):
    rows = build_rows(run, count_parameters(run.config))
    latex = render_latex(
        rows,
        caption="c",
        label="tab:model_hparams",
        generator="analysis_training.arch_table",
    )

    assert latex.count("\\\\") == len(rows) + 1  # one per row, plus the header
    assert "\\label{tab:model_hparams}" in latex
    assert "analysis_training.arch_table" in latex
