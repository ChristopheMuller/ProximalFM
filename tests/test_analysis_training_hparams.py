"""The training table is generated from the launch script, so pin the parse.

The table's whole reason for existing is that a hand-written one went stale
against the run; these tests pin the two ways the generator could go stale in
the same way -- resolving a variable to the wrong value, and silently accepting
a flag the trainer no longer has.
"""

from __future__ import annotations

import pytest

from analysis_training.hparam_table import DEFAULT_SCRIPT, build_rows, render_latex
from analysis_training.run_config import ScriptParseError, parse_script

SCRIPT = """#!/usr/bin/env bash
set -euo pipefail

RUN=${RUN:-whole_1}
CKPT=${CKPT:-ckpt_tabicl/backbone.ckpt}
INTERVAL=${INTERVAL:-100}

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

python -m proximalfm.train \\
    --max_steps 75000 --batch_size 64 --micro_batch_size 16 \\
    --loss_type noise_aware --lambda_cate 0.5 \\
    --nbins 2000 --cate_vmin -5 --cate_vmax 5 \\
    --hl_final_sigma_bins 2.0 \\
    --checkpoint_path "${CKPT}" \\
    --checkpoint_dir "${RUN}" \\
    --synt_eval_interval "${INTERVAL}"
"""


def _write(tmp_path, text=SCRIPT):
    path = tmp_path / "run.sh"
    path.write_text(text)
    return path


def test_variables_resolve_from_script_defaults(tmp_path):
    run = parse_script(_write(tmp_path))

    assert run.config.checkpoint_dir == "whole_1"
    assert run.config.checkpoint_path == "ckpt_tabicl/backbone.ckpt"
    assert run.config.synt_eval_interval == 100


def test_overrides_beat_script_defaults(tmp_path):
    run = parse_script(_write(tmp_path), {"RUN": "whole_2", "INTERVAL": "250"})

    assert run.config.checkpoint_dir == "whole_2"
    assert run.config.synt_eval_interval == 250
    # Untouched variables keep the script's default.
    assert run.config.checkpoint_path == "ckpt_tabicl/backbone.ckpt"


def test_unset_flags_fall_back_to_trainer_defaults(tmp_path):
    run = parse_script(_write(tmp_path))

    # The script passes neither, so the table reports what the trainer used.
    assert run.config.scheduler == "cosine_warmup"
    assert run.config.weight_decay == 0
    assert set(run.explicit_flags) >= {"max_steps", "nbins", "checkpoint_dir"}
    assert "scheduler" not in run.explicit_flags


def test_unknown_flag_is_refused(tmp_path):
    path = _write(tmp_path, SCRIPT.replace("--nbins 2000", "--nbins 2000 --gone 1"))

    with pytest.raises(SystemExit):
        parse_script(path)


def test_script_without_invocation_is_refused(tmp_path):
    with pytest.raises(ScriptParseError):
        parse_script(_write(tmp_path, "RUN=${RUN:-x}\necho hi\n"))


def test_production_script_renders(tmp_path):
    """The real script parses and its derived quantities are consistent."""
    run = parse_script(DEFAULT_SCRIPT)
    rows = build_rows(run)
    values = {row.label: row.value for row in rows}

    c = run.config
    # Bin width and final sigma are derived, not transcribed.
    width = (c.cate_vmax - c.cate_vmin) / c.nbins
    assert values["Bin width $w$"] == f"${width:g}$"
    assert values["Final smoothing $\\sigma$"].startswith(
        f"${c.hl_final_sigma_bins * width:g}$"
    )

    latex = render_latex(rows, caption="c", label="tab:training_hparams")
    assert latex.count("\\\\") == len(rows) + 1  # one per row, plus the header
    assert "\\label{tab:training_hparams}" in latex
