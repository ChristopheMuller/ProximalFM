"""Tests for ``analysis_training.time_table`` and its wandb fetcher.

Nothing here touches the network: :func:`analysis_training.time_table.build`
takes an injectable ``fetch_fn``, and the fetcher's own tests exercise the parts
that are pure (the paging-artifact correction, the interruption arithmetic).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from analysis_training import time_table, wandb_history
from analysis_training.hparam_table import num
from analysis_training.run_config import parse_script
from analysis_training.wandb_history import Gap, History, RunMeta

STEPS = 1000


def _meta(**overrides) -> RunMeta:
    start = dt.datetime(2026, 8, 12, 20, 41, tzinfo=dt.timezone.utc)
    base = dict(
        path="e/p/abc",
        id="abc",
        name="whole_x",
        state="finished",
        final_step=STEPS,
        runtime_s=1000.0,
        started_at=start,
        ended_at=start + dt.timedelta(seconds=1500),
        gpu="NVIDIA H100 NVL",
        gpu_count_host=4,
        host="morph",
        gaps=(Gap(step=500, seconds=500.0, resumed_at=start),),
    )
    base.update(overrides)
    return RunMeta(**base)


def _history(**columns) -> History:
    frame = pd.DataFrame({"_step": range(1, len(next(iter(columns.values()))) + 1)})
    for key, values in columns.items():
        frame[key] = values
    return History(frame=frame, keys=tuple(frame.columns))


def _fetch_fn(train=0.2, prior=0.5, n_prior_eval=5, n_synt_eval=10, eval_s=1.0):
    def fetch_fn(keys, run=None, **_):
        if "train_time" in keys:
            return _history(train_time=[train] * STEPS, prior_time=[prior] * STEPS)
        if "Charts/prior_eval_time" in keys:
            return _history(**{"Charts/prior_eval_time": [eval_s] * n_prior_eval})
        return _history(**{"Charts/synt_eval_time": [eval_s] * n_synt_eval})

    return fetch_fn


@pytest.fixture
def run():
    return parse_script(time_table.DEFAULT_SCRIPT, {})


# --------------------------------------------------------------------------- #
# the decomposition
# --------------------------------------------------------------------------- #


def test_components_sum_to_occupancy(run):
    """Every hour of occupancy is attributed to exactly one row."""
    rows = time_table.build(run, _meta(), fetch_fn=_fetch_fn())
    compute = [r for r in rows if r.group == "Compute"]
    assert sum(r.hours for r in compute) == pytest.approx(1000.0)
    assert sum(r.share for r in compute) == pytest.approx(1.0)


def test_residual_is_occupancy_minus_measured(run):
    """0.2 + 0.5 per step over 1000 steps, plus 15 s of eval, leaves 285 s."""
    rows = time_table.build(run, _meta(), fetch_fn=_fetch_fn())
    other = next(r for r in rows if "Checkpointing" in r.label)
    assert other.hours == pytest.approx(1000.0 - 700.0 - 15.0)


def test_percentages_are_of_occupancy_not_calendar_span(run):
    """The idle hours are not a denominator; charging the run for the time it
    spent dead would dilute every component."""
    rows = time_table.build(run, _meta(), fetch_fn=_fetch_fn())
    generation = next(r for r in rows if r.label == "Prior generation")
    assert generation.share == pytest.approx(500.0 / 1000.0)  # not /1500


def test_interruptions_do_not_reach_the_table(run):
    """The gap is what selects the clock, not something the appendix reports:
    the table reads as the uninterrupted run it is equivalent to."""
    interrupted = time_table.build(run, _meta(), fetch_fn=_fetch_fn())
    clean = time_table.build(run, _meta(gaps=()), fetch_fn=_fetch_fn())
    assert interrupted == clean
    assert not [r for r in interrupted if "idle" in r.label or "span" in r.label]


# --------------------------------------------------------------------------- #
# the refusals
# --------------------------------------------------------------------------- #


def test_refuses_overlapping_slices(run):
    """A negative residual means the components are not disjoint intervals of
    one clock, so the table would be fiction."""
    with pytest.raises(ValueError, match="not disjoint"):
        time_table.build(run, _meta(runtime_s=100.0), fetch_fn=_fetch_fn())


def test_refuses_a_prefetching_dataloader(run):
    """Worker processes overlap generation with training, which is exactly the
    assumption the residual rests on."""
    run.config.train_dataloader_num_workers = 4
    with pytest.raises(ValueError, match="serial with training"):
        time_table.build(run, _meta(), fetch_fn=_fetch_fn())


# --------------------------------------------------------------------------- #
# scale rows and rendering
# --------------------------------------------------------------------------- #


def test_datasets_consumed_is_batch_size_times_steps(run):
    rows = time_table.build(run, _meta(), fetch_fn=_fetch_fn())
    row = next(r for r in rows if "datasets consumed" in r.label)
    assert row.detail == num(run.config.batch_size * STEPS)


def test_csv_carries_the_scale_numbers(run, tmp_path):
    """`detail` is LaTeX, so the csv reads `raw` -- and a count stays a count."""
    rows = time_table.build(run, _meta(), fetch_fn=_fetch_fn())
    path = tmp_path / "cost.csv"
    time_table.write_csv(rows, path)
    body = path.read_text()
    assert f",{run.config.batch_size * STEPS}\n" in body  # not 4.8e+06
    assert "75\\,000" not in body


def test_renders_a_complete_float(run):
    rows = time_table.build(run, _meta(), fetch_fn=_fetch_fn())
    latex = time_table.render_latex(rows, caption="c", label="tab:x")
    assert latex.count("\\midrule") == 3  # Compute | Total | Scale
    assert "\\label{tab:x}" in latex
    assert latex.rstrip().endswith("\\end{table}")
    # Multi-row groups get a `\multirow` span; a one-row group takes the label
    # plainly, because `\multirow{1}` reads as a mistake.
    assert "\\multirow{5}{*}{Compute}" in latex
    assert "\\multirow{3}{*}{Scale}" in latex
    assert "\\multirow" not in latex.split("Total")[1].split("\\midrule")[0]


# --------------------------------------------------------------------------- #
# the fetcher
# --------------------------------------------------------------------------- #


def test_total_corrects_for_dropped_pages():
    """``scan_history`` drops ~one row per page boundary. A raw sum would
    under-count; ``mean x n_steps`` does not."""
    frame = pd.DataFrame({"_step": [1, 2, 4, 5], "train_time": [2.0, 2.0, 2.0, 2.0]})
    history = History(frame=frame, keys=("_step", "train_time"))
    assert history.total("train_time") == pytest.approx(8.0)  # raw
    assert history.total("train_time", 5) == pytest.approx(10.0)  # corrected
    assert history.coverage(5) == pytest.approx(0.8)


def test_meta_separates_occupancy_from_calendar_span():
    meta = _meta()
    assert meta.runtime_s == pytest.approx(1000.0)
    assert meta.span_s == pytest.approx(1500.0)
    assert meta.idle_s == pytest.approx(500.0)


def test_meta_round_trips_through_json():
    meta = _meta()
    assert RunMeta.from_dict(meta.to_dict()) == meta


def test_run_path_resolution():
    class FakeApi:
        default_entity = "ent"

    api = FakeApi()
    assert wandb_history._resolve_path("a/b/c", api) == "a/b/c"
    assert wandb_history._resolve_path("Proj/xyz", api) == "ent/Proj/xyz"
    assert wandb_history._resolve_path("xyz", api) == "ent/ProximalFM/xyz"
