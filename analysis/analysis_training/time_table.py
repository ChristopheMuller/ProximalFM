"""Render the training cost table (``tab:training_cost``).

```bash
uv run python -m analysis.analysis_training.time_table --print
uv run python -m analysis.analysis_training.time_table --run ProximalFM/<id> --script scripts/other.sh
```

Where the run's GPU-hours went. The point of the table is the *first* row: prior
generation, not the forward/backward pass, is the dominant cost of training a
PFN, because the prior **is** the dataset and is generated once per use. That is
what ``--gen_reuse_passes`` exists to trade against, and it is the number a
reader needs in order to price a replication.

The decomposition is exact rather than indicative, and that rests on how the run
was configured: ``--prior_device cuda:0`` (the same device as training) with
``--train_dataloader_num_workers 0`` means generation is *serial* with the
optimizer step, not overlapped in a worker process. So the four measured
components are disjoint intervals of one wall clock, and what is left over is
genuinely everything else. A run with a prefetching dataloader would need a
different table, because the slices would overlap and the residual could go
negative -- :func:`build` refuses that case rather than printing it.

**The table reports occupancy, and says nothing about scheduling.** wandb keeps
two clocks: ``_runtime`` accumulates process lifetimes, ``_timestamp`` is
absolute. ``whole_1`` was interrupted after step 51 400 and resumed 4.9 h later,
so its calendar span (71.05 h) exceeds its occupancy (66.15 h) by that idle time.
Only the occupancy is compute, and it is the only figure here: the total and
every percentage are against ``meta.runtime_s``, so the table reads as the
uninterrupted run it is equivalent to. Where the hours fell on a calendar is a
fact about the cluster's queue, not about the method, and the appendix does not
report it. :attr:`RunMeta.gaps` still locates the boundaries -- that is what
establishes which of the two clocks to use -- it just does not reach the table.

Sources are split on purpose: what was *configured* (steps, batch size) comes
from the launch script through :mod:`analysis.analysis_training.run_config`, the same path
:mod:`analysis.analysis_training.hparam_table` reads, so the two tables cannot disagree;
what it *cost* comes from wandb. Timing keys, all logged by
``proximalfm.train._run``: ``prior_time`` (per step, block generation amortized
over the steps that reuse the block), ``train_time`` (per step), and
``Charts/{prior,synt}_eval_time`` (per validation pass).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from analysis.analysis_training.hparam_table import num
from analysis.analysis_training.run_config import TrainingRun, parse_script
from analysis.analysis_training.wandb_history import (
    DEFAULT_RUN,
    RunMeta,
    fetch,
    fetch_meta,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPT = REPO_ROOT / "scripts" / "pretrain_prod.sh"
DEFAULT_OUT = (
    REPO_ROOT / "analysis" / "analysis_training" / "results" / "training_cost.tex"
)
LABEL = "tab:training_cost"

CAPTION = (
    "Cost of the \\modelname training run, as measured GPU-hours. "
    "Validation is the two frozen evaluation datasets, scored at their logging "
    "intervals throughout training."
)


@dataclass(frozen=True)
class CostRow:
    group: str
    label: str
    hours: float | None
    share: float | None  # Fraction of occupancy.
    detail: str = ""  # Typeset value, when the row is not a duration.
    raw: float | None = None  # `detail`'s number, for the csv: `detail` is LaTeX.


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #


def build(
    run: TrainingRun,
    meta: RunMeta,
    *,
    wandb_run: str = DEFAULT_RUN,
    fetch_fn=fetch,
) -> List[CostRow]:
    """Measure the components and assemble the rows.

    ``fetch_fn`` is injectable so the two refusals below can be exercised
    without a network round-trip; nothing else should pass it.
    """
    config = run.config

    if config.train_dataloader_num_workers:
        raise ValueError(
            "This table assumes generation is serial with training "
            f"(--train_dataloader_num_workers 0); the script says "
            f"{config.train_dataloader_num_workers}. With worker processes the "
            "components overlap and the residual is not interpretable."
        )

    steps = meta.final_step

    # Per-step keys share a cadence; the two eval surfaces do not, so each is
    # its own scan (see wandb_history's docstring).
    per_step = fetch_fn(["train_time", "prior_time"], run=wandb_run)
    prior_eval = fetch_fn(["Charts/prior_eval_time"], run=wandb_run)
    synt_eval = fetch_fn(["Charts/synt_eval_time"], run=wandb_run)

    generation = per_step.total("prior_time", steps)
    optimizer = per_step.total("train_time", steps)
    # Validation passes are counted, not extrapolated: a pass either ran at a
    # given step or it did not, and the scan's row set is that ground truth.
    eval_prior = prior_eval.total("Charts/prior_eval_time")
    eval_synt = synt_eval.total("Charts/synt_eval_time")

    measured = generation + optimizer + eval_prior + eval_synt
    other = meta.runtime_s - measured
    if other < 0:
        raise ValueError(
            f"measured components ({measured / 3600:.2f} h) exceed occupancy "
            f"({meta.runtime_s / 3600:.2f} h) -- the slices are not disjoint, so "
            "this decomposition does not describe this run."
        )

    total = meta.runtime_s
    rows: List[CostRow] = [
        CostRow("Compute", "Prior generation", generation, generation / total),
        CostRow("Compute", "Forward / backward", optimizer, optimizer / total),
        CostRow(
            "Compute",
            f"Validation, prior set ({num(len(prior_eval.frame))} passes)",
            eval_prior,
            eval_prior / total,
        ),
        CostRow(
            "Compute",
            f"Validation, realistic set ({num(len(synt_eval.frame))} passes)",
            eval_synt,
            eval_synt / total,
        ),
        CostRow("Compute", "Checkpointing, logging, startup", other, other / total),
        CostRow(
            "Total",
            f"GPU-hours ($1 \\times$ {meta.gpu.replace('NVIDIA ', '')})",
            total,
            1.0,
        ),
    ]

    datasets = config.batch_size * steps
    rows += [
        CostRow("Scale", "Training steps", None, None, num(steps), float(steps)),
        CostRow(
            "Scale",
            "Synthetic datasets consumed",
            None,
            None,
            num(datasets),
            float(datasets),
        ),
        CostRow(
            "Scale",
            "Seconds per step",
            None,
            None,
            f"${total / steps:.2f}$",
            total / steps,
        ),
    ]
    return rows


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def _csv_number(value: float) -> str:
    """Counts stay counts: ``%g`` would render 4 800 000 as ``4.8e+06``."""
    return str(int(value)) if float(value).is_integer() else f"{value:.6g}"


def _cells(row: CostRow) -> tuple[str, str]:
    if row.hours is None:
        return row.detail, ""
    return f"${row.hours / 3600:.2f}$", f"${100 * row.share:.1f}$"


def render_latex(rows: Sequence[CostRow], caption: str, label: str = LABEL) -> str:
    """A ``table`` float shaped like ``tab:training_hparams``."""
    lines = [
        "% Generated by `python -m analysis.analysis_training.time_table` -- do not edit by hand.",
        "\\begin{table}[t]",
        "    \\centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        "    \\begin{tabular}{llrr}",
        "        \\toprule",
        "        & & Hours & \\% \\\\",
        "        \\midrule",
    ]

    groups: List[tuple[str, List[CostRow]]] = []
    for row in rows:
        if not groups or groups[-1][0] != row.group:
            groups.append((row.group, []))
        groups[-1][1].append(row)

    for index, (group, block) in enumerate(groups):
        if index:
            lines.append("        \\midrule")
        # `\multirow{1}` is a no-op that reads as a mistake; a one-row block
        # takes the label plainly.
        if len(block) == 1:
            lines.append(f"        {group}")
        else:
            lines.append(f"        \\multirow{{{len(block)}}}{{*}}{{{group}}}")
        for row in block:
            value, share = _cells(row)
            lines.append(f"            & {row.label} & {value} & {share} \\\\")

    lines += ["        \\bottomrule", "    \\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def write_csv(rows: Sequence[CostRow], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["group", "component", "hours", "percent_of_occupancy", "value"]
        )
        for row in rows:
            writer.writerow(
                [
                    row.group,
                    row.label,
                    "" if row.hours is None else f"{row.hours / 3600:.4f}",
                    "" if row.share is None else f"{100 * row.share:.2f}",
                    "" if row.raw is None else _csv_number(row.raw),
                ]
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--run", default=DEFAULT_RUN, help="wandb run to measure.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--label", default=LABEL)
    parser.add_argument("--caption", default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="VAR=VALUE",
        help="Shell variable override for the launch script. Repeatable.",
    )
    parser.add_argument("--force", action="store_true", help="Refetch from wandb.")
    parser.add_argument("--print", dest="to_stdout", action="store_true")
    args = parser.parse_args(argv)

    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            parser.error(f"--set expects VAR=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = value

    run = parse_script(args.script, overrides)
    meta = fetch_meta(run=args.run, force=args.force)
    rows = build(run, meta, wandb_run=args.run)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    latex = render_latex(rows, caption=args.caption or CAPTION, label=args.label)
    args.out.write_text(latex)
    write_csv(rows, args.out.with_suffix(".csv"))

    if args.to_stdout:
        print(latex)
    print(f"wrote {args.out} and {args.out.with_suffix('.csv')}  [{meta.name}]")


if __name__ == "__main__":
    main()
