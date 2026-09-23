"""Render the training hyperparameter table (``tab:training_hparams``).

```bash
uv run python -m analysis_training.hparam_table                     # defaults below
uv run python -m analysis_training.hparam_table --script scripts/pretrain_prod.sh \
    --out docs/latex/assets/training/training_hparams.tex
```

The table is generated from ``scripts/pretrain_prod.sh`` rather than written by
hand, because the appendix has already gone stale against the run once: its
architecture table records ``B = 1000`` over ``[-4, 4]`` where the run used
``--nbins 2000 --cate_vmin -5 --cate_vmax 5``. Anything a reader could check
against the launch script is therefore read *from* the launch script, and the
derived quantities that matter for reading the run -- the bin width $w$, the
final $\\sigma$ in absolute units, the gradient-accumulation factor -- are
computed with the trainer's own helpers (``hl_bin_width``,
``resolve_hl_final_sigma``) rather than recomputed here.

Rows are declared in :func:`build_rows` as ``(group, label, value)``; the
renderer is generic, so adding a row is one entry there. Values reach the table
through :class:`~analysis_training.run_config.TrainingRun`, whose ``config`` is
the script's flags layered over ``build_parser()``'s defaults -- so a knob the
script never passes (``--weight_decay``, ``--scheduler``) is still reported at
the value the run actually used.
"""

from __future__ import annotations

import argparse
import csv
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from proximalfm.train._train_config import hl_bin_width, resolve_hl_final_sigma

from analysis_training.run_config import TrainingRun, parse_script

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCRIPT = REPO_ROOT / "scripts" / "pretrain_prod.sh"
DEFAULT_OUT = REPO_ROOT / "analysis_training" / "results" / "training_hparams.tex"
LABEL = "tab:training_hparams"

# Written for the appendix, not for the tool: a caption is read next to the
# table, never next to the command that produced it, so it names the prior, the
# objective and the selection rule rather than the launch script.
CAPTION = (
    "Training configuration of \\modelname: the prior the model is trained on, "
    "the objective and its smoothing schedule, the optimizer, the pre-trained "
    "backbone it is initialized from, and the validation sets used for "
    "checkpoint selection."
)


@dataclass(frozen=True)
class Row:
    group: str
    label: str
    value: str


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #

_TEX_SPECIAL = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$"}


def tt(text: str) -> str:
    """Typeset ``text`` verbatim, escaping the specials a path/flag contains."""
    return "\\texttt{" + "".join(_TEX_SPECIAL.get(c, c) for c in str(text)) + "}"


def num(value: int) -> str:
    """Thin-space digit grouping, e.g. ``75000`` -> ``75\\,000``."""
    return f"{int(value):,}".replace(",", "\\,")


def sci(value: float, digits: int = 0) -> str:
    """Scientific notation as math, e.g. ``1e-4`` -> ``$1 \\times 10^{-4}$``."""
    if value == 0:
        return "$0$"
    exponent = 0
    mantissa = float(value)
    while abs(mantissa) >= 10:
        mantissa /= 10
        exponent += 1
    while abs(mantissa) < 1:
        mantissa *= 10
        exponent -= 1
    head = f"{mantissa:.{digits}f}"
    if exponent == 0:
        return f"${head}$"
    return f"${head} \\times 10^{{{exponent}}}$"


def dec(value: float, digits: int = 3) -> str:
    """Fixed-point math, trailing zeros trimmed."""
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return f"${text}$"


def episodes(cap: int | None) -> str:
    """Episode cap, where ``None`` means the whole set rather than zero."""
    return "all" if cap is None else num(cap)


def rng(lo, hi) -> str:
    return f"${lo}$--${hi}$"


def yesno(flag: bool) -> str:
    return "yes" if flag else "no"


# --------------------------------------------------------------------------- #
# rows
# --------------------------------------------------------------------------- #


def build_rows(run: TrainingRun) -> List[Row]:
    """The table's content: one ``Row`` per line, grouped by stage."""
    c: Namespace = run.config

    width = hl_bin_width(c)
    final_sigma = resolve_hl_final_sigma(c)
    accum = c.batch_size / c.micro_batch_size

    rows: List[Row] = []

    def add(group: str, label: str, value: str) -> None:
        rows.append(Row(group, label, value))

    # --- prior / data -------------------------------------------------------
    # --prior_type is not reported: tree_scm/mix_scm are dead weight, so the
    # flag is legacy and naming it would suggest a choice the paper never makes.
    add("Prior", "Confounder dimension $d_U$", rng(c.min_dim_U, c.max_dim_U))
    add(
        "Prior",
        "Treatment-confounding proxy dimension $d_Z$",
        rng(c.min_dim_Z, c.max_dim_Z),
    )
    add(
        "Prior",
        "Outcome-confounding proxy dimension $d_W$",
        rng(c.min_dim_W, c.max_dim_W),
    )
    add("Prior", "Covariate dimension $d_X$", rng(c.min_dim_X, c.max_dim_X))
    sampling = "log-uniform" if c.log_seq_len else "uniform"
    add(
        "Prior",
        "Context units $N$",
        f"{rng(num(c.min_seq_len), num(c.max_seq_len))} ({sampling})",
    )
    add("Prior", "Query units $n_q$", num(c.n_query))
    add("Prior", "Monte Carlo worlds $K$", num(c.mc_cate))
    add(
        "Prior",
        "MC targets on context units",
        yesno(c.mc_targets_on_context),
    )

    # --- objective ----------------------------------------------------------
    add("Objective", "Loss", tt(c.loss_type))
    add("Objective", "Mean penalty $\\lambda_{\\mathrm{mean}}$", dec(c.lambda_cate))
    add("Objective", "Bins $B$", num(c.nbins))
    add(
        "Objective",
        "Grid $[v_{\\min}, v_{\\max}]$",
        f"$[{c.cate_vmin:g}, {c.cate_vmax:g}]$",
    )
    add("Objective", "Bin width $w$", dec(width, 4))
    add("Objective", "Initial smoothing $\\sigma_0$", dec(c.hl_start_sigma))
    add(
        "Objective",
        "$\\sigma$ held constant for",
        f"{num(c.hl_warmup_sigma_steps)} steps",
    )
    add(
        "Objective",
        "$\\sigma$ linear decay over",
        f"{num(c.hl_linear_decay_steps)} steps",
    )
    add(
        "Objective",
        "Final smoothing $\\sigma$",
        f"{dec(final_sigma, 4)} (${c.hl_final_sigma_bins:g}w$)",
    )

    # --- optimization -------------------------------------------------------
    add("Optimization", "Optimizer", "AdamW")
    add("Optimization", "Learning rate", sci(c.lr))
    add("Optimization", "Schedule", tt(c.scheduler))
    warmup = (
        f"{num(c.warmup_steps)} steps"
        if c.warmup_proportion == -1
        else f"{c.warmup_proportion:g} of total steps"
    )
    add("Optimization", "Warmup", warmup)
    add("Optimization", "Weight decay", dec(c.weight_decay))
    add("Optimization", "Gradient clipping", dec(c.gradient_clipping))
    add("Optimization", "Training steps", num(c.max_steps))
    add(
        "Optimization",
        "Datasets per step",
        f"{num(c.batch_size)} ({num(c.micro_batch_size)} $\\times$ {accum:g} accumulated)",
    )
    add("Optimization", "Precision", f"{tt(c.dtype)} (AMP)")
    add("Optimization", "Seed (numpy / torch)", f"{c.np_seed} / {c.torch_seed}")

    # --- initialization -----------------------------------------------------
    # Paths are reported repo-relative, root included: `transfer_rules.yaml`
    # alone does not tell a reader where in the checkout to find it.
    add("Initialization", "Init. checkpoint", tt(c.checkpoint_path))
    add("Initialization", "Transfer rules", tt(c.transfer_rules))
    add("Initialization", "Weights only (fresh optimizer)", yesno(c.only_load_model))

    # --- validation / selection --------------------------------------------
    # A surface with no directory is disabled, and a row reporting its interval
    # would claim a validation pass that never ran.
    if c.prior_eval_dir:
        add("Validation", "Prior set", tt(Path(c.prior_eval_dir).name))
        add(
            "Validation",
            "Prior set: episodes / interval",
            f"{episodes(c.prior_eval_max_episodes)} / {num(c.prior_eval_interval)} steps",
        )
    if c.synt_eval_dir:
        add("Validation", "Realistic set", tt(Path(c.synt_eval_dir).name))
        add(
            "Validation",
            "Realistic set: episodes / interval",
            f"{episodes(c.synt_eval_max_episodes)} / {num(c.synt_eval_interval)} steps",
        )
    add(
        "Validation",
        "Selection metric",
        f"{tt(c.best_metric)} ({c.best_metric_mode})",
    )

    return rows


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def render_latex(
    rows: Sequence[Row],
    caption: str,
    label: str = LABEL,
    generator: str = "analysis_training.hparam_table",
) -> str:
    """A complete ``table`` float, grouped by ``Row.group`` with ``multirow``.

    Matches the shape of ``tab:model_hparams`` so the two read as a pair:
    booktabs rules, a ``multirow`` group column, values right-aligned. That is
    also why ``analysis_training.arch_table`` renders through this function rather
    than keeping its own copy, passing its own ``generator`` for the header
    comment so the file names the command that regenerates it.
    """
    lines = [
        f"% Generated by `python -m {generator}` -- do not edit by hand.",
        "\\begin{table}[t]",
        "    \\centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        "    \\begin{tabular}{llr}",
        "        \\toprule",
        "        Stage & Hyperparameter & Value \\\\",
        "        \\midrule",
    ]

    groups: List[tuple[str, List[Row]]] = []
    for row in rows:
        if not groups or groups[-1][0] != row.group:
            groups.append((row.group, []))
        groups[-1][1].append(row)

    for index, (group, block) in enumerate(groups):
        if index:
            lines.append("        \\midrule")
        lines.append(f"        \\multirow{{{len(block)}}}{{*}}{{{group}}}")
        for row in block:
            lines.append(f"            & {row.label} & {row.value} \\\\")

    lines += ["        \\bottomrule", "    \\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def write_csv(rows: Sequence[Row], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "hyperparameter", "value"])
        for row in rows:
            writer.writerow([row.group, row.label, row.value])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--script",
        type=Path,
        default=DEFAULT_SCRIPT,
        help="Training launch script to read (default: scripts/pretrain_prod.sh).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output .tex path; a sibling .csv is written next to it.",
    )
    parser.add_argument(
        "--label",
        default=LABEL,
        help=f"LaTeX label of the table (default: {LABEL}).",
    )
    parser.add_argument(
        "--caption",
        default=None,
        help="Override the generated caption.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="VAR=VALUE",
        help="Shell variable override, as if exported before the script "
        "(e.g. --set RUN=whole_2). Repeatable. The ambient environment is "
        "never read, so the table does not depend on the calling shell.",
    )
    parser.add_argument(
        "--print",
        dest="to_stdout",
        action="store_true",
        help="Also print the table to stdout.",
    )
    args = parser.parse_args(argv)

    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            parser.error(f"--set expects VAR=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = value

    run = parse_script(args.script, overrides)
    rows = build_rows(run)

    caption = args.caption or CAPTION

    args.out.parent.mkdir(parents=True, exist_ok=True)
    latex = render_latex(rows, caption=caption, label=args.label)
    args.out.write_text(latex)
    write_csv(rows, args.out.with_suffix(".csv"))

    if args.to_stdout:
        print(latex)
    print(f"wrote {args.out} ({len(rows)} rows) and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
