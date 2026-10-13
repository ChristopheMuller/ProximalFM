"""Output helpers for the prior audits: backend and saving.

This module owns *where things go*; :mod:`analysis.analysis_prior.design` owns *how they
look*. Keeping them apart means a change to the visual identity never risks
touching the I/O, and the design module can import ``plt`` from here without a
cycle.

Centralises two things so individual analyses don't repeat them:

- A headless Matplotlib backend. Analyses run in scripts, CI and notebooks; none
  of them should require a display, and importing this module guarantees that.
- Save helpers that write each artifact in the paper-ready *and* the
  machine-readable format at once: figures as both ``.pdf`` (for LaTeX) and
  ``.png`` (for quick viewing), tables as both ``.tex`` and ``.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

# Must be set before pyplot is imported. Agg renders to files without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import pandas as pd  # noqa: E402


def save_figure(fig: "plt.Figure", outdir: Path, name: str) -> List[Path]:
    """Save ``fig`` as both PDF and PNG under ``outdir``; return the paths.

    The figure is saved at exactly the size it was created with -- see
    :mod:`analysis.analysis_prior.design` for why that matters and how to pick it. The
    figure is closed afterwards so that generating many analyses in one process
    does not leak figure handles.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = outdir / f"{name}.{ext}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def save_table(
    df: pd.DataFrame,
    outdir: Path,
    name: str,
    caption: str = "",
    label: str = None,
    escape: bool = True,
    wrap_columns: List[str] = None,
    small: bool = False,
    centering: bool = True,
    here: bool = False,
    group_column: str = None,
    column_format: str = None,
    multicolumn_format: str = None,
) -> List[Path]:
    """Save ``df`` as both a LaTeX table and a CSV under ``outdir``.

    The ``.tex`` is a complete float (``\\begin{table}...\\end{table}``) so a
    section can ``\\input`` it and refer to it with ``\\Cref{<label>}``; it is
    the only tracked format, the ``.csv`` (gitignored) being for eyeballing and
    downstream reuse. A ``label`` is baked in only when given, so tables keep a
    single source of truth for their caption and reference key.

    ``escape`` (default ``True``) protects plain-text cells that contain LaTeX
    specials such as underscores. Pass ``escape=False`` when the frame's cells
    and headers are *deliberately* LaTeX (e.g. ``$d_U$``); with escaping on,
    that math would be printed verbatim rather than typeset.

    ``wrap_columns`` names the columns that should line-wrap to fit
    ``\\textwidth`` instead of forcing the table wider than the page -- pass the
    ones holding prose or long identifiers (e.g. long ``\\texttt{...}`` names,
    free-text descriptions). Plain ``l`` columns in a LaTeX ``tabular`` never
    break a line no matter how long the cell is; wrapped columns render as
    ragged-right ``tabularx`` ``X`` columns instead, which is a portrait-page
    fix for an over-wide table and preferable to rotating the page (landscape
    breaks the reading flow of an appendix meant to be read section by
    section). Requires ``\\usepackage{tabularx}`` in the document preamble.

    ``small`` wraps the table body in ``\\small`` -- combine with
    ``wrap_columns`` for tables with many columns of running text.

    ``centering`` inserts ``\\centering`` right after ``\\begin{table}``.

    ``here`` tags the float ``\\begin{table}[h]``, hinting LaTeX to place it
    approximately where it occurs in the source rather than floating it.

    ``group_column`` names a column that carries a group label on the first
    row of each block and ``""`` on every row after -- the sparsified pattern
    ``_grouped_rows`` in :mod:`analysis.analysis_prior.analyses.hyperparameters`
    already produces for the CSV. Rendered as an ordinary ``l`` column, that
    label wastes a full column's width on text that appears only a handful of
    times; pulling it out and turning each label into a bold, full-width
    ``\\multicolumn`` header row instead gives that width back to the columns
    that actually need it (e.g. the wrapped ones), which for a table with many
    rows is often the difference between fitting on one page and not. The
    column stays in the CSV; only the LaTeX body is affected.

    ``column_format`` overrides the auto-generated LaTeX column spec (e.g.
    ``"r|rrr|rrr"`` to rule off column groups). It is mutually exclusive with
    ``wrap_columns``, which builds its own spec. ``multicolumn_format`` sets the
    alignment of the ``\\multicolumn`` group spanners a MultiIndex-columned
    frame emits; pass e.g. ``"|r"`` so a ``|`` in ``column_format`` runs
    continuously through the spanner row instead of breaking at it.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"{name}.csv"
    df.to_csv(csv_path, index=False)

    body_df = df
    groups = None
    if group_column:
        if group_column not in df.columns:
            raise ValueError(f"group_column {group_column!r} not in frame")
        groups = df[group_column]
        body_df = df.drop(columns=[group_column])

    if wrap_columns:
        if column_format is not None:
            raise ValueError("pass either wrap_columns or column_format, not both")
        unknown = set(wrap_columns) - set(body_df.columns)
        if unknown:
            raise ValueError(f"wrap_columns not in frame: {sorted(unknown)}")
        column_format = "".join(
            ">{\\raggedright\\arraybackslash}X" if col in wrap_columns else "l"
            for col in body_df.columns
        )

    extra = (
        {} if multicolumn_format is None else {"multicolumn_format": multicolumn_format}
    )
    latex = body_df.to_latex(
        index=False,
        escape=escape,
        caption=caption or None,
        label=label,
        float_format="%.3f",
        column_format=column_format,
        **extra,
    )
    if wrap_columns:
        # to_latex always emits `tabular`; swap in `tabularx` for the same
        # column_format string, which the substitution above already primed
        # with X columns. `{\textwidth}` is tabularx's extra mandatory arg.
        latex = latex.replace("\\begin{tabular}", "\\begin{tabularx}{\\textwidth}")
        latex = latex.replace("\\end{tabular}", "\\end{tabularx}")
    if groups is not None:
        latex = _insert_group_headers(latex, groups, ncols=len(body_df.columns))
    if small:
        env = "tabularx" if wrap_columns else "tabular"
        latex = latex.replace(f"\\begin{{{env}}}", f"\\small\n\\begin{{{env}}}")
    if centering:
        latex = latex.replace("\\begin{table}", "\\begin{table}\n\\centering", 1)
    if here:
        latex = latex.replace("\\begin{table}", "\\begin{table}[h]", 1)

    tex_path = outdir / f"{name}.tex"
    tex_path.write_text(latex)
    return [tex_path, csv_path]


def _insert_group_headers(latex: str, groups: pd.Series, ncols: int) -> str:
    """Turn a sparsified group column into bold full-width header rows.

    ``groups`` has one entry per data row: the group label on the first row
    of a block, ``""`` on the rest. Splits the ``to_latex`` output on the
    (single, pre-group-column) ``\\midrule``/``\\bottomrule`` markers to find
    the data-row lines -- one line per row, in order, since ``to_latex`` never
    wraps a row across lines -- and inserts a ``\\multicolumn`` header line
    before each non-empty label, with a ``\\midrule`` separating it from the
    previous block.
    """
    lines = latex.split("\n")
    start = lines.index("\\midrule") + 1
    end = lines.index("\\bottomrule")
    body, rest = lines[start:end], lines[end:]

    values = list(groups)
    if len(body) != len(values):
        raise ValueError("group_column length does not match table rows")

    new_body: List[str] = []
    for label, line in zip(values, body):
        if label:
            if new_body:
                new_body.append("\\midrule")
            new_body.append(
                f"\\multicolumn{{{ncols}}}{{l}}{{\\textbf{{{label}}}}} \\\\"
            )
        new_body.append(line)

    return "\n".join(lines[:start] + new_body + rest)
