"""Output helpers for the prior audits: backend and saving.

This module owns *where things go*; :mod:`analysis_prior.design` owns *how they
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
    :mod:`analysis_prior.design` for why that matters and how to pick it. The
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
    df: pd.DataFrame, outdir: Path, name: str, caption: str = ""
) -> List[Path]:
    """Save ``df`` as both a LaTeX table and a CSV under ``outdir``.

    The ``.tex`` is paper-ready (it is the only tracked format -- ``.csv`` is
    gitignored). The ``.csv`` is for eyeballing and downstream reuse.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"{name}.csv"
    df.to_csv(csv_path, index=False)

    tex_path = outdir / f"{name}.tex"
    tex_path.write_text(
        df.to_latex(
            index=False, escape=True, caption=caption or None, float_format="%.3f"
        )
    )
    return [tex_path, csv_path]
