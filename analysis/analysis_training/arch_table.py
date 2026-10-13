"""Render the architecture table (``tab:model_hparams``).

```bash
uv run python -m analysis.analysis_training.arch_table --print
uv run python -m analysis.analysis_training.arch_table --script scripts/pretrain_prod.sh \\
    --out docs/latex/assets/training/model_hparams.tex
```

Same construction as :mod:`analysis.analysis_training.hparam_table`, for the same reason:
the hand-written architecture table had already drifted from the run it
describes. Measured against ``scripts/pretrain_prod.sh`` it was wrong in four
places -- it reported ``4`` attention heads in the column embedder and ``4`` in
the in-context learner where all three stages use ``8``, and ``B = 1000`` over
``[-4, 4]`` where the run passes ``--nbins 2000 --cate_vmin -5 --cate_vmax 5``.
The parameter total was stale as a *consequence* of the last one: a 2000-bin
head is 2.6M parameters, not 1.3M, so ``28.7``M understated the model by nearly
a million.

Two sources, and neither is this file:

- **Trainer flags** come from :mod:`analysis.analysis_training.run_config`, which layers
  the script's flags over ``build_parser()``'s defaults. The architecture block
  is the case that machinery exists for: ``pretrain_prod.sh`` passes *none* of
  it, so every row here is a parser default -- reported at the value the run
  used, and raising rather than printing a stale number if the flag is renamed.
- **Parameter counts** come from instantiating :class:`ProximalFM` on that
  config, on the ``meta`` device so no weights are allocated. They are therefore
  a property of the code plus the script, and the head's share tracks ``--nbins``
  automatically. ``--no-params`` drops those rows (and the torch import cost).

Rows are declared in :func:`build_rows` as ``(group, label, value)``; the
``Row`` dataclass, the renderer and the number formatters come from its sibling
:mod:`~analysis.analysis_training.hparam_table` rather than being copied, so the two
tables stay typographically identical -- they are read as a pair, and they read
the same launch script through the same :mod:`~analysis.analysis_training.run_config`, so
they cannot disagree about the run.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
from typing import List, Sequence

from analysis.analysis_training.hparam_table import (
    Row,
    num,
    render_latex,
    tt,
    write_csv,
)
from analysis.analysis_training.run_config import TrainingRun, parse_script

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPT = REPO_ROOT / "scripts" / "pretrain_prod.sh"
DEFAULT_OUT = (
    REPO_ROOT / "analysis" / "analysis_training" / "results" / "model_hparams.tex"
)
LABEL = "tab:model_hparams"

CAPTION = (
    "Architectural hyperparameters of \\modelname. The three stages are those "
    "of TabICLv2, kept at its values so that its pre-trained weights initialize "
    "every component we did not modify; the output head is ours. Parameter "
    "counts are per stage, the in-context learner's including the head."
)

# The trainer spells the softmax variant as a flag value; the appendix names it
# in words. Anything unrecognised falls through verbatim rather than being
# silently prettified into a claim the flag does not make.
_SSMAX_NAMES = {
    "none": "none",
    "ssmax": "scalable softmax",
    "ssmax-mlp": "MLP-scaled",
    "ssmax-mlp-elementwise": "MLP-scaled, elementwise",
    "qassmax-mlp": "query-aware",
    "qassmax-mlp-elementwise": "query-aware, elementwise",
}

# Which stage each top-level module belongs to, in table order.
STAGES = (
    ("Column embedder", "col_embedder"),
    ("Row interactor", "row_interactor"),
    ("In-context learner", "icl_predictor"),
)

# The subset of the trainer config `ProximalFM.__init__` consumes. Listed rather
# than filtered by signature so a constructor argument that gains a trainer flag
# has to be added deliberately; `_run.py::build_model` holds the same list.
MODEL_KEYS = (
    "loss_type",
    "nbins",
    "vmin",
    "vmax",
    "cate_vmin",
    "cate_vmax",
    "embed_dim",
    "col_num_blocks",
    "col_nhead",
    "col_num_inds",
    "col_affine",
    "col_ssmax",
    "row_num_blocks",
    "row_nhead",
    "row_num_cls",
    "row_rope_base",
    "row_rope_interleaved",
    "icl_num_blocks",
    "icl_nhead",
    "icl_ssmax",
    "ff_factor",
    "dropout",
    "activation",
    "norm_first",
    "bias_free_ln",
    "recompute",
)


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #


def ssmax(value) -> str:
    """Name the scalable-softmax variant, or report the flag verbatim."""
    if value is True:
        value = "qassmax-mlp-elementwise"
    elif value is False:
        value = "none"
    text = _SSMAX_NAMES.get(str(value))
    return text if text is not None else tt(value)


def mega(count: int, digits: int = 1) -> str:
    """Parameter count in millions, e.g. ``28312528`` -> ``$28.3$\\,M``."""
    return f"${count / 1e6:.{digits}f}$\\,M"


def power(value: float) -> str:
    """An exact power of ten as ``$10^{k}$``; anything else as a plain number.

    ``--row_rope_base 100000`` reads as $10^{5}$ in the appendix, but a base
    that is *not* a power of ten must not be rounded into one, so the test is
    exact and the fallback prints the digits.
    """
    if value > 0 and float(value).is_integer():
        digits = f"{int(value)}"
        if digits[0] == "1" and set(digits[1:]) <= {"0"}:
            return f"$10^{{{len(digits) - 1}}}$"
        return num(int(value))
    return f"${value:g}$"


# --------------------------------------------------------------------------- #
# parameter counts
# --------------------------------------------------------------------------- #


def count_parameters(config: Namespace) -> dict:
    """Parameters per stage, plus ``head`` and ``total``, for ``config``.

    Built on the ``meta`` device: shapes are what the table reports, so no
    storage is allocated and the count costs nothing.
    """
    import torch

    from proximalfm._model.proximalfm import ProximalFM

    kwargs = {key: getattr(config, key) for key in MODEL_KEYS}
    with torch.device("meta"):
        model = ProximalFM(**kwargs)

    def total(module) -> int:
        return sum(p.numel() for p in module.parameters())

    counts = {attr: total(getattr(model, attr)) for _, attr in STAGES}
    counts["head"] = total(model.icl_predictor.decoder)
    counts["total"] = total(model)
    return counts


# --------------------------------------------------------------------------- #
# rows
# --------------------------------------------------------------------------- #


def build_rows(run: TrainingRun, counts: dict | None = None) -> List[Row]:
    """The table's content: one ``Row`` per line, grouped by stage."""
    from proximalfm._model.proximalfm import CATE_SINGLE_LOSS_TYPES

    c: Namespace = run.config
    rows: List[Row] = []

    def add(group: str, label: str, value: str) -> None:
        rows.append(Row(group, label, value))

    # --- shared -------------------------------------------------------------
    add("Shared", "Embedding dimension $E$", num(c.embed_dim))
    add("Shared", "Feedforward expansion factor", num(c.ff_factor))
    add("Shared", "Activation", str(c.activation).upper())
    add("Shared", "Normalization", "pre-norm" if c.norm_first else "post-norm")
    add("Shared", "Dropout", f"${c.dropout:.1f}$")

    # --- column embedder ----------------------------------------------------
    add("Column embedder", "Blocks", num(c.col_num_blocks))
    add("Column embedder", "Attention heads", num(c.col_nhead))
    add("Column embedder", "Inducing points", num(c.col_num_inds))
    add("Column embedder", "Scalable softmax", ssmax(c.col_ssmax))

    # --- row interactor -----------------------------------------------------
    add("Row interactor", "Blocks", num(c.row_num_blocks))
    add("Row interactor", "Attention heads", num(c.row_nhead))
    add("Row interactor", "CLS tokens $n_{\\mathrm{cls}}$", num(c.row_num_cls))
    add("Row interactor", "RoPE base", power(c.row_rope_base))

    # --- in-context learner -------------------------------------------------
    # The ICL stage runs on the concatenated CLS tokens, so its width is derived
    # rather than configured; the same product `_model` computes as `icl_dim`.
    add("In-context learner", "Blocks", num(c.icl_num_blocks))
    add("In-context learner", "Attention heads", num(c.icl_nhead))
    add(
        "In-context learner",
        "Model dimension $E \\cdot n_{\\mathrm{cls}}$",
        num(c.embed_dim * c.row_num_cls),
    )

    # --- output head --------------------------------------------------------
    # `cate_single`/`noise_aware` put one head on tau, on its own grid; the
    # two-head losses score Y(a) on [vmin, vmax]. Reporting the wrong pair would
    # misstate the support constraint the appendix turns on.
    if c.loss_type in CATE_SINGLE_LOSS_TYPES:
        vmin, vmax = c.cate_vmin, c.cate_vmax
    else:
        vmin, vmax = c.vmin, c.vmax
    add("Output head", "Bins $B$", num(c.nbins))
    add("Output head", "Grid $[v_{\\min}, v_{\\max}]$", f"$[{vmin:g}, {vmax:g}]$")

    # --- parameters ---------------------------------------------------------
    if counts is not None:
        for label, attr in STAGES:
            add("Parameters", label, mega(counts[attr]))
        add("Parameters", "Total", mega(counts["total"]))

    return rows


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
        "(e.g. --set CKPT=other.ckpt). Repeatable. The ambient environment is "
        "never read, so the table does not depend on the calling shell.",
    )
    parser.add_argument(
        "--no-params",
        dest="params",
        action="store_false",
        help="Skip the parameter-count rows (does not instantiate the model).",
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
    counts = count_parameters(run.config) if args.params else None
    rows = build_rows(run, counts)

    caption = args.caption or CAPTION

    args.out.parent.mkdir(parents=True, exist_ok=True)
    latex = render_latex(
        rows,
        caption=caption,
        label=args.label,
        generator="analysis.analysis_training.arch_table",
    )
    args.out.write_text(latex)
    write_csv(rows, args.out.with_suffix(".csv"))

    if args.to_stdout:
        print(latex)
    print(f"wrote {args.out} ({len(rows)} rows) and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
