"""
prepare_chamber_csv.py — Turn a raw Causal Chambers export into a canonical,
benchmark-ready CSV.

Raw chamber files carry hardware column names (``led_3_uv``, ``ir_3``, ``pol_1``
…), a treatment coded in raw DAC units (``{0, 2047}``), and an observed outcome
(``ir_3``) that is a *separate physical measurement* from the forced
counterfactuals ``Y0``/``Y1``. This script rewrites them into the canonical
schema the benchmark expects:

    U_1..U_p, X_1..X_q, W_1..W_r, Z_1..Z_s, A, Y, Y0, Y1

with ``A`` binary ``{0, 1}`` and (by default) ``Y = Y1 if A else Y0``, so that
consistency ``Y = Y_A`` holds *exactly* rather than up to measurement noise.

NOT standardized — on purpose
─────────────────────────────
Standardization already happens downstream in
``benchmarks/data.py::build_benchmark_dict`` (``safe_standardize``), and
it is deliberately computed from the **first ``train_size`` rows only** so that
no test-set statistics leak into the context the methods see. Standardizing
here would (a) double-standardize and (b) use *all* rows, destroying exactly
that no-leakage property. So this script leaves values on their natural scale.

Usage
─────
    python benchmarks/causal_chambers/prepare_chamber_csv.py --all
    python benchmarks/causal_chambers/prepare_chamber_csv.py \\
        --input  benchmarks/causal_chambers/data/out_binaryA_....csv \\
        --output benchmarks/causal_chambers/data/prepared/strong_linear.csv

Alongside each CSV it writes a ``.diagnostics.json`` reporting the true ATE,
SD(tau), the SD(tau)/ATE ratio and the naive (unadjusted) bias — the numbers
that say whether a file is a meaningful CATE benchmark, an ATE-only benchmark,
or too weakly confounded to discriminate methods at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Default mapping, mirroring load_cc_data() in
# "Proximal Causal Inference on Causal Chambers.Rmd".
CHAMBER_DEFAULTS = {
    "U": ["green"],
    "X": ["red"],
    "W": ["pol_1"],
    "Z": ["led_2_uv"],
    "A": "led_3_uv",
    "Y": "ir_3",
    "Y0": "Y0",
    "Y1": "Y1",
}

DATA_DIR = Path(__file__).resolve().parent / "data"


def read_raw_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV, dropping a leading unnamed index column if present."""
    df = pd.read_csv(path)
    if len(df.columns) and str(df.columns[0]).startswith("Unnamed:"):
        df = df.drop(columns=df.columns[0])
    return df


def binarize(a: np.ndarray, col_name: str) -> np.ndarray:
    """Map a two-valued treatment column to {0, 1}, failing loudly otherwise.

    The whole benchmark stack assumes a binary treatment (e.g. T_Learner splits
    arms with exact ``A == 0`` / ``A == 1`` masks), so a column with three or
    more levels must not be silently rescaled into the unit interval — rows at
    intermediate values would be dropped from both arms without any warning.
    """
    levels = np.unique(a)
    if len(levels) != 2:
        raise ValueError(
            f"treatment column {col_name!r} has {len(levels)} distinct value(s) "
            f"{levels[:10]}; a binary treatment is required."
        )
    return (a == levels[1]).astype(np.float32)


def prepare(
    df: pd.DataFrame,
    mapping: dict,
    *,
    y_source: str = "potential",
) -> tuple[pd.DataFrame, dict]:
    """Return the canonical frame plus a diagnostics dict."""

    def block(key: str) -> list[str]:
        names = mapping.get(key) or []
        names = [names] if isinstance(names, str) else list(names)
        missing = [c for c in names if c not in df.columns]
        if missing:
            raise ValueError(
                f"{key}: column(s) {missing} not in CSV {list(df.columns)}"
            )
        return names

    out = pd.DataFrame(index=df.index)
    for key in ("U", "X", "W", "Z"):
        for i, col in enumerate(block(key), start=1):
            out[f"{key}_{i}"] = df[col].to_numpy(dtype=np.float64)

    a_col, y_col = mapping["A"], mapping["Y"]
    y0_col, y1_col = mapping["Y0"], mapping["Y1"]

    A = binarize(df[a_col].to_numpy(), a_col)
    Y0 = df[y0_col].to_numpy(dtype=np.float64)
    Y1 = df[y1_col].to_numpy(dtype=np.float64)

    if y_source == "potential":
        # Consistency Y = Y_A holds exactly (no independent measurement noise).
        Y = np.where(A > 0.5, Y1, Y0)
    elif y_source == "observed":
        # The physically measured outcome; differs from Y_A by sensor noise.
        Y = df[y_col].to_numpy(dtype=np.float64)
    else:
        raise ValueError(
            f"y_source must be 'potential' or 'observed', got {y_source!r}"
        )

    out["A"] = A
    out["Y"] = Y
    out["Y0"] = Y0
    out["Y1"] = Y1

    tau = Y1 - Y0
    treated, control = A > 0.5, A <= 0.5
    naive = float(Y[treated].mean() - Y[control].mean())
    ate = float(tau.mean())

    diagnostics = {
        "n_rows": int(len(out)),
        "n_treated": int(treated.sum()),
        "y_source": y_source,
        "ate_true": ate,
        "sd_tau": float(tau.std()),
        # < ~0.05 means the CATE metrics (nPEHE / CATE R² / Shape R², all
        # normalised by SD(tau)) are dividing by noise and are not meaningful.
        "sd_tau_over_ate": float(tau.std() / ate) if ate else None,
        "naive_ate": naive,
        # The confounding bias a method is actually being asked to remove.
        "naive_bias_pct_of_ate": float(100 * (naive - ate) / ate) if ate else None,
    }
    u_cols = block("U")
    if u_cols:
        diagnostics["corr_U1_A"] = float(np.corrcoef(out[f"U_1"], A)[0, 1])
    return out, diagnostics


def _resolve_mapping(args: argparse.Namespace) -> dict:
    mapping = dict(CHAMBER_DEFAULTS)
    for key in ("U", "X", "W", "Z"):
        val = getattr(args, key.lower())
        if val is not None:
            mapping[key] = [c for c in val.split(",") if c]
    for key in ("A", "Y", "Y0", "Y1"):
        val = getattr(args, key.lower())
        if val is not None:
            mapping[key] = val
    return mapping


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--input", help="Raw chamber CSV.")
    p.add_argument("--output", help="Destination canonical CSV.")
    p.add_argument(
        "--all",
        action="store_true",
        help="Process every out_*.csv in the chamber data dir into data/prepared/.",
    )
    p.add_argument(
        "--y-source",
        choices=("potential", "observed"),
        default="potential",
        help="'potential' (default): Y = Y1 if A else Y0, exact consistency. "
        "'observed': keep the measured outcome column.",
    )
    for key in ("u", "x", "w", "z"):
        p.add_argument(f"--{key}", help=f"Comma-separated {key.upper()} column(s).")
    for key in ("a", "y", "y0", "y1"):
        p.add_argument(f"--{key}", help=f"{key.upper()} column name.")
    args = p.parse_args()

    mapping = _resolve_mapping(args)

    if args.all:
        jobs = []
        out_dir = DATA_DIR / "prepared"
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(DATA_DIR.glob("out_*.csv")):
            stem = src.stem.replace("out_binaryA_counterfactuals_", "")
            jobs.append((src, out_dir / f"{stem}.csv"))
        if not jobs:
            raise SystemExit(f"No out_*.csv files found in {DATA_DIR}")
    else:
        if not args.input or not args.output:
            raise SystemExit("Provide --input and --output, or use --all.")
        jobs = [(Path(args.input), Path(args.output))]

    for src, dst in jobs:
        df = read_raw_csv(src)
        out, diag = prepare(df, mapping, y_source=args.y_source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dst, index=False)
        diag["source_file"] = str(src)
        diag["columns"] = list(out.columns)
        with open(dst.with_suffix(".diagnostics.json"), "w") as f:
            json.dump(diag, f, indent=2)

        ratio = diag["sd_tau_over_ate"]
        flag = (
            "" if ratio is None or ratio > 0.05 else "   <- CATE metrics NOT meaningful"
        )
        print(
            f"{dst.name:34s} n={diag['n_rows']:5d} "
            f"ATE={diag['ate_true']:9.1f} SD(tau)/ATE={ratio:7.4f} "
            f"naive bias={diag['naive_bias_pct_of_ate']:+7.2f}%{flag}"
        )


if __name__ == "__main__":
    main()
