"""Generate exhaustive semisynthetic CATE-nPEHE appendix tables.

The three replicates of each source dataset are averaged first. Table entries
then macro-average those source-level means, so each of the twelve source
datasets receives equal weight. Statistical comparisons are likewise paired
at the source-dataset level rather than treating the 36 episodes as
independent.

Run from the repository root::

    uv run python -m analysis_eval.exhaustive_tables
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .design import method_style, ordered_methods
from .semisynt import CELL_ORDER, discover_cells

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX_ROOT = REPO_ROOT / "data" / "semisynt_eval" / "semisynt_val_v1"
DEFAULT_OUT = REPO_ROOT / "analysis_eval" / "results" / "semisynt_exhaustive_npehe.tex"
DEFAULT_CONTEXTS = (128, 512, 1024, 2048)
DEFAULT_EXTENDED_CASE = "nonlinear_conf_high_proxy_low"
DEFAULT_EXTENDED_CONTEXTS = (128, 256, 512, 1024, 2048, 4096, 8192)
DEFAULT_ALPHA = 0.05
DEFAULT_REPLICATES = 3
# ProximalFM-FT belongs only to the targeted large-context experiment, not to
# the exhaustive factorial tables at the four shared context sizes.
EXCLUDED_METHODS = frozenset({"ProximalFM-FT"})

RESULT_COLUMNS = {
    "method",
    "episode",
    "n_ctx",
    "target",
    "estimand",
    "is_reference",
    "failed",
    "npehe",
}


def _episode_identity(cell) -> pd.DataFrame:
    """Return benchmark episode numbers joined to source and replicate IDs."""
    manifest_path = cell.results_path.parents[2] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("episodes")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{manifest_path}: 'episodes' must be a non-empty list")

    rows = []
    for episode, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{manifest_path}: episode {episode} is not an object")
        source = record.get("source_file")
        replicate = record.get("replicate")
        if not isinstance(source, str) or not source:
            raise ValueError(f"{manifest_path}: episode {episode} has no source_file")
        if isinstance(replicate, bool) or not isinstance(replicate, int):
            raise ValueError(f"{manifest_path}: episode {episode} has no replicate")
        rows.append({"episode": episode, "source_file": source, "replicate": replicate})
    identity = pd.DataFrame(rows)
    if identity.duplicated(["source_file", "replicate"]).any():
        raise ValueError(f"{manifest_path}: duplicate source/replicate identity")
    return identity


def load_results(
    matrix_root: Path,
    *,
    dimension: int | None = None,
    selected_case: str | None = None,
) -> pd.DataFrame:
    """Load CATE(X) rows and attach their hierarchical episode identities."""
    frames = []
    for cell in discover_cells(
        Path(matrix_root), dimension=dimension, selected_case=selected_case
    ):
        results_path = cell.results_path
        frame = pd.read_csv(results_path)
        missing = RESULT_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{results_path}: missing columns {sorted(missing)}")
        frame = frame[
            frame["target"].eq("cate")
            & frame["estimand"].eq("cate(x)")
            & ~frame["is_reference"].astype(bool)
        ].copy()
        frame["npehe"] = pd.to_numeric(frame["npehe"], errors="coerce")
        frame.loc[frame["failed"].astype(bool), "npehe"] = np.nan
        frame = frame.merge(
            _episode_identity(cell), on="episode", how="left", validate="many_to_one"
        )
        if frame[["source_file", "replicate"]].isna().any().any():
            unknown = sorted(set(frame.loc[frame["source_file"].isna(), "episode"]))
            raise ValueError(
                f"{results_path}: result rows have unknown episodes {unknown}"
            )
        frame["case"] = cell.case
        frame["family"] = cell.family
        frame["confounding"] = cell.confounding
        frame["proxy"] = cell.proxy
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    keys = ["case", "method", "episode", "n_ctx"]
    duplicates = result.duplicated(keys, keep=False)
    if duplicates.any():
        sample = result.loc[duplicates, keys].head().to_dict("records")
        raise ValueError(f"duplicate benchmark rows, e.g. {sample}")
    return result


def aggregate_replicates(
    frame: pd.DataFrame, *, expected_replicates: int = DEFAULT_REPLICATES
) -> pd.DataFrame:
    """Average complete replicate triples within each source dataset."""
    keys = [
        "case",
        "family",
        "confounding",
        "proxy",
        "method",
        "n_ctx",
        "source_file",
    ]

    def complete_mean(group: pd.DataFrame) -> pd.Series:
        scores = group["npehe"].to_numpy(dtype=float)
        replicates = group["replicate"].to_numpy(dtype=int)
        complete = (
            len(group) == expected_replicates
            and len(np.unique(replicates)) == expected_replicates
            and np.isfinite(scores).all()
        )
        return pd.Series(
            {
                "source_mean": float(scores.mean()) if complete else np.nan,
                "n_replicates": int(np.isfinite(scores).sum()),
                "complete": bool(complete),
            }
        )

    return (
        frame.groupby(keys, observed=True, sort=False)
        .apply(complete_mean)
        .reset_index()
    )


@lru_cache(maxsize=None)
def _sign_matrix(n: int) -> np.ndarray:
    return np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)))


def exact_sign_flip_pvalue(differences: Iterable[float]) -> float:
    """One-sided exact paired randomization p-value for a positive mean."""
    values = np.asarray(list(differences), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("differences must be a non-empty finite vector")
    observed = float(values.mean())
    permuted = _sign_matrix(values.size) @ values / values.size
    tolerance = np.finfo(float).eps * max(1.0, abs(observed)) * 16
    return float(np.mean(permuted >= observed - tolerance))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Return Holm family-wise-error adjusted p-values."""
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (name, pvalue) in enumerate(ordered):
        running = max(running, (m - rank) * pvalue)
        adjusted[name] = min(1.0, running)
    return adjusted


def summarize(
    source_means: pd.DataFrame,
    *,
    contexts: Sequence[int] = DEFAULT_CONTEXTS,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Build table cells and paired comparisons for each case/context."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    contexts = tuple(int(value) for value in contexts)
    available = set(source_means["n_ctx"].unique())
    absent = set(contexts) - available
    if absent:
        raise ValueError(f"requested context sizes are absent: {sorted(absent)}")

    methods = ordered_methods(source_means["method"].unique())
    sources_by_case = {
        case: sorted(group["source_file"].unique())
        for case, group in source_means.groupby("case", observed=True)
    }
    rows = []
    for case, n_ctx in itertools.product(
        [
            f"{family}_conf_{confounding}_proxy_{proxy}"
            for family, confounding, proxy in CELL_ORDER
        ],
        contexts,
    ):
        cell = source_means[
            source_means["case"].eq(case) & source_means["n_ctx"].eq(n_ctx)
        ]
        expected_sources = sources_by_case[case]
        complete: dict[str, pd.Series] = {}
        for method in methods:
            values = (
                cell[cell["method"].eq(method)]
                .set_index("source_file")["source_mean"]
                .reindex(expected_sources)
            )
            if len(values) == len(expected_sources) and values.notna().all():
                complete[method] = values

        if not complete:
            raise ValueError(f"{case}, n_ctx={n_ctx}: no complete methods")
        best = min(complete, key=lambda name: (complete[name].mean(), name))
        raw = {
            method: exact_sign_flip_pvalue(complete[method] - complete[best])
            for method in complete
            if method != best
        }
        adjusted = holm_adjust(raw)

        factors = cell[["family", "confounding", "proxy"]].drop_duplicates()
        if len(factors) != 1:
            raise ValueError(f"{case}, n_ctx={n_ctx}: inconsistent factorial labels")
        family, confounding, proxy = next(factors.itertuples(index=False, name=None))
        for method in methods:
            values = complete.get(method)
            rows.append(
                {
                    "case": case,
                    "family": family,
                    "confounding": confounding,
                    "proxy": proxy,
                    "n_ctx": n_ctx,
                    "method": method,
                    "mean_npehe": float(values.mean())
                    if values is not None
                    else np.nan,
                    "n_sources": len(values) if values is not None else 0,
                    "best": method == best,
                    "not_significantly_worse": (
                        method == best or adjusted.get(method, 0.0) >= alpha
                    )
                    if values is not None
                    else False,
                    "pvalue_vs_best": raw.get(method, np.nan),
                    "pvalue_holm": adjusted.get(method, np.nan),
                }
            )
    return pd.DataFrame(rows)


def summarize_extended_case(
    source_means: pd.DataFrame,
    *,
    case: str = DEFAULT_EXTENDED_CASE,
    expected_contexts: Sequence[int] | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Summarize every available context size for one extended-run case."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    cell_case = source_means[source_means["case"].eq(case)]
    if cell_case.empty:
        available = sorted(str(value) for value in source_means["case"].unique())
        raise ValueError(
            f"unknown extended case {case!r}; available cases are: "
            + ", ".join(available)
        )

    # Begin from the full factorial method union. After constructing all
    # context columns, methods never evaluated in the extended run are removed;
    # methods with partial coverage retain honest dashes for missing contexts.
    methods = ordered_methods(source_means["method"].unique())
    expected_sources = sorted(cell_case["source_file"].unique())
    contexts = sorted(int(value) for value in cell_case["n_ctx"].unique())
    if expected_contexts is not None:
        expected = {int(value) for value in expected_contexts}
        absent = expected - set(contexts)
        if absent:
            raise ValueError(
                f"{case}: extended context sizes are absent: {sorted(absent)}. "
                "Run scripts/run_semisynt_selection_benchmarks.sh before "
                "generating the selection table."
            )
        contexts = sorted(expected)
    rows = []
    for n_ctx in contexts:
        cell = cell_case[cell_case["n_ctx"].eq(n_ctx)]
        complete: dict[str, pd.Series] = {}
        for method in methods:
            values = (
                cell[cell["method"].eq(method)]
                .set_index("source_file")["source_mean"]
                .reindex(expected_sources)
            )
            if len(values) == len(expected_sources) and values.notna().all():
                complete[method] = values

        if not complete:
            raise ValueError(f"{case}, n_ctx={n_ctx}: no complete methods")
        best = min(complete, key=lambda name: (complete[name].mean(), name))
        raw = {
            method: exact_sign_flip_pvalue(complete[method] - complete[best])
            for method in complete
            if method != best
        }
        adjusted = holm_adjust(raw)

        factors = cell[["family", "confounding", "proxy"]].drop_duplicates()
        if len(factors) != 1:
            raise ValueError(f"{case}, n_ctx={n_ctx}: inconsistent factorial labels")
        family, confounding, proxy = next(factors.itertuples(index=False, name=None))
        for method in methods:
            values = complete.get(method)
            rows.append(
                {
                    "case": case,
                    "family": family,
                    "confounding": confounding,
                    "proxy": proxy,
                    "n_ctx": n_ctx,
                    "method": method,
                    "mean_npehe": float(values.mean())
                    if values is not None
                    else np.nan,
                    "n_sources": len(values) if values is not None else 0,
                    "best": method == best,
                    "not_significantly_worse": (
                        method == best or adjusted.get(method, 0.0) >= alpha
                    )
                    if values is not None
                    else False,
                    "pvalue_vs_best": raw.get(method, np.nan),
                    "pvalue_holm": adjusted.get(method, np.nan),
                }
            )
    result = pd.DataFrame(rows)
    ran_methods = set(result.loc[result["mean_npehe"].notna(), "method"])
    return result[result["method"].isin(ran_methods)].reset_index(drop=True)


def _tex(text: str) -> str:
    specials = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    return "".join(specials.get(char, char) for char in str(text))


def _number(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value >= 1000:
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"${mantissa}\!\times\!10^{{{int(exponent)}}}$"
    return f"{value:.3f}"


def _method_family(method: str) -> str:
    if method.startswith("S-"):
        return "S-learners"
    if method.startswith("T-"):
        return "T-learners"
    if method.startswith("X-"):
        return "X-learners"
    if method in {
        "CausalFM",
        "DoPFN",
        "CausalPFN",
        "CausalPFN-WZ-kNN-16",
        "CausalPFN-WZ-Kernel-16",
    }:
        return "Causal ICL"
    if method.startswith("P-Kernel-"):
        return "P-learners"
    if method.startswith(("KPV-", "PMMR-")):
        return "KPV and PMMR"
    if method.startswith("NMMR-"):
        return "NMMR"
    if method.startswith("ProximalFM"):
        return "ProximalFM"
    return "Other"


def _table_methods(methods: Iterable[str]) -> list[str]:
    """Group methods by the appendix families while preserving registry order."""
    canonical = ordered_methods(methods)
    family_order = {
        family: index
        for index, family in enumerate(
            (
                "S-learners",
                "T-learners",
                "X-learners",
                "Causal ICL",
                "P-learners",
                "KPV and PMMR",
                "NMMR",
                "ProximalFM",
                "Other",
            )
        )
    }
    registry_order = {method: index for index, method in enumerate(canonical)}
    return sorted(
        canonical,
        key=lambda method: (
            family_order[_method_family(method)],
            registry_order[method],
        ),
    )


def render_table(summary: pd.DataFrame, n_ctx: int, *, alpha: float) -> str:
    """Render one context-size table as a two-column LaTeX float."""
    table = summary[summary["n_ctx"].eq(n_ctx)]
    case_order = [
        f"{family}_conf_{confounding}_proxy_{proxy}"
        for family, confounding, proxy in CELL_ORDER
    ]
    methods = _table_methods(table["method"].unique())
    lookup = table.set_index(["method", "case"])
    lines = [
        "% Generated by `python -m analysis_eval.exhaustive_tables`; do not edit.",
        "\\begin{table*}[p]",
        "    \\centering",
        "    \\scriptsize",
        "    \\setlength{\\tabcolsep}{3.2pt}",
        "    \\renewcommand{\\arraystretch}{0.92}",
        "    \\caption{Exhaustive semisynthetic CATE nPEHE results at "
        + rf"$n_{{\mathrm{{ctx}}}}={n_ctx:,}$. "
        + "Each entry first averages the three replicates of each source dataset, "
        + "then averages equally over the 12 sources; lower is better. The lowest "
        + "mean in each column is bold. Other underlined methods are not "
        + "significantly worse than that empirical best under one-sided exact "
        + "paired sign-flip tests across source-level means, with Holm correction "
        + rf"within the column ($\alpha={alpha:.2f}$).}}",
        rf"    \label{{tab:semisynt_exhaustive_npehe_{n_ctx}}}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{@{}lrrrrrrrr@{}}",
        "        \\toprule",
        "        & \\multicolumn{4}{c}{Linear} & \\multicolumn{4}{c}{Nonlinear} \\\\",
        "        \\cmidrule(lr){2-5} \\cmidrule(l){6-9}",
        "        & \\multicolumn{2}{c}{Low confounding} & \\multicolumn{2}{c}{High confounding}"
        " & \\multicolumn{2}{c}{Low confounding} & \\multicolumn{2}{c}{High confounding} \\\\",
        "        \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(l){8-9}",
        "        Method & Low proxy & High proxy & Low proxy & High proxy"
        " & Low proxy & High proxy & Low proxy & High proxy \\\\",
        "        \\midrule",
    ]
    previous_family = None
    for method in methods:
        family = _method_family(method)
        if previous_family is not None and family != previous_family:
            lines.append("        \\addlinespace[2pt]")
        previous_family = family
        cells = []
        for case in case_order:
            row = lookup.loc[(method, case)]
            value = _number(float(row["mean_npehe"]))
            if bool(row["best"]):
                value = rf"\textbf{{{value}}}"
            elif bool(row["not_significantly_worse"]):
                value = rf"\underline{{{value}}}"
            cells.append(value)
        lines.append(
            "        "
            + _tex(method_style(method).label)
            + " & "
            + " & ".join(cells)
            + " \\\\"
        )
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}%",
        "    }",
        "\\end{table*}",
        "",
    ]
    return "\n".join(lines)


def render_tables(
    summary: pd.DataFrame,
    *,
    contexts: Sequence[int] = DEFAULT_CONTEXTS,
    alpha: float = DEFAULT_ALPHA,
) -> str:
    return "\n".join(render_table(summary, n_ctx, alpha=alpha) for n_ctx in contexts)


def render_extended_table(
    summary: pd.DataFrame, *, alpha: float, replicates: int
) -> str:
    """Render one case with context sizes as columns."""
    cases = summary["case"].drop_duplicates().tolist()
    if len(cases) != 1:
        raise ValueError("extended table summary must contain exactly one case")
    case = cases[0]
    contexts = sorted(int(value) for value in summary["n_ctx"].unique())
    methods = _table_methods(summary["method"].unique())
    lookup = summary.set_index(["method", "n_ctx"])
    n_sources = int(summary["n_sources"].max())
    column_spec = "l" + "r" * len(contexts)
    header = " & ".join(rf"${value:,}$" for value in contexts)
    lines = [
        "% Generated by `python -m analysis_eval.selection_tables`; do not edit.",
        "\\begin{table*}[p]",
        "    \\centering",
        "    \\scriptsize",
        "    \\setlength{\\tabcolsep}{4.0pt}",
        "    \\renewcommand{\\arraystretch}{0.92}",
        "    \\caption{Extended-context semisynthetic CATE nPEHE results for "
        "nonlinear mechanisms with high confounding and low proxy reliability. "
        + f"Each entry first averages the {replicates} replicates of each source "
        + f"dataset, then averages equally over the {n_sources} sources; lower "
        + "is better. The lowest "
        "mean in each column is bold. Other underlined methods are not "
        "significantly worse than that empirical best under one-sided exact "
        "paired sign-flip tests across source-level means, with Holm correction "
        + rf"within the column ($\alpha={alpha:.2f}$).}}",
        "    \\label{tab:semisynt_extended_context_npehe}",
        "    \\resizebox{\\textwidth}{!}{%",
        f"    \\begin{{tabular}}{{@{{}}{column_spec}@{{}}}}",
        "        \\toprule",
        rf"        Method & \multicolumn{{{len(contexts)}}}{{c}}{{Context size $n_{{\mathrm{{ctx}}}}$}}"
        + r" \\",
        f"        \\cmidrule(l){{2-{len(contexts) + 1}}}",
        "        & " + header + " \\\\",
        "        \\midrule",
    ]
    previous_family = None
    for method in methods:
        family = _method_family(method)
        if previous_family is not None and family != previous_family:
            lines.append("        \\addlinespace[2pt]")
        previous_family = family
        cells = []
        for n_ctx in contexts:
            row = lookup.loc[(method, n_ctx)]
            value = _number(float(row["mean_npehe"]))
            if bool(row["best"]):
                value = rf"\textbf{{{value}}}"
            elif bool(row["not_significantly_worse"]):
                value = rf"\underline{{{value}}}"
            cells.append(value)
        lines.append(
            "        "
            + _tex(method_style(method).label)
            + " & "
            + " & ".join(cells)
            + " \\\\"
        )
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}%",
        "    }",
        "\\end{table*}",
        "",
    ]
    return "\n".join(lines)


def write_summary_csv(summary: pd.DataFrame, path: Path) -> None:
    fields = list(summary.columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary.to_dict("records"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--contexts", type=int, nargs="+", default=DEFAULT_CONTEXTS)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print", dest="to_stdout", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out.suffix.lower() != ".tex":
        raise ValueError("--out must be a .tex path")
    shared_frame = load_results(args.matrix_root, dimension=args.dim)
    shared = aggregate_replicates(shared_frame, expected_replicates=args.replicates)
    shared = shared[~shared["method"].isin(EXCLUDED_METHODS)]
    summary = summarize(shared, contexts=args.contexts, alpha=args.alpha)
    latex = render_tables(summary, contexts=args.contexts, alpha=args.alpha)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(latex, encoding="utf-8")
    csv_path = args.out.with_suffix(".csv")
    write_summary_csv(summary, csv_path)
    print(f"wrote {args.out}")
    print(f"wrote {csv_path}")
    if args.to_stdout:
        print(latex, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
