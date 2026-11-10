"""Plot binned rank-test rejections as the number of rows per arm increases."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.ticker import NullLocator, PercentFormatter

from analysis.analysis_prior.generate import generate
from analysis.analysis_prior.plotting import plt
from rebuttal.completeness_rank import (
    ALPHA,
    PILOT_FRACTION,
    quantile_cells,
    rank_test,
    scalar_prior_spec,
)

OUTPUT = Path(__file__).with_name("results") / "completeness_rank_sweep.png"


def tally_scm(
    u: np.ndarray,
    z: np.ndarray,
    treatment: np.ndarray,
    k: int,
    results: dict[int, Counter[str]],
) -> None:
    split = int(PILOT_FRACTION * len(treatment))
    for arm in (0, 1):
        pilot = treatment[:split] == arm
        evaluation = treatment[split:] == arm
        if not pilot.any():
            for tally in results.values():
                tally["short_arm"] += 1
            continue

        u_eval = u[split:][evaluation][: max(results)]
        z_eval = z[split:][evaluation][: max(results)]
        cells = quantile_cells(u[:split][pilot], z[:split][pilot], u_eval, z_eval, k)
        for n, tally in results.items():
            if n > len(cells):
                tally["short_arm"] += 1
                continue
            counts = np.bincount(cells[:n], minlength=k**2).reshape(k, k)
            if not counts.sum(axis=1).all() or not counts.sum(axis=0).all():
                tally["empty_bin"] += 1
                continue
            sigma_min, cutoff = rank_test(counts, ALPHA)
            tally["rejected" if sigma_min > cutoff else "unresolved"] += 1


def save_plot(results: dict[int, Counter[str]], n_scm: int, k: int, path: Path) -> None:
    total = 2 * n_scm
    sizes = list(results)
    rates = [results[n]["rejected"] / total for n in sizes]
    fig, ax = plt.subplots(figsize=(6.2, 3.7), layout="constrained")
    ax.set_title(
        "Rank-test rejections by sample size", loc="left", weight="bold", pad=23
    )
    ax.text(
        0,
        1.026,
        f"{n_scm} prior SCMs · {k} bins per variable · nominal 5% test",
        transform=ax.transAxes,
        fontsize=10,
        color="#4B5563",
    )
    ax.plot(sizes, rates, marker="o", color="#126C83", linewidth=2)
    for n, rate in zip(sizes, rates, strict=True):
        tally = results[n]
        skipped = tally["short_arm"] + tally["empty_bin"]
        label = f"{tally['rejected']}/{total}"
        if skipped:
            label += f"\n{skipped} skipped"
        ax.annotate(
            label,
            (n, rate),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#374151",
        )
    ax.set_xscale("log")
    ax.margins(x=0.12)
    ax.set_xticks(
        sizes, [f"{n:,}" for n in sizes], rotation=35 if len(sizes) > 6 else 0
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-0.04, 1.15)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel("Evaluation rows per treatment arm")
    ax.set_ylabel("Share rejecting rank deficiency")
    ax.grid(axis="y", color="#E2E8F0")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-scm", type=int, required=True, help="Distinct prior SCMs")
    parser.add_argument(
        "--k", type=int, required=True, help="Quantile bins per variable"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        nargs="+",
        required=True,
        help="Evaluation sample sizes per treatment arm",
    )
    parser.add_argument(
        "--rows-per-scm",
        type=int,
        help="Total generated rows per SCM before splitting into arms",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Figure path")
    args = parser.parse_args()
    if args.n_scm < 1 or args.k < 2 or min(args.n_samples) < 1:
        parser.error("Require at least one SCM, two bins, and positive sample sizes")

    sizes = sorted(set(args.n_samples))
    rows = args.rows_per_scm
    if rows is None:
        rows = max(10_000, 16 * sizes[-1])
    if rows < 5:
        parser.error("Require at least five generated rows per SCM")

    spec = scalar_prior_spec(args.n_scm, rows)
    results: dict[int, Counter[str]] = {n: Counter() for n in sizes}
    for index in range(args.n_scm):
        print(f"Generating SCM {index + 1}/{args.n_scm} ({rows:,} rows)", flush=True)
        record = generate(spec, indices=[index], progress=False)[0]
        u = record.U[:, 0] if record.U.shape[1] else np.zeros(record.n_rows)
        tally_scm(u, record.Z[:, 0], record.A, args.k, results)

    total = 2 * args.n_scm
    print("\nRows per arm    Rejected    Tested    Short arms    Empty bins")
    for n, tally in results.items():
        tested = tally["rejected"] + tally["unresolved"]
        print(
            f"{n:>12,}    {tally['rejected']:>4}/{total:<5} {tested:>6}"
            f" {tally['short_arm']:>13} {tally['empty_bin']:>13}"
        )
    save_plot(results, args.n_scm, args.k, args.output)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
