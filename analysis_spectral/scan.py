"""Per-episode ill-posedness table for a whole ``val/`` set.

    uv run python -m analysis_spectral.scan --val_dir data/prior_eval/prior_val_v2_small/val

Writes one row per episode to ``analysis_spectral/results/<name>.csv``, keyed by
``index`` -- the episode's position in the set, which is the same ordering the
benchmark harness produces its per-episode scores in, so the two frames join on
that column and nothing else is needed to correlate difficulty with error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from analysis_spectral._load import iter_episodes
from analysis_spectral.decay import episode_spectra, episode_summary

RESULTS = Path(__file__).parent / "results"


def scan(
    val_dir: str | Path,
    *,
    n_bins: int = 8,
    x_bins: int = 1,
    n_perm: int = 200,
    quantile: float = 0.95,
    limit: int | None = None,
    min_points: int = 3,
) -> pd.DataFrame:
    rows = []
    for ep in tqdm(iter_episodes(val_dir, limit=limit), desc="episodes"):
        cells = episode_spectra(
            ep,
            n_bins=n_bins,
            x_bins=x_bins,
            n_perm=n_perm,
            quantile=quantile,
            seed=ep.index,
        )
        summary = episode_summary(cells, min_points=min_points)
        if not summary:
            continue
        summary["index"] = ep.index
        summary["p_treated"] = float(ep.A.mean())
        rows.append(summary)
    df = pd.DataFrame(rows).set_index("index")
    df.attrs["n_bins"] = n_bins
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val_dir", default="data/prior_eval/prior_val_v2_small/val")
    p.add_argument("--name", default=None, help="output stem (default: set name)")
    p.add_argument(
        "--n_bins",
        type=int,
        nargs="+",
        default=[8],
        help=(
            "proxy categories; several values run the whole sweep and write one "
            "CSV per resolution. L is a choice, not a property of the data, so a "
            "single value is only ever readable next to its neighbours."
        ),
    )
    p.add_argument("--x_bins", type=int, default=1)
    p.add_argument("--n_perm", type=int, default=200)
    p.add_argument("--quantile", type=float, default=0.95)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    RESULTS.mkdir(exist_ok=True)
    stem = args.name or Path(args.val_dir).parent.name
    frames = {}

    for n_bins in args.n_bins:
        df = scan(
            args.val_dir,
            n_bins=n_bins,
            x_bins=args.x_bins,
            n_perm=args.n_perm,
            quantile=args.quantile,
            limit=args.limit,
        )
        out = RESULTS / f"{stem}_L{n_bins}.csv"
        df.to_csv(out)
        frames[n_bins] = df
        print(f"{len(df)} episodes -> {out}")

    print(
        f"\n{'L':>4} {'meas':>6} {'sigma2 med':>11} {'eff_rank med':>13} "
        f"{'poly_rate med':>14} {'zero-signal':>12}"
    )
    for n_bins, df in frames.items():
        m = df[df.measurable]
        print(
            f"{n_bins:>4} {df.measurable.mean():>6.2f} {df.sigma2.median():>11.3f} "
            f"{df.eff_rank.median():>13.2f} {m.poly_rate.median():>14.3f} "
            f"{(df.eff_rank == 0).mean():>12.2f}"
        )

    if len(frames) > 1:
        # The absolute values move with L by construction -- a bigger table has
        # a bigger noise floor. What has to survive is the ORDERING of episodes,
        # since that is all the join against method error ever uses.
        from scipy.stats import spearmanr

        print("\nrank stability across L (Spearman on the same episodes)")
        ls = list(frames)
        print(f"{'pair':>14} {'sigma2':>8} {'eff_rank':>9}")
        for i, a in enumerate(ls):
            for b in ls[i + 1 :]:
                j = frames[a].index.intersection(frames[b].index)

                def r(c, a=a, b=b, j=j):
                    return spearmanr(frames[a].loc[j, c], frames[b].loc[j, c]).statistic

                print(
                    f"{f'L={a} vs {b}':>14} {r('sigma2'):>8.2f} {r('eff_rank'):>9.2f}"
                )


if __name__ == "__main__":
    main()
