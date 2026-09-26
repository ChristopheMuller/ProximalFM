"""Inspect the estimated matrices for a few episodes.

    uv run python -m analysis_spectral --val_dir data/prior_eval/prior_val_v2_small/val
"""

from __future__ import annotations

import argparse

import numpy as np

from analysis_spectral import conditional_matrices, load_episodes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val_dir", default="data/prior_eval/prior_val_v2_smalldim_50k/val")
    p.add_argument("--limit", type=int, default=8, help="episodes to load")
    p.add_argument("--n_bins", type=int, default=8, help="proxy categories")
    p.add_argument("--x_bins", type=int, default=1, help="bins per covariate dim")
    p.add_argument("--smoothing", type=float, default=0.0)
    p.add_argument("--show", type=int, default=6, help="singular values to print")
    args = p.parse_args()

    eps = load_episodes(args.val_dir, limit=args.limit)
    print(eps[0], "\n")
    for ep in eps:
        cells = conditional_matrices(
            ep, n_bins=args.n_bins, x_bins=args.x_bins, smoothing=args.smoothing
        )
        for cm in cells:
            s = cm.singular_values()[: args.show]
            print(
                f"ep {ep.index:>4}  a={cm.a} x={cm.x_cell}  n={cm.n:>5}  "
                f"{cm.shape[0]}x{cm.shape[1]}  "
                f"sv={np.array2string(s, precision=3, floatmode='fixed')}  "
                f"chi2/n={cm.chi2_over_n():.4f} (null {cm.independence_floor():.4f})"
            )


if __name__ == "__main__":
    main()
