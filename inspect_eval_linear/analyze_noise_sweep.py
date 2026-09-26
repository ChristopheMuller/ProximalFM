"""Summarize and plot adjustment error across proxy informativeness levels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .sidecar import load_linear_meta

DEFAULT_ROOT = Path("data/inspect_eval_linear/proxy_sweep")
LADDER = {
    "Q-X-naive": "Q_naive_X",
    "Q-W-adj": "Q_adj_W",
    "Q-Z-adj": "Q_adj_Z",
    "Q-WZ-adj": "Q_adj_WZ",
    "Q-true": "Q_true",
}
DISPLAY_LABELS = {"P2SLS": "P2SLS (ATE)"}


def _benchmark_prediction_files(root: Path) -> dict[str, Path]:
    """Return fitted-method caches, excluding the exported Q references."""
    pred_dir = root / "bench" / "preds"
    files = {}
    for path in sorted(pred_dir.glob("*_preds.npz")):
        method = path.name.removesuffix("_preds.npz")
        if method not in LADDER:
            files[method] = path
    return files


def summarize(root: Path) -> list[dict[str, float | int | str]]:
    meta = load_linear_meta(root)
    if meta["global"].get("set_kind") != "inspect_linear":
        raise ValueError(f"{root} is not an inspect_eval_linear set")
    grouped: dict[float, list[dict]] = {}
    for record in meta["episodes"]:
        grouped.setdefault(float(record["proxy_noise_std"]), []).append(record)

    rows = []
    for noise in sorted(grouped):
        records = grouped[noise]
        truth = np.concatenate([r["Q_true"].numpy() for r in records])
        reliability = float(records[0]["proxy_reliability"])
        for label, field in LADDER.items():
            values = np.concatenate([r[field].numpy() for r in records])
            error = values - truth
            rows.append(
                {
                    "proxy_noise_std": noise,
                    "proxy_reliability": reliability,
                    "method": label,
                    "source": "population",
                    "bias": float(error.mean()),
                    "absolute_bias": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "n_episodes": len(records),
                    "n_query_values": len(error),
                }
            )
    prediction_files = _benchmark_prediction_files(root)
    if not prediction_files:
        return rows

    n_episodes = len(meta["episodes"])
    n_query = int(meta["global"]["n_q"])
    episode_noise = np.array(
        [float(record["proxy_noise_std"]) for record in meta["episodes"]]
    )
    truth = np.stack([record["Q_true"].numpy() for record in meta["episodes"]])
    y_std = np.array([float(record["y_context_std"]) for record in meta["episodes"]])[
        :, None
    ]

    for method, path in prediction_files.items():
        with np.load(path) as cached:
            predictions_std = np.asarray(cached["preds"], dtype=np.float64)
        expected = (n_episodes, n_query)
        if predictions_std.shape != expected:
            raise ValueError(
                f"benchmark cache {path} has shape {predictions_std.shape}, "
                f"expected {expected}"
            )
        if not np.isfinite(predictions_std).all():
            raise ValueError(f"benchmark cache {path} contains non-finite predictions")
        # Benchmark methods predict treatment effects in context-standardized
        # outcome units. A treatment contrast has no mean shift, so multiplying
        # by the frozen context y standard deviation returns raw outcome units.
        predictions = predictions_std * y_std
        for noise in sorted(grouped):
            selected = episode_noise == noise
            error = predictions[selected].reshape(-1) - truth[selected].reshape(-1)
            rows.append(
                {
                    "proxy_noise_std": noise,
                    "proxy_reliability": float(grouped[noise][0]["proxy_reliability"]),
                    "method": method,
                    "source": "benchmark",
                    "bias": float(error.mean()),
                    "absolute_bias": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "n_episodes": int(selected.sum()),
                    "n_query_values": len(error),
                }
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    colors = {
        "Q-X-naive": "#7f7f7f",
        "Q-W-adj": "#4c78a8",
        "Q-Z-adj": "#f58518",
        "Q-WZ-adj": "#d62728",
        "Q-true": "#2ca02c",
    }
    methods = list(LADDER) + sorted(
        {str(row["method"]) for row in rows if row["method"] not in LADDER}
    )
    benchmark_colors = plt.get_cmap("tab10")
    for benchmark_index, method in enumerate(methods):
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["proxy_reliability"],
        )
        x = [row["proxy_reliability"] for row in selected]
        is_benchmark = selected[0]["source"] == "benchmark"
        color = colors.get(method, benchmark_colors(benchmark_index % 10))
        if method == "Q-WZ-adj":
            style = {"linewidth": 3.0, "zorder": 5, "linestyle": "-"}
        elif is_benchmark:
            style = {"linewidth": 2.0, "linestyle": "--", "alpha": 0.9}
        else:
            style = {"linewidth": 1.8, "linestyle": "-"}
        axes[0].plot(
            x,
            [row["rmse"] for row in selected],
            marker="o",
            label=DISPLAY_LABELS.get(method, method),
            color=color,
            **style,
        )
        axes[1].plot(
            x,
            [row["bias"] for row in selected],
            marker="o",
            label=DISPLAY_LABELS.get(method, method),
            color=color,
            **style,
        )
    axes[0].set_ylabel("RMSE against CATE*")
    axes[1].set_ylabel("Mean bias against CATE*")
    axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    for axis in axes:
        axis.set_xlabel(r"Proxy reliability $R^2_{U,proxy\mid X}$")
        axis.set_xlim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    title = (
        "Population adjustment and fitted estimators vs proxy information"
        if any(row["source"] == "benchmark" for row in rows)
        else "Ordinary adjustment as W and Z become more informative"
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(
    root: Path, *, csv_path: Path | None = None, plot_path: Path | None = None
) -> tuple[Path, Path]:
    root = Path(root)
    out = root / "analysis"
    csv_path = Path(csv_path) if csv_path else out / "proxy_noise_summary.csv"
    plot_path = Path(plot_path) if plot_path else out / "proxy_noise_sweep.png"
    rows = summarize(root)
    if not rows:
        raise ValueError("sidecar contains no episodes")
    write_csv(rows, csv_path)
    write_plot(rows, plot_path)
    return csv_path, plot_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args(argv)
    csv_path, plot_path = analyze(args.root, csv_path=args.csv, plot_path=args.plot)
    print(f"summary -> {csv_path}")
    print(f"plot    -> {plot_path}")


if __name__ == "__main__":
    main()
