import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Method Styling Configuration
# ─────────────────────────────────────────────────────────────────────────────
# One entry per `name:` in config_*.yaml (active *and* commented, so styles
# survive when a method is re-enabled), plus the neural model name.
#
# Convention: colour = method family, marker = variant within the family.
#   OLS .......... Blue    #1f77b4      KPV .......... Cyan   #17becf
#   RF ........... Green   #2ca02c      PMMR ......... Brown  #8c564b
#   TabICL ....... Purple  #9467bd      NMMR ......... Pink   #e377c2
#   P2SLS ........ Orange  #ff7f0e      DFPV ......... Olive  #bcbd22
#   Neural causal models (CausalFM / DoPFN / CausalPFN / ProximalFM) . Red #d62728
#
# Proximal strategy markers: paired/XW = "o", independent/X-Wind = "s",
# conditional/X-cond = "^", marginal/ATE = "D".
METHOD_STYLES = {
    # ── OLS baselines (Blue) ──
    "S-OLS": {"color": "#1f77b4", "marker": "o"},  # circle
    "S-OLS+AX": {"color": "#1f77b4", "marker": "s"},  # square
    "T-OLS": {"color": "#1f77b4", "marker": "^"},  # triangle up
    # ── RF baselines (Green) ──
    "S-RF": {"color": "#2ca02c", "marker": "o"},
    "T-RF": {"color": "#2ca02c", "marker": "^"},
    "X-RF": {"color": "#2ca02c", "marker": "P"},  # plus (filled)
    "P-RF": {"color": "#2ca02c", "marker": "X"},  # cross (filled)
    # ── TabICL baselines (Purple) ── (Oracle variants use a star)
    "S-TabICL": {"color": "#9467bd", "marker": "o"},
    "S-TabICL-Oracle": {"color": "#9467bd", "marker": "*", "s": 180},
    "T-TabICL": {"color": "#9467bd", "marker": "^"},
    "T-TabICL-Oracle": {"color": "#9467bd", "marker": "X"},
    "X-TabICL": {"color": "#9467bd", "marker": "v"},  # triangle down
    "X-TabICL-Oracle": {"color": "#9467bd", "marker": "P"},
    # ── P2SLS proximal baselines (Orange) ──
    "P2SLS": {"color": "#ff7f0e", "marker": "o"},
    "P2SLS+AX": {"color": "#ff7f0e", "marker": "s"},
    # ── KPV (Cyan) ──
    "KPV-XW": {"color": "#17becf", "marker": "o"},
    "KPV-X-Wind": {"color": "#17becf", "marker": "s"},
    "KPV-X-Wcond": {"color": "#17becf", "marker": "^"},
    "KPV-ATE": {"color": "#17becf", "marker": "D"},
    # ── PMMR (Brown) ──
    "PMMR-XW": {"color": "#8c564b", "marker": "o"},
    "PMMR-X-Wind": {"color": "#8c564b", "marker": "s"},
    "PMMR-X-Wcond": {"color": "#8c564b", "marker": "^"},
    "PMMR-ATE": {"color": "#8c564b", "marker": "D"},
    # ── NMMR (Pinks: V-statistic light, U-statistic dark) ──
    "NMMR-V-XW": {"color": "#e377c2", "marker": "o"},
    "NMMR-V-Wind": {"color": "#e377c2", "marker": "s"},
    "NMMR-V-Wcond": {"color": "#e377c2", "marker": "^"},
    "NMMR-U-XW": {"color": "#c51b8a", "marker": "o"},
    "NMMR-U-Wind": {"color": "#c51b8a", "marker": "s"},
    "NMMR-U-Wcond": {"color": "#c51b8a", "marker": "^"},
    # ── DFPV (Olive) ──
    "DFPV-XW": {"color": "#bcbd22", "marker": "o"},
    "DFPV-X-Wind": {"color": "#bcbd22", "marker": "s"},
    "DFPV-X-cond": {"color": "#bcbd22", "marker": "^"},
    "DFPV-ATE": {"color": "#bcbd22", "marker": "D"},
    # ── Other forests ──
    "CausalForest": {"color": "#2ca02c", "marker": "D"},
    # ── Tabular & Neural Models (Red) ──
    "CausalFM": {"color": "#d62728", "marker": "X"},  # cross
    "DoPFN": {"color": "#d62728", "marker": "D"},  # diamond
    "CausalPFN": {"color": "#d62728", "marker": "P"},  # plus
    "ProximalFM3": {"color": "#d62728", "marker": "*", "s": 200},  # big star
    "ProximalFM4": {"color": "#d62728", "marker": "p", "s": 160},  # pentagon
    "ProximalFM": {"color": "#d62728", "marker": "*", "s": 200},
}


def style_for(method: str) -> dict:
    """Style lookup: exact name, else longest prefix match, else gray.

    Prefix matching covers checkpoint-suffixed model names (e.g.
    'ProximalFM3_55350' inherits the 'ProximalFM3' style).
    """
    if method in METHOD_STYLES:
        return METHOD_STYLES[method]
    prefixes = [k for k in METHOD_STYLES if method.startswith(k)]
    if prefixes:
        return METHOD_STYLES[max(prefixes, key=len)]
    return {"color": "gray", "marker": "o"}


def default_target(csv_path: str | Path) -> str | None:
    """Which scoring track the plots speak in unless told otherwise.

    ``results.csv`` holds one row per (method, episode, target). The CATE target
    is the primary one whenever the set carries it; pre-track result files have
    no ``target`` column at all, and those are read whole.
    """
    df = pd.read_csv(csv_path, nrows=200)
    if "target" not in df.columns:
        return None
    return "cate" if (pd.read_csv(csv_path)["target"] == "cate").any() else "ite"


def load_results(csv_path: str | Path, target: str | None) -> pd.DataFrame:
    """Read results.csv, keeping only the rows of one scoring track."""
    df = pd.read_csv(csv_path)
    if target is None or "target" not in df.columns:
        return df
    df = df[df["target"] == target]
    if df.empty:
        raise SystemExit(f"No rows with target={target!r} in {csv_path}.")
    return df


def _suffixed(path: Path, target: str | None) -> Path:
    """Per-track output filename, so the two tracks' plots never overwrite."""
    if target is None:
        return path
    return path.with_name(f"{path.stem}_{target}{path.suffix}")


def plot_ate_bias_vs_npehe(results_dir: Path):
    """
    Scatter comparing the (absolute) normalised ATE bias against the median
    nPEHE per method.  Ideal performance is at the origin (0, 0): no level bias
    and CATE error well below the constant-ATE baseline (nPEHE = 1).
    """
    summary_csv = results_dir / "summary.csv"

    if not summary_csv.exists():
        warnings.warn(f"summary.csv not found in {results_dir}. Skipping bias plot.")
        return

    df = pd.read_csv(summary_csv, index_col="method")
    df = df[["ATE Bias", "nPEHE"]].dropna()
    df["abs_bias"] = df["ATE Bias"].abs()

    # Group the legend by method family: follow METHOD_STYLES order (families
    # are contiguous there), with unstyled stragglers appended alphabetically.
    known = [k for k in METHOD_STYLES]
    df = df.sort_index(
        key=lambda idx: [
            (
                known.index(
                    max((k for k in known if m.startswith(k)), key=len, default="")
                )
                if any(m.startswith(k) for k in known)
                else len(known),
                m,
            )
            for m in idx
        ]
    )

    plt.figure(figsize=(7, 7))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    for row in df.itertuples():
        method = row.Index
        style = style_for(method)

        plt.scatter(
            row.abs_bias,
            row.nPEHE,
            color=style.get("color"),
            marker=style.get("marker"),
            s=style.get("s", 120),
            label=method,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )

    # nPEHE = 1 is the "predict the constant ATE" reference.
    plt.axhline(
        1.0,
        color="red",
        linestyle="--",
        alpha=0.5,
        zorder=0,
        label="nPEHE = 1 (constant-ATE baseline)",
    )
    plt.axvline(0, color="black", linestyle="-", alpha=0.4, zorder=0)

    plt.title(
        "ATE Bias vs CATE Error\n[Closer to Origin is Better]",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("|Normalised ATE bias|  (median, ÷ SD(τ))")
    plt.ylabel("nPEHE  (median)")

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        title="Methods",
        frameon=False,
        fontsize=9,
        ncol=2 if len(df) > 18 else 1,
    )

    output_file = results_dir / "plot_ate_bias_vs_npehe.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved successfully to: {output_file}")
    plt.close()


def plot_npehe_boxplot(csv_path: str, target: str | None = None):
    df = load_results(csv_path, target)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    df_valid = df.dropna(subset=["npehe"]).copy()
    if df_valid.empty:
        warnings.warn("No heterogeneous episodes. Skipping nPEHE boxplot.")
        return

    # Order methods by median nPEHE (best/lowest at the top).
    method_order = (
        df_valid.groupby("method")["npehe"].median().sort_values(ascending=True).index
    )

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=df_valid,
        x="npehe",
        y="method",
        order=method_order,
        palette="viridis",
    )

    ax.axvline(
        1.0,
        color="red",
        linestyle="--",
        alpha=0.6,
        label="nPEHE = 1 (constant-ATE baseline)",
    )
    ax.set_xlim(0, min(3.0, df_valid["npehe"].quantile(0.99)))

    plt.title("CATE Estimation: nPEHE Distribution", fontsize=16, fontweight="bold")
    plt.xlabel("nPEHE  (√(mean residual² / Var(τ)), lower is better)")
    plt.ylabel("")
    plt.legend(loc="lower right")

    output_file = _suffixed(Path(csv_path).parent / "plot_npehe_box.png", target)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved successfully to: {output_file}")
    plt.close()


def plot_cd_diagram(csv_path: str, target: str | None = None):
    """Critical-difference diagram of methods ranked by per-episode nPEHE.

    Runs a Friedman omnibus test + Nemenyi post-hoc (via autorank) over the
    per-episode nPEHE ranks.  Methods are placed on a rank axis (lower/left =
    better) and any group whose average ranks differ by less than the critical
    difference (CD) is joined by a horizontal bar — i.e. not significantly
    different at α = 0.05.

    The test requires a *complete block*: every method must have a valid nPEHE
    on every episode used.  Homogeneous episodes (NaN for all methods) and any
    episode where a method failed are therefore dropped to keep the comparison
    paired.
    """
    try:
        from autorank import autorank, plot_stats
    except ImportError:
        warnings.warn("autorank not installed; skipping CD diagram.")
        return

    df = load_results(csv_path, target)
    if "npehe" not in df.columns:
        warnings.warn("No 'npehe' column found; skipping CD diagram.")
        return
    # Oracles are anchors, not competitors: including them would shift every
    # method's average rank and inflate the omnibus test.
    if "is_reference" in df.columns:
        df = df[~df["is_reference"].astype(bool)]

    n_episodes = df["episode"].nunique()

    # Build an episode × method matrix of nPEHE (lower is better).
    pivot = df.pivot_table(index="episode", columns="method", values="npehe")
    pivot = pivot.dropna(axis=1, how="all")  # drop never-evaluated methods
    pivot = pivot.dropna(axis=0, how="any")  # keep only fully-paired episodes

    if pivot.shape[1] < 3:
        warnings.warn("Need >=3 methods with paired nPEHE for a CD diagram. Skipping.")
        return
    if pivot.shape[0] < 5:
        warnings.warn(
            f"Only {pivot.shape[0]} fully-paired episodes; CD diagram needs more "
            "for a meaningful Friedman test. Skipping."
        )
        return

    dropped = n_episodes - len(pivot)
    if dropped:
        print(
            f"CD diagram: {len(pivot)} complete episodes used "
            f"({dropped} dropped for homogeneity/failed methods)."
        )

    # order='ascending' → smaller nPEHE ranks better; force the nonparametric
    # path so we always get a Friedman+Nemenyi CD diagram.  autorank still runs
    # normality/variance tests for its report; silence their spurious warnings
    # (constant-nPEHE methods trigger zero-range Shapiro/Levene) since the
    # forced nonparametric path discards those results anyway.
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        result = autorank(
            pivot,
            alpha=0.05,
            order="ascending",
            force_mode="nonparametric",
            verbose=False,
        )

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    height = max(2.5, 0.4 * pivot.shape[1])
    fig, ax = plt.subplots(figsize=(11, height))
    plot_stats(result, allow_insignificant=True, ax=ax)

    fig.suptitle(
        "Critical-Difference Diagram — methods ranked by nPEHE (left = better)\n"
        f"Friedman + Nemenyi, α=0.05, CD={result.cd:.3f}, "
        f"{pivot.shape[0]} episodes",
        fontsize=13,
        fontweight="bold",
        y=1.05,
    )

    output_file = _suffixed(Path(csv_path).parent / "plot_cd_npehe.png", target)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved successfully to: {output_file}")
    plt.close()


def plot_true_vs_pred_ate(csv_path: str, target: str | None = None):
    df = load_results(csv_path, target)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    min_val = min(
        df["ate_true"].min(), max(df["ate_pred"].min(), 2 * df["ate_true"].min())
    )
    max_val = max(
        df["ate_true"].max(), min(df["ate_pred"].max(), 2 * df["ate_true"].max())
    )
    pad = (max_val - min_val) * 0.05  # 5% padding
    axis_limits = (min_val - pad, max_val + pad)

    g = sns.FacetGrid(df, col="method", col_wrap=4, height=3.5, aspect=1.0)

    def plot_scatter_with_ref(x, y, **kwargs):
        sns.scatterplot(x=x, y=y, alpha=0.6, s=20, edgecolor=None, **kwargs)

        ax = plt.gca()
        ax.plot(
            [axis_limits[0], axis_limits[1]],
            [axis_limits[0], axis_limits[1]],
            ls="--",
            c="black",
            alpha=0.6,
            zorder=0,
        )

    g.map(plot_scatter_with_ref, "ate_true", "ate_pred", color="#2c7fb8")

    g.set_axis_labels("True ATE", "Predicted ATE")
    g.set_titles(col_template="{col_name}", fontweight="bold")
    g.set(xlim=axis_limits, ylim=axis_limits)

    plt.suptitle(
        "True vs. Predicted ATE across Episodes", y=1.02, fontsize=16, fontweight="bold"
    )

    output_file = _suffixed(Path(csv_path).parent / "plot_ate_scatter.png", target)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved successfully to: {output_file}")
    plt.close()


def plot_shape_r2_boxplot(csv_path: str, target: str | None = None):
    df = load_results(csv_path, target)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    df_valid = df.dropna(subset=["shape_r2"]).copy()

    if df_valid.empty:
        warnings.warn(
            "No heterogeneity in treatment effect. Skipping shape R2 boxplot."
        )
        return

    # Order methods by median shape R² (best at the top).
    method_order = (
        df_valid.groupby("method")["shape_r2"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    plt.figure(figsize=(10, 6))

    # Create the horizontal boxplot
    ax = sns.boxplot(
        data=df_valid,
        x="shape_r2",
        y="method",
        order=method_order,
        palette="viridis",
    )

    # Add reference lines for R^2 = 0 (predicting the mean) and R^2 = 1 (perfect)
    ax.axvline(0, color="black", linestyle="--", alpha=0.6, label="R² = 0 (Baseline)")
    ax.axvline(1, color="green", linestyle="--", alpha=0.6, label="R² = 1 (Perfect)")

    # Clip the X-axis so massive negative outliers don't squash the plot
    ax.set_xlim(-2.05, 1.05)

    # The plotted column is shape_r2, not cate_r2 — label it as such. (CATE R²
    # is 1 − nPEHE² and is a different curve; it lives in summary_*.csv.)
    plt.title("CATE Estimation: Shape R² Distribution", fontsize=16, fontweight="bold")
    plt.xlabel("Shape R² (pure shape recovery, ATE bias removed)")
    plt.ylabel("")
    plt.legend(loc="lower left")

    output_file = _suffixed(Path(csv_path).parent / "plot_shape_r2_box.png", target)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved successfully to: {output_file}")
    plt.close()


def plot_single_episode_cate(
    exp_dir: Path, episode_idx: int = 0, target: str | None = None
):
    # 1. Load the ground truth for this specific episode. Prefer the Monte-Carlo
    #    CATE: against the realized ITE this scatter is mostly a noise cloud,
    #    because Var(ITE | X) is typically most of Var(ITE).
    data_path = exp_dir / "data.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data.pt at {data_path}")

    data = torch.load(data_path, weights_only=False)
    if target == "ite" or data.get("true_CATEs") is None:
        true_cate, target = data["true_ITEs"][episode_idx], "ite"
    else:
        true_cate, target = data["true_CATEs"][episode_idx], "cate"

    true_cate_eval = np.array(
        true_cate.cpu() if hasattr(true_cate, "cpu") else true_cate
    )
    if np.allclose(true_cate_eval, true_cate_eval[0]):
        warnings.warn(
            f"No heterogeneity in treatment effect for episode {episode_idx}. Skipping plot."
        )
        return

    preds_dir = exp_dir / "preds"
    records = []

    for npz_file in preds_dir.glob("*_preds.npz"):
        method_name = npz_file.stem.replace("_preds", "")
        method_preds = np.load(npz_file)["preds"]

        # Check if the episode failed (NaNs)
        if np.isnan(method_preds[episode_idx]).all():
            print(f"Skipping {method_name} - Episode {episode_idx} failed (NaN).")
            continue

        pred_cate = method_preds[episode_idx]

        for t, p in zip(true_cate, pred_cate):
            records.append({"method": method_name, "true_cate": t, "pred_cate": p})

    if not records:
        print("No valid predictions found to plot.")
        return

    df = pd.DataFrame(records)

    # 3. Set up the FacetGrid
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    min_val = min(df["true_cate"].min(), df["pred_cate"].min())
    max_val = max(df["true_cate"].max(), df["pred_cate"].max())
    pad = (max_val - min_val) * 0.1
    axis_limits = (min_val - pad, max_val + pad)

    g = sns.FacetGrid(df, col="method", col_wrap=4, height=3.5, aspect=1.0)

    def plot_scatter_with_ref(x, y, **kwargs):
        sns.scatterplot(x=x, y=y, alpha=0.8, s=40, edgecolor="w", **kwargs)
        # Reference line y=x
        ax = plt.gca()
        ax.plot(
            [axis_limits[0], axis_limits[1]],
            [axis_limits[0], axis_limits[1]],
            ls="--",
            c="black",
            alpha=0.6,
            zorder=0,
        )

    g.map(plot_scatter_with_ref, "true_cate", "pred_cate", color="#e6550d")

    truth = "tau*(x)" if target == "cate" else "realized ITE"
    g.set_axis_labels(f"True {truth} (Ep {episode_idx})", "Predicted CATE")
    g.set_titles(col_template="{col_name}", fontweight="bold")
    g.set(xlim=axis_limits, ylim=axis_limits)

    plt.suptitle(
        f"CATE Estimation vs {truth}: Single Episode (Index: {episode_idx})",
        y=1.02,
        fontsize=16,
        fontweight="bold",
    )

    # 4. Save
    output_file = _suffixed(
        exp_dir / "results" / f"plot_cate_scatter_ep{episode_idx}.png", target
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved successfully to: {output_file}")
    plt.close()


def resolve_exp_dir(exp: str) -> Path:
    """Resolve an experiment name to its benchmark output directory.

    Two layouts exist:
      * fixed synthetic builds  -> data/synt_eval/<exp>/bench (config_val.yaml)
      * generated-data runs     -> results/<exp>              (config_gen.yaml)
    The synthetic layout wins if both are present; `synthetic/<exp>/bench` is
    still accepted for builds made before the default moved under `data/`.
    """
    candidates = [
        Path("data/synt_eval") / exp / "bench",
        Path("synthetic") / exp / "bench",
        Path("results") / exp,
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise SystemExit(
        f"No experiment directory found for '{exp}' "
        f"(tried: {', '.join(str(c) for c in candidates)}). "
        "Run benchmarks/run.py first, or pass --path explicitly."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot True vs Predicted ATE")

    parser.add_argument(
        "--exp",
        default="first_test",
        help="Experiment name, resolved to data/synt_eval/<exp>/bench, "
        "synthetic/<exp>/bench or results/<exp> (first match wins).",
    )
    parser.add_argument(
        "--path",
        default=None,
        help=(
            "Path to a benchmark output directory (containing data.pt, preds/ and "
            "a results/ sub-folder), e.g. data/synt_eval/<name>/bench. "
            "Overrides --exp for experiments stored elsewhere."
        ),
    )
    parser.add_argument(
        "--target",
        choices=["cate", "ite"],
        default=None,
        help=(
            "Which scoring track to plot: 'cate' scores against the "
            "Monte-Carlo tau*(x) = E[Y1-Y0 | X], 'ite' against the realized "
            "Y1-Y0. Defaults to cate when the run has it. Outputs are suffixed "
            "with the track, so both can coexist."
        ),
    )
    args = parser.parse_args()

    exp_dir = Path(args.path) if args.path else resolve_exp_dir(args.exp)
    results_dir = exp_dir / "results"
    full_csv_path = results_dir / "results.csv"
    if not full_csv_path.exists():
        raise SystemExit(
            f"{full_csv_path} not found — the benchmark has not produced results "
            f"in {exp_dir} yet. Run benchmarks/run.py first."
        )

    target = args.target or default_target(full_csv_path)

    # Existing Plots
    plot_true_vs_pred_ate(full_csv_path, target)
    plot_shape_r2_boxplot(full_csv_path, target)
    plot_npehe_boxplot(full_csv_path, target)
    plot_ate_bias_vs_npehe(results_dir)
    plot_cd_diagram(full_csv_path, target)

    for i in range(5):
        plot_single_episode_cate(exp_dir, episode_idx=i, target=target)
