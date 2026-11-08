"""Regenerate tables and PNGs from the published forward-pass CSVs.

Use --audit-predictions after run_forward.py to also recompute every metric
from the locally saved prediction arrays. The published results/audit.json
records that full check for the reported run. Default CSV-only checks are
written separately to results/summary_checks.json.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--audit-predictions", action="store_true")
args = parser.parse_args()
OUT = ROOT / "results"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": "#23364a", "text.color": "#23364a",
                     "axes.titleweight": "bold", "savefig.facecolor": "white"})
METRICS = ["npehe", "shape_r2", "level_error", "heterogeneity_error", "npehe_squared"]
audit = {}
frames = {}
for mode, expected in [("dimensions", 720), ("shape", 1152)]:
    folder = ROOT / "raw/results" / mode
    df = pd.read_csv(folder / "metrics.csv")
    assert len(df) == expected
    assert not df.duplicated(["source", "case", "replicate", "total_dimension", "context"]).any()
    assert np.isfinite(df[METRICS].to_numpy()).all()
    assert np.allclose(df.npehe ** 2, df.npehe_squared, rtol=1e-10, atol=1e-10)
    assert np.allclose(df.npehe_squared, df.level_error + df.heterogeneity_error, rtol=1e-10, atol=1e-10)
    assert np.allclose(df.heterogeneity_error, 1 - df.shape_r2, rtol=1e-10, atol=1e-10)
    differences = []
    for row in (df.itertuples() if args.audit_predictions else []):
        data = np.load(folder / "predictions" / row.prediction_file)
        p, t = data["prediction"].astype(float), data["truth"].astype(float)
        assert np.isfinite(p).all() and len(p) == 256 and t.var() > 0
        assert not np.intersect1d(data["context_rows"], data["query_rows"]).size
        e = p - t
        actual = [np.sqrt(np.mean(e**2) / t.var()), 1 - e.var() / t.var(),
                  e.mean()**2 / t.var(), e.var() / t.var(), np.mean(e**2) / t.var()]
        differences.extend(abs(v - getattr(row, k)) for v, k in zip(actual, METRICS))
        assert abs(actual[-1] - actual[2] - actual[3]) < 1e-10
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["parameters_unchanged"] and metadata["completed"]
    assert metadata["model_parameters"] == 29587656
    assert metadata["model_config"]["row_num_cls"] == 4
    assert metadata["checkpoint_sha256"] == "2dc76f3dbe001af9816ce90e642108cd9d41a0ca59060b15395e382a7582b5df"
    audit[mode] = dict(episodes=len(df), csv_metric_identities_verified=True,
                       predictions_recomputed=args.audit_predictions,
                       frozen_parameters_reported=metadata["parameters_unchanged"],
                       checkpoint_sha256=metadata["checkpoint_sha256"])
    if args.audit_predictions:
        assert max(differences) < 1e-10
        audit[mode].update(recomputed_max_error=float(max(differences)),
                           disjoint_context_query=True)
    frames[mode] = df

d = frames["dimensions"]
by_source = d.groupby(["source", "total_dimension"])[METRICS].mean().reset_index()
by_source.to_csv(OUT / "dimension_by_source.csv", index=False)
ds = by_source.groupby("total_dimension")[METRICS].mean()
ds["median_ms"] = d.groupby("total_dimension").seconds.median() * 1000
ds["mean_ms"] = d.groupby("total_dimension").seconds.mean() * 1000
ds["d_X"] = ds.index - 2 * (ds.index // 3)
ds["d_W"] = ds.index // 3
ds["d_Z"] = ds.index // 3
ds.to_csv(OUT / "dimension_summary.csv")
palette = {"Allstate_Claims_Severity": "#245d96", "superconduct": "#d77531", "MiniBooNE": "#39826e"}
labels = {"Allstate_Claims_Severity": "Allstate Claims", "superconduct": "Superconduct", "MiniBooNE": "MiniBooNE"}
fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.5))
for name, color in palette.items():
    g = by_source[by_source.source == name]
    for ax, metric in zip(axes[:2], ["npehe", "shape_r2"]):
        ax.plot(g.total_dimension, g[metric], "o-", color=color, label=labels[name], lw=1.8, ms=3.8)
    timing = d[d.source == name].groupby("total_dimension").seconds.median() * 1000
    axes[2].plot(timing.index, timing, "o-", color=color, lw=1.8, ms=3.8)
for ax, title, ylabel in zip(axes, ["CATE accuracy", "Heterogeneity recovery", "Forward-pass time"],
                            ["nPEHE (lower is better)", r"Shape $R^2$ (higher is better)", "Milliseconds per episode"]):
    ax.set(title=title, xlabel=r"Total input features $d_X+d_W+d_Z$", ylabel=ylabel)
    ax.set_xticks([5, 15, 25, 35, 50])
    ax.grid(axis="y", color="#dce3e9", linewidth=.6)
    ax.set_axisbelow(True)
axes[0].set_ylim(0, 1.05)
axes[1].set_ylim(0, 1.0)
axes[2].set_ylim(0, 28)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(.52, 1.02))
fig.subplots_adjust(left=.065, right=.99, bottom=.19, top=.79, wspace=.36)
fig.savefig(FIG / "dimension_sweep.png", dpi=220, bbox_inches="tight")
plt.close(fig)

s = frames["shape"]
source = s.groupby(["source", "case", "family", "confounding", "proxy", "context"])[METRICS].mean().reset_index()
source.to_csv(OUT / "shape_by_source.csv", index=False)
summary = source.groupby(["case", "family", "confounding", "proxy", "context"])[METRICS].mean().reset_index()
summary.to_csv(OUT / "shape_summary.csv", index=False)
overall = source.groupby("context")[METRICS].mean()
overall.to_csv(OUT / "shape_overall.csv")
order = [f"{f}_conf_{c}_proxy_{p}" for f in ["linear", "nonlinear"] for c in ["low", "high"] for p in ["low", "high"]]
matrix = summary.pivot(index="case", columns="context", values="shape_r2").loc[order]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.8, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(4), ["128", "512", "1,024", "2,048"])
ax.set_yticks(range(8), [f"{f.capitalize()} | {c} / {p}" for f in ["linear", "nonlinear"] for c in ["low", "high"] for p in ["weak", "strong"]])
ax.set_xlabel("Context observations")
ax.set_title(r"Shape $R^2$ across all eight settings", pad=12)
ax.text(-.01, 1.01, "Row labels: mechanism | confounding / proxies", transform=ax.transAxes, fontsize=8, va="bottom")
for i in range(8):
    for j in range(4):
        value = matrix.iloc[i, j]
        ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value > .60 else "#23364a", fontsize=10)
ax.axhline(3.5, color="white", lw=3)
for spine in ax.spines.values(): spine.set_visible(False)
x = np.arange(4)
bx.bar(x, overall.heterogeneity_error, color="#2e76a9", width=.62, label="Heterogeneity error")
bx.bar(x, overall.level_error, bottom=overall.heterogeneity_error, color="#d98b49", width=.62, label="Level error")
for i, value in enumerate(overall.npehe_squared):
    bx.text(i, value + .025, f"{value:.3f}", ha="center", fontsize=10)
bx.set_xticks(x, ["128", "512", "1,024", "2,048"])
bx.set(xlabel="Context observations", ylabel=r"Mean squared nPEHE", ylim=(0, 1.08))
bx.set_title("Both error components improve", pad=12)
bx.legend(frameon=False, loc="upper right", fontsize=9)
bx.grid(axis="y", color="#dce3e9", linewidth=.6)
bx.set_axisbelow(True)
fig.subplots_adjust(left=.24, right=.99, bottom=.15, top=.87, wspace=.34)
fig.savefig(FIG / "shape_r2.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Four main-context tables: every cell pairs metrics from exactly the same predictions.
lines = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
         r"& & \multicolumn{2}{c}{128} & \multicolumn{2}{c}{512} & \multicolumn{2}{c}{1024} & \multicolumn{2}{c}{2048}\\",
         r"Mechanism & Conf./proxy & nPEHE & $R^2_s$ & nPEHE & $R^2_s$ & nPEHE & $R^2_s$ & nPEHE & $R^2_s$\\", r"\midrule"]
for case in order:
    g = summary[summary.case == case].set_index("context")
    item = g.iloc[0]
    values = " & ".join(f"{g.loc[n,'npehe']:.3f} & {g.loc[n,'shape_r2']:.3f}" for n in [128, 512, 1024, 2048])
    lines.append(f"{item.family.title()} & {item.confounding}/{('weak' if item.proxy == 'low' else 'strong')} & {values}" + r"\\")
lines.extend([r"\bottomrule", r"\end{tabular}"])
(OUT / "paired_main_table.tex").write_text("\n".join(lines) + "\n")
decrease = overall.loc[128] - overall.loc[2048]
audit["heterogeneity_share_of_squared_error_reduction"] = float(decrease.heterogeneity_error / decrease.npehe_squared)
audit["shape_increases_in_every_regime_and_context_step"] = bool((matrix.diff(axis=1).iloc[:, 1:] > 0).all().all())
audit_file = "audit.json" if args.audit_predictions else "summary_checks.json"
(OUT / audit_file).write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
