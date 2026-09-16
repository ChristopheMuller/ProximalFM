# `analysis_eval` — evaluation-section artifacts

This folder builds paper-ready descriptions of benchmark runs. It does not run
estimators or read prediction caches, with one exception: the posterior
calibration runner below evaluates a ProximalFM checkpoint against fixed
evaluation data.

## Posterior calibration

```bash
uv run python -m analysis_eval.posterior_calibration \
  --data-root data/semisynt_eval/semisynt_posterior_calibration_v1/d-5/nonlinear_conf_high_proxy_low \
  --ckpt path/to/checkpoint-or-directory \
  --device mps
```

The runner supports analytic-CATE semi-synthetic dataset roots and frozen
`val/` datasets with `frozen_meta.pt`. By default it scores nested context
prefixes from 128 through the dataset maximum against the same query rows. It
uses a single forward posterior prediction, with the comparison scale frozen
at the largest context size. Use `--contexts` and `--levels` to override the
context sizes and central coverage levels, `--max-episodes` to limit a run,
`--out-dir` to choose its destination, and `--force` to replace existing
outputs.

Each run writes `points.parquet` (per-query posterior and target values),
`metrics.csv` (distributional scores by context size), `coverage.csv`
(pooled and per-episode interval coverage), `pit_ecdf.csv`, `run.json` with
dataset/checkpoint provenance, and `posterior_calibration.pdf` / `.png`. The
three-panel figure shows: empirical coverage at each nominal level as context
grows; nominal-versus-empirical calibration curves with the 45° reference;
and posterior-width contraction. The contraction curve is the mean row-level
posterior SD, with a shaded IQR computed over episode-level mean widths. That
band describes between-episode heterogeneity, rather than the sampling error
of the pooled mean. `pit_ecdf.csv` remains available for PIT diagnostics even
though it is not a figure panel. The runner requires a single
CATE-distribution head; two-potential-outcome checkpoints are not supported
for this evaluation.

Once a calibration run exists, regenerate only its figure—without loading the
model or running inference—with `--plot-only`:

```bash
uv run python -m analysis_eval.posterior_calibration \
  --data-root data/prior_eval/prior_calibration_v1 \
  --ckpt whole_1/best_model.ckpt \
  --plot-only
```

### Prior versus semi-synthetic comparison

`analysis_eval.compare_posterior_calibration` is also cache-only. It combines
two completed calibration output directories into a two-row, three-column
figure: prior results on the first row and semi-synthetic results on the
second. Each row contains the same coverage, calibration, and contraction
panels described above.

```bash
uv run python -m analysis_eval.compare_posterior_calibration \
  --prior-dir analysis_eval/results/posterior_calibration/prior_calibration_v1__whole_1-best_model \
  --semisynt-dir analysis_eval/results/posterior_calibration/nonlinear_conf_high_proxy_low__whole_1-best_model \
  --out-dir analysis_eval/results/posterior_calibration/comparison
```

Use `--name` to change the output basename; the command writes both PDF and
PNG files and never recomputes posterior predictions.

## Method table

```bash
uv run python -m analysis_eval.method_table \
  --config benchmarks/config_methods.yaml
```

The command writes `analysis_eval/results/methods.tex` and a machine-readable
`methods.csv`. Use `--out <path>.tex` to place both files elsewhere and
`--print` to also print the LaTeX.

The benchmark YAML is the source of truth for which methods were tested. The
artifact intentionally keeps only outputs with the exact canonical estimand
`cate(x)`: proxy-conditional CATEs, oracle-conditioned outputs, and ATE outputs
are omitted. A multi-output entry therefore becomes one row for each of its
CATE(X) variants. Config-dependent properties are derived from the config and
the benchmark's own estimand helper. The compact LaTeX table is designed to fit
within one paper column/page width and contains only the benchmark name, method
label, implementation/prediction variant, and citation; the CSV retains all
resolved fields for detailed appendix tables.

[`methods.json`](methods.json) is the curated source for stable facts that do
not belong in a run config: display family, whether a method is proximal or a
foundation model, and its citation. Citation objects contain readable `text`
for CSV output and an optional `bibtex_key`; the LaTeX renderer uses
`\citep{...}` whenever a key is present. The keys shipped here match
`docs/latex/references.bib`. Add metadata for a new benchmark `cls` before
generating the table—the command fails on missing entries rather than silently
printing an incomplete row.

## Semisynthetic learning curves

```bash
uv run python -m analysis_eval.learning_curves
```

This reads dimension 5 of the completed eight-case matrix at
`data/semisynt_eval/semisynt_val_v1`, validates the full linearity × confounding
× proxy-reliability factorial, and writes
`analysis_eval/results/semisynt_learning_curves.{pdf,png}`. Curves show median
CATE nPEHE over episodes at each context size; optionally, faint bands show
episode interquartile ranges. By default it plots the compact, data-independent
`design.RELEVANT_METHODS` subset; `--methods` provides a figure-specific
override. Use `--matrix-root`, `--dim`, `--out-dir`, `--y-max`, and
`--confidence-interval {false,iqr,std,ste}` to override the other defaults. The
`std` and `ste` bands are centered on the median curve and use one standard
deviation or one standard error, respectively.

[`design.py`](design.py) is the visual source of truth for method labels,
colours, markers, line styles, emphasis, and legend order. Every evaluation
figure should use that registry so a method retains the same identity across
the paper. Unknown methods deliberately raise until a style is added.

To plot one matrix configuration as a single subplot, pass its case name (the
`--config` spelling is also accepted):

```bash
uv run python -m analysis_eval.learning_curves \
  --case linear_conf_low_proxy_low \
  --name semisynt_learning_curve_linear_conf_low_proxy_low
```

The case names follow
`<linear|nonlinear>_conf_<low|high>_proxy_<low|high>`. The Python API accepts
the same selector through `make_learning_curve_figure(..., case=...)`. In
single-case mode, only the selected case needs a completed benchmark result;
the default full-grid mode still requires all eight cases.

The semi-synthetic builder stores each dimension below `d-<dimension>` and
keeps the matrix manifest at its parent. Pass that parent as `--matrix-root`;
`--dim 5` is the default for this layout:

```bash
uv run python -m analysis_eval.learning_curves \
  --matrix-root data/semisynt_eval/semisynt_val_v1 \
  --dim 5
```

The error-decomposition, combined, and speed/rank commands use the same
dimension selection. Direct `d-<dimension>` paths remain accepted for
backward-compatible one-dimension invocations.

## Semisynthetic error decomposition

```bash
uv run python -m analysis_eval.error_decomposition
```

This writes `analysis_eval/results/semisynt_error_decomposition.{pdf,png}`.
The same eight factorial cells show trajectories in squared normalized ATE-bias
(level error) versus `1 - Shape R²` (heterogeneity error), with arrows and
increasing marker area indicating context sizes 128 → 2048. Dotted contours
show the square root of the two displayed contributions.

The default robust view separately takes the episode median of each component.
Consequently its contour value is a *decomposed robust score*, not median
nPEHE. `--aggregation mean` instead preserves the additive nPEHE² identity
exactly, but is intentionally optional because a few unstable episode fits can
make the mean view span several orders of magnitude. The default method subset
is the fixed, data-independent set in `design.RELEVANT_METHODS`; override it
with `--methods`. Each panel uses data-dependent x and y ranges so its
trajectories remain legible; use `--xlim MIN MAX` or `--ylim MIN MAX` to impose
a common range for either dimension. Trajectory segments crossing an edge stop
at that edge, without drawing their out-of-grid nodes.

## Combined learning curve and error decomposition

To place one case's learning curve and error decomposition side by side with a
shared legend, run:

```bash
uv run python -m analysis_eval.learning_curves_error_decomposition \
  --case nonlinear_conf_high_proxy_low \
  --matrix-root data/semisynt_eval/d4_10k \
  --name semisynt_learning_curves_error_decomposition_d4_10k \
  --y-max 2
```

This writes matching PDF and PNG files under `analysis_eval/results/`. The
combined command accepts the learning curve's `--confidence-interval` option
and the decomposition's `--aggregation`, `--xlim`, and `--ylim` options.

## Semisynthetic speed versus rank

```bash
uv run python -m analysis_eval.speed_rank
```

This writes `analysis_eval/results/semisynt_speed_rank.{pdf,png}`. At the
largest available context size (2048 in the production matrix), separate low-
and high-confounding panels rank every method within each `(factorial cell,
episode)` block. Each panel therefore summarizes four linearity/proxy cells
independently rather than inheriting ranks from a pooled eight-cell comparison.
The y-axis shows median paired rank with an
interquartile whisker; the x-axis shows mean wall-clock seconds per episode on
a shared log scale. Thus the upper-left is better. All complete methods remain
in the ranking and are plotted, but only the fixed `design.RELEVANT_METHODS`
subset is coloured and labelled; the remaining methods form a light-grey
context layer.

A filled marker uses the complete measured runtime: estimator construction,
hyperparameter search, final fit, prediction, and diagnostics. For methods with
explicit tuning, a connected hollow marker uses `time_post_tune`, the measured
total after subtracting search-only regions. It answers the deployment-like
question “what if the selected configuration were already known?” without
mixing that hypothetical cost into the end-to-end benchmark. Historical tuned
caches cannot reconstruct this split and are shown only at their total time;
run `FORCE=1 scripts/run_semisynt_benchmarks.sh` to populate the second timing.
Use `--context-size` or `--methods` to override the defaults.

## Causal Chambers ATE learning curve

```bash
uv run python -m analysis_eval.causal_chambers \
  --summary-csv benchmarks/causal_chambers/results/ate/summary.csv
```

This reads the Causal Chambers runner's `summary.csv` and writes
`analysis_eval/results/causal_chambers_rmse.{pdf,png}`. The figure shows one
ATE RMSE line per method over the available context sizes. Use `--methods`,
`--out-dir`, or `--name` to customize the figure.
