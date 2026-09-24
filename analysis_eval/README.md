# `analysis_eval` — evaluation-section artifacts

This folder builds paper-ready descriptions of benchmark runs. It does not run
estimators or read prediction caches, with one exception: the posterior
calibration runner below evaluates a ProximalFM checkpoint against fixed
evaluation data.

## Main-text practical-results figure

```bash
.venv/bin/python -m analysis_eval.main_results_figure
```

This cache-only command writes `docs/latex/assets/main/practical_results.pdf`
and `.png`. Panel (a) reuses the complete-case ranks and dedicated sequential
timings from `speed_rank.py` at context size 2048. Panel (b) takes the nominal
90% coverage slice from the existing prior and semi-synthetic calibration
caches. Panel (c) reads the source-averaged extended-context CSV used by the
appendix table; the current raw matrix CSV only reaches 2048. Each input path
and the output directory can be overridden with CLI flags. Run the upstream
analyses first if an input cache is absent.

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
override. The default y-axis maximum is 2. Use `--matrix-root`, `--dim`,
`--out-dir`, `--y-max`, and
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

## Exhaustive semisynthetic nPEHE tables

```bash
uv run python -m analysis_eval.exhaustive_tables
```

This writes four LaTeX tables for the context sizes shared by all eight
factorial cells (`128`, `512`, `1024`, and `2048`), followed by one table for
all seven context sizes in the targeted nonlinear/high-confounding/low-proxy
experiment, to
`analysis_eval/results/semisynt_exhaustive_npehe.tex`, together with a
machine-readable CSV for the shared tables and a sibling `_extended.csv` for
the extended table. Pass
`--out docs/latex/assets/results/semisynt_exhaustive_npehe.tex` to refresh the
copy included by the paper. Every table contains every CATE(X) method from the
shared factorial experiment;
`ProximalFM-FT`, which belongs only to the targeted large-context experiment,
is deliberately excluded from the four shared tables and included in the
extended one. The extended table automatically omits methods with no complete
result at any of its seven context sizes. A method with partial coverage is
retained and its missing or incomplete context-size entries are printed as
dashes.

The extended table reads the regular dimension-five
`nonlinear_conf_high_proxy_low/bench/results/results.csv`, not the separate
`d4_10k` experiment. Because benchmark scripts share that output path,
`scripts/run_semisynt_selection_benchmarks.sh` must be the last benchmark
script run before generating the tables. The command validates that all seven
expected context sizes are present and otherwise fails with that instruction
rather than silently writing a four-context table. Use `--extended-case` or
`--extended-contexts` to override those defaults.

Aggregation respects the experimental hierarchy: nPEHE is first averaged over
the three replicates of each source dataset and then macro-averaged over the 12
sources. The lowest mean in a configuration is bold. Other underlined methods
are not significantly worse under a one-sided exact paired sign-flip test on
the 12 source-level means, with Holm correction across comparisons to the
empirical best within that configuration and context size. This is a
failure-to-reject set, not an equivalence set. Use `--alpha`, `--contexts`,
`--extended-contexts`, `--matrix-root`, `--dim`, or `--out` to override the
defaults.

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

This writes `analysis_eval/results/speed/semisynt_speed_rank.{pdf,png}`. It
computes paired CATE nPEHE ranks from the completed matrix, but takes every
runtime coordinate exclusively from the sequential timing run made by
`scripts/run_semisynt_speed_benchmarks.sh`. The x-axis is therefore the
benchmark's measured end-to-end latency, rather than incidental timings from
another experimental run. `ProximalFM-CPU` and `ProximalFM-MPS` appear as two
separate measured speed points at the common `ProximalFM` accuracy rank, joined
by an orange dotted hardware-comparison line. The CPU point carries the rank
IQR; MPS is its hollow endpoint. Filled points are end-to-end runtime; hollow
points connected by a dotted segment are post-selection runtime with
hyperparameter search removed.

The default `--scope overall` pools all eight configurations. Use one of these
to choose what ranks are summarized:

```bash
# One configuration
uv run python -m analysis_eval.speed_rank --scope case \
  --case nonlinear_conf_high_proxy_low

# Two explicit configurations, side by side
uv run python -m analysis_eval.speed_rank --scope pair --cases \
  linear_conf_low_proxy_low nonlinear_conf_high_proxy_high

# Two factorial groups, side by side
uv run python -m analysis_eval.speed_rank --scope linearity
uv run python -m analysis_eval.speed_rank --scope confounding
uv run python -m analysis_eval.speed_rank --scope proxy
```

All plots use the largest context size common to the matrix and speed run by
default; override it with `--context-size`. `--speed-dir`, `--methods`, and
`--out-dir` support alternate timing runs, subsets, and destinations.

## Semisynthetic speed by context size

```bash
uv run python -m analysis_eval.speed_context
```

This reads the timing arrays written by
`scripts/run_semisynt_speed_benchmarks.sh` and writes
`analysis_eval/results/speed/semisynt_speed_by_context.{pdf,png}`. It shows
median wall-clock latency per episode against context size on log scales, with
an interquartile ribbon across episodes. A fixed representative set is styled
and labelled; every other completed benchmark method is retained as a muted
reference curve. Use `--methods METHOD ...` to choose the highlighted set or
`--all` for a fully labelled (denser) view. `--bench-dir` and `--out-dir` make
the command reusable for another speed run or output location.

## Causal Chambers ATE learning curves

```bash
uv run python -m analysis_eval.causal_chambers
```

This reads the Causal Chambers runner's long-form `results.csv` and writes
`analysis_eval/results/causal_chambers/causal_chambers_absolute_error.{pdf,png}`.
The figure has one panel for each cell of the physical 2 x 2 x 3 factorial
design. Each panel shows absolute ATE error in sensor counts as context grows;
both axes are logarithmic and share their scales across panels. The default
method set is the fixed `design.SELECTION_CAUSAL_CHAMERS` subset. Use
`--results-csv`, `--methods`, `--out-dir`, or `--name` to customize it.

To compare methods on each Causal Chambers dataset at 1,000 datapoints, use
the long-form results file:

```bash
uv run python -m analysis_eval.causal_chambers_rank
```

This writes `analysis_eval/results/causal_chambers_rank_grid.{pdf,png}`. Rows
are sorted by average within-dataset RMSE rank; columns are grouped as linear
versus non-linear, then strong versus weak confounding, then low/mid/high proxy
informativeness. Use `--context-size`, `--methods`, `--out-dir`, or `--name`
to customize the figure.
