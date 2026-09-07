# `analysis_eval` — evaluation-section artifacts

This folder builds paper-ready descriptions of benchmark runs. It does not run
estimators or read prediction caches.

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

This reads the completed eight-case matrix at
`data/semisynt_eval/semisynt_val_v1`, validates the full linearity × confounding
× proxy-reliability factorial, and writes
`analysis_eval/results/semisynt_learning_curves.{pdf,png}`. Curves show median
CATE nPEHE over episodes at each context size; optionally, faint bands show
episode interquartile ranges. By default it plots the compact, data-independent
`design.RELEVANT_METHODS` subset; `--methods` provides a figure-specific
override. Use `--matrix-root`, `--out-dir`, `--y-max`, and
`--confidence-interval {false,iqr,std,ste}` to override the other defaults. The
`std` and `ste` bands are centered on the median curve and use one standard
deviation or one standard error, respectively.

[`design.py`](design.py) is the visual source of truth for method labels,
colours, markers, line styles, emphasis, and legend order. Every evaluation
figure should use that registry so a method retains the same identity across
the paper. Unknown methods deliberately raise until a style is added.

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

## Semisynthetic speed versus rank

```bash
uv run python -m analysis_eval.speed_rank
```

This writes `analysis_eval/results/semisynt_speed_rank.{pdf,png}`. At the
largest available context size (2048 in the production matrix), separate
linear- and nonlinear-mechanism panels rank every method within each
`(factorial cell, episode)` block. Each panel therefore summarizes four
confounding/proxy cells independently rather than inheriting ranks from a
pooled eight-cell comparison. The y-axis shows median paired rank with an
interquartile whisker; the x-axis shows mean wall-clock seconds per episode on
a shared log scale. Thus the upper-left is better.

A filled marker uses the complete measured runtime: estimator construction,
hyperparameter search, final fit, prediction, and diagnostics. For methods with
explicit tuning, a connected hollow marker uses `time_post_tune`, the measured
total after subtracting search-only regions. It answers the deployment-like
question “what if the selected configuration were already known?” without
mixing that hypothetical cost into the end-to-end benchmark. Historical tuned
caches cannot reconstruct this split and are shown only at their total time;
run `FORCE=1 scripts/run_semisynt_benchmarks.sh` to populate the second timing.
Use `--context-size` or `--methods` to override the defaults.
