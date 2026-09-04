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
episode interquartile ranges. Use `--matrix-root`, `--out-dir`, `--methods`,
`--y-max`, and `--confidence-interval {false,iqr,std,ste}` to override those
defaults. The `std` and `ste` bands are centered on the median curve and use
one standard deviation or one standard error, respectively.

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
is ProximalFM, CausalPFN, conditional-W NMMR, T-OLS, and X-RF; override it with
`--methods`. Use `--xlim MIN MAX` and `--ylim MIN MAX` to fix the shared grid;
trajectory segments crossing an edge stop at that edge, without drawing their
out-of-grid nodes.
