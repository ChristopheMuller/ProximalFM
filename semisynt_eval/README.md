# Empirical-root proximal evaluation matrix

`semisynt_eval` combines empirical `(X, U)` rows from CSV files with known
synthetic mechanisms for `W`, `Z`, `A`, and `Y`. The default format-v4 recipe
builds a controlled eight-case factorial:

| Family | Direct-U confounding | W/Z U-information |
|---|---:|---:|
| linear | 0.25 | 0.25 |
| linear | 0.25 | 0.75 |
| linear | 0.75 | 0.25 |
| linear | 0.75 | 0.75 |
| nonlinear | 0.25 | 0.25 |
| nonlinear | 0.25 | 0.75 |
| nonlinear | 0.75 | 0.25 |
| nonlinear | 0.75 | 0.75 |

[`config.yaml`](config.yaml) is the complete, version-controlled definition of
the matrix. Python supplies validation, not hidden generative defaults.

## Build and consume

List available CSV sources and configured cases:

```bash
uv run python -m semisynt_eval.build --list
```

Build every case:

```bash
NAME=semisynt_matrix_v1 ./scripts/build_semisynt_eval.sh
```

The script uses plain `python` through `scripts/_env.sh`, so it works on the
cluster where `uv run` is unavailable. Build-level settings are environment
variable overridable, for example:

```bash
NAME=semisynt_matrix_v1 N_CONTEXT=1024 N_QUERY=256 \
  REPLICATES_PER_CSV=24 SEED=20260903 \
  ./scripts/build_semisynt_eval.sh
```

Use `CONFIG=path/to/recipe.yaml` for another scientific matrix and `FORCE=1`
to replace builder-owned files for the selected name.

The result is one matrix root and eight independent benchmark sets:

```text
data/semisynt_eval/semisynt_matrix_v1/
  manifest.json
  config.yaml
  linear_conf_low_proxy_low/
    manifest.json
    config.yaml
    val/
  ... seven more cases ...
```

Point a benchmark at one case, never at the matrix root:

```bash
uv run benchmarks/run.py --config benchmarks/config_methods.yaml \
  --val-dir data/semisynt_eval/semisynt_matrix_v1/linear_conf_low_proxy_low/val \
  --context-sizes 128 512
```

[`scripts/run_semisynt_benchmarks.sh`](../scripts/run_semisynt_benchmarks.sh) runs the shared method
configuration over all eight cases. Set `MATRIX_ROOT` or `METHODS_CONFIG` to override its defaults.

Each case config is a complete one-case format-v4 recipe and can be rebuilt
with `--config`; override `--name` to keep the original output intact. Source
SHA-256 values, resolved case settings, paired episode IDs, and content digests
are recorded in the manifests. `--force` replaces only builder-owned files.

Every `*.csv` directly in `semisynt_eval/datasets/` is discovered in stable
filename order. Before any data-dependent preprocessing, `n_calibration`
(default 1,000) rows are selected deterministically for that source. Numeric
imputation medians, 0.5th/99.5th-percentile clipping thresholds, categorical
codes, and standardization constants are fitted on those rows only. The
remaining rows form the evaluation pool. Without explicit `--x_columns` and
`--u_columns`, a stable hash split selects the requested number of X and U
columns from features that have positive calibration variance.

Every context/query episode is sampled without replacement exclusively from
the evaluation pool. Calibration rows are fixed per source and shared by all
replicates and cases; they never appear in an episode. A source must therefore
contain at least `n_calibration + n_context + n_query` usable rows. Partition
seeds, source-row identities, preprocessing constants, and episode row hashes
are recorded in the manifests. `replicates_per_csv` means independently seeded
benchmark episodes, not Monte Carlo rows.

Mechanisms follow an explicit fit-then-generate lifecycle. For each episode,
`fit_mechanism(X_cal, U_cal, ...)` constructs the mechanism using only the
calibration partition, and the returned object generates `W`, `Z`, `A`, and
`Y` from sampled evaluation rows through `mechanism.generate(X_eval, U_eval)`.
This is currently an interface boundary: the legacy random-basis families use
the calibration tensors only to establish their dimensions and retain their
existing episode-local normalizations. Ayu-style fitted basis, latent, and
proxy-noise constants will be added behind this boundary separately.

The row split has one spelling: `n_context` context rows plus `n_query` query
rows. The total episode size is derived as their sum; there is no `seq_len` or
`prop_train` ratio to resolve.

## Causal contract and target

The interfaces structurally enforce

```text
W    <- U, optional X
Z    <- U, optional X
A    <- U, optional X, optional Z
Y(a) <- U, optional X, optional W, a
tau  <- optional X
```

Z is therefore the treatment-inducing proxy and cannot enter Y; W is the
outcome-inducing proxy and cannot enter A. All default arrows are enabled.

Both families use `Y(1) = Y(0) + tau(X)`. The linear family gives `tau` a
linear random basis; the nonlinear family uses the same mixed basis as its
nuisance equations. Outcome noise is shared across arms, so `Y1 - Y0` equals
the analytic CATE exactly. The explicit `CATE` field activates CATE scoring in
the benchmark without within-row Monte Carlo replication.

## Families, normalization, and interactions

The linear family uses identity terms. The nonlinear family randomly chooses a
term for every input/output coefficient from the configured mixture of:

- identity
- quadratic
- signed `log1p(abs(x))`
- clipped exponential
- sine

Every transformed term is centered and scaled before its random coefficient is
applied. Exponential inputs are clipped at the configured bound first. This
makes coefficient and component scales comparable rather than allowing the
chosen function's raw range to determine strength.

`interactions_default` supplies the default for every equation. Each of `w`,
`z`, `a`, `y`, and `tau` may override it with `interactions: true/false/null`,
where `null` inherits the common value. Enabling interactions adds all pairwise
products of normalized transformed parents. A term involving U is classified
as latent even when its other factor is X, Z, or W. The resolved case configs
always contain concrete booleans.

## Strength and overlap controls

For treatment and outcome, the random regression is separated into a direct-U
component and an observed-parent component. Each is normalized, then combined
as

```text
sqrt(u_variance_share) * latent
  + sqrt(1 - u_variance_share) * observed.
```

The separate knobs are `a_u_variance_share` and `y_u_variance_share`. They
describe structural component variance, not an observational partial R-squared;
correlation between empirical X and U can still change measured confounding.

For each proxy, reliability is calibrated against its normalized U signal:

```text
reliability = Var(U signal) / (Var(U signal) + Var(noise)).
```

`w_u_reliability` and `z_u_reliability` can be varied independently. Optional
X contributions have their own `x_strength`, so adding X does not redefine the
U-to-noise calibration. Final proxy normalization preserves these relative
component scales.

Treatment uses a normalized score, configurable logit scale (inverse
temperature), random-intercept range, logit clipping, and a final
`overlap_epsilon` clamp before Bernoulli sampling. Outcome noise and CATE scale
remain separate knobs because they control different difficulty dimensions.

## Editing the matrix

The `common` block owns coefficients, the nonlinear palette, and complete
equation defaults. Every entry under `cases` selects a family and the four
calibrated strength values, with an optional `equations` override map. Add or
edit cases there; do not encode the scientific matrix in shell loops or CLI
flags. Case names become directory names and must not contain `/`.

Cases are paired by `(source CSV, replicate)`: all cases receive the same
episode seed and therefore identical empirical X/U rows. Cases of the same
family and structure also reuse coefficient and base-noise draws, so changing a
low/high strength rescales the same underlying mechanism rather than drawing a
new one.
