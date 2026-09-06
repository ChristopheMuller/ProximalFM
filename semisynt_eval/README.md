# Empirical-root proximal evaluation matrix

`semisynt_eval` combines empirical `(X, U)` rows from CSV files with known
synthetic mechanisms for `W`, `Z`, `A`, and `Y`. The schema defines a
controlled eight-case `calibrated_latent_proxy` factorial:

| Family | Confounding (`alpha`, `gamma`) | Proxy reliability |
|---|---:|---:|
| linear | low (0.35, 0.5) | low (0.2) |
| linear | low (0.35, 0.5) | high (0.8) |
| linear | high (1.25, 1.5) | low (0.2) |
| linear | high (1.25, 1.5) | high (0.8) |
| nonlinear | low (0.35, 0.5) | low (0.2) |
| nonlinear | low (0.35, 0.5) | high (0.8) |
| nonlinear | high (1.25, 1.5) | low (0.2) |
| nonlinear | high (1.25, 1.5) | high (0.8) |

[`config.yaml`](config.yaml) is the complete, version-controlled definition of
the matrix. Python supplies validation, not hidden generative defaults.

Mechanism identity and function family are separate configuration axes:
`common.mechanism` is `calibrated_latent_proxy`, while each case selects
`family: linear` or `family: nonlinear`. All numeric mechanism constants and
the low/high level mappings live in the YAML.

## Build and consume

List available CSV sources and configured cases:

```bash
uv run python -m semisynt_eval.build --list
```

Build every case:

```bash
NAME=semisynt_matrix_v1 ./scripts/build_semisynt_eval.sh
```

Pass `--case NAME` one or more times to the Python entry point to build a
subset. Omitting it builds all eight cases. Case selection does not change a
case's content digest or generated tensors.

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

Each case config snapshots the complete matrix recipe and can be
rebuilt with `--config`; override `--name` to keep the original output intact.
Source SHA-256 values, resolved case settings, paired episode IDs, and content
digests are recorded in the manifests. `--force` replaces only builder-owned
files.

Generated configs include a canonical `resolved_cases` snapshot containing the
mechanism name and every resolved numeric setting. Loading such a
config verifies the snapshot against `common` and `cases` and rejects stale or
hand-edited disagreements. The identical resolved payload is embedded in case
and matrix manifests and contributes to every case content digest.

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

Mechanisms follow an explicit fit-prepare-generate lifecycle. For each
`(source, replicate)`, evaluation rows are sampled once and exactly two
mechanisms are fitted on the calibration partition: one per function family.
Each mechanism prepares its augmented latent state, X functions, CATE, and
named innovations once, then generates the four confounding/proxy cases from
that shared state.

The row split has one spelling: `n_context` context rows plus `n_query` query
rows. The total episode size is derived as their sum; there is no `seq_len` or
`prop_train` ratio to resolve.

## Causal contract and target

The calibrated latent proxy mechanism has the fixed graph

```text
U <- U_source, latent innovation
W <- U, X, measurement noise
Z <- U, X, measurement noise
A <- X, shared latent index
Y(a) <- X, shared latent index, outcome noise, a
tau <- X
```

W and Z are measurements: Z does not directly cause treatment, and W does not
directly cause either potential outcome. Both treatment and outcome use one
shared latent direction. Shared potential-outcome noise makes
`Y1 - Y0 = tau(X)` exactly, so the explicit `CATE` field activates benchmark
CATE scoring without row replication.

The `linear` family uses identity latent/proxy transforms and an identity X
basis. The `nonlinear` family uses `asinh` latent/proxy transforms and a fitted
X basis drawn from the YAML-declared identity, `asinh`, sine, and quadratic
terms, with at most the configured number of interactions.

Proxy reliability is the calibration signal fraction
`Var(signal) / (Var(signal) + Var(noise))`. Treatment propensity is clipped to
the configured lower/upper bounds. Outcome-noise SD and CATE location/scale are
separate explicit settings.

## Editing the matrix

The `common` block owns every numeric mechanism constant and the named
confounding/proxy levels. Each case contains exactly `family`, `confounding`,
and `proxy`. Add or edit cases there; do not encode the scientific matrix in
shell loops or CLI flags. Case names become directory names and must not
contain `/`.

Cases are paired by `(source CSV, replicate)`: all cases reuse identical
ordered empirical rows. Cases in one family reuse the fitted mechanism,
augmented latent state, X functions, CATE, and named innovations, so changing a
low/high strength modifies only the quantities downstream of that setting.
The `U` stored in `val/` is the innovation-augmented hidden latent; empirical
`U_source` remains diagnostic-only and is never exposed to benchmark methods.
