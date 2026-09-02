# Adjustment-inspection evaluation sets

`inspect_eval` is an inspection-oriented copy of `synt_eval`. It retains the
twelve realistic proximal mechanisms and the standard `val/` output, but its
replicate branch additionally computes the population limits of ordinary
adjustment:

\[
Q_X^{\mathrm{naive}},\quad
Q_W^{\mathrm{adj}},\quad
Q_Z^{\mathrm{adj}},\quad
Q_{WZ}^{\mathrm{adj}},\quad
Q^{\mathrm{true}}.
\]

The original `synt_eval` implementation is unchanged.

## Build

```bash
uv run python -m inspect_eval.build_proximal_datasets \
  --name adjustment_small \
  --n_datasets 12 \
  --n_max 512 \
  --n_q 32 \
  --mc_cate 1000 \
  --seed 20260902
```

Outputs are written below `data/inspect_eval/<name>/` by default. The standard
observational episodes and true MC CATE target are in `val/`. The adjustment
ladder is stored per episode in `frozen_meta.pt`, in both raw outcome units and
head-standardized units:

- `Q_naive_X`, `Q_naive_X_std`
- `Q_adj_W`, `Q_adj_W_std`
- `Q_adj_Z`, `Q_adj_Z_std`
- `Q_adj_WZ`, `Q_adj_WZ_std`
- `Q_true`, `Q_true_std`

Context entries are not stored in these vectors: every sidecar vector is
aligned with `query_index` and has length `n_q`.

## Export to the benchmark

The inspection values can be scored and plotted like benchmark methods:

```bash
uv run python -m inspect_eval.export_ladder_predictions
uv run benchmarks/run.py --config benchmarks/config_inspect.yaml
uv run benchmarks/plots.py --path data/inspect_eval/adjustment_small/bench
```

The exporter reads the standardized sidecar fields and writes
`Q-{X-naive,W-adj,Z-adj,WZ-adj,true}_preds.npz` below `bench/preds/`.
`config_inspect.yaml` contains corresponding `cls: Precomputed` entries. Such
entries are cache-only: the benchmark loads them even under `--force` and gives
a direct error naming the exporter if a file is missing.

## Estimation

For a proxy adjustment set `C`, the target is

\[
Q_C^{\mathrm{adj}}(x)
= E_{C\mid x}\{E[Y\mid A=1,x,C]-E[Y\mid A=0,x,C]\}.
\]

The simulator exposes the fine propensity
`pi = P(A=1 | U,X,Z,...)`. The observed-set propensity is estimated as

\[
e_C(X,C)=E[\pi\mid X,C]
\]

using deterministic cross-fitted histogram gradient boosting with `pi` as a
soft target. The adjustment functional is then evaluated with the
self-normalized IPW identity

\[
E\left[\frac{\pi Y_1}{e_C(X,C)}
-\frac{(1-\pi)Y_0}{1-e_C(X,C)}\mid X=x\right].
\]

`Q_naive_X` does not require a nuisance fit because every replicate group holds
`X=x` exactly. `Q_true` is the ordinary paired Monte Carlo mean of `Y1-Y0` and
is checked against `MU1-MU0` before writing.

The propensity regression is part of the Monte Carlo approximation, not a
claim that the adjustment functional itself is identified as causal. Its
settings participate in the set digest and are exposed through the
`--adjustment_*` arguments. A production analysis should check stability across
replicate counts and nuisance settings before treating the resulting bias gaps
as settled.
