# Linear-Gaussian proxy inspection

`inspect_eval_linear` is a controlled companion to [`inspect_eval`](../inspect_eval/README.md).
It replaces the twelve hand-written mechanisms with one scalar linear-Gaussian
SCM and varies only the independent measurement noise in `W` and `Z`. This
makes the behavior of ordinary adjustment as proxy information changes directly
visible.

The mechanism is

\[
X\sim N(0,1),\qquad U=\beta_{UX}X+\epsilon_U,
\]
\[
W=\lambda U+\epsilon_W,\qquad Z=\lambda U+\epsilon_Z,
\]
\[
A=1\{\alpha_0+\alpha_X X+\alpha_U U+\epsilon_A>0\},
\]
\[
Y(a)=\beta_0+\beta_X X+\beta_U U+a(\tau_0+\tau_X X)+\epsilon_Y.
\]

All errors are independent Gaussian variables. W and Z are pure measurements:
they do not directly enter treatment or outcome. The exact CATE is therefore
`tau_0 + tau_x * X`. The treatment uses a latent probit model, allowing
`P(A=1 | X,C)` for `C` equal to W, Z, or WZ to be computed analytically rather
than fitted with a nuisance model.

## Build and inspect

```bash
uv run python -m inspect_eval_linear.build \
  --name proxy_sweep \
  --n_datasets 120 \
  --n_max 512 \
  --n_q 128 \
  --mc_cate 1000 \
  --proxy_noise_stds 0 0.25 0.5 1 2 4

uv run python -m eval_datasets.prior_eval.verify \
  --dir data/inspect_eval_linear/proxy_sweep

uv run python -m inspect_eval_linear.analyze_noise_sweep \
  --root data/inspect_eval_linear/proxy_sweep
```

To add the shared fitted methods to the same reliability plot, run the benchmark
and regenerate the analysis:

```bash
uv run benchmarks/run.py --config benchmarks/config_methods.yaml \
  --val-dir data/inspect_eval_linear/proxy_sweep/val
uv run python -m inspect_eval_linear.analyze_noise_sweep \
  --root data/inspect_eval_linear/proxy_sweep
```

The analysis automatically discovers non-Q prediction caches in `bench/preds/`,
converts their context-standardized treatment effects back to raw outcome units,
and draws them as dashed lines alongside the solid population-functional curves.

The analysis writes:

- `analysis/proxy_noise_summary.csv`, with bias, mean absolute bias, and RMSE
  against CATE* for every Q target and noise level;
- `analysis/proxy_noise_sweep.png`, plotting RMSE and signed bias against the
  conditional proxy reliability
  `lambda^2 sigma_U^2 / (lambda^2 sigma_U^2 + sigma_proxy^2)`.

The default noise grid spans reliability 1 (a perfect measurement) through
approximately 0.059. At zero noise, W or Z identifies U exactly and every
proxy-adjusted Q equals CATE*. With finite noise, WZ combines two independent
measurements and has lower posterior uncertainty about U than either one alone.

## Stored targets

The standard observational episode and Monte Carlo potential-outcome targets
are written to `val/`. The aligned query-only inspection targets live in each
episode of `frozen_meta.pt`, in raw and context-standardized outcome units:

- `Q_naive_X`
- `Q_adj_W`
- `Q_adj_Z`
- `Q_adj_WZ`
- `Q_true`

For `C` equal to W, Z, or WZ, the adjustment functional is

\[
Q_C(x)=E_{C\mid x}\{E[Y\mid A=1,x,C]-E[Y\mid A=0,x,C]\}.
\]

It is evaluated with the same self-normalized IPW identity as `inspect_eval`,
but its observed-set propensity is the exact Gaussian/probit conditional
probability. Monte Carlo error remains only in the outer integration over the
latent and proxy draws.

Every structural coefficient is exposed as a builder flag and participates in
the set's content digest. A saved `config.yaml` can be replayed with `--config`;
explicit CLI flags override it.

## Benchmark export

```bash
uv run python -m inspect_eval_linear.export_ladder_predictions \
  --root data/inspect_eval_linear/proxy_sweep
```

This creates the standard cache files under `bench/preds/` for `Q-X-naive`,
`Q-W-adj`, `Q-Z-adj`, `Q-WZ-adj`, and `Q-true`.
