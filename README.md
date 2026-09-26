# ProximalFM

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ProximalFM estimates **conditional average treatment effects (CATEs)** when treatment and outcome share an unobserved confounder. It uses observed covariates $X$, a treatment proxy $Z$, an outcome proxy $W$, treatment $A$, and outcome $Y$ to estimate $\tau(x) = \mathbb{E}[Y(1)-Y(0)\mid X=x]$.

The model learns from synthetic datasets and makes predictions on a new dataset **in context**, without fitting its weights to that dataset.

## How it works

1. A synthetic structural causal model generates datasets with hidden confounding and proxies. For training targets, it averages repeated potential-outcome draws at each $x$ to approximate $\tau(x)$ rather than using a single noisy individual effect. See [`proximalfm/prior/`](proximalfm/prior/).
2. A tabular transformer reads the observed dataset, distinguishes the roles of $X$, $Z$, $W$, and $A$, and predicts effects on query rows. The production loss trains a CATE distribution while accounting for noise in the Monte Carlo target. See [`proximalfm/_model/`](proximalfm/_model/).
3. Fixed synthetic and realistic evaluation sets measure effect accuracy and predictive uncertainty. See [`proximalfm/train/metrics.md`](proximalfm/train/metrics.md) for metric definitions and caveats.

## Get started

From a clone of this repository:

```bash
uv sync
uv run python -m pytest tests/
```

The production training recipe is [`scripts/pretrain_prod.sh`](scripts/pretrain_prod.sh). It expects a backbone checkpoint and built evaluation sets; the corresponding builders are [`scripts/build_prior_eval.sh`](scripts/build_prior_eval.sh) and [`scripts/build_synt_eval.sh`](scripts/build_synt_eval.sh). For training options and defaults, use `uv run python -m proximalfm.train --help` or read [`proximalfm/train/_train_config.py`](proximalfm/train/_train_config.py). The script holds the production flag values; they are not repeated here.

For method comparisons, use [`benchmarks/config_methods.yaml`](benchmarks/config_methods.yaml) and the [benchmark guide](benchmarks/README.md):

```bash
uv run benchmarks/run.py --config benchmarks/config_methods.yaml \
  --val-dir data/synt_eval/<name>/val
```

## More detail

- [Prior audits](analysis_prior/README.md): assumptions, diagnostics, and figures.
- [Training analysis](analysis_training/README.md): run configuration, learning curves, and posterior figures.
- [Evaluation analysis](analysis_eval/README.md): benchmark figures, tables, and posterior calibration.
- [Synthetic proxy sweep](analysis_synthetic_proxy_sweep_linear/README.md): linear study of proxy strength and CATE accuracy.
- [Evaluation metrics](proximalfm/train/metrics.md): what each reported score means.
- [Benchmark guide](benchmarks/README.md): methods, configuration, tables, and plots.
- [Semisynthetic datasets](semisynt_eval/README.md): additional evaluation sets.
