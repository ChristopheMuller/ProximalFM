# Empirical-root proximal evaluation sets

This package builds semi-synthetic proximal datasets whose paired `(X, U)` rows
come from arbitrary CSV files, while `W`, `Z`, `A`, and `Y` come from a known
synthetic SCM. Every `*.csv` placed directly in `semi_synt_eval/datasets/` is
discoverable without adding Python code:

```bash
uv run python -m semi_synt_eval.build --list
```

## Quick start

```bash
uv run python -m semi_synt_eval.build \
  --csv bank \
  --name bank_linear_v1 \
  --drop_columns y \
  --x_columns age,education,marital \
  --u_columns balance,housing,loan \
  --mechanism linear \
  --n_max 1024 \
  --n_q 256 \
  --seed 20260903
```

Without explicit column lists, the loader converts every usable CSV column to
one numeric column, keeps at most `--max_features`, and creates a deterministic
X/U split controlled by `--u_fraction` and `--split_seed`. Numeric missing
values are median-imputed; categorical values use stable sorted category codes.
This generic encoding is meant as a reproducible baseline. A scientifically
chosen split should use `--x_columns`, `--u_columns`, and `--drop_columns`.

The output lands in `data/real_eval/<name>/` and uses the standard sparse
`val/` format:

```yaml
data:
  source: load
  val_dir: data/real_eval/bank_linear_v1/val
```

The resolved X/U assignment, source SHA-256, exact target kind, and all
generative settings are recorded in `manifest.json` and `config.yaml`.

## Causal contract

The mechanism interfaces enforce

```text
W, Z <- (X, U)
A    <- (X, U, Z)
Y(a) <- (X, U, W, a)
```

The bundled `linear` and `nonlinear` families use
`Y(1) = Y(0) + tau(X)`, so `tau(X)` is the exact CATE and is written directly
to each batch as `CATE`. There is no Monte Carlo parameter, no replication, and
no claim that the builder can sample `U | X`.

A future mechanism with no analytic CATE declares
`target_capability = "ite_only"`, returns no CATE, and is evaluated using its
realized `Y1 - Y0`. If a future source provides a genuine conditional model for
`U | X`, Monte Carlo CATE would be a separate capability rather than an
overload of this one.

## Extending mechanisms

Implement the small interface in `mechanisms/base.py`, register its builder in
`mechanisms/basic.py` (or a new registry module), and keep the method signatures
unchanged. The signatures make it impossible for W to enter treatment or Z to
enter the outcome accidentally.

The builder and generator are self-contained within `semi_synt_eval`; their only
project-level dependency is the canonical low-level `write_batch_file` format
writer used by every benchmark dataset producer.
