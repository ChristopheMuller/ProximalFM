# Empirical-root proximal evaluation sets

This package builds semi-synthetic proximal datasets whose paired `(X, U)` rows
come from arbitrary CSV files, while `W`, `Z`, `A`, and `Y` come from a known
synthetic SCM. Every `*.csv` placed directly in `semisynt_eval/datasets/` is
discoverable without adding Python code:

```bash
uv run python -m semisynt_eval.build --list
```

## Quick start

```bash
uv run python -m semisynt_eval.build \
  --csv heloc \
  --name heloc_linear_v1 \
  --n_features_x 7 \
  --n_features_u 5 \
  --mechanism linear \
  --n_max 1024 \
  --n_q 256 \
  --seed 20260903
```

[`config.yaml`](config.yaml) is the version-controlled source of truth. It
contains every build default and complete `linear` and `linear_quadratic`
sub-configs. Python code validates this schema but supplies no hidden mechanism
defaults. Pass `--config path/to/recipe.yaml` to build from another complete
recipe; explicit CLI arguments override its `build:` section.

Without explicit column lists, the loader converts every usable CSV column to
one numeric column and selects exactly `--n_features_x` columns for X and
`--n_features_u` columns for U, deterministically under `--split_seed`. The
builder raises if the CSV does not have enough non-constant usable columns.
Numeric missing values are median-imputed; categorical values use stable sorted
category codes. This generic encoding is meant as a reproducible baseline. A
scientifically chosen split should use `--x_columns`, `--u_columns`, and
`--drop_columns`; the explicit lists must match the requested counts.

The output lands in `data/real_eval/<name>/` and uses the standard sparse
`val/` format:

```yaml
data:
  source: load
  val_dir: data/real_eval/heloc_linear_v1/val
```

Every output directory receives a fully resolved `config.yaml` containing the
actual build values, both complete mechanism sub-configs, and the source CSV's
SHA-256. That saved file can be passed back to `--config` directly. The builder
checks the recorded source digest before regenerating, while `manifest.json`
also records the resolved X/U assignment, episode seeds, target kind, and
content digest.

## Causal contract

The mechanism interfaces enforce

```text
W, Z <- (X, U)
A    <- (X, U, Z)
Y(a) <- (X, U, W, a)
```

The bundled `linear` and `linear_quadratic` families use
`Y(1) = Y(0) + tau(X)`, so `tau(X)` is the exact CATE and is written directly
to each batch as `CATE`. There is no Monte Carlo parameter, no replication, and
no claim that the builder can sample `U | X`.

By default, W and Z are normalized after adding their structural noise, the
treatment score is normalized before its logistic link, and the noisy untreated
outcome baseline is normalized before adding `tau(X)` to form `Y(1)`. Thus the
normalization never changes the closed-form treatment effect.

A future mechanism with no analytic CATE declares
`target_capability = "ite_only"`, returns no CATE, and is evaluated using its
realized `Y1 - Y0`. If a future source provides a genuine conditional model for
`U | X`, Monte Carlo CATE would be a separate capability rather than an
overload of this one.

## Extending mechanisms

Each SCM family has its own module: `mechanisms/linear.py` contains the random
linear equations and `mechanisms/linear_quadratic.py` contains the mixed-power
family. In the latter, every coefficient is independently assigned either a
linear or quadratic input term when an episode's mechanism is constructed.
There are no hidden nonlinearities or cross terms.

The repository `config.yaml` is the single generative configuration surface.
Its `mechanisms:` mapping contains one complete sub-config per family, including
the input blocks used by each equation (for example, whether W enters Y),
coefficient distributions and scales, noise levels, normalization switches,
treatment overlap, and quadratic allocation probabilities. Change defaults or
add mechanism-level settings there rather than adding duplicate CLI flags.

To add a family, implement the small interface in `mechanisms/base.py`, put the
SCM in a new mechanism module, add its typed sub-config to `config.py`, and
register it in `mechanisms/__init__.py`. The signatures make it impossible for
W to enter treatment or Z to enter the outcome accidentally.

The builder and generator are self-contained within `semisynt_eval`; their only
project-level dependency is the canonical low-level `write_batch_file` format
writer used by every benchmark dataset producer.
