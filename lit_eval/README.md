# Literature evaluation datasets

`lit_eval` turns published data-generating mechanisms into the same `val/`
format used by the prior and realistic evaluation sets. Each build contains
independent episodes from one mechanism; mechanisms are never mixed within a
set.

## Sverdrup Appendix B

Build the paper's 4,000-context / 2,000-query design with:

```bash
uv run python -m lit_eval.build \
  --dataset sverdrup_appB \
  --name sverdrup_appB_v1 \
  --n_datasets 240 \
  --n_max 4000 \
  --n_q 2000 \
  --mc_cate 2 \
  --seed 20260811
```

The result is written under `data/lit_eval/sverdrup_appB_v1/`. Appendix B has
an analytic paired effect, `exp(X1) - 3 X2`, so two replicates are sufficient
to activate the benchmark's CATE-target contract while producing zero Monte
Carlo variance.

Point a benchmark config at the generated validation directory:

```yaml
data:
  source: load
  val_dir: data/lit_eval/sverdrup_appB_v1/val
```

Then run the benchmark normally:

```bash
uv run benchmarks/run.py --config benchmarks/config_val.yaml --force
```

The build also writes a resolved `config.yaml`, a human-readable
`manifest.json`, and a `frozen_meta.pt` sidecar with a content digest. Re-run a
saved build with `--config data/lit_eval/sverdrup_appB_v1/config.yaml`.

## Wu Appendix H.2

Wu et al.'s high-dimensional experiment has continuous treatment. The adapter
uses the endpoint contrast 1 versus 0 and replaces the paper's continuous law
`A_cont = Lambda(L) + 0.25 U_w` with the new binary law
`P(A=1 | X,Z,U_w) = Lambda(L + 0.25 U_w)`, where
`L = 3 X' beta_x + 3 Z' beta_z`. This retains overlap but is an explicit
adaptation, not an equivalent rewriting. Its analytic target is
`tau*(x) = 2.2 + x_1`, so `mc_cate=2` is sufficient. With `mc_cate=1` the paired
effect is still exact, but the builder intentionally writes no distributional
sidecar and the benchmark treats the set as ITE-only.

Build a set at the paper's `n=1000` sample-size regime with:

```bash
uv run python -m lit_eval.build \
  --dataset wu_appH2 \
  --name wu_appH2_n1000_v1 \
  --n_datasets 240 \
  --n_max 1000 \
  --n_q 128 \
  --mc_cate 2 \
  --option dim_x=100 \
  --option dim_z=10 \
  --option dim_w=10 \
  --seed 20260811
```

The paper also reports `n=2000`; build that as a separate named set with
`--n_max 2000`. The query count is specific to this benchmark harness because
Wu et al. do not define a held-out query split. These binary-CATE results are
not directly comparable to the paper's continuous-dose cMSE values.
The three dimensions are dataset options and therefore become part of the
resolved config and content digest. `dim_x` must be at least 1; `dim_z` and
`dim_w` must each be at least 2. The defaults `100/10/10` reproduce the paper;
other values define dimension-scaled variants of its mechanism.

## Adding a mechanism

1. Add one module under `lit_eval/datasets/` implementing
   `ProximalDataset.sample_episode`.
2. Decorate the class with `@register_dataset` and import it from
   `lit_eval/datasets/__init__.py`.
3. Put every data-changing option in `option_defaults()` so it participates in
   the set's content digest.
4. Add equation-level tests: reconstruct the published residuals and verify
   the stated treatment-effect target, not just tensor shapes.

The shared base handles replicate layout, paired target reduction, validation,
and stacking. The shared builder handles deterministic episode seeds, the
canonical sparse writer, metadata, manifests, and sidecars.
