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
