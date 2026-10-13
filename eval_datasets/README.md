# Evaluation datasets

The evaluation builders and offline scoring tools live under `eval_datasets/`:

- `prior_eval/`: frozen datasets sampled from the training prior, with verification,
  posterior scoring, stratification, and context sweeps.
- `semisynt_eval/`: empirical CSV sources combined with known causal mechanisms.
- `synt_eval/`: synthetic proximal mechanisms with optional Monte Carlo CATE targets.

Run module commands from the repository root:

```sh
uv run python -m eval_datasets.prior_eval.build --help
uv run python -m eval_datasets.semisynt_eval.build --list
uv run python -m eval_datasets.synt_eval.build_proximal_datasets --help
```

The scripts in `scripts/` use these module paths. The semi-synthetic recipe is
`eval_datasets/semisynt_eval/config.yaml`, and its input CSVs live in
`eval_datasets/semisynt_eval/datasets/` (download with
`uv run python scripts/download_tabular_benchmark.py`).

Generated datasets remain under `data/prior_eval/`, `data/semisynt_eval/`, and
`data/synt_eval/`. Their formats, content digests, and the training flags
`--prior_eval_dir` and `--synt_eval_dir` are unchanged. Existing datasets can be
loaded without rebuilding them.

Python imports and commands that previously started with `prior_eval`,
`semisynt_eval`, or `synt_eval` now start with `eval_datasets.`; for example,
`python -m prior_eval.score` becomes `python -m eval_datasets.prior_eval.score`.
