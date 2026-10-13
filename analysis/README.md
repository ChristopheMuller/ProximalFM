# Analysis tools

The analysis packages live together under `analysis/`. Run module commands from
the repository root with the `analysis.` prefix:

```sh
uv run python -m analysis.analysis_prior.main --list-analyses
uv run python -m analysis.analysis_eval.learning_curves
uv run python -m analysis.analysis_training.hparam_table
uv run python -m analysis.analysis_synthetic_proxy_sweep_linear.plot_results
```

- [Prior analysis](analysis_prior/README.md): prior diagnostics and appendix artifacts.
- [Evaluation analysis](analysis_eval/README.md): benchmark figures and tables.
- [Training analysis](analysis_training/README.md): training figures and tables.
- [Synthetic proxy sweep](analysis_synthetic_proxy_sweep_linear/README.md): paired proxy-strength experiments.

Each package keeps its existing data, caches, and results beside its scripts.
Repository inputs such as `data/`, `scripts/`, `benchmarks/`, and `whole_1/`
remain relative to the repository root. Explicit paths passed on the command
line remain relative to the working directory.
