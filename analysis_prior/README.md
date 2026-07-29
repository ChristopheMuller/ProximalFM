# `analysis_prior` — Appendix B artifacts

Generates the figures/tables of [`docs/latex/appendices/B_prior.tex`](../docs/latex/appendices/B_prior.tex)
from the synthetic prior (datasets generated with all latents kept). Never trains anything.
See [`docs/prior_appendix_plan.md`](../docs/prior_appendix_plan.md) for status/tracking; this file is
the "how do I regenerate artifact X" cheat sheet.

```
uv run python -m analysis_prior.main --spec <preset> --analysis <name> [--n-datasets N] [--n-jobs -1] [--force]
uv run python -m analysis_prior.main --list-analyses | --list-caches | --list-families
```

## Specs, families, caching

`config.py` holds the two objects that drive everything:

- **`PriorSpec`** (`--spec`, from `config.PRESETS`) — a reproducible prior configuration: dimension
  ranges, `seq_len`, `prior_type`, `mcmc_cate`, HP overrides.
- **`SpecFamily`** (`--family`, from `config.FAMILIES`) — a base spec plus named single-knob
  interventions, for comparison studies; each resolves to its own `PriorSpec`.

**Caching is content-addressed on generative fields only.** `name` and `n_datasets` are excluded from
`PriorSpec.cache_key` (`generative_digest()`), so:

- specs differing only in those share one cache file `cache/<hash>.pt` (+ `.json` sidecar);
- each dataset's seed is `(digest, index)`, so dataset `i` is bit-identical however many were
  requested — asking for fewer takes a prefix, asking for more appends only the missing indices;
- bumping `CACHE_FORMAT_VERSION` invalidates every cache, for when a generator change would make old
  caches *wrong* rather than merely differently labelled;
- adding a new per-dataset diagnostic to the cache payload needs one `--force` regeneration (data
  stays bit-identical, the diagnostic is simply recorded alongside).

Two more cache layers sit under `cache/`: `regression/` (cross-fitted out-of-fold predictions from
[`regression.py`](regression.py), so re-renders never refit) and `frames/`
([`frame_cache.py`](frame_cache.py), the assembled analysis frame — a re-render becomes one file read;
its key folds in a source hash, so editing a measurement invalidates automatically).

Presets: `general_ite` (n=1024, `seq_len=2048`, **K=1**) is the workhorse and the base of both
families; `mc_targets` (n=100, **K=100**) is the only `K>1` preset; `default` mirrors training runs;
`smoke` is for dev; `no_u_augment` / `no_categorization` are ablations. Families: `dim_ablation`
(`triangular` / `uniform`) and `cat_ablation` (`default` / `no_enforcement`), both on `general_ite`.

## Artifacts

Outputs land in `results/<spec-or-family>/<analysis>/` (gitignored, fully regenerable) alongside a
`summary.json` carrying the numbers quoted in the prose.

| § | Module → analysis (`--analysis`) | Spec/family | Outputs → label |
|---|---|---|---|
| B.2.1 | `hyperparameters.py` → `hyperparameters` | `--spec general_ite` | `structural_hyperparameters.tex` → `tab:prior_structural_hp`; `sampled_hyperparameters.tex` → `tab:prior_sampled_hp`; `fixed_hyperparameters.tex` → `tab:prior_fixed_hp` |
| B.2.2 + B.2.4 | `dimensions.py` → `dimension_comparison` | `--family dim_ablation` | `dimension_marginals.pdf` → `fig:prior_dim_marginals`; `dimension_sampling.tex` → `tab:prior_dim_sampling`; `dimension_joint_heatmaps_comparison.pdf` → `fig:prior_dimension_joint_heatmaps` |
| B.2.3 | `mechanism_inputs.py` → `mechanism_inputs` | `--spec general_ite` | `mechanism_input_composition.pdf` → `fig:prior_mechanism_input_composition`; `mechanism_input_composition_table.tex` → `tab:prior_mechanism_inputs` |
| B.2.3 | `mechanism_outputs.py` → `mechanism_outputs` | `--spec general_ite` | `mechanism_output_dependence.pdf` → `fig:prior_mechanism_output_dependence`; `mechanism_output_dependence_table.tex` → `tab:prior_mechanism_outputs` |
| B.2.4 | `categorical.py` → `categorical_completeness` | `--family cat_ablation` | `category_count_marginals.pdf` → `fig:prior_category_count_marginals`; `proxy_enforcement_vs_u_demand.pdf` → `fig:prior_proxy_enforcement_vs_u_demand`; `categorical_completeness_survival.tex` → `tab:prior_categorical_completeness_survival` |
| B.2.5 | `treatment.py` → `treatment_walkthrough` | `--spec general_ite` | `treatment_walkthrough.pdf` → `fig:prior_treatment_walkthrough` |
| B.2.6 | `outcome.py` → `heterogeneity` | `--spec general_ite` | `heterogeneity_spread.pdf` → `fig:prior_heterogeneity_spread` |
| B.2.6 | `mc_targets.py` → `mc_targets` | `--spec mc_targets` | `cate_vs_ite.pdf` → `fig:prior_cate_vs_ite`; `mc_signal_share.tex` → `tab:prior_mc_signal_share` |
| B.2.x | `overview.py` → `prior_overview` | `--spec general_ite` | `prior_feature_distributions.pdf`, `prior_feature_pairs.pdf`, `prior_feature_pairs_cross.pdf` → `fig:prior_feature_{distributions,pairs,pairs_cross}` |
| B.3.1 | `treatment.py` → `treatment_overlap` | `--spec general_ite` | `overlap.pdf` → `fig:prior_overlap` |
| B.3.2 | `exclusion.py` → `exclusion` | `--spec general_ite` | `exclusion.pdf` → `fig:prior_exclusion`; `exclusion_table.tex` → `tab:prior_exclusion` |
| B.3.3 | `confounding.py` → `confounding_strength` | `--spec general_ite` | `confounding_ladder.pdf` → `fig:prior_confounding`; `confounding_vs_share.pdf` → `fig:prior_confounding_share`; `confounding_table.tex` → `tab:prior_confounding` |
| B.3.4 | `proxy_relevance.py` → `proxy_relevance` | `--spec general_ite` | `proxy_relevance.pdf` → `fig:prior_relevance`; `proxy_relevance_table.tex` → `tab:prior_relevance` |
| — | `dimensions.py` → `dimensions` | any spec | single-spec `dimension_histograms.*`, `dimension_joint_heatmaps.*`, `completeness_heuristic.tex`; superseded by `dimension_comparison` |

Cost: everything above is seconds-to-minutes except the regression-based audits, which are
`rf`-based (~1.5 s/dataset, ≈25 min over the full cache) — plus `mechanism_outputs`, which is TabICL
and needs `--n-datasets` capped. Set `analysis.model = "tabicl"` on the B.3 audits for a robustness
pass.

## Regenerating everything

From the repo root. Add `--force` to rebuild the dataset cache, `--n-jobs -1` to parallelise
generation.

```bash
# B.2.1 Parameters of the prior (spec-only: reads the HP config, no dataset cache needed)
uv run python -m analysis_prior.main --spec general_ite --analysis hyperparameters

# B.2.2 Dimensions + B.2.4 numeric completeness (one run produces both)
uv run python -m analysis_prior.main --family dim_ablation --analysis dimension_comparison

# B.2.4 Categorical completeness
uv run python -m analysis_prior.main --family cat_ablation --analysis categorical_completeness

# B.2.3 Mechanism input composition (needs the mechanism-diagnostics payload)
uv run python -m analysis_prior.main --spec general_ite --analysis mechanism_inputs

# B.2.3 Mechanism output dependence (cross-fitted TabICL; uses EVERY dataset in
# the cache, so cap it — the default 1024 is ~8 h. 256 is served as a prefix of
# the existing cache, no regeneration.)
uv run python -m analysis_prior.main --spec general_ite --n-datasets 256 --analysis mechanism_outputs

# B.2.5 Treatment assignment walkthrough (needs the raw f_A logits in the cache)
uv run python -m analysis_prior.main --spec general_ite --analysis treatment_walkthrough

# B.2.6 Heterogeneity dial (gamma is drawn inside the generator, so the prior's
# own uniform draw *is* the sweep; K=1 is fine)
uv run python -m analysis_prior.main --spec general_ite --analysis heterogeneity

# B.2.6 Monte Carlo CATE targets (the only K>1 command; generates its own cache)
uv run python -m analysis_prior.main --spec mc_targets --analysis mc_targets --n-jobs -1

# B.2.x Visualize the prior
uv run python -m analysis_prior.main --spec general_ite --analysis prior_overview

# B.3.1 Overlap (pure summary of cached A_probs, no regression)
uv run python -m analysis_prior.main --spec general_ite --analysis treatment_overlap

# B.3.2 Exclusion restrictions
uv run python -m analysis_prior.main --spec general_ite --analysis exclusion

# B.3.3 Confounding strength (needs the mechanism-diagnostics payload)
uv run python -m analysis_prior.main --spec general_ite --analysis confounding_strength

# B.3.4 Proxy relevance
uv run python -m analysis_prior.main --spec general_ite --analysis proxy_relevance
```

Fast dev demo of the `mechanism_outputs` figure (a few datasets, random forest, ~seconds; writes to
`results/general_ite/mechanism_outputs_small/`, not the paper path):

```bash
uv run python -m analysis_prior.examples.mechanism_outputs_small
```

The tex `\input`s / `\includegraphics` from `assets/prior/`; if your build doesn't read straight from
`results/`, copy or symlink the outputs there before building LaTeX.
