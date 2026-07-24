# Appendix B reproduction commands

Maps every table/figure currently used (or planned) in `docs/latex/appendices/B_prior.tex`
to the exact `analysis_prior` CLI invocation that produces it. See
[`docs/prior_appendix_plan.md`](../docs/prior_appendix_plan.md) for the full status/tracking doc —
this file is just the "how do I regenerate artifact X" cheat sheet.

All commands run from the repo root:

```
uv run python -m analysis_prior.main <args>
```

Add `--force` to regenerate the cache instead of reusing it, and `--n-jobs -1` to parallelise
generation across all cores.

---

## `PriorSpec`, `SpecFamily`, and how caching works

`config.py` defines two dataclasses that drive everything above:

- **`PriorSpec`** is the single source of truth for "which datasets are we analysing" —
  dimension ranges, `seq_len`, `prior_type`, `mcmc_cate`, HP overrides, etc. It's JSON-serialisable
  and hashes to a stable cache key. `--spec <name>` looks one up from `config.PRESETS` (e.g.
  `general_ite`, `smoke`, `no_u_augment`).
- **`SpecFamily`** is a base `PriorSpec` plus a dict of named interventions (small field overrides)
  used for comparison studies — "what does changing one knob do to the prior?" `--family <name>`
  looks one up from `config.FAMILIES` (e.g. `dim_ablation`, which compares `triangular` /
  `uniform` / `unenforced` dimension sampling against the same `general_ite` base). Each
  intervention resolves to its own `PriorSpec` via `SpecFamily.resolve()`.

**Cache keys are content-addressed, not name-addressed.** A spec's `name` and `n_datasets` are
cosmetic — they don't affect what any individual dataset looks like — so they're excluded from the
cache key. `PriorSpec.cache_key` is the hash of the *generative* fields only (`generative_digest()`
in `config.py`). Practically:

- Two specs that differ only by `name` or `n_datasets` share the same cache file under
  `cache/<hash>.pt` (+ a `.json` sidecar with the generative spec, for `--list-caches`).
- Each dataset's seed is derived from `(generative_digest, index)`, not from `n_datasets`, so
  dataset `i` is bit-identical regardless of how many datasets were requested. Asking for fewer
  just takes a prefix; asking for more (`load_or_generate` in `generate.py`) generates only the
  missing indices and appends them — existing datasets are never touched or regenerated.
- Bumping `CACHE_FORMAT_VERSION` in `config.py` invalidates every existing cache file (it's
  folded into the hash), for when a generation-code change would make an old cache wrong instead
  of just differently-labelled.

This is why `--family dim_ablation`'s `triangular` variant (the identity intervention, `{}`) and
`--spec general_ite` resolve to the *same* cache file: they're generatively identical specs that
only differ in name.

### Presets and families (current)

| preset | n | seq_len | K (`mcmc_cate`) | notes |
|---|---|---|---|---|
| `default` | 64 | 2048 | 100 | mirrors training runs |
| `general_ite` | **1024** | 2048 | **1** | ITE-phase workhorse; base of `dim_ablation` and `cat_ablation` |
| `smoke` | 4 | 100 | 8 | fast CI/dev |
| `no_u_augment` | 64 | 2048 | 100 | ablation: `confounding_dim_control=False` (f_A/f_Y see raw full U) |
| `no_categorization` | 64 | 2048 | 100 | ablation: `cat_col_prob` pinned to 0 |

| family | base | interventions |
|---|---|---|
| `dim_ablation` | `general_ite` | `triangular` (identity), `uniform` (`dim_U_distribution=uniform`), `unenforced` (`enforce_completeness_heuristic=False`) |
| `cat_ablation` | `general_ite` | `full` (identity), `ablation` (`enforce_categorical_completeness=False`), `baseline` (`corrupt_U=False`, `enforce_categorical_completeness=False`) |
| `smoke_family` | `smoke` | `triangular`, `uniform` |

---

## Implemented and wired into the tex

Four analyses are registered today (`analyses/__init__.py`): `dimensions`, `dimension_comparison`,
`hyperparameters`, `categorical_completeness`. The subsections below match the *current*
`B_prior.tex` numbering (B.2 = "Generative Model, Step-by-Step").

### B.2.1 Hyperparameters — `tab:prior_fixed_hp`, `tab:prior_sampled_hp`

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis hyperparameters` |
| **Config group** | preset `general_ite` (`config.PRESETS["general_ite"]`) |
| **Analysis module** | `analyses/hyperparameters.py` → `HyperparametersAnalysis` (`name="hyperparameters"`) |
| **Output dir** | `results/general_ite/hyperparameters/` |
| **Table outputs** | `sampled_hyperparameters.tex` → `tab:prior_sampled_hp`; `fixed_hyperparameters.tex` → `tab:prior_fixed_hp` |
| **Appendix location** | §B.2.1 "Hyperparameters: Fixed vs Sampled" (`app:prior_hyperparams`) |

This analysis is spec-only (reads the resolved HP config, not generated data), so any `--spec`
works and no dataset cache is needed.

### B.2.2 Dimensions & Dimension Heuristic — `fig:prior_dim_marginals`, `tab:prior_dim_sampling`

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --family dim_ablation --analysis dimension_comparison` |
| **Config group** | family `dim_ablation` (base preset `general_ite`, variants `triangular` / `uniform` / `unenforced`) |
| **Analysis module** | `analyses/dimensions.py` → `DimensionComparisonAnalysis` (`name="dimension_comparison"`) |
| **Output dir** | `results/dim_ablation/dimension_comparison/` |
| **Figure output** | `dimension_marginals.pdf` → `fig:prior_dim_marginals` |
| **Table output** | `dimension_sampling.tex` → `tab:prior_dim_sampling` |
| **Appendix location** | §B.2.2 "Dimensions & Dimension Heuristic" (`app:prior_dimensions`) |

### B.2.4 Post-process (numeric completeness) — `fig:prior_dimension_joint_heatmaps`

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --family dim_ablation --analysis dimension_comparison` |
| **Config group** | family `dim_ablation` (same run as B.2.2 — one invocation produces both) |
| **Analysis module** | `analyses/dimensions.py` → `DimensionComparisonAnalysis` (`name="dimension_comparison"`) |
| **Output dir** | `results/dim_ablation/dimension_comparison/` |
| **Figure output** | `dimension_joint_heatmaps_comparison.pdf` → `fig:prior_dimension_joint_heatmaps` |
| **Appendix location** | §B.2.4 "Post-Process" (`app:prior_postprocess`), "Numeric completeness" paragraph |

> **Removed:** the numeric survival table (`completeness_survival_unenforced.tex` →
> `tab:prior_completeness_survival`) is **no longer rendered** — its rendering was deleted in
> commit `9bcd8db`, and the tex no longer `\input`s it. Any `completeness_survival_unenforced.tex`
> / `.csv` still sitting under `results/dim_ablation/dimension_comparison/` are stale leftovers and
> can be deleted; a fresh run will not regenerate them.

### B.2.4 Post-process (categorical completeness) — `fig:prior_category_count_marginals`, `fig:prior_proxy_stats_by_u_cardinality`, `fig:prior_proxy_composition_vs_u_demand`, `tab:prior_categorical_completeness_survival`

Discrete analogue of the dimension heuristic: a proxy block is *categorically complete* w.r.t.
`U` when every column of `U` can be matched one-to-one to a proxy column that is at least as
informative (continuous dominates any finite categorical; a categorical column dominates another only
if its category count is `>=`). Since the categorisation metadata isn't persisted in the cache, each
block's profile is reconstructed from realised values (a column with `<= max_categories` distinct
values counts as categorical) — see the module docstring in `analyses/categorical.py` for why that's
unambiguous at the shipped cap.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --family cat_ablation --analysis categorical_completeness` |
| **Config group** | family `cat_ablation` (base preset `general_ite`, variants `full` / `ablation` / `baseline` — enforcement and `U`-categorisation toggled via `enforce_categorical_completeness` / `corrupt_U`) |
| **Analysis module** | `analyses/categorical.py` → `CategoricalCompletenessAnalysis` (`name="categorical_completeness"`) |
| **Output dir** | `results/cat_ablation/categorical_completeness/` |
| **Figure outputs** | `category_count_marginals.pdf` → `fig:prior_category_count_marginals`; `proxy_stats_by_u_cardinality.pdf` → `fig:prior_proxy_stats_by_u_cardinality`; `proxy_composition_vs_u_demand.pdf` → `fig:prior_proxy_composition_vs_u_demand` |
| **Table output** | `categorical_completeness_survival.tex` → `tab:prior_categorical_completeness_survival` |
| **Appendix location** | §B.2.4 "Post-Process" (`app:prior_postprocess`), paragraphs "Categorical completeness" / "Enforcement scales with `U`'s demand" / "Survival" |

The composition and cardinality figures (`proxy_composition_vs_u_demand`, `proxy_stats_by_u_cardinality`)
only plot the categorised-`U` arms (`full`, `ablation`) — `baseline` has a continuous `U` by
construction and is dropped from those two views, though it still appears in the category-marginals
figure and the survival table.

---

## Implemented but not currently referenced in the tex

### Single-spec dimension audit (no comparison)

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis dimensions` |
| **Config group** | preset `general_ite` (any single preset works — swap `--spec`) |
| **Analysis module** | `analyses/dimensions.py` → `DimensionsAnalysis` (`name="dimensions"`) |
| **Output dir** | `results/<spec-name>/dimensions/` |
| **Outputs** | `dimension_histograms.*`, `dimension_joint_heatmaps.*` (single-spec marginal/joint dim plots), `completeness_heuristic.tex` (numeric-survival-by-`d_U` table) |
| **Appendix location** | none — superseded by the `dimension_comparison` family run above |

---

## Planned, not yet implemented (Tracks 1–3 of the appendix plan)

No CLI command exists yet — the modules below are named in
[`docs/prior_appendix_plan.md`](../docs/prior_appendix_plan.md) §3/§4 as future `analyses/`
additions, not currently in `analyses/__init__.py`'s registry (the four listed above are all that
is registered today). Listed here so the mapping is complete once they land.

| Appendix § | Label | Planned module (`name=`) | Planned config group | Target artifact | Plan track |
|---|---|---|---|---|---|
| B.2.3 mechanism inputs | `app:prior_mechanisms` | `mechanism_inputs` | `general_ite` | `fig:prior_mechanism_signal_share` | T1.1 |
| B.2.5 treatment walkthrough | `app:prior_treatment` | `treatment_overlap` | `general_ite` (1 SCM) | `fig:prior_treatment_walkthrough` | T1.2 |
| B.2.6 outcome / heterogeneity | `app:prior_outcome` | `heterogeneity` | K-sweep family (K>1) | `fig:prior_heterogeneity_spread`, `fig:prior_cate_vs_ite_K` | T1.3 |
| B.2.x visualize the prior | `app:prior_visualize` | `prior_overview` | `general_ite` | `fig:prior_summary_metrics`, `fig:prior_feature_distributions`, `fig:prior_feature_pairs` | T1.4 |
| B.3.1 Overlap | `app:prior_overlap` | `treatment_overlap` | `general_ite` | Fig audit(b) + overlap table | T2.1 |
| B.3.2 Exclusion | `app:prior_exclusion` | `exclusion` | `general_ite` | Fig audit(a) | T2.4 |
| B.3.3 Confounding strength | `app:prior_confounding` | `confounding_strength` | `general_ite` | Fig audit(c), `tab:prior_confounding` | T2.2 |
| B.3.4 Proxy relevance | `app:prior_relevance` | `proxy_relevance` | `general_ite` | Fig audit(d) | T2.3 |
| B.3.5 Ill-posedness | `app:prior_illposed` | `illposedness` | `general_ite` | `fig:prior_illposedness` | T2.5 |
| B.3.6 Identification stress test | `app:prior_identification` | `identification` | `general_ite` | `fig:prior_identification` | T2.6 |
| B.3 assembly | `app:prior_audit` | `audit_summary` | `general_ite` (K=1) | 4-panel audit fig, `tab:prior_audit_summary` | T2.7 |
| B.5 MC error | `app:prior_extensions` | `mc_error` | K-sweep family, K∈{1,5,25,100,200,5000} | `fig:mc_error` | T3.1 |
| B.5 structural coverage | `app:prior_extensions` | *(hand-written, no module)* | — | `tab:prior_structural_coverage` | T3.2 |
| B.5 memory budget | `app:prior_extensions` | `mem_budget` | `default` (T=2048) | `tab:prior_mem_budget` | T3.3 |
| B.4 downstream ablation | `app:prior_downstream` | *(needs training runs)* | — | `tab:prior_downstream_ablation` | T3.4 (deferred) |

The regression-based audits (B.3.2/B.3.3/B.3.4) share a **cross-fitted regression helper** (plan
T0.1) that does not exist yet — build it before those modules.

> **Removed module.** `analyses/confounding.py` (`confounding_dim`, added in `4801ef3`) was **deleted**
> in `a77c467`. It has no replacement in the registry; the old analytic `eq:conf_dim` /
> `tab:prior_conf_dim` / `fig:prior_conf_dim` deliverables are dropped from the plan and re-scoped as
> the empirical `mechanism_inputs` analysis (T1.1). Any `results/*/confounding_dim/` and
> `results/*/input_sampler/` directories on disk are **orphans** (no module produces them) and can be
> deleted.

**Note on `general_ite` vs. final data.** All the `general_ite`-based commands above (K=1) are fine
for drafting the ITE-phase (B.2/B.3) artifacts. For the CATE-phase artifacts that need `K>1`
(`mc_error`, `heterogeneity`), reuse the same underlying SCMs as `general_ite` — a base spec
identical apart from `mcmc_cate` — rather than generating a second, unrelated batch, so Fig A8 and
the B.2.6 figures share the same datasets.

---

## Reproducing the appendix: full command list

Everything below regenerates every table/figure currently wired into `B_prior.tex` (see the
per-artifact sections above for what each command produces and where it lands). Run from the repo
root:

```
# B.2.1 Hyperparameters — tab:prior_fixed_hp, tab:prior_sampled_hp
PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis hyperparameters

# B.2.2 Dimensions + B.2.4 numeric completeness — fig:prior_dim_marginals, tab:prior_dim_sampling,
# fig:prior_dimension_joint_heatmaps
PYTHONPATH=. uv run python -m analysis_prior.main --family dim_ablation --analysis dimension_comparison

# B.2.4 Categorical completeness — fig:prior_category_count_marginals,
# fig:prior_proxy_stats_by_u_cardinality, fig:prior_proxy_composition_vs_u_demand,
# tab:prior_categorical_completeness_survival
PYTHONPATH=. uv run python -m analysis_prior.main --family cat_ablation --analysis categorical_completeness

# B.2.3 Mechanism input composition — fig:prior_mechanism_input_composition,
# tab:prior_mechanism_inputs.
# Needs the per-dataset mechanism diagnostics: a cache written before they
# existed must be regenerated once with --force (same data, diagnostics added).
PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis mechanism_inputs
```

Add `--force` to any of the three to regenerate the cache instead of reusing it, and `--n-jobs -1`
to parallelise generation across all cores.

The `hyperparameters`, `dim_ablation`, and `cat_ablation` outputs land under the paths the tex
`\input`s / `\includegraphics` from `assets/prior/`. If your build reads figures straight from
`results/`, no copying is needed; otherwise copy/symlink the PDFs into the appendix figures tree
before building LaTeX.

Everything in "Planned, not yet implemented" above has no CLI command yet — those modules don't
exist in `analyses/__init__.py`'s registry.

## Discovery commands

```
uv run python -m analysis_prior.main --list-analyses   # registered analyses
uv run python -m analysis_prior.main --list-caches     # cached specs on disk
uv run python -m analysis_prior.main --list-families   # registered comparison families
```
