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

---

## Implemented and wired into the tex

### B.2 Hyperparameters — `tab:prior_fixed_hp`, `tab:prior_sampled_hp`

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis hyperparameters` |
| **Config group** | preset `general_ite` (`config.PRESETS["general_ite"]`) |
| **Analysis module** | `analyses/hyperparameters.py` → `HyperparametersAnalysis` (`name="hyperparameters"`) |
| **Output dir** | `results/general_ite/hyperparameters/` |
| **Table outputs** | `sampled_hyperparameters.tex` → `tab:prior_sampled_hp`; `fixed_hyperparameters.tex` → `tab:prior_fixed_hp` |
| **Appendix location** | §B.2 "Hyperparameters" (`app:prior_hyperparams`) |

### B.2.1 Prior Dimensions — `fig:prior_dim_marginals`, `tab:prior_dim_sampling`

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --family dim_ablation --analysis dimension_comparison` |
| **Config group** | family `dim_ablation` (base preset `general_ite`, variants `triangular` / `uniform` / `unenforced`) |
| **Analysis module** | `analyses/dimensions.py` → `DimensionComparisonAnalysis` (`name="dimension_comparison"`) |
| **Output dir** | `results/dim_ablation/dimension_comparison/` |
| **Figure output** | `dimension_marginals.pdf` → `fig:prior_dim_marginals` |
| **Table output** | `dimension_sampling.tex` → `tab:prior_dim_sampling` |
| **Appendix location** | §B.2.1 "Prior Dimensions" (`app:prior_dimensions`) |

### B.3 Postprocessing — `fig:prior_dimension_joint_heatmaps`, `tab:prior_completeness_survival`

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --family dim_ablation --analysis dimension_comparison` |
| **Config group** | family `dim_ablation` (same run as above — one invocation produces both B.2.1 and B.3 artifacts) |
| **Analysis module** | `analyses/dimensions.py` → `DimensionComparisonAnalysis` (`name="dimension_comparison"`) |
| **Output dir** | `results/dim_ablation/dimension_comparison/` |
| **Figure output** | `dimension_joint_heatmaps_comparison.pdf` → `fig:prior_dimension_joint_heatmaps` |
| **Table output** | `completeness_survival_unenforced.tex` → `tab:prior_completeness_survival` |
| **Appendix location** | §B.3 "Postprocessing" (`app:prior_postprocessing`) |

**Note:** `figures_app/prior/` doesn't exist yet in the LaTeX tree — copy/symlink the two PDFs above
from `results/dim_ablation/dimension_comparison/` before building (Track A item 1 in the plan).

### B.3 Categorical completeness — `fig:prior_category_count_marginals`, `fig:prior_proxy_stats_by_u_cardinality`, `fig:prior_proxy_composition_vs_u_demand`, `tab:prior_categorical_completeness_survival`

Discrete analogue of the dimension heuristic above: a proxy block is *categorically complete* w.r.t.
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
| **Appendix location** | §B.3 "Postprocessing" (`app:prior_postprocessing`), paragraphs "Categorical completeness" / "Enforcement tracks `U`'s categorical demand" / "Survival of categorical completeness" |

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
| **Outputs** | `dimension_histograms.*`, `dimension_joint_heatmaps.*` (marginal/joint dim plots for one spec, not a comparison across variants) |
| **Appendix location** | none — superseded by the family comparison above |

---

## Planned, not yet implemented (Track B–F of the appendix plan)

No CLI command exists yet — the modules below are named in `docs/prior_appendix_plan.md` §1/§5 as
future `analyses/` additions, not currently in `analyses/__init__.py`'s registry
(only `dimensions` and `hyperparameters` are registered today). Listed here so the mapping is
complete once they land.

| Appendix § | Label | Planned module (`name=`) | Planned config group | Target artifact |
|---|---|---|---|---|
| B.4 assembly | `app:prior_audit` | `audit_summary` | `general_ite` (K=1) | Table 6 |
| B.4.1 Overlap | `app:prior_overlap` | `overlap` | `general_ite` | Fig audit(b) |
| B.4.2 Exclusion | `app:prior_exclusion` | `exclusion` | `general_ite` | Fig audit(a) |
| B.4.3 Confounding strength | `app:prior_confounding` | `confounding_strength` | `general_ite` | Fig audit(c), Table 7 |
| B.4.4 Proxy relevance | `app:prior_relevance` | `proxy_relevance` | `general_ite` | Fig audit(d) |
| B.4.5 Ill-posedness | `app:prior_illposed` | `illposedness` | `general_ite` | Fig A4 |
| B.4.6 Identification stress test | `app:prior_identification` | `identification` | `general_ite` | Fig A5 |
| B.5.1 Input duplication | `app:prior_duplication` | `confounding_duplication` | new K=1 family off `no_u_augment` | Fig A7 |
| B.5.2 Heterogeneity remix | `app:prior_heterogeneity` | `heterogeneity` | new K>1 preset | Fig A6 |
| B.6 MC error | *(no label yet)* | `mc_error` | K-sweep family, K∈{1,5,25,100,200,5000} | Fig A8 |
| B.6 memory budget | *(no label yet)* | `mem_budget` | `default` (T=2048) | Table 9 |
| B.6 structural coverage | *(no label yet)* | *(hand-written, no module)* | — | Table 10 |

**Note on `general_ite` vs. final data.** All the `general_ite`-based commands above (K=1) are fine
for drafting. For the final version there's no need to keep `mc_error`'s K-sweep on a *separate*
high-K dataset from the rest of B.4/B.5 — the K-sweep family can reuse the same underlying SCMs as
the other sections (base spec identical to `general_ite` apart from `mcmc_cate`), rather than
generating a second, unrelated batch of complete datasets just for Fig A8.

---

## Reproducing the appendix: full command list

Everything below regenerates every table/figure currently wired into `B_prior.tex` (see the
per-artifact sections above for what each command produces and where it lands). Run from the repo
root, in order:

```
# B.2 Hyperparameters — tab:prior_fixed_hp, tab:prior_sampled_hp
PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis hyperparameters

# B.2.1 Prior Dimensions + B.3 Postprocessing — fig:prior_dim_marginals, tab:prior_dim_sampling,
# fig:prior_dimension_joint_heatmaps, tab:prior_completeness_survival
PYTHONPATH=. uv run python -m analysis_prior.main --family dim_ablation --analysis dimension_comparison

# B.3 Categorical completeness — fig:prior_category_count_marginals,
# fig:prior_proxy_stats_by_u_cardinality, fig:prior_proxy_composition_vs_u_demand,
# tab:prior_categorical_completeness_survival
PYTHONPATH=. uv run python -m analysis_prior.main --family cat_ablation --analysis categorical_completeness
```

Add `--force` to any of the three to regenerate the cache instead of reusing it, and `--n-jobs -1`
to parallelise generation across all cores.

**Before building the LaTeX**, copy/symlink the dimension-ablation PDFs into the appendix figures
tree (`figures_app/prior/` doesn't exist yet — Track A item 1 in
[`docs/prior_appendix_plan.md`](../docs/prior_appendix_plan.md)):

```
results/dim_ablation/dimension_comparison/dimension_marginals.pdf
results/dim_ablation/dimension_comparison/dimension_joint_heatmaps_comparison.pdf
```

The `cat_ablation` and `hyperparameters` outputs already land under paths the tex expects, so no
extra copying is needed for those.

Everything in "Planned, not yet implemented" above has no CLI command yet — those modules don't
exist in `analyses/__init__.py`'s registry.

## Discovery commands

```
uv run python -m analysis_prior.main --list-analyses   # registered analyses
uv run python -m analysis_prior.main --list-caches     # cached specs on disk
uv run python -m analysis_prior.main --list-families   # registered comparison families
```
