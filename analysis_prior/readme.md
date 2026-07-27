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
| `mc_targets` | 128 | 256 | **100** | B.2.6 Monte Carlo targets; the only `K>1` preset in use. Cheap on purpose — precision comes from paired within-row contrasts, so raise `seq_len` (not `n_datasets`) to tighten the variance self-check |
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

Ten analyses are registered today (`analyses/__init__.py`): `dimensions`, `dimension_comparison`,
`hyperparameters`, `categorical_completeness`, `mechanism_inputs`, `mechanism_outputs`,
`treatment_walkthrough`, `heterogeneity`, `mc_targets`, `prior_overview`. The subsections below match the *current*
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

### B.2.3 Mechanism input composition — `fig:prior_mechanism_input_composition`, `tab:prior_mechanism_inputs`

Off the prior's own statistics (no regression): for `f_A` and `f_Y`, each parent block's share of the
mechanism input, measured two ways — *wiring* (column-count share, equal to the input-variance share
since parent blocks are standardised before concatenation) and *first-layer* injected variance (exact
from the realised weights) — plotted against the sampled knob steering each channel.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis mechanism_inputs` |
| **Config group** | preset `general_ite` (reads the mechanism-diagnostics payload in `sampled_hp` / `mechanisms`) |
| **Analysis module** | `analyses/mechanism_inputs.py` → `MechanismInputsAnalysis` (`name="mechanism_inputs"`) |
| **Output dir** | `results/general_ite/mechanism_inputs/` |
| **Figure output** | `mechanism_input_composition.pdf` → `fig:prior_mechanism_input_composition` |
| **Table output** | `mechanism_input_composition_table.tex` → `tab:prior_mechanism_inputs` |
| **Appendix location** | §B.2.3 "MLP Generation, Noise & Edge Sampling" (`app:prior_mechanisms`), "Delivered input composition" paragraph |

Needs the per-dataset mechanism diagnostics (`sampled_hp`, `mechanisms`); a cache written before they
existed must be regenerated once with `--force` (data is bit-identical, the diagnostics are simply
recorded alongside).

### B.2.3 Mechanism output dependence — `fig:prior_mechanism_output_dependence`, `tab:prior_mechanism_outputs`

The output-side companion: what each mechanism's output *depends on*, via a cross-fitted TabICL
regression on the delivered data. Leave-one-block-out ΔR² gives each parent block's unique
contribution — `f_A` predicts `logit(π)` from `{U, X, Z}`, `f_Y` predicts `Y0` from `{U, X, W}`; the
exogenous-noise channel, having no block to drop, is read as the unexplained share `1 − R²`. A
per-dataset tqdm progress bar tracks the run.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --n-datasets 256 --analysis mechanism_outputs` |
| **Config group** | preset `general_ite`. **Uses every dataset in the cache**, so the count is set by `--n-datasets` (the default 1024 ≈ 8 h of TabICL — cap it; `--n-datasets 256` is served as a prefix of the existing cache, no regeneration) |
| **Analysis module** | `analyses/mechanism_outputs.py` → `MechanismOutputsAnalysis` (`name="mechanism_outputs"`) |
| **Regression helper** | `analysis_prior/regression.py` — cross-fitted out-of-fold, model `linear`/`rf`/`tabicl` (attr `MechanismOutputsAnalysis.model`, default `tabicl`); predictions cached under `cache/regression/`, so re-renders and resumed runs are instant |
| **Output dir** | `results/general_ite/mechanism_outputs/` |
| **Figure output** | `mechanism_output_dependence.pdf` → `fig:prior_mechanism_output_dependence` |
| **Table output** | `mechanism_output_dependence_table.tex` → `tab:prior_mechanism_outputs` |
| **Fast demo** | `PYTHONPATH=. uv run python -m analysis_prior.examples.mechanism_outputs_small` (a few datasets, random forest, ~seconds; writes to `results/general_ite/mechanism_outputs_small/`, not the paper path; `--model tabicl --n 16` for a quick TabICL taste) |
| **Appendix location** | §B.2.3 "MLP Generation, Noise & Edge Sampling" (`app:prior_mechanisms`), "Output dependence" paragraph |

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

### B.2.4 Post-process (categorical completeness) — `fig:prior_category_count_marginals`, `fig:prior_proxy_enforcement_vs_u_demand`, `tab:prior_categorical_completeness_survival`

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
| **Figure outputs** | `category_count_marginals.pdf` → `fig:prior_category_count_marginals`; `proxy_enforcement_vs_u_demand.pdf` → `fig:prior_proxy_enforcement_vs_u_demand` |
| **Table output** | `categorical_completeness_survival.tex` → `tab:prior_categorical_completeness_survival` |
| **Appendix location** | §B.2.4 "Post-Process" (`app:prior_postprocess`), paragraphs "Categorical completeness" / "Enforcement scales with `U`'s demand" / "Survival" |

`proxy_enforcement_vs_u_demand` is a single three-panel figure: (a, b) condition on one `U` column's
own cardinality (proxy continuous fraction, proxy mean category count), (c) conditions at the dataset
level on `|cats(U)|` (proxy categorical fraction). It replaces the former standalone
`proxy_stats_by_u_cardinality` (panels a, b) and `proxy_composition_vs_u_demand` (panel c) — any
leftovers of those two under `results/cat_ablation/categorical_completeness/` are stale and will not
regenerate. All three panels plot only the categorised-`U` arms (`full`, `ablation`) — `baseline` has
a continuous `U` by construction and is dropped, though it still appears in the category-marginals
figure and the survival table.

### B.2.5 Treatment assignment walkthrough — `fig:prior_treatment_walkthrough`

A single-SCM illustration of `sample_A`'s pipeline: raw `f_A` logit → standardise (`eq:A_normalise`)
→ shift/scale (`eq:A_affine`) → sigmoid+clamp (`eq:A_sigmoid`/`eq:A_clamp`) → Bernoulli draw, tracking
one highlighted unit (red) through all five stages. Pure function of the raw logits now retained in
the cache (`record.a_logits`, plus the population `a_logit_mean`/`a_logit_std` that `sample_A`
standardises by) and the per-dataset propensity knobs; it re-applies the exact transform, so the
reconstructed `π` matches the stored `A_probs` to ~1e-7 (asserted in the test and reported as
`reconstruction_residual` in `summary.json`). The illustrated dataset is auto-selected for a legible,
symmetric clamp (balanced prevalence, moderate scale); its index/seed land in `summary.json`. Panel (d)
draws the clamp at an enlarged `DISPLAY_EPSILON` (0.05) — the production `propensity_epsilon` (1e-3)
sits flush against 0/1 and is invisible on a `[0,1]` axis; only the display bound is enlarged, and the
`A_probs` validation still uses the true 1e-3.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis treatment_walkthrough` |
| **Config group** | preset `general_ite` (`K=1`, so the observed slice is the whole population and the standardisation is exact) |
| **Analysis module** | `analyses/treatment.py` → `TreatmentWalkthroughAnalysis` (`name="treatment_walkthrough"`) |
| **Output dir** | `results/general_ite/treatment_walkthrough/` |
| **Figure output** | `treatment_walkthrough.pdf` → `fig:prior_treatment_walkthrough` |
| **Appendix location** | §B.2.5 "Treatment Assignment" (`app:prior_treatment`) |

The raw logits are collected only when the cache carries mechanism diagnostics. A cache written before
they existed (i.e. before `a_logits` was added) must be regenerated once with `--force` — the data is
bit-identical, the logits are simply recorded alongside. The B.3.1 overlap audit (`app:prior_overlap`)
will later add a second analysis (`treatment_overlap`) to this same `treatment.py` module.

### B.2.6 Heterogeneity dial — `fig:prior_heterogeneity_spread`

> **Rendered but not yet `\input` into `B_prior.tex`.** The figure exists; the `\cm` at
> `app:prior_outcome` also asks for `fig:prior_cate_vs_ite_K` (still planned, see below), so the
> marker stays until both land.

Does the remix dial `gamma` control the spread of the delivered effect? Substituting `eq:remix`
collapses the per-row weights to `tau = Y1 - Y0 = gamma * tau_raw + (1 - gamma) * tau_bar` — the
per-row `alpha` cancels — so `sd(tau) = gamma * sd(tau_raw)` exactly.

**`gamma` cannot be pinned from a `PriorSpec`**: it is drawn inside the generator
(`torch.rand(1)` in `_dataset.py`), not from the sampled-HP config, so there is no way to hold an
SCM fixed and turn the dial. It *is* recorded per dataset (`record.gamma_hetero`) and drawn
uniformly, so one cache is already a sweep — the analysis reads the delivered `gamma` and asks
whether dispersion tracks it **across** the prior. Two consequences worth knowing before reading
the figure:

- the ATE-preservation half of the claim is **not** testable this way (the ATE varies by orders of
  magnitude across SCMs, swamping any `gamma` effect) — only the spread claim is;
- `Y` is not standardised by `Corrupt`, so every statistic is a ratio. Dispersion is
  `sd(tau)/sd([Y0;Y1])`; `summary.json` reports `spearman_gamma_vs_scale` (0.04 on `general_ite`)
  as the check that this denominator is `gamma`-neutral and the panel-(c) overlay compares like
  with like.

Three panels: **(a)** two effect densities on a shared outcome-scaled axis, one gathered
(`gamma ≈ 0.1`) and one spread (`gamma ≈ 0.9`) — the pair is chosen for *matched* SCM dispersion
`c = dispersion/gamma`, so the contrast is about `gamma` rather than SCM luck; **(b)**
`corr(Y0, Y1)` vs `gamma`, whose `gamma → 0` endpoint is pinned at exactly 1 by the algebra
(`Y1 = Y0 + tau_bar`) and needs no scale reference; **(c)** dispersion violins per `gamma` bin
against the proportionality `gamma * median(c)`.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis heterogeneity` |
| **Config group** | preset `general_ite` (reuses the same cache as `mechanism_inputs` / `treatment_walkthrough` / `prior_overview`) |
| **Analysis module** | `analyses/outcome.py` → `HeterogeneitySpreadAnalysis` (`name="heterogeneity"`) |
| **Output dir** | `results/general_ite/heterogeneity/` |
| **Figure output** | `heterogeneity_spread.pdf` → `fig:prior_heterogeneity_spread` |
| **Appendix location** | §B.2.6 "Potential Outcomes, Heterogeneity Remix & CATE Targets" (`app:prior_outcome`) |

Statistics use the ITE `Y1 - Y0` of the realised replicate, which satisfies the collapse above at
any `K`. At `K = 1` the Monte Carlo CATE target `MU1 - MU0` **is** that same quantity, so on a
`general_ite` cache the panels read equally as a CATE view. Needs only `gamma_hetero` from the
diagnostics payload (a cache predating it must be regenerated once with `--force`), no regression.

### B.2.6 Monte Carlo CATE targets — `fig:prior_cate_vs_ite`, `tab:prior_mc_signal_share`

> **Rendered but not yet `\input` into `B_prior.tex`.** Together with
> `fig:prior_heterogeneity_spread` above, this closes the `\cm` at `app:prior_outcome`.
> Note the label is `fig:prior_cate_vs_ite`, not the planned `..._K`: there is no measured `K`
> sweep in it (see below).

How much of the training target is real? The target `Q_hat_K = MU1 - MU0` is unbiased at every
`K`, so a small `K` costs variance, not bias. Writing `sigma^2(x) = Var(tau|x)` for the
within-unit replicate variance and `Var_x(Q)` for the between-unit CATE variation,

```
Var_rows(Q_hat_K) = Var_x(Q) + E[sigma^2]/K      →      S(K) = 1/(1 + NSR/K)
```

with `NSR = E[sigma^2]/Var_x(Q)`. `S(K)` is the fraction of the target's variance that is signal —
equivalently its R² against the true CATE.

**The one trick that makes this free.** The retained replicate is one of the `K` that went into
`MU`, so the mean of the *other* `K-1` is recoverable exactly and is statistically **independent**
of the retained one: `Q_loo = (K*Q_hat_K - ITE)/(K-1)`. That gives two independent measurements of
the same `Q(x)` — parallel-forms reliability — hence two subtraction-free estimators:
`Var_x(Q) = Cov(ITE, Q_loo)` and `E[sigma^2] = mean[(ITE - Q_loo)^2]*(K-1)/K`. Neither differences
near-equal variances, which is why a short `seq_len` works: precision comes from `T` paired
contrasts.

**Measured vs extrapolated.** Only `NSR` is measured, once per dataset. `S(K)` then follows
analytically (it is the variance of a sample mean, exact under the i.i.d. replicate structure the
generator guarantees) — it is **not** a measured sweep, and cannot be: the cache pins measured
anchors only at `K=1`, `K-1` and `K`, and intermediate `K` would need the per-replicate matrix.
That is why `S(K)` is reported as a *table* rather than drawn as a band of curves. `gamma` cancels
from `NSR` (both variances carry `gamma^2`), so no faceting is needed; datasets with
`gamma < 0.05` are dropped only because both variances then vanish numerically.

Two panels: **(a)** one dataset's `K=1` target against the held-out `Q_loo`, with the identity and
the fitted regression. The informative feature is the **slope**, not the diagonal cloud — a
"CATE target" that were merely the ITE plus noise would regress with slope 1, whereas replicate
averaging shrinks it to `Var(Q)/Var(ITE) = S(1)`; the gap between the two lines is what the Monte
Carlo integration removed. **(b)** signal share across the prior at `K=1` vs the production `K`.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec mc_targets --analysis mc_targets --n-jobs -1` |
| **Config group** | preset `mc_targets` (`n=128`, `seq_len=256`, **`K=100`** — the only preset with `K>1`; deliberately cheap, raise `n_datasets` to append to the same cache) |
| **Analysis module** | `analyses/mc_targets.py` → `McTargetsAnalysis` (`name="mc_targets"`) |
| **Output dir** | `results/mc_targets/mc_targets/` |
| **Figure output** | `cate_vs_ite.pdf` → `fig:prior_cate_vs_ite` |
| **Table output** | `mc_signal_share.tex` → `tab:prior_mc_signal_share` |
| **Appendix location** | §B.2.6 "Potential Outcomes, Heterogeneity Remix & CATE Targets" (`app:prior_outcome`) |

Rejects a `K=1` cache outright (there is no replicate structure to measure). `summary.json` reports
two implied identities as `*_residual`; they are not equally informative — `var_ite_residual` is
near-algebraic given how the estimators are built, while `var_cate_residual` is the real check, and
is floored by the sampling error of a variance over `T` rows (`~sqrt(2/T)`, so ~9% at
`seq_len=256`). **That floor is the reason to raise `seq_len`**, not the number of datasets.

### B.2.x Visualize the prior — `fig:prior_feature_distributions`, `fig:prior_feature_pairs`, `fig:prior_feature_pairs_cross`

Three galleries of the delivered data, all off the prior's own statistics (no training):

- **`prior_feature_distributions`** — one column per block (`X, U, W, Z`, `π` for `A`, `Y0`/`Y1` for `Y`),
  `k` rows, each the marginal of one feature column sampled from a random dataset (solid = continuous,
  dashed = categorical; the `Y` column overlays both potential outcomes in two colours).
- **`prior_feature_pairs`** — within-block dependence: one column per block, `k` rows, each a scatter of
  two same-block features (`A`: `π` vs the draw; `Y`: `Y0` vs `Y1`).
- **`prior_feature_pairs_cross`** — cross-block dependence: one column per chosen block pair
  (`U`–`W`, `U`–`Z`, `W`–`Z`, `U`–`π`), `k` rows, each a scatter of one feature from each block.

All three sample datasets with a fixed seed (`PriorOverviewAnalysis.seed`), so every figure is
reproducible.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis prior_overview` |
| **Config group** | preset `general_ite` (reuses the same cache as `mechanism_inputs` / `treatment_walkthrough`) |
| **Analysis module** | `analyses/overview.py` → `PriorOverviewAnalysis` (`name="prior_overview"`) |
| **Output dir** | `results/general_ite/prior_overview/` |
| **Figure outputs** | `prior_feature_distributions.pdf` → `fig:prior_feature_distributions`; `prior_feature_pairs.pdf` → `fig:prior_feature_pairs`; `prior_feature_pairs_cross.pdf` → `fig:prior_feature_pairs_cross` |
| **Appendix location** | §B.2.x "Visualize the prior" (`app:prior_visualize`) |

Uses only the cached feature blocks and the `max_categories` field of `sampled_hp` (for the
continuous/categorical split); no regression or mechanism-diagnostic payload, so any `general_ite` cache
works without `--force`.

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
T0.1), `analysis_prior/regression.py` — built for the B.2.3 `mechanism_outputs` analysis
(fold-stable out-of-fold predictions, pluggable `linear`/`rf`/`tabicl` model, disk-cached
predictions under `cache/regression/`). Reuse it for those modules.

> **Removed module.** `analyses/confounding.py` (`confounding_dim`, added in `4801ef3`) was **deleted**
> in `a77c467`. It has no replacement in the registry; the old analytic `eq:conf_dim` /
> `tab:prior_conf_dim` / `fig:prior_conf_dim` deliverables are dropped from the plan and re-scoped as
> the empirical `mechanism_inputs` analysis (T1.1). Any `results/*/confounding_dim/` and
> `results/*/input_sampler/` directories on disk are **orphans** (no module produces them) and can be
> deleted.

**Note on `general_ite` vs. final data.** All the `general_ite`-based commands above (K=1) are fine
for drafting the ITE-phase (B.2/B.3) artifacts, and `heterogeneity` needs nothing more (the
`gamma` collapse is `K`-independent). Only the Monte Carlo artifacts (`mc_error`,
`fig:prior_cate_vs_ite_K`) need `K>1`.

> **A `K` sweep across specs does not hold the SCM fixed.** Changing `mcmc_cate` resizes the
> `T*K` tensors the generator draws, which shifts the RNG stream, so the mechanism weights differ
> between arms — per-dataset error curves across `K` specs would be comparing different SCMs.
> A single `K>1` cache avoids this entirely: it stores both the retained replicate
> (`Y1 - Y0`, variance `Var(Q) + E[sigma^2]`) and the `K`-average (`MU1 - MU0`, variance
> `Var(Q) + E[sigma^2]/K`), two equations that solve per dataset for the CATE signal and the
> within-`x` replicate noise — hence the whole signal-share curve `1/(1 + NSR/K)` with no gold
> standard and no matched SCMs. The MC analysis wants many datasets, not long ones, so a shorter
> `seq_len` at `K=100` is much cheaper than reusing `general_ite`'s `seq_len=2048`.

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

# B.2.3 Mechanism output dependence — fig:prior_mechanism_output_dependence,
# tab:prior_mechanism_outputs.
# Cross-fitted TabICL leave-one-block-out R2 on the delivered data (slow: a few
# thousand in-context regressions the first time; predictions are cached under
# cache/regression/ so re-renders are instant). Uses ALL datasets in the cache,
# so pass --n-datasets to control the count: general_ite's default is 1024
# (~8h of TabICL), so cap it. --n-datasets 256 is served as a prefix of the
# existing 1024-dataset cache — no regeneration.
PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --n-datasets 256 --analysis mechanism_outputs

# ^ Fast demo of the same figure (a few datasets, random forest, ~seconds) for
# development — writes to results/general_ite/mechanism_outputs_small/, not the
# paper path. Pass --model tabicl --n 16 for a quick TabICL taste.
PYTHONPATH=. uv run python -m analysis_prior.examples.mechanism_outputs_small

# B.2.5 Treatment assignment walkthrough — fig:prior_treatment_walkthrough,
# tab:prior_treatment_walkthrough.
# Reuses the same general_ite cache as mechanism_inputs (both need the raw f_A
# logits, added alongside the mechanism diagnostics — regenerate once with
# --force if the cache predates them).
PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis treatment_walkthrough

# B.2.6 Heterogeneity dial — fig:prior_heterogeneity_spread.
# Reads the delivered gamma (record.gamma_hetero) off the same general_ite
# cache; gamma cannot be pinned from a spec, so the sweep is the prior's own
# uniform draw. K=1 is fine — the gamma collapse is K-independent.
PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis heterogeneity

# B.2.6 Monte Carlo CATE targets — fig:prior_cate_vs_ite,
# tab:prior_mc_signal_share.
# The only command needing a K>1 cache (preset mc_targets, K=100), which it
# generates on first run. Everything is arithmetic on MU0/MU1 and the retained
# replicate -- no per-replicate matrix, no regression.
PYTHONPATH=. uv run python -m analysis_prior.main --spec mc_targets --analysis mc_targets --n-jobs -1

# B.2.x Visualize the prior — fig:prior_feature_distributions,
# fig:prior_feature_pairs, fig:prior_feature_pairs_cross.
# Reuses the same general_ite cache; needs no diagnostics payload (only the
# feature blocks and the max_categories field), so no --force is required.
PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis prior_overview
```

Add `--force` to any of these to regenerate the cache instead of reusing it, and `--n-jobs -1`
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
