# `benchmarks/` — method comparison

Runs a fixed set of datasets through every CATE estimator (baselines, proximal methods,
foundation-model competitors, ProximalFM itself) and reports how well each one recovers the
treatment effect. One YAML drives the whole thing. ProximalFM has no separate pipeline: it is an
entry under `methods:` like the rest — see §4.

```bash
uv run benchmarks/run.py --config benchmarks/config_val.yaml
uv run benchmarks/plots.py --path data/synt_eval/<name>/bench
```

---

## 1. What is being scored

Each dataset ("episode") is split into a **context** the methods may fit on and **query** rows they
must predict for. There are two different ground truths on those query rows, and the difference is
the single most important thing to understand before reading any number here:

| target | what it is | why it matters |
|---|---|---|
| **`cate`** *(primary)* | `tau*(x) = E[Y1 − Y0 \| X = x]`, estimated by the generator's Monte-Carlo average `MU1 − MU0` over `K_eff` replicates | the estimand ProximalFM is trained on, and the best any method seeing only `X` can attain |
| **`ite`** *(secondary)* | the realized draw `Y1 − Y0` for that row, carrying its own `U` and noise | irreducibly noisy: `E[Var(Y1−Y0 \| X)]` is typically ~95% of `Var(ITE)`, which flattens every method onto nPEHE ≈ 1 |

The CATE target only exists when the set was built with replicates (`--mc_cate K`, `K > 1`).
Without them the run falls back to the ITE task and says so on the console.

### The three tasks

Methods do not all answer the same question, so they are not all ranked in the same table. Each
table is one `Task` object in [`metrics.py`](metrics.py) — the single place that says which target it
scores, which methods it admits, what it ranks on and which columns it shows. Every
prediction output carries an **estimand tag** that spells out what it answers — a *kind* and the
variables the prediction reads at **predict** time, in the canonical order `x,w,z,u`:

```
cate(x)   cate(x,w)   cate(x,z)   cate(x,w,z)   cate(x,u)   cate(x,w,z,u)   ate(x)   ate(x,w,z)
```

`cate` means the prediction varies per row, `ate` that it is one scalar per episode. Only
*predict-time* use counts: `PLearner` fits nuisances on both proxies and is still `cate(x)`, because
its prediction is a function of `x` alone. `u` appears when `is_oracle: true` concatenates the true
confounder onto `X`. Tags are derived from the config by
[`methods.method_estimand`](methods.py) (override with e.g. `estimand: cate(x)` on a method or an
`outputs[]` entry; the legacy `x`/`xw`/`ate` spellings still parse, as `cate(x)`/`cate(x,w)`/`ate(x)`).

Which table a tag lands in is then two checks and nothing more — the kind, and whether `w` or `z` is
in the parenthesis:

| tag | appears in |
|---|---|
| `cate(…)` without `w`/`z` — e.g. `cate(x)`, `cate(x,u)` | **`summary_cate.csv`** (headline), `summary_ite.csv`, `summary_ate.csv` |
| `cate(…)` with `w` or `z` — e.g. `cate(x,w)`, `cate(x,w,z)` | `summary_ite.csv`, `summary_ate.csv` |
| `ate(…)` | `summary_ate.csv` only |

The routing applies to *methods*; the reference rows below bypass it.

`cate(x,w)` is a genuinely finer conditional, not a better estimate of `tau(x)`. On the ITE target that
extra information is a real advantage; on the CATE target it is answering a different question. It
therefore gets its own row in the ITE table and none in the CATE ranking — the "they had the
advantage and still lost" comparison.

**Reference rows** are computed with no fitting. Unlike every other output they are **not** routed by
their tag: the table above is a fairness rule for competitors, and a reference row competes with
nobody, so all three oracles appear in all three tables — shown but never ranked, and never eligible
for a bolded best-of cell. Anchoring each table with all three is the point: reading `Oracle-ITE` and
`Oracle-ATE` beside each other on the CATE table brackets what a `tau(x)` method should even attempt.

- **`Oracle-CATE(X)`** — the true `tau*(x)`. Scores nPEHE ≈ 0 on the cate task (a self-check on the
  whole pipeline; if it does not, something is wrong with the target plumbing) and the **irreducible
  noise floor** on the ite task. That floor is what makes the `cate(x,w)` methods' advantage readable.
- **`Oracle-ATE`** — the best per-episode constant, and so the nPEHE = 1 / Shape R² = 0 anchor. Its
  tag is `ate(x)`; a *method* with that tag would be routed to `summary_ate.csv` only, but as a
  reference row it is drawn in all three, which is where the "1.0 is the do-nothing baseline" claim
  becomes something you can read off the same column rather than take on faith.
- **`Oracle-ITE`** — the realized `Y1 − Y0` itself, the mirror image of `Oracle-CATE(X)`. Perfect on
  the ite task (nPEHE = 0) and, on the cate task, the **price of the noise**: it predicts
  `tau*(x)` plus that row's own `U` and noise, so its nPEHE is `SD(ITE − tau*)/SD(tau*)`, normally
  well above 1, with a strongly negative Shape R². Read beside `Oracle-ATE` it says how much of
  `Var(ITE)` no `tau(x)` method should be trying to explain — and it is the reason a good method can
  beat a *perfect ITE predictor* on the CATE table without anything being wrong. At the ATE level it
  is near-unbiased (only the row-average of the noise survives), so its `ATE RMSE` is the
  per-episode sampling floor for that column. Only emitted when the set carries a Monte-Carlo CATE
  target; without one it would duplicate `Oracle-CATE(X)` exactly.

`u` in a reference row's tag is therefore never disqualifying, and neither is the `ate` kind — those
rows are anchors for reading a table, which is the whole reason they are computed.

### Metrics

Per episode, everything relative to `SD(tau)` so episodes can be pooled
([`metrics.py`](metrics.py) has the formulas and the caveats):

- `npehe` = `sqrt(mean((tau_hat − tau)²) / Var(tau))` — 1.0 is the constant-ATE baseline.
- `cate_r2` = `1 − npehe²`; `shape_r2` = `1 − Var(resid)/Var(tau)` (heterogeneity recovery, level
  error removed — the most discriminative one).
- `ate_bias_norm`, `ate_rmse` — the level component.
- `s2_over_var` — the share of the CATE target's variance that is Monte-Carlo noise *in the target
  itself*. **Reported, never subtracted**: ~0.004 at `K=1000`, ~0.2 at `K=8`. A warning fires past
  5%; past that, every method scores worse than it is and the set wants a larger `--mc_cate`.

Episodes with `Var(tau) < 1e-6` are dropped from the scale-free metrics (same cutoff as the
trainer's `CausalEvaluator`, pinned by a test — so the benchmark's cate-task nPEHE and wandb's
`synt_point/cate/npehe` are the same quantity).

---

## 2. Files

| file | what it does |
|---|---|
| **[`run.py`](run.py)** | The entry point. Loads the data, hands each method its resolved compute budget and runs it, scores all three tasks, writes `results/`. |
| **[`compute.py`](compute.py)** | The single owner of device and thread settings: `ComputeConfig`, the global/per-method resolution, and the session-wide `WorkerManager` that hands out process pools. See §3.1 — read it before adding a knob that touches threads, devices or process counts. |
| **[`runners.py`](runners.py)** | How one method's episodes actually execute. `run_method` dispatches on `compute.execution`: fan out through the manager, or loop in-process with a device held. One code path for both, so there is no sequential copy to drift. |
| **[`data.py`](data.py)** | Builds the standardized benchmark dict from the raw sparse `val/` batches (`batch_*.pt`). Standardization uses **context-only statistics** — preserving that no-test-leakage property is the point of the module. Emits `true_ITEs` always and `true_CATEs`/`cate_s2`/`cate_K_eff` when the set has replicates. Imports nothing from `proximalfm` on purpose (`sparse2dense` is vendored, kept in sync by a test) so the benchmark runs where proximalfm is not installed. |
| **[`data_csv.py`](data_csv.py)** | The same dict from a single real-world CSV (Causal Chambers), given a column→variable-block mapping. Always one episode. No CATE target exists here, so this path is ITE-only by construction. |
| **[`methods/`](methods/)** | Every estimator, behind one `CATEEstimator` interface (`base.py`: `fit(X, A, Y, W=, Z=)` / `predict(X, W=, Z=)` / `predict_variants`). `__init__.py` is the registry — the thin wrappers, `build_method(cfg, compute)` dispatching on the config's `cls:` (the device and `n_jobs` an estimator gets come from that second argument and nowhere else), `method_estimand`, and `normalize_method_cfg`. The implementations are the sibling modules: `kpv.py`, `pmmr.py`, `nmmr.py`, `dfpv.py`, `plearner.py` (+ shared `kernels.py`), and `proximalfm.py`. Was `methods.py` + `baselines/`; ProximalFM joined them when the `models:` block was removed. |
| **[`estimands.py`](estimands.py)** | The estimand-tag algebra (`format_estimand`/`parse_estimand`/`canonical_estimand`). Its own module because both `methods/` and `metrics.py` need it and neither may import the other. |
| **[`metrics.py`](metrics.py)** | Pure numpy/pandas scoring, in four layers: metric primitives (`mse`/`rmse`/`nrmse`/`r2`/`shape_r2`/`bias`/`norm_bias`), the `Metric` descriptor (aggregation, direction, colour), the `Task` registry (`TASKS` — the **single source of truth** for what each table scores, admits, ranks and shows), and `TaskReport` (`summary()`/`rich_table()`/`latex()`). No torch, no model — that isolation is why the point metrics are duplicated rather than shared with `train/_eval.py`. Adding a metric is one `Metric` in one `Task`; there is no second column spec anywhere. |
| **[`tables.py`](tables.py)** | A `# %%` script: reads `summary_{cate,ite,ate}.csv` and prints one LaTeX table each through `TaskReport.latex()`, top-3 bolded (direction-aware, oracles excluded). Set `experiment_dir` at the top. It holds no column spec of its own — the columns, formats, bolding direction and caption all live on the `Task`. |
| **[`plots.py`](plots.py)** | Every figure, as a registry: `--list` names them, `--only`/`--skip` select, and each *task-scope* figure is drawn once per task with the task's own rank metric. See §6. |
| **[`plot_frames.py`](plot_frames.py)** | The numbers behind the figures — `BenchmarkRun`, which owns loading, the complete-block pivots, the ranks, and the per-episode difficulty covariates derived from `data.pt`. No figure computes its own. |
| **[`plot_style.py`](plot_style.py)** | Colour (family) and marker (variant) per method, the reference lines each metric is read against (`METRIC_ANCHORS`), and the theme/save chores. |
| **[`tuning.py`](tuning.py)** | Method-agnostic hyperparameter search scaffolding. A method supplies `fit_score(hparams) -> float` (lower better); this owns the split, the grid/random enumeration and the selection. Used by NMMR's `tune:` block. |
| **[`causal_chambers/`](causal_chambers/)** | Real-data benchmark (Causal Chambers), notebook/Rmd driven, separate from the main run. `prepare_chamber_csv.py` turns a raw export into the canonical CSV. |
| `config_val.yaml` | The main config: a fixed synthetic build and the full method list (ProximalFM checkpoints included). |
| `config_chamber.yaml` | The single-CSV real-data path. |

---

## 3. Config

```yaml
data:
  source: load                                  # load | csv
  val_dir: data/synt_eval/<name>/val            # a build's val/ directory
  output_dir: data/synt_eval/<name>/bench       # where data.pt, preds/, results/ go

run:
  seed: 999999999
  force_rerun: false
  compute:                                      # the run-wide default budget
    device: cpu                                 # cpu | mps | cuda
    n_workers: 6                                # episode fan-out width
    torch_threads: 1
    blas_threads: 1
    n_jobs: 1

methods:                                        # baselines; comment one out to disable
  - name: S-OLS
    cls: S_Learner
    base: linear
    use_W: false
    use_Z: false

  - name: ProximalFM5-last                      # this project's model: a method like any other
    cls: ProximalFM
    checkpoint: mc_run_5/step-70000.ckpt        # a file, or a directory (best_model.ckpt / top step)
    all: false                                  # true: score every ckpt in the directory
    compute: { device: mps }                    # → execution: inline; see §3.1
```

Build a set to point `val_dir` at with **`--mc_cate K` (K > 1)**, or there is no CATE task:

```bash
uv run python -m synt_eval.build_proximal_datasets --name proximal_val_v1 \
  --n_datasets 480 --n_max 2048 --n_q 128 --mc_cate 200 --seed 123
```

`prior_eval.build` sets work too — same on-disk format, and they always carry replicates.

**Keys the `data:` block takes per `source:`.**

| `source:` | keys |
|---|---|
| `load` *(default)* | `val_dir` |
| `csv` | `csv_path`, `columns`, `prop_train`, `shuffle`, `normalize_treatment` |

The benchmark scores **pre-built sets only**. There is no on-the-fly prior sampling: a set that
exists only inside one run is one nobody else can score against, so the row split, the replicate
count and the seed belong to a builder (`prior_eval.build`, `synt_eval/build_proximal_datasets.py`,
`lit_eval.build`) whose output is on disk and has a `content_digest`. `source: generate` and its
`n_context`/`n_query`/`n_batch`/`prior:` keys are gone; a config still asking for it is **refused
with an error** rather than quietly falling back to a cache. `prop_train` survives only under `csv`,
where the row count is a property of the file and a ratio is the only thing there is to give.

**Common method keys.** `name` (display + output filename), `cls` (dispatch), `base`
(`linear` | `rf` | `tabicl`), `use_W`/`use_Z` (whether proxies enter the design matrix — **default
`True`** for the learners and the PFN wrappers), `interactions`, `is_oracle: true` (concatenates the
true `U` onto `X` — an upper bound, not a competitor), and `estimand:` to override the derived tag.

**`base: tabicl` speed.** TabICL dominates the wall clock for a reason that is entirely outside the
model: with no cache, every `predict()` re-runs the whole context (2048 rows × 8 ensemble members)
through the network, and an S-learner predicts twice off one fit. The `tabicl:` block passes options
straight to `TabICLRegressor`; the benchmark defaults to `kv_cache: repr`, which caches that context
pass at `fit()` time and reproduces the uncached predictions **bit for bit**. 480 episodes, 6 workers:

| `kv_cache`        | 480 episodes | vs uncached                                  | peak RSS / worker |
|-------------------|--------------|----------------------------------------------|-------------------|
| `false` (library) | 30.6 min     | —                                            | 1.61 GiB          |
| `repr` (default)  | 22.7 min     | bit-identical                                | 1.54 GiB          |
| `true`            | 16.6 min     | max abs Δ ≈ 3e-7 — 2 float32 ulp             | 2.44 GiB          |

```yaml
  - name: S-TabICL
    cls:   S_Learner
    base:  tabicl
    tabicl: { kv_cache: true }      # 1.8x, at float32 accumulation noise
    compute: { torch_threads: 2 }   # a further 1.2x; see §3.1
```

`kv_cache: true` additionally caches the ICL layers' KV projections; the residual difference is
accumulation order, not a change of estimator. Check the memory column against `n_workers` before
using it.

### 3.1 Compute

**[`compute.py`](compute.py) is the single owner of device and thread settings.** Nothing below it
picks either for itself: no estimator calls `torch.set_num_threads`, no estimator auto-selects a
device, and the BLAS thread count is a config field rather than an import-time side effect. That
was not true before — three layers used to set threads without knowing about each other, and NMMR
grabbed MPS inside every pool worker while DFPV hardcoded an mps→cpu downgrade to survive it.

Control is inverted relative to the old `run_method_parallel`: `run.py` resolves a `ComputeConfig`
per method and calls it, and the method decides whether to fan its episodes out through the
session-wide `WorkerManager` or to hold a device in-process. `run.compute` is the run-wide default;
any `methods:` entry may overlay its own `compute:` block, and unset fields inherit.

| key | meaning |
|---|---|
| `device` | `cpu` \| `mps` \| `cuda`. Validated on startup — an unavailable device falls back to CPU with a warning instead of failing inside a worker. |
| `n_workers` | Episode fan-out width. `1` runs in the calling process with no pool at all. |
| `torch_threads` | Per-process torch threads. Only the torch-backed methods care; the sklearn ones don't use them. |
| `blas_threads` | OMP/MKL/OpenBLAS per process. Set in the pool initializer, so it is part of a pool's identity. |
| `n_jobs` | Intra-method sklearn/joblib fan-out (CausalForest, the P-learner's forest). Nests *inside* a worker: what lands on the machine is `n_workers × n_jobs`. |
| `execution` | `pool` \| `inline`. Defaults to `inline` for a non-CPU device or a single worker. |

Two rules the config enforces rather than documents. **A non-CPU device with `execution: pool` and
more than one worker is refused** — concurrent multi-process Metal/CUDA access stalls, so "on the
GPU" and "fanned out over processes" are mutually exclusive. And **the pre-`compute:` spellings
raise**: `run.n_workers`, `run.device`, and per-method `torch_threads:` / `n_jobs:` are errors with
a message naming the replacement, not silently-accepted aliases. `run.device` in particular used to
reach the (now removed) `models:` block *only*, so the honest translation is a `compute:` block on
each ProximalFM method entry rather than a global default. A config still carrying `models:` is
likewise refused, with the `cls: ProximalFM` translation.

Pools are shared across the whole session, one per distinct `(n_workers, blas_threads)`, created on
first use. That is why `torch_threads` is deliberately *not* part of a pool's identity: it is
applied per task, so one pool serves methods that want different thread counts, and consecutive
methods on the same budget reuse warm workers with their module and checkpoint caches intact. The
cost is that a worker which ran CausalFM may later be handed NMMR — `methods._switch_src_context`
exists to unpick exactly that, and is the first thing to suspect if a submodule-backed method
misbehaves under a shared pool.

`--no-parallel` is now an override of the global budget (`n_workers=1`, `execution=inline`), not a
separate code path; the duplicated sequential runner is gone.

**Don't bother chasing the device.** Once the context pass is cached, every arrangement converges on
the same ceiling, because the irreducible work is one forward of 8 ensemble members over the 2048
context rows per episode. Real episodes, 480 of them:

| arrangement                          | ep/s  | 480 episodes |
|--------------------------------------|-------|--------------|
| cpu, 6 workers × 1 thread, no cache  | 0.255 | 31.3 min     |
| cpu, 6 workers × 1 thread, `kv`      | 0.467 | 17.1 min     |
| cpu, 6 workers × 2 threads, `kv`     | 0.556 | 14.4 min     |
| mps, 2–3 workers × 2 threads, `kv`   | 0.55  | 14.5 min     |
| mps, 1 process × 4 threads, `kv`     | 0.569 | 14.1 min     |

(The two mps rows predate the `execution` rule and are no longer expressible for a *method* — a
non-CPU device now implies `inline`. They are kept because they are what the ceiling argument rests
on.) `device: mps` is ~5x faster *per episode* than one CPU worker (2.1 s vs 15.0 s), which is why a
single GPU process keeps up with six CPU ones — but it does not beat them, so it is worth choosing
only for its memory profile (one 2.4 GiB process instead of six). Breaking the ~0.57 ep/s ceiling
needs episodes batched into the model's batch dimension, the way the ProximalFM path already does
it; that means bypassing `TabICLRegressor` and padding the ragged `d_X`, so it would no longer be
the same predictions.

**Multi-output methods.** KPV / PMMR / NMMR / DFPV fit once and emit several predictions from that
one fit via an `outputs:` list — much cheaper than four separate entries:

```yaml
  - name: PMMR
    cls: PMMR
    outputs:
      - { name: PMMR-XW,      strategy: paired }       # -> xw
      - { name: PMMR-X-Wind,  strategy: independent }  # -> x
      - { name: PMMR-X-Wcond, strategy: conditional }  # -> x
      - { name: PMMR-ATE,     strategy: marginal }     # -> ate
```

---

## 4. Methods

`cls:` values accepted by `build_method`, and what each one's prediction is actually a function of:

| `cls` | prediction | estimand | notes |
|---|---|---|---|
| `S_Learner` | `f(x,·,1) − f(x,·,0)` | `cate(x…)` | one model with `A` as a feature; `interactions: true` adds `A·X` |
| `T_Learner` | one model per arm | `cate(x…)` | |
| `X_Learner` | propensity-weighted two-stage | `cate(x…)` | |
| `CausalForest` | EconML `DROrthoForest.effect(X)` | `cate(x)` | `use_W`/`use_Z` feed EconML's *nuisance controls*, not the heterogeneity features |
| `P2SLS` | a single fitted ATE, broadcast | `ate(x)` | proximal two-stage least squares |
| `PLearner` | `E[gamma \| X]` from a cross-fit proximal score | `cate(x)` | `ProximalPLearner`; W/Z used at fit, discarded at predict |
| `KernelPV` | kernel proxy-variable bridge | per strategy | `methods/kpv.py` |
| `PMMR` | proximal maximum-moment-restriction bridge | per strategy | `methods/pmmr.py` |
| `NMMR` | neural MMR bridge | per strategy | `methods/nmmr.py`; supports `tune:`; no `marginal`; torch — for a GPU give it `compute: { device: mps, execution: inline }` |
| `DFPV` | deep feature proxy variable | per strategy | `methods/dfpv.py`; torch, same GPU note as NMMR (it used to be pinned to CPU unconditionally) |
| `CausalFM`, `DoPFN`, `CausalPFN` | in-context tabular causal FMs | `cate(x…)` | imported from the git submodules via `_switch_src_context`; init them first. `CausalFM` and `CausalPFN` honour `compute.device`; `DoPFN` is CPU-only because `DoPFNRegressor` exposes no device argument |
| `ProximalFM` | this project's model, in-context | `cate(x)` | receives the full sequence but **masks the query rows' proxies** (`_hide_query_proxies`), so it answers `tau(x)` using `W`/`Z` only through the context. `checkpoint:` (file or directory), `all: true` (one output per checkpoint in the directory, all sharing one stored context), `ensemble: N` + `transforms:`. CPU: fans out over the episode pool like everything else. GPU: `compute: { device: mps }` |

**Strategies** (the four proximal bridge methods) decide what is done with `W` at prediction time:

- `paired` → `h(1,x,w) − h(0,x,w)` at the query row's own `w`. **`cate(x,w)`.**
- `independent` → marginalizes `W` over its context empirical marginal (assumes `W ⟂ X`). **`cate(x)`.**
- `conditional` → marginalizes over `W|X` by kernel smoothing on `X`. **`cate(x)`.**
- `marginal` → averages over the empirical joint `(X, W)`: one scalar. **`ate(x)`.**

Note that `independent`/`conditional`/`marginal` marginalize over the **context** sample, so they
assume the query rows come from the same distribution.

---

## 5. Outputs

Written under `output_dir`:

```
data.pt                     the standardized dict (cached; see below)
preds/<output>_preds.npz    per-method `preds` [B, n_test] + `times` [B]
results/results.csv         long-form: one row per (method, episode, task)
results/summary_cate.csv    the headline table
results/summary_ite.csv     the ITE task (adds the proxy-conditioned methods)
results/summary_ate.csv     level only (where `ate` methods live)
results/summary.csv         a copy of the primary task's table
results/plots/<figure>_{cate,ite,ate}.png   task-scope figures (§6)
results/plots/<figure>.png                  run-scope figures (§6)
```

`results.csv` carries a row only for the tasks whose table can rank that method
(§1), so its `target` column names the **task** and the rows in it are exactly the rows the tables
show. Each row carries only that task's metrics — the `ate` rows have no `npehe`, because a
level-only view has no shape.

Every metric summarized as a median also carries `<label> p25` / `<label> p75` companions in
`summary_*.csv` — the same convention as the trainer's `synt_point/*` keys, and for the same reason:
the R²-style metrics are unbounded below, so a mean is not robust and a single median hides the
spread. `ATE RMSE` is the exception, declared `pooled_rms`: it already pools the bias and the
episode-to-episode spread, so it is not a quantile of anything and gets no companions.

**Everything is resumable and cached.** `data.pt` is reused if present, per-method predictions land
in `preds/` incrementally and finished methods are skipped on re-run. `--force` (or
`run.force_rerun`) recomputes. **When results look stale, suspect the cache first** — and note that a
`data.pt` cached before the CATE target existed has no `true_CATEs`, so the run silently drops to the
ITE task (it does print the banner); delete it or pass `--force`.

Predictions are per-episode and independent, so a method that raises on some episodes leaves NaN
there and is reported with a non-zero failure rate rather than taking the run down.

---

## 6. Figures

```bash
uv run benchmarks/plots.py --path data/synt_eval/<name>/bench    # everything
uv run benchmarks/plots.py --exp <name> --list                   # what "everything" is
uv run benchmarks/plots.py --exp <name> --only scorecard iso_npehe --tasks cate
```

Figures are a **registry**, and the task-scope ones are drawn once per task from
[`metrics.py`](metrics.py)'s `TASKS` — the same object that decides the tables. A figure asks the
task what it ranks on, which metrics it shows and which methods it admits, so `ate` is not a special
case and a new `Metric` appears in the ECDF grid and the scorecard without an edit here. Options:
`--tasks` (default: every task the run scored), `--only`/`--skip`, `--top` (how many methods the
one-line-per-method figures keep), `--baseline` (what `paired_delta` compares against; defaults to
ProximalFM), `--episodes`, `--meta` (a frozen set's root, adding its `sampled_hp` as stratification
axes), `--dpi`. A figure that cannot be drawn says why and the rest still render.

Two invariants live in [`plot_frames.py`](plot_frames.py), not in the figures: **oracles annotate
and never compete** (no rank, no win rate, no Pareto front), and every paired statistic runs on a
**complete block** — an episode any method missed is dropped for all of them, with the dropped count
printed on the figure. Get either wrong and the picture is quietly wrong rather than visibly broken.

Start at **`scorecard`**: one row per method with its rank profile, error ECDF, error anatomy,
position on all three tasks and cost, aligned so the columns compare. Then:

| figure | what it answers |
|---|---|
| `ecdf` | the whole distribution per metric — and the missing mass above a curve is that method's unscored share |
| `rank_profile` / `cd_diagram` / `head2head` | how often a method wins, whether the gap is significant, and who beats whom |
| `performance_profile` | peak vs robustness: share of episodes within a factor τ of the per-episode best |
| `paired_delta` | the same-episode difference against one baseline — difficulty cancels, so it separates what the ECDFs overlap |
| `decomposition` / `iso_npehe` | is the error level or shape (see below) |
| `excess_over_oracle` | error ÷ the oracle's on the same episode; the only readable view of the ITE task |
| `strata` | where a method wins — weak proxies, strong confounding, poor overlap, high noise |
| `robustness` | failure rate, and how often a method loses to the trivial predictor |
| `cost_accuracy` | accuracy against wall-clock, with the Pareto front |
| `bias_caterpillar` / `ate_scatter` | the ATE level: systematic bias vs episode-to-episode spread, which `ATE RMSE` pools |
| `cross_task_ranks` | consistency across ATE / CATE / ITE — a method high on ATE and low on CATE gets the level right and the shape wrong |
| `task_compression` | why the benchmark leads with CATE: the ITE axis is squeezed onto ≈1 by the noise floor |
| `target_noise` | whether the Monte-Carlo target is good enough to rank on at all |
| `episode_fit` | row-level predictions on an easy, a median and a hard episode |

**The identity two of them are drawn on.** Per episode, with `b` the normalised ATE bias and `s` the
shape R²:

```
nPEHE² = b² + (1 − s)
```

exactly — level error plus shape error. `decomposition` is that as a stacked bar and `iso_npehe` as
a plane with *analytic* contours (nothing is fitted). It also fixes the aggregation: the components
are averaged as squares, so `sqrt` of the total is an RMS nPEHE and the point lands on its own
contour; medians of the parts would not. `mean(nPEHE²)` has an unbounded tail (a near-homogeneous
episode divides by a Var(τ) just above the threshold), so the worst 5% of episodes are trimmed per
method — the identity is linear in episodes and survives it, which is why the trim happens in one
place rather than per figure.

---

## 7. Related

- Distributional scoring of ProximalFM (CRPS, coverage, PIT) is **not** here — the benchmark's
  interface is point predictions and no baseline has a posterior. Use `python -m prior_eval.score`
  on the same set; it has the frozen comparison scale and the head's grid.
- [`../proximalfm/train/metrics.md`](../proximalfm/train/metrics.md) is the catalogue of the
  *training-side* metric namespace and the estimand traps behind it. Read it before adding a metric
  on either side.
