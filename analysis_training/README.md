# `analysis_training` — training-section artifacts

Headquarters for every figure and table that describes the **training run** —
Appendix C (`docs/latex/appendices/C_model_training.tex`) and the one main-paper
training figure. Sibling of [`analysis_prior/`](../analysis_prior) (Appendix B, the prior).
Nothing here trains anything: each artifact is derived either from a *training config*
(`scripts/pretrain_prod.sh`) or from a run's logged history / checkpoints.

Outputs land in `analysis_training/results/` (gitignored, fully regenerable) as a paper-ready
`.tex` plus a machine-readable `.csv`. Pass `--out` to write straight into the paper's assets
directory instead.

## Artifacts

| Appendix ref | Command | Module |
|---|---|---|
| `\Cref{tab:model_hparams}` | `uv run python -m analysis_training.arch_table` | [`arch_table.py`](arch_table.py) |
| `\Cref{tab:training_hparams}` | `uv run python -m analysis_training.hparam_table` | [`hparam_table.py`](hparam_table.py) |
| `\Cref{tab:training_cost}` | `uv run python -m analysis_training.time_table` | [`time_table.py`](time_table.py) |
| `\Cref{fig:training_curves}` | `uv run python -m analysis_training.loss_figure` | [`loss_figure.py`](loss_figure.py) |
| `\Cref{fig:point_validation}` | `uv run python -m analysis_training.point_figure` | [`point_figure.py`](point_figure.py) |
| `\Cref{fig:distributional_validation}` | `uv run python -m analysis_training.distributional_figure` | [`distributional_figure.py`](distributional_figure.py) |
| `\Cref{fig:posterior_snapshot}` | `uv run python -m analysis_training.posterior_snapshot --ckpt ... --data-dir ... --dataset-index ... --n-ctx ... --query-index ...` | [`posterior_snapshot.py`](posterior_snapshot.py) |
| `\Cref{fig:posterior_contraction}` | `uv run python -m analysis_training.posterior_contraction --ckpt-dir ... --data-dir ...` | [`posterior_contraction.py`](posterior_contraction.py) |
| `\Cref{fig:posterior_evolution}` | `uv run python -m analysis_training.posterior_evolution --data-dir ... --dataset-index ... --query-index ...` | [`posterior_evolution.py`](posterior_evolution.py) |

Artifacts derived from a *run* rather than from a config go through
[`wandb_history.py`](wandb_history.py), which fetches and caches history into
`results/history/`. Figures get their look from [`design.py`](design.py).

## `hparam_table` — the training hyperparameter table

```bash
uv run python -m analysis_training.hparam_table                          # -> results/training_hparams.{tex,csv}
uv run python -m analysis_training.hparam_table --print                  # also to stdout
uv run python -m analysis_training.hparam_table --script scripts/other.sh --set RUN=whole_2
uv run python -m analysis_training.hparam_table --out <paper>/assets/training/training_hparams.tex
```

**The table is generated, not written**, because the hand-written one already went stale: the
appendix's `tab:app_arch_hparams` records `B = 1000` over `[-4, 4]` where the production run used
`--nbins 2000 --cate_vmin -5 --cate_vmax 5`. Anything a reader could check against the launch
script is therefore read *from* the launch script.

Three things worth knowing before changing it:

- **The trainer's parser is the schema.** [`run_config.py`](run_config.py) resolves the script's
  `${VAR:-default}` assignments and line continuations, then feeds the flag tokens through
  `proximalfm.train._train_config.build_parser()`. So (a) a flag the script passes that the trainer
  no longer has **raises** rather than printing a stale table, and (b) a knob the script never
  passes (`--scheduler`, `--weight_decay`, the whole architecture block) is still reported at the
  value the run actually used, because it comes back as a parser default.
- **The ambient environment is never read.** `RUN=x` in your shell does not change the table; pass
  `--set RUN=x`, which is what a `VAR=` prefix on the launch would have done. Otherwise the same
  command would print different numbers on different machines.
- **Derived quantities use the trainer's own helpers.** The bin width `w`, the final `σ` in
  absolute units and its ratio to `w` come from `hl_bin_width` / `resolve_hl_final_sigma`, not from
  arithmetic repeated here — `σ` is a bias on the learned posterior width under
  `noise_aware` (see `notes_loss_and_sigma.tex`), so a table that disagreed with the
  trainer about it would misreport the one number that objective turns on.

Rows are declared in `build_rows` as `(group, label, value)` and the renderer is generic: adding a
row is one entry there. Groups become `\multirow` blocks, matching the shape of `tab:model_hparams`
so the two read as a pair. The table needs `booktabs` and `multirow` (both already in
`docs/latex/preamble.tex`) and `\modelname`.

The script parser covers exactly the shell the production scripts use — simple assignments, one
`python -m proximalfm.train` invocation, line continuations. A conditionally built flag list or a
command substitution feeding the invocation raises `ScriptParseError` rather than being guessed at.

## `arch_table` — the architecture table

```bash
uv run python -m analysis_training.arch_table --print        # -> results/model_hparams.{tex,csv}
uv run python -m analysis_training.arch_table --no-params    # skip instantiating the model
uv run python -m analysis_training.arch_table --out <paper>/assets/training/model_hparams.tex
```

`tab:model_hparams` — the architecture — built the same way as `hparam_table` above and from the
same launch script through the same `run_config`, so the two tables cannot disagree about the run.
It is the case that machinery exists for: `pretrain_prod.sh` passes **no architecture flag at all**,
so every row comes back as a `build_parser()` default, reported at the value the run actually used.

The hand-written table it replaces was wrong in four places, which is the argument for generating
it: it reported **4** attention heads in the column embedder and **4** in the in-context learner
where all three stages use the default **8**, and `B = 1000` over `[-4, 4]` where the run passes
`--nbins 2000 --cate_vmin -5 --cate_vmax 5`. The parameter total was stale as a *consequence* of the
last one — a 2000-bin head is 2.6M parameters, not 1.3M — so `28.7`M understated the model by nearly
a million. The current figures are 0.9M / 0.4M / 28.3M for the three stages, **29.6M** in total.

Three things worth knowing before changing it:

- **Parameter counts are measured, not transcribed.** `count_parameters` instantiates `ProximalFM`
  on the resolved config on the `meta` device — shapes are all the table needs, so no storage is
  allocated — and sums per stage. That is what makes the head's share track `--nbins` on its own.
  `MODEL_KEYS` lists the constructor's arguments explicitly, mirroring `_run.py::build_model`, so a
  new architecture flag has to be added here deliberately rather than appearing by signature magic.
  `--no-params` drops those rows and the torch import with them.
- **The reported grid follows `--loss_type`.** Under `noise_aware`/`cate_single` the single head
  scores `tau` on `[cate_vmin, cate_vmax]`; the two-head losses score `Y(a)` on `[vmin, vmax]`.
  Printing the wrong pair would misstate the hard support constraint §C.1 turns on.
- **Nothing is prettified past its value.** `power` renders `100000` as `$10^{5}$` only on an exact
  power of ten, and an unrecognised `--col_ssmax` is typeset verbatim rather than given a name the
  flag does not carry.

`Row`, the formatters and `render_latex` are shared with `hparam_table` rather than duplicated;
`render_latex` takes a `generator=` argument only so each generated file names the command that
regenerates it. `tests/test_analysis_training_arch.py` pins the derivations.

## `wandb_history` — getting a run's history off wandb

```bash
uv run python -m analysis_training.wandb_history --run ENTITY/PROJECT/RUN_ID --summary
uv run python -m analysis_training.wandb_history --run ENTITY/PROJECT/RUN_ID --keys loss/total synt_dist/crps
```

Pass the run as `--run ENTITY/PROJECT/RUN_ID` or set `PROXIMALFM_WANDB_RUN`.
Use the run ID rather than its display name: `--wandb_name` gets the launch
timestamp appended and a resume can change it.

The run logged with `--wandb_mode online`, so its history is on wandb's servers and **nothing has to
be copied off the cluster**. Fetches are cached as parquet in `results/history/` (gitignored), keyed
by run id and key set; `--force` refetches. Four traps, all handled in the module and all worth
knowing before you write your own query:

- **`run.history()` is a ~500-row downsample; `run.scan_history()` is exact.** Summing `train_time`
  over the former under-counts by two orders of magnitude. Only `scan_history` is used here.
- **`scan_history` pages, and each page boundary duplicates one row and drops another.** At
  `page_size=P` a 75k-step run comes back with `75000/P` duplicates and as many missing steps, and
  the missing indices *move* when `P` changes — which is how it was diagnosed. `fetch` de-duplicates
  on `_step`, and `History.total(key, n_steps)` reconstructs a sum as `mean × n_steps`, which the
  artifact cannot bias.
- **Keys on different logging intervals must be scanned separately.** `scan_history(keys=[...])`
  returns only rows carrying *every* requested key, so `train_time` (every step) plus
  `Charts/prior_eval_time` (every 200) yields their intersection — 376 rows instead of 75 000.
- **Two clocks.** `_runtime` accumulates process lifetimes; `_timestamp` is absolute.
  Interruptions make the calendar span longer than the compute occupancy. `RunMeta.gaps`
  locates the steps where the two clocks disagree.

`RunMeta` also carries the device from wandb's run metadata. Note `gpu_count` there is the **host's**
GPU count, not the run's: `pretrain_prod.sh` passes `--device cuda:0`, so
GPU-hours are occupancy times one device.

## `time_table` — where the GPU-hours went

```bash
uv run python -m analysis_training.time_table --print     # -> results/training_cost.{tex,csv}
```

| Component | Hours | % |
|---|---|---|
| Prior generation | 43.46 | 65.7 |
| Forward / backward | 18.12 | 27.4 |
| Validation, prior set (376 passes) | 1.27 | 1.9 |
| Validation, realistic set (751 passes) | 2.34 | 3.5 |
| Checkpointing, logging, startup (residual) | 0.96 | 1.5 |
| **GPU-hours** (1 × H100 NVL) | **66.15** | 100 |
| Training steps / datasets consumed / s per step | 75 000 · 4 800 000 · 3.17 | |

The headline is the first row: **prior generation is 2.4× the forward/backward cost**, because the
prior *is* the training set and is drawn once per use. That is what `--gen_reuse_passes` trades
against, and it is what a reader needs in order to price a replication.

The decomposition is exact, not indicative, and that rests on the run's configuration:
`--prior_device cuda:0` with `--train_dataloader_num_workers 0` makes generation *serial* with the
optimizer step, so the four measured components are disjoint intervals of one clock and the residual
is genuinely everything else. `build` **refuses** a script with dataloader workers (the slices would
overlap) and refuses a negative residual, rather than printing either.

**The table reports occupancy and says nothing about scheduling.** Only occupancy is compute,
and it is the only figure the table carries — the total and every percentage are against it.
Where the hours fell on a calendar is a fact about the cluster's queue, not about the method.
`RunMeta.gaps` still locates the boundaries, since that is what establishes *which* clock to use; it
just does not reach the table.

Sources are split deliberately: what was *configured* (steps, batch size) comes from the launch
script through `run_config`, the same path `hparam_table` reads, so the two tables cannot disagree
about the run; what it *cost* comes from wandb.

## `design` — the figure identity

Appendix C's figures sit next to Appendix B's, so they must be typographically the *same* figures.
[`design.py`](design.py) therefore **re-exports [`analysis_prior.design`](../analysis_prior/design.py)**
— text width, 9 pt serif, Okabe–Ito, constrained layout — rather than keeping a second copy, which
is what that module's own docstring argues against. It adds only two things:

- **`SERIES`**, a registry of colour per *logged quantity*, one level up from `analysis_prior`'s
  per-structural-variable `BLOCK_COLORS`: `loss/total` must be the same colour in every Appendix C
  figure. Same rule as there — never name a colour inline, ask the module. Unknown keys raise.
- **`binned` / `draw_binned`**, the reduction every per-step metric goes through: median and p25–p75
  band per log- or linearly-spaced bin. 75 000 raw points are noise; a fixed-width rolling window is
  the wrong smoother on a log axis (over-smooths the transient, under-smooths the plateau); and a
  *median* rather than a mean because the first steps spike far above the trend. The band is the
  gradient noise, which a single line would hide.

`reference_line` labels a "learned nothing" baseline on the line itself instead of in a legend, and
`panel_tag` moves a panel title left with an `(a)`/`(b)` prefix.

## `loss_figure` — the training curves

```bash
uv run python -m analysis_training.loss_figure     # -> results/training_curves.{pdf,png}
```

Four panels, each answering something the others cannot:

- **(a) the objective and its two terms, log step.** All the schedule's action is in the first 3 500
  of 75 000 steps, so a linear axis would render it as a vertical line. The reference line is
  `log(v_max − v_min) = log 10 = 2.3026`, the differential entropy of a **uniform density over the
  grid** — and the head's first logged step is 2.3037, i.e. it is initialised flat. That line is what
  makes the panel a scale rather than arbitrary units, and it is the distributional twin of (c)'s
  do-nothing line. The panel also shows that the distributional term *is* the objective to within a
  few percent: the mean penalty is a regulariser, not a co-equal half.
- **(b) the same, linear step, past the anneal.** Where a reader can see the objective still
  descending at 75 000 steps — the fact the training budget is an argument about. Carries the
  figure's only legend.
- **(c) the two schedules**, σ and the learning rate on twin axes. This panel exists for what it
  explains in (a): under `noise_aware` the loss is a cross-entropy against a target
  convolved with a kernel of width σ, so **shrinking σ sharpens the target and raises the attainable
  loss**. The level of the distributional term is therefore not comparable across the anneal, and a
  curve that moves inside the shaded window is the schedule, not the model. The shading in (a) marks
  that window; nothing else in the appendix would tell a reader to distrust that segment.
- **(d) the weighted mean term in detail.** This is the exact same
  `loss/cate_mse_weighted = lambda_cate × loss/cate_mse` curve shown in orange in (a) and (b),
  magnified on its own log y-axis; the grey region again marks the σ anneal, and the curve ends at
  **0.018**.

`(c)` sits under `(a)` on purpose, and `build_figure` pins the log-step limits of every panel that
uses one rather than letting them autoscale: a binned panel stops at its last bin's median step and a
raw one at the final step, so autoscaled limits put the shaded σ window at a *slightly different* x in
(a) than in (c) — the one thing the column layout exists to prevent. A test pins it.

### Why (d) is separate

The objective identity in (a) and (b),
`loss/total = loss/cate_dist + loss/cate_mse_weighted`, is visually meaningful because all three
curves share one y-axis. Putting the much smaller mean term on a twin axis in (b) would make its
apparent height and slope depend on arbitrary axis limits and obscure that it is a regularizer rather
than a co-equal half of the objective. Panel (d) preserves the shared scale in the objective panels
while giving the same orange curve enough vertical resolution to inspect its convergence.
