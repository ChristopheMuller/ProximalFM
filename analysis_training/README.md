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
| `\Cref{tab:training_hparams}` | `uv run python -m analysis_training.hparam_table` | [`hparam_table.py`](hparam_table.py) |
| `\Cref{tab:training_cost}` | `uv run python -m analysis_training.time_table` | [`time_table.py`](time_table.py) |

Artifacts derived from a *run* rather than from a config go through
[`wandb_history.py`](wandb_history.py), which fetches and caches history into
`results/history/`.

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
row is one entry there. Groups become `\multirow` blocks, matching the shape of
`tab:app_arch_hparams` so the two read as a pair. The table needs `booktabs` and `multirow`
(both already in `docs/latex/preamble.tex`) and `\modelname`.

The script parser covers exactly the shell the production scripts use — simple assignments, one
`python -m proximalfm.train` invocation, line continuations. A conditionally built flag list or a
command substitution feeding the invocation raises `ScriptParseError` rather than being guessed at.

## `wandb_history` — getting a run's history off wandb

```bash
uv run python -m analysis_training.wandb_history --summary
uv run python -m analysis_training.wandb_history --keys loss/total synt_dist/crps
```

The production run is `whole_1` = **`ProximalFM/9n413rsf`**, the module's `DEFAULT_RUN`. It is
identified by **id, not name**: `--wandb_name` gets the launch timestamp appended and a resume
rewrites it, so `whole_1_2026-08-14_23-44-45` is not a stable handle.

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
- **Two clocks.** `_runtime` accumulates process lifetimes; `_timestamp` is absolute. `whole_1` was
  interrupted after step 51 400 and resumed 4.9 h later, so `_runtime` (66.15 h) is the occupancy
  and the `_timestamp` span (71.05 h) is the calendar span. `RunMeta.gaps` locates such boundaries as
  the steps where the two clocks disagree.

`RunMeta` also carries the device from wandb's run metadata. Note `gpu_count` there is the **host's**
GPU count (4 H100 NVL on `morph`), not the run's: `pretrain_prod.sh` passes `--device cuda:0`, so
GPU-hours are occupancy × 1.

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

**The table reports occupancy and says nothing about scheduling.** `whole_1` was interrupted after
step 51 400 and resumed 4.9 h later, so its calendar span (71.05 h) exceeds its occupancy (66.15 h).
Only the occupancy is compute, and it is the only figure the table carries — the total and every
percentage are against it, so the table reads as the uninterrupted run it is equivalent to. Where
the hours fell on a calendar is a fact about the cluster's queue, not about the method.
`RunMeta.gaps` still locates the boundaries, since that is what establishes *which* clock to use; it
just does not reach the table.

Sources are split deliberately: what was *configured* (steps, batch size) comes from the launch
script through `run_config`, the same path `hparam_table` reads, so the two tables cannot disagree
about the run; what it *cost* comes from wandb.
