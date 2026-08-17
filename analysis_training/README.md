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
