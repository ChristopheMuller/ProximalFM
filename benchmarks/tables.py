# %%
"""tables.py — Export the benchmark summaries as LaTeX tables.

run.py writes one summary per *task* — the three views of the same run:

    summary_cate.csv   tau*(x) = E[Y1-Y0 | X], the Monte-Carlo CATE target.
                       The headline table: methods that answer tau(x), ranked
                       on the estimand ProximalFM is trained on.
    summary_ite.csv    the realized Y1-Y0. Adds the methods that read the query
                       row's own proxies (an estimand naming w or z), which is a
                       genuine advantage here and no help at all on the CATE
                       target; read them against the Oracle-CATE(X) row, the
                       floor no tau(x) method can go below.
    summary_ate.csv    the level only, and the one table where a method that
                       returns a single scalar per episode (P2SLS, the
                       `marginal` strategies) says anything.

This script is a thin front-end: the columns, their formats, the bolding
direction and the caption all live on the ``Task`` objects in
[metrics.py](metrics.py), so the console table and these LaTeX tables cannot
drift apart. Methods are ordered by exact predict-time input block and then by
Mean Rank computed only within that block. Reference rows (the oracles) carry
no Mean Rank and are printed last: they are anchors, not competitors.
"""
from pathlib import Path

from benchmarks.metrics import TASKS, TaskReport

# %% Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Resolve paths relative to the project root so the script works regardless of
# the current working directory (e.g. run as a `# %%` cell or from benchmarks/).
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:  # __file__ is undefined in some interactive runners
    PROJECT_ROOT = Path.cwd()
    while (
        not (PROJECT_ROOT / "pyproject.toml").exists()
        and PROJECT_ROOT != PROJECT_ROOT.parent
    ):
        PROJECT_ROOT = PROJECT_ROOT.parent

experiment_dir = PROJECT_ROOT / "results/test_tbd/results/"
methods_to_include: list[str] = []  # empty → all methods


# %% Emit one table per available task
# ─────────────────────────────────────────────────────────────────────────────
def main(results_dir: Path, methods: list[str] | None = None) -> None:
    found = False
    for name, task in TASKS.items():
        path = results_dir / f"summary_{name}.csv"
        if not path.exists():
            continue
        found = True
        print(f"% ── {name} ──")
        print(TaskReport.from_summary_csv(path, task).latex(methods))

    if not found:
        raise FileNotFoundError(
            f"No summary_*.csv in {results_dir}. Run run.py first, or fix "
            "experiment_dir above."
        )


main(experiment_dir, methods_to_include)
