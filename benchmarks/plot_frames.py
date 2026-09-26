"""plot_frames.py — the numbers behind the figures.

One class, :class:`BenchmarkRun`, wrapping a finished benchmark output directory
(``results/results.csv``, ``data.pt``, ``preds/``). Every figure in
:mod:`benchmarks.plots` asks it for a frame and draws; none of them touches a
path, a pivot or a rank of its own.

Three things are done here exactly once, because each is easy to get subtly
wrong and impossible to spot afterwards in a picture:

**Complete blocks.** ``results.csv`` leaves a metric NaN on a *homogeneous*
episode (its denominator is Var(τ) ≈ 0) and on a *failed* one. A mean rank, a
Friedman test, a win-rate and a performance profile are all paired statistics:
they mean nothing unless every method compared is scored on the same episodes.
:meth:`BenchmarkRun.pivot` therefore drops any episode where any method is NaN,
and *reports how many it dropped* so a figure can say so out loud.

**References are not competitors.** ``Oracle-CATE(X)`` and friends are in the
frame as anchors, so they never enter a ranking, a win-rate or a Pareto front —
:meth:`competitors` excludes them and :meth:`references` is the separate list a
figure annotates with. This mirrors ``TaskReport.summary``, which ranks
``~is_reference`` for the same reason.

**Episode covariates are derived here, not stored.** ``data.pt`` carries the
latents (``U`` included), so the difficulty axes a stratified figure needs —
proxy informativeness, confounding strength, overlap, noise share — are
computable from it without a sidecar. They are deliberately *linear* R²
summaries: cheap, and honest about being a coarse ordering rather than a
structural quantity. A frozen set's ``frozen_meta.pt`` can be merged on top
(:meth:`covariates` ``meta=``) to add the generator's own sampled
hyperparameters as further axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch

from benchmarks.metrics import TASKS, Task, primary_task

#: Reference rows, in the order they should be annotated.
ORACLE_CATE = "Oracle-CATE(X)"
ORACLE_ATE = "Oracle-ATE"
ORACLE_ITE = "Oracle-ITE"


# ─────────────────────────────────────────────────────────────────────────────
# Small numeric helpers
# ─────────────────────────────────────────────────────────────────────────────


def _r2(design: np.ndarray, y: np.ndarray) -> float:
    """Out-of-nothing linear R² of ``y`` on ``design`` (intercept added).

    Averaged over the columns of a 2-D ``y``. In-sample on purpose: this is a
    descriptive ordering of episodes, not an estimate, and the context has
    thousands of rows against a handful of columns.
    """
    if y.ndim == 1:
        y = y[:, None]
    keep = y.var(axis=0) > 1e-12
    if not keep.any():
        return float("nan")
    y = y[:, keep]
    A = np.c_[np.ones(len(design)), design] if design.size else np.ones((len(y), 1))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(np.mean(1.0 - resid.var(axis=0) / y.var(axis=0)))


def _active(block: np.ndarray) -> np.ndarray:
    """Drop an episode block's zero-padding columns (kept: any variance)."""
    if block.ndim != 2 or block.shape[1] == 0:
        return block
    return block[:, block.var(axis=0) > 1e-12]


def terciles(values: pd.Series) -> pd.Series:
    """Low/Mid/High labels by tercile, NaN-safe, collapsing on ties.

    A near-degenerate axis (``d_W`` when every episode has the same width) has
    no terciles; it comes back all-NaN and the caller drops the axis rather than
    drawing three identical points.
    """
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().sum() < 6 or v.nunique(dropna=True) < 3:
        return pd.Series(np.nan, index=values.index, dtype="object")
    try:
        cut = pd.qcut(v, 3, labels=["Low", "Mid", "High"], duplicates="drop")
    except ValueError:
        return pd.Series(np.nan, index=values.index, dtype="object")
    if cut.cat.categories.size < 3:
        return pd.Series(np.nan, index=values.index, dtype="object")
    return cut.astype("object")


# ─────────────────────────────────────────────────────────────────────────────
# The run
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BenchmarkRun:
    """A finished benchmark output directory, as the figures want to read it."""

    dir: Path
    results: pd.DataFrame
    _data: Optional[dict] = field(default=None, repr=False)
    _covariates: Optional[pd.DataFrame] = field(default=None, repr=False)

    # ── Loading ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, exp_dir: Path | str) -> "BenchmarkRun":
        exp_dir = Path(exp_dir)
        csv = exp_dir / "results" / "results.csv"
        if not csv.exists():
            raise SystemExit(
                f"{csv} not found — the benchmark has not produced results in "
                f"{exp_dir} yet. Run benchmarks/run.py first."
            )
        df = pd.read_csv(csv)
        if "target" not in df.columns:
            raise SystemExit(
                f"{csv} predates the three-task scoring (no 'target' column). "
                "Re-run benchmarks/run.py — it rescores from the cached preds/ "
                "and does not refit anything."
            )
        for col in ("is_reference", "failed", "homogeneous"):
            if col in df.columns:
                df[col] = df[col].astype(bool)
        return cls(exp_dir, df)

    @property
    def results_dir(self) -> Path:
        return self.dir / "results"

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plots"

    # ── Tasks and frames ─────────────────────────────────────────────────────

    @property
    def tasks(self) -> list[Task]:
        """The tasks this run actually scored, in registry order."""
        present = set(self.results["target"])
        return [t for t in TASKS.values() if t.name in present]

    @property
    def primary(self) -> Task:
        return TASKS[primary_task(sorted(set(self.results["target"])))]

    @property
    def n_ctx(self) -> int:
        """Context size shown by the standard figures (the largest run)."""
        if "n_ctx" in self.results.columns:
            return int(self.results["n_ctx"].max())
        return int(self.data["N_TRAIN"])

    def frame(self, task: Task) -> pd.DataFrame:
        rows = self.results[self.results["target"] == task.name]
        if "n_ctx" in rows.columns:
            rows = rows[rows["n_ctx"] == self.n_ctx]
        return rows

    def competitor_frame(self, task: Task) -> pd.DataFrame:
        f = self.frame(task)
        return f[~f["is_reference"]]

    def reference_frame(self, task: Task) -> pd.DataFrame:
        f = self.frame(task)
        return f[f["is_reference"]]

    def references(self, task: Task) -> list[str]:
        f = self.reference_frame(task)
        order = [ORACLE_CATE, ORACLE_ATE, ORACLE_ITE]
        present = list(dict.fromkeys(f["method"]))
        return sorted(present, key=lambda m: (order.index(m) if m in order else 99, m))

    def estimands(self, task: Task) -> dict[str, str]:
        f = self.frame(task)
        return dict(zip(f["method"], f["estimand"]))

    def n_episodes(self, task: Task) -> int:
        return int(self.frame(task)["episode"].nunique())

    # ── Pivots, ranks, order ─────────────────────────────────────────────────

    def pivot(
        self,
        task: Task,
        key: str,
        *,
        complete: bool = True,
        include_references: bool = False,
    ) -> tuple[pd.DataFrame, int]:
        """Episode × method matrix of one metric, plus the episodes dropped.

        With ``complete=True`` (the default, and what every paired statistic
        needs) an episode survives only if *every* remaining method scored it.
        Methods that scored nothing at all are dropped first, so one broken
        method cannot empty the block.
        """
        f = self.frame(task) if include_references else self.competitor_frame(task)
        if key not in f.columns:
            return pd.DataFrame(), 0
        wide = f.pivot_table(index="episode", columns="method", values=key)
        wide = wide.dropna(axis=1, how="all")
        n_before = len(wide)
        if complete:
            wide = wide.dropna(axis=0, how="any")
        return wide, n_before - len(wide)

    def ranks(self, task: Task, *, complete: bool = True) -> pd.DataFrame:
        """Episode × method ranks on ``task.rank_by`` (1 = best), competitors
        only — an oracle taking rank 1 would shift every real method."""
        wide, _ = self.pivot(task, task.rank_by, complete=complete)
        if wide.empty:
            return wide
        return wide.rank(axis=1, method="min", ascending=True)

    def mean_rank(self, task: Task, *, complete: bool = True) -> pd.Series:
        r = self.ranks(task, complete=complete)
        return r.mean().sort_values() if not r.empty else pd.Series(dtype=float)

    def method_order(self, task: Task) -> list[str]:
        """Competitors best-first: by mean rank, falling back to the median of
        the ranking metric when no complete block exists."""
        order = self.mean_rank(task)
        if not order.empty:
            return list(order.index)
        f = self.competitor_frame(task)
        if task.rank_by not in f.columns:
            return sorted(f["method"].unique())
        med = f.groupby("method")[task.rank_by].median().sort_values()
        return list(med.index)

    def summary_stat(self, task: Task, key: str, how: str = "median") -> pd.Series:
        """One number per method (references included) for metric ``key``."""
        f = self.frame(task)
        if key not in f.columns:
            return pd.Series(dtype=float)
        return getattr(f.groupby("method")[key], how)()

    def failure_rate(self, task: Task) -> pd.Series:
        return self.frame(task).groupby("method")["failed"].mean()

    def mean_time(self, task: Task) -> pd.Series:
        return self.frame(task).groupby("method")["time"].mean()

    # ── Raw data and predictions ─────────────────────────────────────────────

    @property
    def data(self) -> dict:
        if self._data is None:
            path = self.dir / "data.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found — the row-level figures need the cached "
                    "standardized data. Re-run benchmarks/run.py."
                )
            self._data = torch.load(path, weights_only=False)
        return self._data

    def targets(self) -> dict[str, np.ndarray]:
        """The per-row ground truths, keyed as the tasks name them."""
        d = self.data
        out = {"ite": np.asarray(d["true_ITEs"], dtype=float)}
        if d.get("true_CATEs") is not None:
            out["cate"] = np.asarray(d["true_CATEs"], dtype=float)
        return out

    def target_for(self, task: Task) -> tuple[Optional[str], Optional[np.ndarray]]:
        targets = self.targets()
        kind = task.target_kind(targets)
        return kind, (None if kind is None else targets[kind])

    def preds(self, method: str) -> Optional[np.ndarray]:
        swept = self.dir / "preds" / f"{method}_nctx-{self.n_ctx}_preds.npz"
        legacy = self.dir / "preds" / f"{method}_preds.npz"
        path = swept if swept.exists() else legacy
        if not path.exists():
            return None
        return np.load(path)["preds"]

    def available_preds(self) -> list[str]:
        d = self.dir / "preds"
        if not d.is_dir():
            return []
        suffix = f"_nctx-{self.n_ctx}_preds"
        swept = [p.stem[: -len(suffix)] for p in d.glob(f"*{suffix}.npz")]
        legacy = [p.stem[: -len("_preds")] for p in d.glob("*_preds.npz")]
        legacy = [name for name in legacy if "_nctx-" not in name]
        return sorted(set(swept + legacy))

    # ── Episode covariates ───────────────────────────────────────────────────

    def covariates(self, meta: Optional[Path | str] = None) -> pd.DataFrame:
        """Per-episode difficulty axes, indexed by episode.

        Everything here is computed from ``data.pt`` alone — the standardized
        blocks *and* the latent ``U``, which the benchmark dict keeps. The
        R²-based axes are linear and in-sample: they order episodes, they do not
        measure a structural quantity, and the column names say which is which.

        ``meta`` optionally points at a frozen set's **root** (the directory
        holding ``val/`` and ``frozen_meta.pt``); its per-episode scalar
        ``sampled_hp`` entries are merged in as extra axes prefixed ``hp_``.
        """
        if self._covariates is None:
            self._covariates = self._derive_covariates()
        cov = self._covariates
        if meta is not None:
            cov = cov.join(self._sidecar_covariates(Path(meta)), how="left")
        return cov

    def _derive_covariates(self) -> pd.DataFrame:
        d = self.data
        n_train = self.n_ctx
        X, W, Z, U = (np.asarray(d[k], dtype=float) for k in ("X", "W", "Z", "U"))
        A = np.asarray(d["A"], dtype=float).reshape(len(X), -1)
        Y = np.asarray(d["Y"], dtype=float).reshape(len(X), -1)
        d_X, d_W, d_Z = (np.asarray(d[k]).astype(int) for k in ("d_X", "d_W", "d_Z"))
        targets = self.targets()

        rows = []
        for b in range(int(d["N_BATCH"])):
            ctx = slice(0, n_train)
            Xb, Wb, Zb = X[b, ctx, : d_X[b]], W[b, ctx, : d_W[b]], Z[b, ctx, : d_Z[b]]
            Ub = _active(U[b, ctx])
            Ab, Yb = A[b, ctx], Y[b, ctx]
            has_u = Ub.shape[1] > 0

            row = {
                "episode": b,
                "d_X": int(d_X[b]),
                "d_W": int(d_W[b]),
                "d_Z": int(d_Z[b]),
                "d_U": int(Ub.shape[1]),
                "treated_frac": float(Ab.mean()),
                # Proxy informativeness: how much of the latent a linear read of
                # each proxy block recovers. Weak proxies are where proximal
                # identification is fragile, so this is the first cut to make.
                "proxy_R2_W": _r2(Wb, Ub) if has_u else np.nan,
                "proxy_R2_Z": _r2(Zb, Ub) if has_u else np.nan,
                # Confounding strength: what U adds *beyond X* to each mechanism.
                # Both must be non-trivial for the episode to be confounded at
                # all — U acting on Y alone is not confounding.
                "confounding_A": (
                    _r2(np.c_[Xb, Ub], Ab) - _r2(Xb, Ab) if has_u else np.nan
                ),
                "confounding_Y": (
                    _r2(np.c_[Xb, Ab[:, None], Ub], Yb)
                    - _r2(np.c_[Xb, Ab[:, None]], Yb)
                    if has_u
                    else np.nan
                ),
            }
            # Overlap, from a linear-probability read of the propensity: the
            # mean of min(p, 1-p), so 0.5 is a coin flip and 0 is deterministic
            # assignment. Crude by construction; it orders episodes.
            design = np.c_[Xb, Ub] if has_u else Xb
            p = np.clip(_fitted(design, Ab), 0.0, 1.0)
            row["overlap"] = float(np.minimum(p, 1 - p).mean())

            if "cate" in targets:
                tau = targets["cate"][b]
                row["hetero_var"] = float(tau.var())
                if "ite" in targets:
                    ite = targets["ite"][b]
                    row["noise_share"] = float(
                        1.0 - tau.var() / ite.var() if ite.var() > 0 else np.nan
                    )
            rows.append(row)

        cov = pd.DataFrame(rows).set_index("episode")
        s2 = self.frame(self.primary).groupby("episode")["s2_over_var"].median()
        if s2.notna().any():
            cov["s2_over_var"] = s2
        return cov

    def _sidecar_covariates(self, root: Path) -> pd.DataFrame:
        """Scalar ``sampled_hp`` entries from a frozen set's ``frozen_meta.pt``.

        Read directly rather than through ``prior_eval`` — this module must keep
        working in a checkout where ``proximalfm`` is not importable, the same
        constraint ``data.py`` carries.
        """
        path = Path(root) / "frozen_meta.pt"
        if not path.exists():
            print(f"  ! no frozen_meta.pt under {root} — skipping sidecar axes")
            return pd.DataFrame()
        meta = torch.load(path, weights_only=False, map_location="cpu")
        rows = {}
        for rec in meta.get("episodes", []):
            hp = dict(rec.get("sampled_hp") or {})
            scalars = {}
            for k, v in hp.items():
                if isinstance(v, torch.Tensor):
                    v = v.item() if v.numel() == 1 else None
                if isinstance(v, (int, float, bool, np.integer, np.floating)):
                    scalars[f"hp_{k}"] = float(v)
            rows[int(rec["index"])] = scalars
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "episode"
        return frame

    def difficulty(self, task: Task) -> pd.Series:
        """Per-episode median of the ranking metric across competitors — how
        hard the field found the episode, and the axis the row-level figures
        pick their example episodes along."""
        wide, _ = self.pivot(task, task.rank_by, complete=False)
        return wide.median(axis=1) if not wide.empty else pd.Series(dtype=float)


def _fitted(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.c_[np.ones(len(y)), design] if design.size else np.ones((len(y), 1))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef


def select_methods(
    order: Iterable[str], top: int, *, always: Iterable[str] = ()
) -> list[str]:
    """The best ``top`` methods, plus any named in ``always`` that exist.

    Figures with one line per method stop being readable past a handful; this is
    the one place that decides which survive, so every such figure cuts the same
    way and the reader can compare them.
    """
    order = list(order)
    keep = order[:top]
    for name in always:
        matches = [m for m in order if m.startswith(name)]
        for m in matches:
            if m not in keep:
                keep.append(m)
    return [m for m in order if m in keep]
