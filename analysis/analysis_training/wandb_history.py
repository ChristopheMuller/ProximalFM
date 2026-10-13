"""Fetch and cache a training run's wandb history.

```bash
uv run python -m analysis.analysis_training.wandb_history --keys train_time prior_time
uv run python -m analysis.analysis_training.wandb_history --summary
```

The run is read from wandb's servers with the public API rather than from the
cluster: `--wandb_mode online` means the history is already there, so nothing
has to be copied off `morph`. Fetches are cached as parquet under
``results/history/`` (gitignored) so a figure re-render costs no network.

Three properties, all load-bearing:

- **``scan_history`` is exact, ``history`` is not.** ``Run.history()`` returns a
  ~500-row *downsample* of the run; summing ``train_time`` over it would
  under-count by two orders of magnitude. Only :func:`fetch` is used here, and
  it goes through ``scan_history``.
- **``scan_history`` pages, and its page boundaries are lossy.** Every page
  boundary duplicates one row and drops a different one: at ``page_size=P`` a
  75k-step run comes back with ``75000/P`` duplicates and as many missing
  steps, and the missing step indices move when ``P`` changes. So a raw sum over
  the scan is wrong by ~1 row per page in each direction. :func:`fetch`
  therefore de-duplicates on ``_step`` and records the *coverage*, and
  :func:`total` reconstructs a sum as ``mean x n_steps`` — an estimator the
  artifact cannot bias, because the dropped rows are dropped for a reason
  (pagination) that has nothing to do with their values.
- **A key logged on an interval must be scanned alone.**
  ``scan_history(keys=[...])`` returns only rows where *every* requested key is
  present, so asking for ``train_time`` (every step) together with
  ``Charts/prior_eval_time`` (every 200 steps) yields their intersection. One
  :func:`fetch` per cadence.

:class:`RunMeta` carries what the history cannot: the device, and the
**interruption structure**. ``whole_1`` was resumed once, and wandb's
``_runtime`` accumulates process lifetimes while ``_timestamp`` is absolute — so
``_runtime`` is the run's *occupancy* (the honest GPU-hours) and the
``_timestamp`` span is the calendar span. Reading the latter as compute would
over-count by the 4.9 h the job spent dead. :attr:`RunMeta.gaps` finds those
boundaries as the steps where the two disagree.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "analysis" / "analysis_training" / "results" / "history"

# The production run. An id, not a name: `wandb_name` has the launch timestamp
# appended and a resume rewrites it, so the name is not a stable handle.
DEFAULT_RUN = "ProximalFM/9n413rsf"  # whole_1

# A page large enough that the boundary artifact costs a handful of rows out of
# 75k, small enough that the API does not time out.
PAGE_SIZE = 10_000

# Gaps below this are step-to-step jitter, not an interruption.
GAP_THRESHOLD_S = 300.0


class WandbUnavailableError(RuntimeError):
    """Raised when the API cannot be reached and no cache is present."""


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Gap:
    """An interruption: wall clock advanced, the run's own clock did not."""

    step: int
    seconds: float
    resumed_at: dt.datetime


@dataclass(frozen=True)
class RunMeta:
    """What a run's history cannot say about itself."""

    path: str
    id: str
    name: str
    state: str
    final_step: int
    runtime_s: float  # wandb `_runtime`: summed process lifetimes.
    started_at: dt.datetime  # First history row, not `created_at`.
    ended_at: dt.datetime
    gpu: str
    gpu_count_host: int  # The *host's* GPUs. The run used --device cuda:0.
    host: str
    gaps: tuple[Gap, ...] = ()

    @property
    def span_s(self) -> float:
        """Calendar span, interruptions included."""
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def idle_s(self) -> float:
        """Wall clock the run existed but was not running."""
        return sum(gap.seconds for gap in self.gaps)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "id": self.id,
            "name": self.name,
            "state": self.state,
            "final_step": self.final_step,
            "runtime_s": self.runtime_s,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "gpu": self.gpu,
            "gpu_count_host": self.gpu_count_host,
            "host": self.host,
            "gaps": [
                {
                    "step": g.step,
                    "seconds": g.seconds,
                    "resumed_at": g.resumed_at.isoformat(),
                }
                for g in self.gaps
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RunMeta":
        return cls(
            **{
                **{k: v for k, v in payload.items() if k != "gaps"},
                "started_at": dt.datetime.fromisoformat(payload["started_at"]),
                "ended_at": dt.datetime.fromisoformat(payload["ended_at"]),
                "gaps": tuple(
                    Gap(
                        step=int(g["step"]),
                        seconds=float(g["seconds"]),
                        resumed_at=dt.datetime.fromisoformat(g["resumed_at"]),
                    )
                    for g in payload.get("gaps", ())
                ),
            }
        )


@dataclass
class History:
    """A fetched key set, de-duplicated and sorted by step."""

    frame: pd.DataFrame
    keys: tuple[str, ...]
    covered: int = field(init=False)

    def __post_init__(self) -> None:
        self.covered = len(self.frame)

    def coverage(self, n_steps: int) -> float:
        return self.covered / n_steps if n_steps else 0.0

    def total(self, key: str, n_steps: int | None = None) -> float:
        """Sum of ``key`` over the whole run.

        ``mean x n_steps`` rather than a raw sum: ``scan_history``'s page
        boundaries drop ~one row per page, and a raw sum would silently
        under-count by that many. With ``n_steps=None`` this is the plain sum.
        """
        series = self.frame[key].dropna()
        if n_steps is None:
            return float(series.sum())
        return float(series.mean()) * n_steps


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #


def _cache_path(run_path: str, keys: Sequence[str], cache_dir: Path) -> Path:
    digest = hashlib.sha256("\x00".join(sorted(keys)).encode()).hexdigest()[:12]
    return cache_dir / f"{run_path.replace('/', '_')}__{digest}.parquet"


def _resolve_path(run: str, api) -> str:
    """Accept ``entity/project/id``, ``project/id`` or a bare id."""
    parts = run.split("/")
    if len(parts) == 3:
        return run
    if len(parts) == 2:
        return f"{api.default_entity}/{run}"
    return f"{api.default_entity}/ProximalFM/{run}"


def _utc(timestamp: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc)


def fetch(
    keys: Sequence[str],
    *,
    run: str = DEFAULT_RUN,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
    page_size: int = PAGE_SIZE,
) -> History:
    """Fetch ``keys`` (plus ``_step``, ``_runtime``, ``_timestamp``), cached.

    Every requested key must share a logging cadence; see the module docstring.
    """
    keys = tuple(dict.fromkeys(["_step", "_runtime", "_timestamp", *keys]))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(run, keys, cache_dir)

    if path.exists() and not force:
        return History(frame=pd.read_parquet(path), keys=keys)

    import wandb

    api = wandb.Api(timeout=300)
    handle = api.run(_resolve_path(run, api))
    rows = list(handle.scan_history(keys=list(keys), page_size=page_size))
    if not rows:
        raise WandbUnavailableError(f"{run} returned no rows for keys {keys}")

    frame = (
        pd.DataFrame(rows)
        .drop_duplicates(subset="_step", keep="first")
        .sort_values("_step")
        .reset_index(drop=True)
    )
    frame.to_parquet(path)
    return History(frame=frame, keys=keys)


def fetch_meta(
    *,
    run: str = DEFAULT_RUN,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
) -> RunMeta:
    """Fetch the run's metadata and its interruption structure."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{run.replace('/', '_')}__meta.json"
    if path.exists() and not force:
        return RunMeta.from_dict(json.loads(path.read_text()))

    import wandb

    api = wandb.Api(timeout=300)
    resolved = _resolve_path(run, api)
    handle = api.run(resolved)
    metadata = handle.metadata or {}

    # `_runtime` and `_timestamp` come from the same scan the timing table uses,
    # so the two never disagree about where the boundaries are.
    frame = fetch(
        ["train_time"], run=run, cache_dir=cache_dir, force=force
    ).frame  # every step

    wall = frame["_timestamp"].diff()
    own = frame["_runtime"].diff()
    idle = wall - own
    # `itertuples` renames leading-underscore columns positionally; index by
    # label instead so `_step` stays `_step`.
    breaks = frame.assign(idle=idle).loc[idle > GAP_THRESHOLD_S]
    gaps = tuple(
        Gap(
            step=int(row["_step"]),
            seconds=float(row["idle"]),
            resumed_at=_utc(float(row["_timestamp"])),
        )
        for _, row in breaks.iterrows()
    )

    meta = RunMeta(
        path=resolved,
        id=handle.id,
        name=handle.name,
        state=handle.state,
        final_step=int(frame["_step"].max()),
        runtime_s=float(frame["_runtime"].iloc[-1]),
        started_at=_utc(float(frame["_timestamp"].iloc[0])),
        ended_at=_utc(float(frame["_timestamp"].iloc[-1])),
        gpu=str(metadata.get("gpu", "unknown")),
        gpu_count_host=int(metadata.get("gpu_count", 0) or 0),
        host=str(metadata.get("host", "unknown")),
        gaps=gaps,
    )
    path.write_text(json.dumps(meta.to_dict(), indent=2))
    return meta


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=DEFAULT_RUN, help=f"(default: {DEFAULT_RUN})")
    parser.add_argument("--keys", nargs="*", default=[], help="Metric keys to fetch.")
    parser.add_argument("--force", action="store_true", help="Ignore the cache.")
    parser.add_argument("--summary", action="store_true", help="Print run metadata.")
    args = parser.parse_args(argv)

    meta = fetch_meta(run=args.run, force=args.force)
    if args.summary or not args.keys:
        print(f"{meta.name}  ({meta.path}, {meta.state})")
        print(f"  steps      {meta.final_step:,}")
        print(f"  device     {meta.gpu} on {meta.host}")
        print(f"  started    {meta.started_at:%Y-%m-%d %H:%M} UTC")
        print(f"  ended      {meta.ended_at:%Y-%m-%d %H:%M} UTC")
        print(f"  occupancy  {meta.runtime_s / 3600:.2f} h")
        print(f"  span       {meta.span_s / 3600:.2f} h")
        for gap in meta.gaps:
            print(
                f"  restart    step {gap.step:,}: idle {gap.seconds / 3600:.2f} h, "
                f"resumed {gap.resumed_at:%Y-%m-%d %H:%M} UTC"
            )

    if args.keys:
        history = fetch(args.keys, run=args.run, force=args.force)
        print(
            f"fetched {len(history.frame):,} rows "
            f"({history.coverage(meta.final_step):.3%} of steps) "
            f"for {', '.join(args.keys)}"
        )


if __name__ == "__main__":
    main()
