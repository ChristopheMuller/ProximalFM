"""compute.py — the single owner of device and thread settings for the benchmark.

Why this module exists
──────────────────────
Compute used to be spread over three places that did not know about each other: an
unconditional BLAS pin at `methods.py` import time, a `torch.set_num_threads` inside the
run script's pool worker, and a handful of estimators re-pinning to 1 in their `__init__`
(so the worker's setting was silently overridden). Device was worse: one global string
that only the ProximalFM path read, while `NMMR` auto-grabbed MPS in *every* pool worker
and `DFPV` hardcoded an mps→cpu downgrade to survive the resulting collision.

The arrangement here inverts that. A method is handed a resolved `ComputeConfig` and is
then free to fan its episodes out through the shared `WorkerManager` or to keep them
in-process — but *nothing* below this module sets a thread count or picks a device on its
own. `apply_compute` is called exactly once per process-and-method, and it is the only
caller of `torch.set_num_threads`.

Three shapes of method have to fit in one schema:

    cheap, CPU, episode-independent   S/T/X learners, P2SLS, KPV, PMMR
        → `execution: pool`, many workers, one thread each
    torch, CPU, warm module cache     CausalFM, DoPFN, CausalPFN, `base: tabicl`
        → `execution: pool`, fewer workers, several torch threads each
    device-resident                   NMMR/DFPV on mps, ProximalFM
        → `execution: inline`, one process that holds the device

The third is why `execution` is a separate field rather than something derived from the
worker count: concurrent multi-process Metal access stalls, so "on the GPU" and "fanned
out over processes" are mutually exclusive and the config has to be able to say so.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Iterable, Iterator

import torch

__all__ = [
    "ComputeConfig",
    "LegacyComputeKeyError",
    "WorkerManager",
    "apply_compute",
    "parse_global_compute",
    "resolve_device",
    "resolve_method_compute",
]

# Execution modes. `pool` fans episodes out over the shared process pool; `inline`
# runs them in this process, one after another, so a device stays resident.
POOL = "pool"
INLINE = "inline"
_EXECUTIONS = (POOL, INLINE)

_DEVICES = ("cpu", "mps", "cuda")

# Names of the BLAS thread-count env vars, set in the pool initializer *before* the
# child imports anything numeric.
_BLAS_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


class LegacyComputeKeyError(KeyError):
    """A config still uses a pre-`compute:` spelling.

    Raised rather than aliased on purpose: the old keys meant something subtly
    different (`run.device` reached the neural models *only*), so silently mapping
    them onto the new schema would change what a config does without saying so.
    """

    def __str__(self) -> str:  # KeyError repr quotes the message; we don't want that
        return self.args[0]


@dataclass(frozen=True)
class ComputeConfig:
    """The resolved compute budget for one method (or for the run as a whole).

    Attributes:
        device: `cpu` | `mps` | `cuda`. Already validated against what this machine
            actually has — see `resolve_device`.
        n_workers: episode fan-out width. 1 means no pool at all.
        torch_threads: intra-process torch threads. Only bites the torch-backed
            methods; the sklearn ones do not use them.
        blas_threads: OMP/MKL/OpenBLAS threads per process. Part of the pool identity,
            because it can only be set before the child imports numpy/torch.
        n_jobs: intra-method sklearn/joblib fan-out (CausalForest, the P-learner's
            forest). Distinct from `n_workers`: this one nests *inside* a worker, so
            `n_workers * n_jobs` is what actually lands on the machine.
        execution: `pool` or `inline`.
    """

    device: str = "cpu"
    n_workers: int = 1
    torch_threads: int = 1
    blas_threads: int = 1
    n_jobs: int = 1
    execution: str = POOL

    def __post_init__(self) -> None:
        if self.device not in _DEVICES:
            raise ValueError(
                f"compute.device must be one of {_DEVICES}, got {self.device!r}."
            )
        if self.execution not in _EXECUTIONS:
            raise ValueError(
                f"compute.execution must be one of {_EXECUTIONS}, "
                f"got {self.execution!r}."
            )
        for name in ("n_workers", "torch_threads", "blas_threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"compute.{name} must be >= 1.")
        # n_jobs follows the sklearn convention where -1 means "every core".
        if self.n_jobs == 0:
            raise ValueError("compute.n_jobs must be >= 1 or -1 (all cores).")
        if self.device != "cpu" and self.execution == POOL and self.n_workers > 1:
            raise ValueError(
                f"compute: device={self.device!r} cannot be combined with "
                f"execution='pool' and n_workers={self.n_workers}. Concurrent "
                "multi-process access to Metal/CUDA from a spawn pool stalls; use "
                "execution: inline to hold the device in one process."
            )

    @property
    def pool_key(self) -> tuple[int, int]:
        """Pools are shared by everything that agrees on these two numbers.

        `torch_threads` is deliberately *not* part of the key — it is set per task by
        `apply_compute`, so one pool serves methods that want different torch thread
        counts. `blas_threads` is, because it can only be set in the initializer.
        """
        return (self.n_workers, self.blas_threads)

    def describe(self) -> str:
        if self.execution == INLINE:
            where = "inline"
        else:
            where = f"{self.n_workers}w"
        return (
            f"{self.device}/{where} "
            f"(torch={self.torch_threads}, blas={self.blas_threads}, "
            f"n_jobs={self.n_jobs})"
        )


_FIELD_NAMES = {f.name for f in fields(ComputeConfig)}


def resolve_device(name: str, *, what: str = "compute") -> str:
    """Validate a requested device against this machine, falling back to CPU.

    The benchmark is routinely carried between a laptop with MPS and a CUDA cluster
    with the same config file, and `torch.device("mps")` on a box without it does not
    fail until the first `.to()`, deep inside a worker, as an opaque error.
    """
    name = str(name).lower()
    if name not in _DEVICES:
        raise ValueError(f"{what}.device must be one of {_DEVICES}, got {name!r}.")
    if name == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            f"{what}: device='cuda' unavailable; falling back to CPU.", stacklevel=2
        )
        return "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        warnings.warn(
            f"{what}: device='mps' unavailable; falling back to CPU.", stacklevel=2
        )
        return "cpu"
    return name


def _default_execution(device: str, n_workers: int) -> str:
    """`inline` whenever fanning out would be pointless or unsafe."""
    if device != "cpu" or n_workers <= 1:
        return INLINE
    return POOL


def _check_unknown(block: dict, what: str) -> None:
    unknown = set(block) - _FIELD_NAMES
    if unknown:
        raise ValueError(
            f"{what}: unknown compute key(s) {sorted(unknown)}. "
            f"Valid keys: {sorted(_FIELD_NAMES)}."
        )


def parse_global_compute(run_cfg: dict) -> ComputeConfig:
    """Build the run-wide default from the `run:` section of a config file.

    Refuses the pre-`compute:` spellings outright.
    """
    for legacy, new in (("n_workers", "n_workers"), ("device", "device")):
        if legacy in run_cfg:
            raise LegacyComputeKeyError(
                f"config `run.{legacy}` has moved to `run.compute.{new}`. Note that "
                f"`run.device` used to apply to the `models:` block only, so the "
                f"direct translation is a `compute: {{device: ...}}` on each model "
                f"entry, not a global default."
            )

    block = dict(run_cfg.get("compute") or {})
    _check_unknown(block, "run.compute")

    device = resolve_device(block.get("device", "cpu"), what="run.compute")
    n_workers = int(block.get("n_workers", max(1, (os.cpu_count() or 4) - 2)))
    execution = block.get("execution") or _default_execution(device, n_workers)
    return ComputeConfig(
        device=device,
        n_workers=n_workers,
        torch_threads=int(block.get("torch_threads", 1)),
        blas_threads=int(block.get("blas_threads", 1)),
        n_jobs=int(block.get("n_jobs", 1)),
        execution=execution,
    )


def resolve_method_compute(base: ComputeConfig, method_cfg: dict) -> ComputeConfig:
    """Overlay one method's `compute:` block on the run-wide default.

    Unset fields inherit. `execution` is re-derived when the method changes the device
    or the worker count without naming it explicitly, so `compute: {device: mps}` on a
    single method does the obviously-intended thing (hold the GPU in one process)
    rather than inheriting a `pool` that would deadlock.
    """
    name = method_cfg.get("name", method_cfg.get("cls", "<unnamed>"))
    for legacy in ("torch_threads", "n_jobs"):
        if legacy in method_cfg:
            raise LegacyComputeKeyError(
                f"method {name!r}: `{legacy}:` has moved into the method's "
                f"`compute:` block — write `compute: {{{legacy}: "
                f"{method_cfg[legacy]!r}}}`."
            )

    block = dict(method_cfg.get("compute") or {})
    _check_unknown(block, f"method {name!r} compute")
    if not block:
        return base

    device = (
        resolve_device(block["device"], what=f"method {name!r} compute")
        if "device" in block
        else base.device
    )
    n_workers = int(block.get("n_workers", base.n_workers))

    if "execution" in block:
        execution = block["execution"]
    elif "device" in block or "n_workers" in block:
        execution = _default_execution(device, n_workers)
    else:
        execution = base.execution

    return replace(
        base,
        device=device,
        n_workers=n_workers,
        torch_threads=int(block.get("torch_threads", base.torch_threads)),
        blas_threads=int(block.get("blas_threads", base.blas_threads)),
        n_jobs=int(block.get("n_jobs", base.n_jobs)),
        execution=execution,
    )


def apply_compute(compute: ComputeConfig) -> None:
    """Apply the thread budget to the current process. The only caller of set_num_threads.

    Safe to call repeatedly (a pool worker calls it once per task, since consecutive
    tasks may belong to methods with different budgets). `set_num_interop_threads`
    raises once the interop pool has been used, which is why it is caught rather than
    guarded — there is nothing to do about it and nothing depends on it.
    """
    try:
        torch.set_num_threads(int(compute.torch_threads))
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _pool_init(blas_threads: int) -> None:
    """Pool worker bootstrap: pin BLAS *before* the child imports anything numeric.

    This runs in the spawned child ahead of any task, which is the only point where
    OMP/MKL/OpenBLAS thread counts still have an effect — they are read at library
    load time. Setting them here is what lets `methods.py` stop doing it
    unconditionally at import.
    """
    for var in _BLAS_ENV_VARS:
        os.environ[var] = str(blas_threads)
    # FAISS (pulled in by CausalPFN) and torch each ship their own OpenMP; loading
    # both trips a duplicate-runtime abort on macOS unless this is set. Both
    # spellings, because the benchmark has always set the typo'd one.
    os.environ.setdefault("KMP_DUPLICATE_OK", "True")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")


class WorkerManager:
    """Session-wide owner of the process pools that methods fan their episodes into.

    One pool per distinct `(n_workers, blas_threads)`, created lazily on first use and
    kept for the rest of the run — so the respawn, the re-import of torch and the
    submodule checkpoint loads are paid once per *pool*, not once per method as they
    used to be, and the module-level model caches in `methods.py` stay warm across
    methods that share a budget.

    The cost of that reuse: a worker that has run CausalFM may later be handed NMMR,
    with the previous competitor's `src.*` still in `sys.modules`.
    `methods._switch_src_context` exists precisely to unpick that and is what makes
    the sharing safe — it is also the first thing to suspect if a submodule-backed
    method starts behaving oddly under a shared pool.
    """

    def __init__(self, *, mp_context: Any = None) -> None:
        self._pools: dict[tuple[int, int], ProcessPoolExecutor] = {}
        self._mp_context = mp_context

    # -- pools ---------------------------------------------------------------

    def get_pool(self, compute: ComputeConfig) -> ProcessPoolExecutor:
        """The pool serving this budget, creating it on first request."""
        key = compute.pool_key
        pool = self._pools.get(key)
        if pool is None:
            kwargs: dict[str, Any] = {
                "max_workers": compute.n_workers,
                "initializer": _pool_init,
                "initargs": (compute.blas_threads,),
            }
            if self._mp_context is not None:
                kwargs["mp_context"] = self._mp_context
            pool = ProcessPoolExecutor(**kwargs)
            self._pools[key] = pool
        return pool

    @property
    def n_pools(self) -> int:
        return len(self._pools)

    # -- work ----------------------------------------------------------------

    def map(
        self,
        fn: Callable[[Any], Any],
        items: Iterable[Any],
        compute: ComputeConfig,
    ) -> Iterator[Any]:
        """Yield `fn(item)` for every item, as completed and in no particular order.

        `execution: inline` (and any single-worker budget) runs in this process and
        never touches a pool — which is what makes `--no-parallel` a config override
        rather than a second code path, and what lets a device-resident method reuse
        the same runner.

        `fn` must be a module-level function and every item picklable: the pool is a
        `spawn` pool, so nothing is inherited from the parent.
        """
        items = list(items)
        if compute.execution == INLINE or compute.n_workers <= 1:
            apply_compute(compute)
            for item in items:
                yield fn(item)
            return

        pool = self.get_pool(compute)
        futures = [pool.submit(fn, item) for item in items]
        for future in as_completed(futures):
            yield future.result()

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        for pool in self._pools.values():
            pool.shutdown(wait=True)
        self._pools.clear()

    def __enter__(self) -> "WorkerManager":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
