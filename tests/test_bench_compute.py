"""Tests for benchmarks/compute.py — the single owner of device and threads.

What is worth pinning here is not the dataclass but the three decisions the module
exists to make, each of which used to be made in several places at once:

1. The hard migration. `run.n_workers` / `run.device` and per-method `torch_threads:`
   / `n_jobs:` are *refused*, not aliased — `run.device` in particular used to reach
   the `models:` block only, so silently mapping it to a global default would change
   what an old config does without saying so.
2. Where a method runs. A non-CPU device must never be combined with process fan-out
   (concurrent multi-process Metal/CUDA access stalls — the constraint that used to be
   hardcoded inside DFPV_Wrapper and ignored by NMMR).
3. Pool sharing. Two methods with the same budget must land on the *same* pool, or the
   session-wide reuse this refactor is for does not happen.
"""

import warnings
import sys
from unittest import mock

import pytest

from benchmarks.compute import (
    INLINE,
    POOL,
    ComputeConfig,
    LegacyComputeKeyError,
    WorkerManager,
    apply_compute,
    parse_global_compute,
    resolve_device,
    resolve_method_compute,
)


def test_apply_compute_limits_loaded_faiss_runtime():
    faiss = mock.Mock()
    with mock.patch.dict(sys.modules, {"faiss": faiss}):
        apply_compute(ComputeConfig(blas_threads=2))
        apply_compute(ComputeConfig(blas_threads=1))
    assert faiss.omp_set_num_threads.call_args_list == [mock.call(2), mock.call(1)]


# ── 1. Hard migration ────────────────────────────────────────────────────────


@pytest.mark.parametrize("key,value", [("n_workers", 6), ("device", "mps")])
def test_legacy_run_keys_raise(key, value):
    with pytest.raises(LegacyComputeKeyError) as exc:
        parse_global_compute({"seed": 1, key: value})
    assert f"run.compute.{key}" in str(exc.value)


@pytest.mark.parametrize("key,value", [("torch_threads", 2), ("n_jobs", -1)])
def test_legacy_method_keys_raise(key, value):
    base = ComputeConfig()
    with pytest.raises(LegacyComputeKeyError) as exc:
        resolve_method_compute(base, {"name": "S-OLS", "cls": "S_Learner", key: value})
    assert "compute:" in str(exc.value)
    assert "S-OLS" in str(exc.value)


def test_unknown_compute_key_raises():
    with pytest.raises(ValueError, match="unknown compute key"):
        parse_global_compute({"compute": {"n_wrokers": 6}})


# ── 2. Resolution ────────────────────────────────────────────────────────────


def test_global_defaults_are_a_cpu_pool():
    c = parse_global_compute({"compute": {"n_workers": 4}})
    assert (c.device, c.n_workers, c.execution) == ("cpu", 4, POOL)
    assert c.torch_threads == c.blas_threads == c.n_jobs == 1


def test_method_block_overrides_and_inherits():
    base = parse_global_compute(
        {"compute": {"n_workers": 6, "blas_threads": 2, "n_jobs": 3}}
    )
    c = resolve_method_compute(base, {"name": "m", "compute": {"torch_threads": 4}})
    assert c.torch_threads == 4  # overridden
    assert (c.n_workers, c.blas_threads, c.n_jobs) == (6, 2, 3)  # inherited
    assert c.execution == base.execution


def test_absent_method_block_returns_the_global_unchanged():
    base = parse_global_compute({"compute": {"n_workers": 6}})
    assert resolve_method_compute(base, {"name": "m"}) is base


def test_execution_defaults_to_inline_for_a_device_or_a_lone_worker():
    # A method that asks for a GPU is asking to hold it in one process; inheriting
    # the global's `pool` would deadlock, so the default is re-derived.
    base = parse_global_compute({"compute": {"n_workers": 6}})
    assert base.execution == POOL

    with mock.patch("torch.backends.mps.is_available", return_value=True):
        gpu = resolve_method_compute(base, {"name": "m", "compute": {"device": "mps"}})
    assert gpu.execution == INLINE

    solo = resolve_method_compute(base, {"name": "m", "compute": {"n_workers": 1}})
    assert solo.execution == INLINE


def test_explicit_execution_is_respected():
    base = parse_global_compute({"compute": {"n_workers": 6}})
    c = resolve_method_compute(
        base, {"name": "m", "compute": {"n_workers": 1, "execution": "pool"}}
    )
    assert c.execution == POOL


def test_device_with_pool_fanout_is_refused():
    """The constraint DFPV used to hardcode and NMMR used to ignore."""
    with pytest.raises(ValueError, match="execution='pool'"):
        ComputeConfig(device="mps", n_workers=6, execution=POOL)
    # ...and the same combination is legal the moment it is inline.
    assert ComputeConfig(device="mps", n_workers=6, execution=INLINE).device == "mps"


def test_unavailable_device_falls_back_to_cpu_with_a_warning():
    with mock.patch("torch.backends.mps.is_available", return_value=False):
        with pytest.warns(UserWarning, match="mps"):
            assert resolve_device("mps") == "cpu"
    with mock.patch("torch.cuda.is_available", return_value=False):
        with pytest.warns(UserWarning, match="cuda"):
            assert resolve_device("cuda") == "cpu"


def test_available_device_is_kept():
    with mock.patch("torch.backends.mps.is_available", return_value=True):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_device("mps") == "mps"


def test_bad_values_are_rejected():
    with pytest.raises(ValueError):
        resolve_device("tpu")
    with pytest.raises(ValueError):
        ComputeConfig(execution="threads")
    with pytest.raises(ValueError):
        ComputeConfig(n_workers=0)


# ── 3. Pool sharing ──────────────────────────────────────────────────────────


def test_pool_key_ignores_torch_threads_but_not_blas():
    # torch threads are set per task, so they must not fragment the pool; BLAS can
    # only be set in the initializer, so it must.
    a = ComputeConfig(n_workers=4, torch_threads=1)
    b = ComputeConfig(n_workers=4, torch_threads=8)
    c = ComputeConfig(n_workers=4, blas_threads=2)
    assert a.pool_key == b.pool_key
    assert a.pool_key != c.pool_key


def test_same_budget_reuses_one_pool():
    manager = WorkerManager()
    a = ComputeConfig(n_workers=2, execution=POOL)
    b = ComputeConfig(n_workers=2, torch_threads=4, execution=POOL)
    c = ComputeConfig(n_workers=3, execution=POOL)
    with mock.patch("benchmarks.compute.ProcessPoolExecutor") as pool_cls:
        pool_cls.side_effect = lambda **kw: mock.Mock(name=str(kw["max_workers"]))
        first = manager.get_pool(a)
        assert manager.get_pool(b) is first  # same key → same pool
        assert manager.get_pool(c) is not first  # different width → new pool
    assert manager.n_pools == 2
    manager.close()
    assert manager.n_pools == 0


def _double(x):
    return x * 2


def test_inline_map_runs_in_process_without_a_pool():
    manager = WorkerManager()
    inline = ComputeConfig(n_workers=8, execution=INLINE)
    with mock.patch("benchmarks.compute.ProcessPoolExecutor") as pool_cls:
        assert sorted(manager.map(_double, [1, 2, 3], inline)) == [2, 4, 6]
        pool_cls.assert_not_called()
    assert manager.n_pools == 0


def test_pooled_map_returns_every_result():
    # Results come back as-completed, so only the multiset is guaranteed.
    with WorkerManager() as manager:
        compute = ComputeConfig(n_workers=2, execution=POOL)
        assert sorted(manager.map(_double, range(4), compute)) == [0, 2, 4, 6]
