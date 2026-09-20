import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.methods.proximalfm import ProximalFM_Estimator, _MODEL_CACHE
from proximalfm._model import ProximalFM
import analysis_synthetic_proxy_sweep_linear.run_ablation as sweep
from analysis_synthetic_proxy_sweep_linear.run_ablation import (
    BACKDOOR_METHODS,
    CAUSALPFN_RUNTIME_SENTINEL,
    ORACLE,
    RESULT_FIELDS,
    fit_backdoor_foundation_model,
    fit_method,
    fit_cache_signature,
    fit_proximalfm,
    ensure_causalpfn_safe_runtime,
    prepare_fit_cache,
    restrict_aligned,
    restrict_context,
)


MODEL_KWARGS = dict(
    embed_dim=16,
    col_num_blocks=1,
    col_nhead=2,
    col_num_inds=8,
    row_num_blocks=1,
    row_nhead=2,
    row_num_cls=4,
    icl_num_blocks=1,
    icl_nhead=2,
    nbins=32,
)


def make_observed(seed=0):
    rng = np.random.default_rng(seed)
    n_ctx, n_q = 12, 5
    n = n_ctx + n_q
    return {
        "row_id": np.arange(n),
        "split": np.r_[np.zeros(n_ctx, dtype=np.uint8), np.ones(n_q, dtype=np.uint8)],
        "X": rng.normal(size=(n, 2)),
        "W": rng.normal(size=(n, 1)),
        "Z": rng.normal(size=(n, 1)),
        "A": (rng.random(n) < 0.5).astype(np.uint8),
        "Y": rng.normal(size=n),
    }


def test_restrict_context_keeps_a_train_prefix_and_every_query_row():
    observed = make_observed()
    restricted = restrict_context(observed, 3)

    np.testing.assert_array_equal(restricted["row_id"], [0, 1, 2, 12, 13, 14, 15, 16])
    np.testing.assert_array_equal(restricted["split"], [0, 0, 0, 1, 1, 1, 1, 1])


def test_linear_methods_fit_only_the_shared_context_prefix():
    observed = make_observed()

    for method in ("proximal_2sls", "backdoor_ols_x", "backdoor_ols_xwz"):
        _, details = fit_method(restrict_context(observed, 11), method)
        assert details["n_train"] == 11
        assert details["n_test"] == 5


def test_oracle_covariates_use_the_same_context_and_query_rows():
    observed = make_observed()
    restricted = restrict_context(observed, 11)
    oracle_u = restrict_aligned(np.arange(17)[:, None], observed["split"], 11)

    _, details = fit_method(restricted, ORACLE, oracle_u)

    assert details["n_train"] == 11
    assert details["n_test"] == 5


def test_backdoor_foundation_models_use_proxy_marginalizer(monkeypatch):
    import benchmarks.methods as benchmark_methods

    observed = make_observed()
    built = []

    class FakeEstimator:
        def fit(self, X, A, Y, *, W=None, Z=None):
            self.fit_args = (X, A, Y, W, Z)
            return self

        def predict(self, X, *, W=None, Z=None):
            self.predict_X = X
            return np.full(len(X), 0.25)

    def fake_build(config, compute):
        estimator = FakeEstimator()
        built.append((config, compute, estimator))
        return config["name"], estimator

    monkeypatch.setattr(benchmark_methods, "build_method", fake_build)
    restricted = restrict_context(observed, 9)

    for method in BACKDOOR_METHODS.values():
        predictions, details = fit_backdoor_foundation_model(
            restricted,
            method,
            device="mps",
            proxy_samples=7,
            seed=13,
            predict_batch_size=64,
            tabicl_estimators=2,
        )

        config, compute, estimator = built[-1]
        assert config["cls"] == "ProxyMarginalizer"
        assert config["integration"] == {
            "method": "empirical",
            "n_points": 7,
            "seed": 13,
        }
        assert config["predict_batch_size"] == 64
        assert config["estimator"]["use_W"] is True
        assert config["estimator"]["use_Z"] is True
        if method == BACKDOOR_METHODS["tabicl"]:
            assert config["estimator"]["cls"] == "S_Learner"
            assert config["estimator"]["base"] == "tabicl"
            assert config["estimator"]["tabicl"] == {
                "kv_cache": True,
                "n_estimators": 2,
            }
        else:
            assert config["estimator"]["cls"] == "CausalPFN"
        assert compute.device == "mps"
        assert compute.execution == "inline"
        assert len(estimator.fit_args[0]) == 9
        assert len(estimator.predict_X) == 5
        np.testing.assert_array_equal(predictions, np.full(5, 0.25))
        assert details["n_train"] == 9
        assert details["n_test"] == 5


def test_causalpfn_relaunches_once_with_safe_thread_environment(monkeypatch):
    relaunched = []
    monkeypatch.delenv(CAUSALPFN_RUNTIME_SENTINEL, raising=False)
    monkeypatch.setattr(
        sweep.os,
        "execvpe",
        lambda executable, argv, environment: relaunched.append(
            (executable, argv, environment)
        ),
    )

    ensure_causalpfn_safe_runtime(["causalpfn"])

    assert len(relaunched) == 1
    environment = relaunched[0][2]
    assert environment[CAUSALPFN_RUNTIME_SENTINEL] == "1"
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert environment[name] == "1"
    monkeypatch.setenv(CAUSALPFN_RUNTIME_SENTINEL, "1")
    ensure_causalpfn_safe_runtime(["causalpfn"])
    ensure_causalpfn_safe_runtime(["tabicl"])
    assert len(relaunched) == 1


def test_previous_results_seed_and_then_load_the_fit_cache(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "results"
    data_dir.mkdir()
    output_dir.mkdir()
    index_path = data_dir / "index.csv"
    index_path.write_text("dimension,level_id\n1,0\n")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    args = SimpleNamespace(
        data_dir=data_dir,
        output_dir=output_dir,
        proximalfm_checkpoint=checkpoint,
        proximalfm_device="mps",
        proximalfm_query_batch_size=512,
        proximalfm_ensemble=1,
        force=False,
    )
    row = {field: "" for field in RESULT_FIELDS}
    row.update(
        dimension="1",
        joint_r2_target="0.5",
        level_id="0",
        method="proximalfm",
        context_size="128",
        status="ok",
        n_train="128",
        n_test="5",
        observed_path="observed.npz",
        truth_path="truth.npz",
    )
    with (output_dir / "method_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    signature = fit_cache_signature(args, index_path)
    metadata = {
        "index_sha256": signature["index_sha256"],
        "numpy": signature["numpy"],
        "scipy": signature["scipy"],
        "proximalfm": {
            "checkpoint": str(checkpoint),
            "device": "mps",
            "query_batch_size": 512,
            "ensemble": 1,
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata))

    seeded, cache_path, _ = prepare_fit_cache(args, index_path)
    loaded, loaded_path, _ = prepare_fit_cache(args, index_path)

    assert len(seeded) == len(loaded) == 1
    assert loaded_path == cache_path
    assert cache_path.is_file()


def test_second_run_computes_only_the_new_context_size(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "results"
    data_dir.mkdir()
    observed = make_observed()
    truth = {
        "CATE": np.zeros(len(observed["split"])),
        "row_id": observed["row_id"],
    }
    np.savez(data_dir / "observed.npz", **observed)
    np.savez(data_dir / "truth.npz", **truth)
    index_path = data_dir / "index.csv"
    index_path.write_text(
        "dimension,joint_r2_target,level_id,observed_path,truth_path,n_train\n"
        "2,0.5,0,observed.npz,truth.npz,12\n"
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    def fake_result(observed, method=None, *args, **kwargs):
        split = observed["split"]
        calls.append((method or "proximalfm", int((split == 0).sum())))
        predictions = np.zeros(int((split == 1).sum()))
        return predictions, {
            "status": "ok",
            "n_train": int((split == 0).sum()),
            "n_test": int((split == 1).sum()),
            "prediction_max_abs": 0.0,
        }

    monkeypatch.setattr(sweep, "fit_method", fake_result)
    monkeypatch.setattr(sweep, "fit_proximalfm", fake_result)
    monkeypatch.setattr(sweep, "fit_backdoor_foundation_model", fake_result)
    args = SimpleNamespace(
        data_dir=data_dir,
        output_dir=output_dir,
        dimensions=None,
        levels=None,
        limit=None,
        context_sizes=[8, 10],
        proximalfm_checkpoint=checkpoint,
        proximalfm_device="mps",
        proximalfm_query_batch_size=512,
        proximalfm_ensemble=1,
        backdoor_methods=["tabicl"],
        backdoor_device="mps",
        backdoor_proxy_samples=16,
        backdoor_seed=0,
        backdoor_predict_batch_size=4096,
        tabicl_estimators=1,
        oracle=False,
        fail_fast=True,
        force=False,
    )

    assert sweep.run(args) == 0
    assert len(calls) == 10
    args.context_sizes = [8, 10, 11]
    assert sweep.run(args) == 0

    assert len(calls) == 15
    assert {context for _, context in calls[10:]} == {11}
    with (output_dir / "run_metadata.json").open() as stream:
        metadata = json.load(stream)
    assert metadata["cache"]["hits"] == 10
    assert metadata["cache"]["computed"] == 5


def test_parallel_workers_match_serial_results(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    observed = make_observed()
    truth = {"CATE": np.zeros(len(observed["split"])), "row_id": observed["row_id"]}
    for setting in range(2):
        np.savez(data_dir / f"observed_{setting}.npz", **observed)
        np.savez(data_dir / f"truth_{setting}.npz", **truth)
    (data_dir / "index.csv").write_text(
        "dimension,joint_r2_target,level_id,observed_path,truth_path,n_train\n"
        + "".join(
            f"2,0.5,{setting},observed_{setting}.npz,truth_{setting}.npz,12\n"
            for setting in range(2)
        )
    )
    args = SimpleNamespace(
        data_dir=data_dir,
        output_dir=tmp_path / "serial",
        dimensions=None,
        levels=None,
        limit=None,
        context_sizes=[12],
        proximalfm_checkpoint=None,
        proximalfm_device="cpu",
        proximalfm_query_batch_size=512,
        proximalfm_ensemble=1,
        backdoor_methods=[],
        backdoor_device="cpu",
        backdoor_proxy_samples=16,
        backdoor_seed=0,
        backdoor_predict_batch_size=4096,
        tabicl_estimators=1,
        oracle=False,
        fail_fast=True,
        force=False,
        workers=1,
    )
    assert sweep.run(args) == 0
    args.output_dir = tmp_path / "parallel"
    args.workers = 2
    assert sweep.run(args) == 0

    def scored_rows(path):
        with path.open(newline="") as stream:
            return [
                {key: value for key, value in row.items() if key != "elapsed_seconds"}
                for row in csv.DictReader(stream)
            ]

    assert scored_rows(tmp_path / "serial" / "method_results.csv") == scored_rows(
        tmp_path / "parallel" / "method_results.csv"
    )
    with (tmp_path / "parallel" / "run_metadata.json").open() as stream:
        assert json.load(stream)["workers"] == 2
    assert sweep.run(args) == 0
    with (tmp_path / "parallel" / "run_metadata.json").open() as stream:
        assert json.load(stream)["cache"]["hits"] == 6
    args.backdoor_methods = ["causalpfn"]
    args.backdoor_device = "mps"
    with pytest.raises(ValueError, match="requires CPU devices"):
        sweep.run(args)


def test_chunked_proximalfm_predictions_match_one_query_pass(tmp_path):
    torch.manual_seed(0)
    model = ProximalFM(**MODEL_KWARGS).eval()
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"config": MODEL_KWARGS, "state_dict": model.state_dict()}, checkpoint)
    observed = make_observed()

    _MODEL_CACHE.clear()
    chunked, details = fit_proximalfm(
        observed,
        checkpoint=checkpoint,
        device="cpu",
        query_batch_size=2,
        ensemble=1,
    )

    train = observed["split"] == 0
    query = ~train
    adapter = ProximalFM_Estimator(str(checkpoint), device="cpu")
    adapter.fit(
        observed["X"][train],
        observed["A"][train],
        observed["Y"][train],
        W=observed["W"][train],
        Z=observed["Z"][train],
    )
    unchunked = adapter.predict(
        observed["X"][query], W=observed["W"][query], Z=observed["Z"][query]
    )

    np.testing.assert_allclose(chunked, unchunked, atol=1e-5)
    assert details["n_train"] == 12
    assert details["n_test"] == 5
