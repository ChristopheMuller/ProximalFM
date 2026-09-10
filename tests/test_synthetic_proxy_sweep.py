import numpy as np
import torch

from benchmarks.methods.proximalfm import ProximalFM_Estimator, _MODEL_CACHE
from proximalfm._model import ProximalFM
from synthetic_proxy_sweep.run_ablation import fit_proximalfm, restrict_context


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
    n_ctx, n_q = 8, 5
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

    np.testing.assert_array_equal(restricted["row_id"], [0, 1, 2, 8, 9, 10, 11, 12])
    np.testing.assert_array_equal(restricted["split"], [0, 0, 0, 1, 1, 1, 1, 1])


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
    assert details["n_train"] == 8
    assert details["n_test"] == 5
