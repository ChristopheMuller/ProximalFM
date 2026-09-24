"""The benchmark adapter preserves ProximalFM's predictions."""

import numpy as np
import torch

from benchmarks.compute import ComputeConfig
from benchmarks.methods import build_method
from proximalfm._model import ProximalFM


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


def test_benchmark_adapter_matches_direct_proximalfm_prediction(tmp_path):
    torch.manual_seed(0)
    model = ProximalFM(**MODEL_KWARGS).eval()
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {"config": MODEL_KWARGS, "state_dict": model.state_dict(), "curr_step": 1},
        checkpoint,
    )
    rng = np.random.default_rng(0)
    n_ctx, n_q = 8, 3
    X, W, Z = (rng.normal(size=(n_ctx + n_q, d)).astype("float32") for d in (3, 2, 2))
    A, Y = rng.integers(0, 2, n_ctx).astype("float32"), rng.normal(size=n_ctx).astype(
        "float32"
    )
    _, adapter = build_method(
        {"name": "PFM", "cls": "ProximalFM", "checkpoint": str(checkpoint)},
        ComputeConfig(),
    )
    adapter.fit(X[:n_ctx], A, Y, W=W[:n_ctx], Z=Z[:n_ctx])
    got = adapter.predict(X[n_ctx:], W=W[n_ctx:], Z=Z[n_ctx:])
    expected = model.predict(
        X=torch.tensor(X)[None],
        W=torch.tensor(W)[None],
        Z=torch.tensor(Z)[None],
        A=torch.tensor(A)[None],
        y_train=torch.tensor(Y)[None],
        d_X=torch.tensor([3]),
        d_W=torch.tensor([2]),
        d_Z=torch.tensor([2]),
        target="cate",
        output_type="mean",
    ).numpy()[0]
    np.testing.assert_allclose(got, expected, atol=1e-5)
