import torch


import pytest


from proximalfm._model import ProximalFM


NBINS = 100


def build_model(**overrides):
    """Build a miniature ProximalFM for fast testing."""
    kwargs = dict(
        embed_dim=16,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=4,
        icl_num_blocks=1,
        icl_nhead=2,
        nbins=NBINS,
    )
    kwargs.update(overrides)
    return ProximalFM(**kwargs)


def build_data(B=2, T=15, train_size=10, H_X=5, H_Z=3, H_W=4, H_A=1):
    """Synthetic inputs matching the miniature model."""
    X = torch.randn(B, T, H_X)
    Z = torch.randn(B, T, H_Z)
    W = torch.randn(B, T, H_W)
    A = torch.randn(B, T, H_A)

    # y_train only covers the first `train_size` rows
    y_train = torch.randn(B, train_size)

    # Simulate dynamic feature counts (assuming full capacity for simplicity here)
    d_X = torch.full((B,), H_X, dtype=torch.long)
    d_Z = torch.full((B,), H_Z, dtype=torch.long)
    d_W = torch.full((B,), H_W, dtype=torch.long)

    return (X, Z, W, A, y_train, d_X, d_Z, d_W), (B, T, train_size)


@pytest.fixture
def setup_model_and_data():
    """
    Fixture to initialize a minimal TabICL model and synthetic data.
    """
    inputs, dims = build_data()
    return build_model(), inputs, dims


def test_tabicl_padding_logic(setup_model_and_data):
    """
    Validates that the model can handle a batch where datasets have fewer
    features than the maximum allocated tensor size.
    """
    model, inputs, dims = setup_model_and_data
    X, Z, W, A, y_train, _, _, _ = inputs
    B, T, train_size = dims
    test_size = T - train_size

    # artificially lower the number of valid features for the batch
    d_X_reduced = torch.tensor([X.shape[2] - 1, X.shape[2] - 2], dtype=torch.long)
    d_Z_reduced = torch.tensor([Z.shape[2] - 1, Z.shape[2] - 1], dtype=torch.long)
    d_W_reduced = torch.tensor([W.shape[2] - 2, W.shape[2] - 1], dtype=torch.long)

    model.train()
    out = model(X, Z, W, A, y_train, d_X=d_X_reduced, d_Z=d_Z_reduced, d_W=d_W_reduced)

    expected_shape = (B, test_size, 2 * NBINS)
    assert (
        out.shape == expected_shape
    ), "Model failed to handle reduced feature counts properly."


def test_cate_single_is_shift_invariant():
    """Shifting every outcome by a constant must not move the predicted CATE.

    tau is a difference, so de-standardisation uses the context *scale* only; the
    context mean cancels. This is the invariant that catches the whole class of
    de-standardisation bug (adding y_context_mean to a difference).
    """
    torch.manual_seed(0)
    model = build_model(loss_type="cate_single")
    inputs, _ = build_data()
    X, Z, W, A, y_train, d_X, d_Z, d_W = inputs

    shift = 100.0
    base = model.predict(X, Z, W, A, y_train, target="cate", d_X=d_X, d_Z=d_Z, d_W=d_W)
    shifted = model.predict(
        X, Z, W, A, y_train + shift, target="cate", d_X=d_X, d_Z=d_Z, d_W=d_W
    )

    assert torch.allclose(base, shifted, atol=1e-4), (
        "CATE prediction changed when all outcomes were shifted by a constant: "
        "the de-standardisation is adding the context mean to a difference."
    )


def test_cate_single_predict_output_types():
    """distribution / mean / quantiles are all supported and mutually consistent."""
    model = build_model(loss_type="cate_single")
    inputs, (B, T, train_size) = build_data()
    X, Z, W, A, y_train, d_X, d_Z, d_W = inputs
    test_size = T - train_size
    kwargs = dict(target="cate", d_X=d_X, d_Z=d_Z, d_W=d_W)

    dist = model.predict(X, Z, W, A, y_train, output_type="distribution", **kwargs)
    assert dist["weights"].shape == (B, test_size, NBINS)
    assert torch.allclose(dist["weights"].sum(-1), torch.ones(B, test_size), atol=1e-5)

    mean = model.predict(X, Z, W, A, y_train, output_type="mean", **kwargs)
    assert mean.shape == (B, test_size)

    qs = model.predict(
        X,
        Z,
        W,
        A,
        y_train,
        output_type="quantiles",
        quantiles=[0.1, 0.5, 0.9],
        **kwargs,
    )
    assert qs.shape == (B, test_size, 3)
    # Quantiles are monotone in the requested levels.
    assert (qs[..., 0] <= qs[..., 1]).all() and (qs[..., 1] <= qs[..., 2]).all()


def test_cate_single_checkpoint_surgery():
    """Warm-starting from a two-head checkpoint drops exactly the final decoder layer."""
    from proximalfm.train._run import get_parameter_state

    torch.manual_seed(0)
    old_model = build_model(loss_type="bins_po")
    new_model = build_model(loss_type="cate_single")

    rules = {"default": "freeze", "icl_predictor.decoder.2.": "scratch"}

    new_state = new_model.state_dict()
    buffer_names = set(dict(new_model.named_buffers()).keys())

    transferred = {}
    for name, weight in old_model.state_dict().items():
        if name in buffer_names:
            continue
        if name in new_state and weight.shape == new_state[name].shape:
            if get_parameter_state(name, rules) in ("finetune", "freeze"):
                transferred[name] = weight

    missing = {
        name
        for name in new_model.state_dict()
        if name not in transferred and name not in buffer_names
    }
    assert missing == {
        "icl_predictor.decoder.2.weight",
        "icl_predictor.decoder.2.bias",
    }, f"Unexpected set of non-transferred parameters: {sorted(missing)}"

    result = new_model.load_state_dict(transferred, strict=False)
    assert result.unexpected_keys == []
