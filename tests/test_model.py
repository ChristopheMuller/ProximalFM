import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from proximalfm.Model import (
    LayerStack,
    PerFeatureTransformerCATE,
    TYPE_X, TYPE_Z, TYPE_W, TYPE_A, TYPE_Y, TYPE_MASK, NUM_TYPES,
    isolate_torch_rng
)

class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    def forward(self, x, **kwargs):
        return self.linear(x)

def test_layer_stack_initialization():
    layer_creator = lambda: DummyLayer()
    num_layers = 5
    stack = LayerStack(
        layer_creator=layer_creator,
        num_layers=num_layers
    )

    assert len(stack.layers) == num_layers
    assert stack.num_layers == num_layers
    assert stack.min_num_layers_layer_dropout == num_layers
    assert not stack.recompute_each_layer

def test_layer_stack_forward_no_dropout():
    layer_creator = lambda: DummyLayer()
    stack = LayerStack(
        layer_creator=layer_creator,
        num_layers=3
    )

    x = torch.randn(2, 10)
    out = stack(x)
    assert out.shape == (2, 10)

def test_layer_stack_forward_half_layers():
    layer_creator = lambda: DummyLayer()
    stack = LayerStack(
        layer_creator=layer_creator,
        num_layers=4
    )

    # Mocking the layers to track calls
    for layer in stack.layers:
        layer.forward = MagicMock(side_effect=lambda x, **kwargs: x + 1)

    x = torch.zeros(2, 10)
    out = stack(x, half_layers=True)

    # Should only pass through half the layers
    assert torch.allclose(out, torch.full_like(out, 2))
    stack.layers[0].forward.assert_called_once()
    stack.layers[1].forward.assert_called_once()
    stack.layers[2].forward.assert_not_called()
    stack.layers[3].forward.assert_not_called()

def test_layer_stack_forward_half_layers_asserts():
    layer_creator = lambda: DummyLayer()
    stack = LayerStack(
        layer_creator=layer_creator,
        num_layers=4,
        min_num_layers_layer_dropout=2
    )

    x = torch.zeros(2, 10)
    with pytest.raises(AssertionError, match="half_layers only works without layer dropout"):
        stack(x, half_layers=True)

@patch("proximalfm.Model.torch.randint")
def test_layer_stack_layer_dropout(mock_randint):
    mock_randint.return_value = torch.tensor([2]) # Execute 2 layers

    layer_creator = lambda: DummyLayer()
    stack = LayerStack(
        layer_creator=layer_creator,
        num_layers=4,
        min_num_layers_layer_dropout=1
    )

    # Mocking layers
    for layer in stack.layers:
        layer.forward = MagicMock(side_effect=lambda x, **kwargs: x + 1)

    x = torch.zeros(2, 10)
    out = stack(x)

    assert torch.allclose(out, torch.full_like(out, 2))
    mock_randint.assert_called_once_with(low=1, high=5, size=(1,))
    stack.layers[0].forward.assert_called_once()
    stack.layers[1].forward.assert_called_once()
    stack.layers[2].forward.assert_not_called()
    stack.layers[3].forward.assert_not_called()

def test_layer_stack_recompute():
    layer_creator = lambda: DummyLayer()
    stack = LayerStack(
        layer_creator=layer_creator,
        num_layers=2,
        recompute_each_layer=True
    )

    x = torch.randn(2, 10, requires_grad=True)

    with patch("proximalfm.Model.checkpoint") as mock_checkpoint:
        mock_checkpoint.side_effect = lambda f, x, **kwargs: f(x)
        out = stack(x)

        assert mock_checkpoint.call_count == 2
        assert out.shape == (2, 10)

@pytest.fixture
def base_model():
    # `features_per_group=1` aligns with the default `LinearInputEncoderStep(num_features=1)`
    # created in `PerFeatureTransformerCATE._make_feature_encoder`
    return PerFeatureTransformerCATE(
        ninp=32, nhead=2, nhid=64, nlayers=2,
        features_per_group=1,
        zero_init=False,
        cache_trainset_representation=False
    )

def test_wrap_dict(base_model):
    # Test None
    assert base_model._wrap_dict(None) is None

    # Test tensor
    t = torch.randn(2, 2)
    wrapped = base_model._wrap_dict(t)
    assert isinstance(wrapped, dict)
    assert "main" in wrapped
    assert torch.equal(wrapped["main"], t)

    # Test existing dict
    d = {"main": t}
    assert base_model._wrap_dict(d) is d

    # Test custom key
    wrapped_custom = base_model._wrap_dict(t, key="custom")
    assert "custom" in wrapped_custom

    # Test missing key assert
    with pytest.raises(AssertionError, match="Key 'main' missing"):
        base_model._wrap_dict({"other": t})

def test_encode_feature_group_padding_and_reshape(base_model):
    # Set features_per_group to 3
    base_model.features_per_group = 3

    seq_len, batch_size, num_features = 4, 2, 5
    tensor = torch.ones(seq_len, batch_size, num_features)
    tensor_dict = {"main": tensor}

    # Expected pad: 3 - (5 % 3) = 1
    # Padded features: 6
    # Reshape b s (f n) -> b s f n -> 2, 4, 2, 3
    # Merge b and f -> s (b f) n -> 4, 4, 3

    mock_encoder = MagicMock()
    # Mock encoder returns a tensor of shape (seq, batch*features, ninp)
    # 4, 4, 32
    mock_encoder.return_value = torch.ones(seq_len, 4, 32)

    out = base_model._encode_feature_group(
        tensor_dict, mock_encoder, seq_len, batch_size, single_eval_pos=2
    )

    # Final output should be rearranged back to: b s f e -> 2, 4, 2, 32
    assert out.shape == (2, 4, 2, 32)

    # Verify the encoder was called with the right tensor dict
    called_dict, *_ = mock_encoder.call_args[0]
    called_tensor = called_dict["main"]

    assert called_tensor.shape == (4, 4, 3) # s, (b*f), n

    # Verify the padding
    # First 5 features are 1, the 6th is 0.
    # Rearranged it's (seq, (batch*feature), n)
    # n is 3. We have 2 batches, each with 2 feature groups.
    # So b*f = 4.
    # The second feature group of each batch contains the 6th feature (which is padding).
    # Specifically, indices 1 and 3 in the batch*feature dimension correspond to the second feature group.
    # The last feature in those groups should be 0.
    assert torch.all(called_tensor[:, 1, -1] == 0)
    assert torch.all(called_tensor[:, 3, -1] == 0)

def test_encode_scalar_group(base_model):
    seq_len, batch_size, num_features = 4, 2, 1
    tensor_dict = {"main": torch.ones(seq_len, batch_size, num_features)}

    mock_encoder = MagicMock()
    # Encoder returns (seq, batch, ninp)
    mock_encoder.return_value = torch.ones(4, 2, 32)

    out = base_model._encode_scalar_group(
        tensor_dict, mock_encoder, single_eval_pos=2
    )

    # Should be transposed to (batch, seq, ninp)
    assert out.shape == (2, 4, 32)

@patch("proximalfm.Model.torch.rand")
def test_add_type_embedding(mock_rand, base_model):
    # base_model.type_embedding is nn.Embedding(NUM_TYPES, 32)
    b, s, f, e = 2, 4, 3, 32
    embedded = torch.zeros(b, s, f, e)

    # No masking
    out = base_model._add_type_embedding(embedded, TYPE_X, mask_prob=0.0)
    assert out.shape == (b, s, f, e)

    # Masking with mask_prob = 1.0 (all TYPE_MASK)
    # mock_rand returns zeros so rand() < 1.0 is True
    mock_rand.return_value = torch.zeros(b, 1, f)
    base_model.train() # Make sure model is in train mode

    out_masked = base_model._add_type_embedding(embedded, TYPE_X, mask_prob=1.0)

    type_mask_emb = base_model.type_embedding(torch.tensor(TYPE_MASK))

    # Assert all elements are equal to the TYPE_MASK embedding
    # since embedded is all zeros.
    assert torch.allclose(out_masked, type_mask_emb.expand_as(out_masked))

    # Masking with mask_prob = 0.0 but mask_prob passed as 0.5 and rand > 0.5 (all TYPE_X)
    mock_rand.return_value = torch.ones(b, 1, f)
    out_unmasked = base_model._add_type_embedding(embedded, TYPE_X, mask_prob=0.5)

    type_x_emb = base_model.type_embedding(torch.tensor(TYPE_X))
    assert torch.allclose(out_unmasked, type_x_emb.expand_as(out_unmasked))

def test_pack_eval_io(base_model):
    N_tr, N_te = 5, 3
    Dx, Dz, Dw = 2, 3, 4

    X_train = torch.randn(N_tr, Dx)
    Z_train = torch.randn(N_tr, Dz)
    W_train = torch.randn(N_tr, Dw)
    A_train = torch.randn(N_tr, 1)
    Y_train = torch.randn(N_tr, 1)

    X_test = torch.randn(N_te, Dx)
    Z_test = torch.randn(N_te, Dz)
    W_test = torch.randn(N_te, Dw)

    x, z, w, a, y, single_eval_pos = base_model._pack_eval_io(
        X_train, Z_train, W_train, A_train, Y_train,
        X_test, Z_test, W_test
    )

    assert single_eval_pos == N_tr

    S = N_tr + N_te
    assert x["main"].shape == (S, 1, Dx)
    assert z["main"].shape == (S, 1, Dz)
    assert w["main"].shape == (S, 1, Dw)
    assert a["main"].shape == (S, 1, 1)
    assert y["main"].shape == (S, 1, 1)

    # A and Y should have NaNs at test positions
    assert torch.isnan(a["main"][N_tr:]).all()
    assert torch.isnan(y["main"][N_tr:]).all()

    # A and Y should not have NaNs at train positions
    assert not torch.isnan(a["main"][:N_tr]).any()
    assert not torch.isnan(y["main"][:N_tr]).any()

def test_add_feature_positional_embeddings(base_model):
    b, s, f, e = 2, 4, 3, 32
    embedded_input = torch.zeros(b, s, f, e)

    # Test None
    base_model.feature_positional_embedding = None
    out = base_model._add_feature_positional_embeddings(
        embedded_input, data_dags=None, num_features=f, seq_len=s
    )
    assert torch.equal(out, embedded_input)

    # Test normal_rand_vec
    base_model.feature_positional_embedding = "normal_rand_vec"
    out1 = base_model._add_feature_positional_embeddings(
        embedded_input, data_dags=None, num_features=f, seq_len=s
    )
    # Check shape
    assert out1.shape == (b, s, f, e)
    # Check it added something
    assert not torch.allclose(out1, embedded_input)
    # Same seed should produce same output
    out2 = base_model._add_feature_positional_embeddings(
        embedded_input, data_dags=None, num_features=f, seq_len=s
    )
    assert torch.equal(out1, out2)

    # Test uni_rand_vec
    base_model.feature_positional_embedding = "uni_rand_vec"
    out_uni = base_model._add_feature_positional_embeddings(
        embedded_input, data_dags=None, num_features=f, seq_len=s
    )
    assert out_uni.shape == (b, s, f, e)

    # Test learned
    base_model.feature_positional_embedding = "learned"
    base_model.feature_positional_embedding_embeddings = nn.Embedding(10, e)
    out_learned = base_model._add_feature_positional_embeddings(
        embedded_input, data_dags=None, num_features=f, seq_len=s
    )
    assert out_learned.shape == (b, s, f, e)

    # Test subspace
    base_model.feature_positional_embedding = "subspace"
    base_model.feature_positional_embedding_embeddings = nn.Linear(e // 4, e)
    out_subspace = base_model._add_feature_positional_embeddings(
        embedded_input, data_dags=None, num_features=f, seq_len=s
    )
    assert out_subspace.shape == (b, s, f, e)

def test_forward_normal_operation(base_model):
    s, b = 5, 2
    Dx, Dz, Dw = 2, 3, 4

    x = torch.randn(s, b, Dx)
    z = torch.randn(s, b, Dz)
    w = torch.randn(s, b, Dw)
    a = torch.randn(s, b, 1)
    y = torch.randn(s, b, 1)

    single_eval_pos = 3

    out = base_model(x, z, w, a, y, single_eval_pos=single_eval_pos)

    assert "cate_mean" in out
    assert "cate_std" in out
    assert "cate_log_var" in out

    assert out["cate_mean"].shape == (b, s, 1)
    assert out["cate_std"].shape == (b, s, 1)

def test_forward_missing_z_w(base_model):
    s, b = 5, 2
    x = torch.randn(s, b, 2)
    a = torch.randn(s, b, 1)
    y = torch.randn(s, b, 1)

    with pytest.raises(ValueError, match="Z \\(treatment-inducing proxy\\) must be provided."):
        base_model(x=x, z=None, w=torch.randn(s, b, 2), a=a, y=y, single_eval_pos=2)

    with pytest.raises(ValueError, match="W \\(outcome-inducing proxy\\) must be provided."):
        base_model(x=x, z=torch.randn(s, b, 2), w=None, a=a, y=y, single_eval_pos=2)

def test_forward_missing_a_y(base_model):
    s, b = 5, 2
    Dx, Dz, Dw = 2, 3, 4
    x = torch.randn(s, b, Dx)
    z = torch.randn(s, b, Dz)
    w = torch.randn(s, b, Dw)

    # Should work even if A or Y are None (they get filled with NaNs)
    out = base_model(x, z, w, a=None, y=None, single_eval_pos=3)

    assert out["cate_mean"].shape == (b, s, 1)

def test_forward_missing_x_raises_type_error(base_model):
    s, b = 5, 2
    Dz, Dw = 3, 4

    z = torch.randn(s, b, Dz)
    w = torch.randn(s, b, Dw)
    a = torch.randn(s, b, 1)
    y = torch.randn(s, b, 1)

    # Model.py uses `x["main"].shape` unconditionally right after wrapping `x`.
    # Therefore, x is mandatory and must not be None.
    with pytest.raises(TypeError):
        base_model(x=None, z=z, w=w, a=a, y=y, single_eval_pos=3)

def test_forward_nan_in_encoded(base_model):
    s, b = 5, 2
    Dx, Dz, Dw = 2, 3, 4

    x = torch.randn(s, b, Dx)
    z = torch.randn(s, b, Dz)
    w = torch.randn(s, b, Dw)
    a = torch.randn(s, b, 1)
    y = torch.randn(s, b, 1)

    # Inject NaN into X
    x[0, 0, 0] = torch.nan

    # LinearInputEncoderStep with replace_nan_by_zero=False will propagate NaNs
    with pytest.raises(ValueError, match="NaN detected in embedded_input after encoding."):
        base_model(x, z, w, a, y, single_eval_pos=3)

def test_forward_cfg_masking(base_model):
    s, b = 5, 2
    Dx, Dz, Dw = 2, 3, 4

    x = torch.randn(s, b, Dx)
    z = torch.randn(s, b, Dz)
    w = torch.randn(s, b, Dw)
    a = torch.randn(s, b, 1)
    y = torch.randn(s, b, 1)

    base_model.train() # Make sure we are in train mode for masking

    with patch("proximalfm.Model.torch.rand") as mock_rand:
        # Provide deterministic mock to force masking
        mock_rand.return_value = torch.zeros(b, 1, 1) # Will be < mask_prob

        # This will trigger the masking paths for x, z, w
        out = base_model(x, z, w, a, y, single_eval_pos=3, zw_mask_prob=1.0, x_mask_prob=1.0)
        assert out["cate_mean"].shape == (b, s, 1)

def test_forward_separate_decoder():
    model = PerFeatureTransformerCATE(
        ninp=32, nhead=2, nhid=64, nlayers=3,
        features_per_group=1,
        use_separate_decoder=True
    )

    s, b = 5, 2
    Dx, Dz, Dw = 2, 3, 4

    x = torch.randn(s, b, Dx)
    z = torch.randn(s, b, Dz)
    w = torch.randn(s, b, Dw)
    a = torch.randn(s, b, 1)
    y = torch.randn(s, b, 1)

    # To test use_separate_decoder properly without hitting the reshape bug of length 0 arrays
    # when s=single_eval_pos (since embedded_input[:, 3:] is empty),
    # we need s > single_eval_pos.
    # But `Model.py` expects `encoder_out.shape[1] == single_eval_pos_`.
    # Let's mock `transformer_encoder` and `transformer_decoder` to just return the expected shapes.
    # This verifies the logic in `Model.py` without worrying about the internal tabpfn implementation details.

    with patch.object(model.transformer_encoder, "forward") as mock_enc:
        with patch.object(model.transformer_decoder, "forward") as mock_dec:
            # Let's say total features is f=11, ninp=e=32
            # mock_enc should return (b, single_eval_pos_, f, e)
            mock_enc.return_value = torch.zeros(b, 3, 11, 32)
            # mock_dec should return (b, s - single_eval_pos_, f, e)
            mock_dec.return_value = torch.zeros(b, s - 3, 11, 32)

            out = model(x, z, w, a, y, single_eval_pos=3)

            assert out["cate_mean"].shape == (b, s, 1)
            mock_enc.assert_called_once()
            mock_dec.assert_called_once()

def test_estimate_cate(base_model):
    N_tr, N_te = 5, 3
    Dx, Dz, Dw = 2, 3, 4

    X_train = torch.randn(N_tr, Dx)
    Z_train = torch.randn(N_tr, Dz)
    W_train = torch.randn(N_tr, Dw)
    A_train = torch.randn(N_tr, 1)
    Y_train = torch.randn(N_tr, 1)

    X_test = torch.randn(N_te, Dx)
    Z_test = torch.randn(N_te, Dz)
    W_test = torch.randn(N_te, Dw)

    # Mocking the forward method to return deterministic output
    b = 1 # estimate_cate uses batch_size = 1 internally
    s = N_tr + N_te # 8

    with patch.object(base_model, "forward") as mock_forward:
        mock_out = {
            "cate_mean": torch.ones(b, s, 1) * 2.0,
            "cate_std": torch.ones(b, s, 1) * 0.5,
        }
        mock_forward.return_value = mock_out

        result = base_model.estimate_cate(
            X_train, Z_train, W_train, A_train, Y_train,
            X_test, Z_test, W_test
        )

        assert "cate_mean" in result
        assert "cate_std" in result

        # It should slice and squeeze the output to just the test positions
        # shape is just [N_te]
        assert result["cate_mean"].shape == (N_te,)
        assert result["cate_std"].shape == (N_te,)

        assert torch.all(result["cate_mean"] == 2.0)
        assert torch.all(result["cate_std"] == 0.5)

        # Verify forward was called with correct single_eval_pos
        mock_forward.assert_called_once()
        kwargs = mock_forward.call_args.kwargs
        assert kwargs["single_eval_pos"] == N_tr
