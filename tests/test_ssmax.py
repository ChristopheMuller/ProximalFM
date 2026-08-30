import math
import pytest
import torch

from proximalfm._model.ssmax import (
    _logn,
    SSMax,
    SSMaxMLP,
    QASSMaxMLP,
    create_ssmax_layer,
)


def test_logn():
    device = torch.device("cpu")
    dtype = torch.float32

    # Normal behavior
    n = 10
    res = _logn(n, device, dtype)
    assert torch.isclose(res, torch.tensor(math.log(10), dtype=dtype))
    assert res.device == device
    assert res.dtype == dtype

    # Boundary conditions (n <= 1 should yield log(1) == 0.0)
    res = _logn(1, device, dtype)
    assert res.item() == 0.0

    res = _logn(0, device, dtype)
    assert res.item() == 0.0

    res = _logn(-5, device, dtype)
    assert res.item() == 0.0


def test_ssmax():
    num_heads = 4
    seq_len = 10
    head_dim = 16
    bs = 2
    n = 20

    model = SSMax(num_heads)

    # Check scales initialization
    assert model.scales.shape == (num_heads,)
    assert torch.allclose(model.scales, torch.ones(num_heads))

    q = torch.randn(bs, num_heads, seq_len, head_dim)

    # Forward pass
    out = model(q, n)

    assert out.shape == q.shape

    # Check math manually
    logn = math.log(max(n, 1))
    expected_out = q * (torch.ones(num_heads).view(1, -1, 1, 1) * logn)
    assert torch.allclose(out, expected_out)


def test_ssmax_mlp_variants():
    num_heads = 4
    seq_len = 10
    head_dim = 16
    bs = 2
    n = 20

    # Test elementwise=False
    model_non_elementwise = SSMaxMLP(num_heads, elementwise=False)
    q = torch.randn(bs, num_heads, seq_len, head_dim)
    out = model_non_elementwise(q, n)
    assert out.shape == q.shape

    # Test elementwise=True
    model_elementwise = SSMaxMLP(num_heads, elementwise=True, head_dim=head_dim)
    out = model_elementwise(q, n)
    assert out.shape == q.shape

    for elementwise in (False, True):
        model = QASSMaxMLP(num_heads, head_dim, elementwise=elementwise)
        assert model(q, n).shape == q.shape

    # Verify ValueError if elementwise=True but head_dim=None
    with pytest.raises(
        ValueError, match="head_dim must be provided when elementwise=True"
    ):
        SSMaxMLP(num_heads, elementwise=True, head_dim=None)


def test_create_ssmax_layer():
    num_heads = 4
    embed_dim = 64  # So head_dim is 16

    assert create_ssmax_layer("none", num_heads, embed_dim) is None

    layer = create_ssmax_layer("ssmax", num_heads, embed_dim)
    assert isinstance(layer, SSMax)

    layer = create_ssmax_layer("ssmax-mlp", num_heads, embed_dim)
    assert isinstance(layer, SSMaxMLP)
    assert not layer.elementwise

    layer = create_ssmax_layer("ssmax-mlp-elementwise", num_heads, embed_dim)
    assert isinstance(layer, SSMaxMLP)
    assert layer.elementwise

    layer = create_ssmax_layer("qassmax-mlp", num_heads, embed_dim)
    assert isinstance(layer, QASSMaxMLP)
    assert not layer.elementwise

    layer = create_ssmax_layer("qassmax-mlp-elementwise", num_heads, embed_dim)
    assert isinstance(layer, QASSMaxMLP)
    assert layer.elementwise

    with pytest.raises(ValueError, match="Unknown ssmax_type='unknown'"):
        create_ssmax_layer("unknown", num_heads, embed_dim)
