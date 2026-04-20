import math
import unittest.mock


import pytest
import torch
import torch.nn as nn
from proximalfm.Structure import ModelConfig
from proximalfm.models.gmm import ProximalGMM

from proximalfm.models.base import (
    ClassEmbedding,
    get_mlp,
    InducedTransformerBlock,
    TransformerBlock,
    Rope,
    QASSMax,
)

def test_get_mlp():
    mlp = get_mlp(10, 20, 5)
    assert isinstance(mlp, nn.Sequential)
    assert len(mlp) == 3
    assert isinstance(mlp[0], nn.Linear)
    assert isinstance(mlp[1], nn.GELU)
    assert isinstance(mlp[2], nn.Linear)

    x = torch.randn(2, 10)
    out = mlp(x)
    assert out.shape == (2, 5)

def test_class_embedding():
    embed = ClassEmbedding(10, 32)
    embed.reset_parameters() # test reset

    y = torch.randint(0, 10, (5, 4, 1)).float()
    out = embed(y)
    assert out.shape == (5, 4, 32)

def test_rope():
    rope = Rope(head_dim=16, theta=10000.0)

    # input shape: (batch_size, num_heads, seq_len, head_dim)
    x = torch.randn(2, 4, 10, 16)
    out = rope(x)
    assert out.shape == (2, 4, 10, 16)

    # Test cache extension by passing a longer sequence
    x_long = torch.randn(2, 4, 20, 16)
    out_long = rope(x_long)
    assert out_long.shape == (2, 4, 20, 16)

def test_qassmax():
    qassmax = QASSMax(num_heads=4, head_dim=16, n_hidden=32)
    q = torch.randn(2, 4, 10, 16)
    out = qassmax(q, n=100)
    assert out.shape == (2, 4, 10, 16)

    # Test edge case with n=0
    out_zero = qassmax(q, n=0)
    assert out_zero.shape == (2, 4, 10, 16)

def test_transformer_block():
    block = TransformerBlock(embed_dim=64, num_heads=4, use_rope=True, ssmax=True)
    q = torch.randn(2, 10, 64)
    kv = torch.randn(2, 10, 64)

    out1 = block(q) # self attention
    assert out1.shape == (2, 10, 64)

    out2 = block(q, kv) # cross attention
    assert out2.shape == (2, 10, 64)

    # with max indices
    out3 = block(q, kv, q_max_idx=5, kv_max_idx=8)
    assert out3.shape == (2, 5, 64)

def test_table_attn_base():
    block = TransformerBlock(embed_dim=64, num_heads=4) # Inherits from TableAttnBase

    # table shape: (batch_size, n_rows, n_cols, embed_dim)
    q = torch.randn(2, 5, 6, 64)

    out_row = block.row_attn(q)
    assert out_row.shape == (2, 5, 6, 64)

    out_col = block.col_attn(q)
    assert out_col.shape == (2, 5, 6, 64)

    kv = torch.randn(2, 5, 6, 64)
    out_row_kv = block.row_attn(q, kv)
    assert out_row_kv.shape == (2, 5, 6, 64)

    out_col_kv = block.col_attn(q, kv)
    assert out_col_kv.shape == (2, 5, 6, 64)

def test_induced_transformer_block():
    block = InducedTransformerBlock(embed_dim=64, num_heads=4, n_inducing=8, ssmax=True)

    q = torch.randn(2, 10, 64)
    out1 = block(q)
    assert out1.shape == (2, 10, 64)

    kv = torch.randn(2, 15, 64)
    out2 = block(q, kv, q_max_idx=5, kv_max_idx=12)
    assert out2.shape == (2, 5, 64)

    # Test table attention wrapper for induced block
    q_table = torch.randn(2, 5, 6, 64)
    out_row = block.row_attn(q_table)
    assert out_row.shape == (2, 5, 6, 64)

def test_proximal_nanotab_icl_v2():
    config = ModelConfig(
        embed_dim=32,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
        col_nhead=4,
        row_nhead=4,
        icl_nhead=4,
        feature_group_size_x=2,
        feature_group_size_w=3,
        feature_group_size_z=2,
        n_cls_cols=4,
        n_cls_rows=16,
        n_gmm_components=2
    )
    model = ProximalGMM(config)

    n_batch = 2
    n_rows = 20
    n_train = 15
    n_cols_x, n_cols_w, n_cols_z = 6, 9, 8

    x = torch.randn(n_batch, n_rows, n_cols_x)
    w = torch.randn(n_batch, n_rows, n_cols_w)
    z = torch.randn(n_batch, n_rows, n_cols_z)

    a = torch.randn(n_batch, n_train)
    y = torch.randn(n_batch, n_train)

    out = model(x, a, w, z, y)
    assert out.shape == (n_batch, n_rows - n_train, 6)

    # Edge case: assertion error on shape mismatch
    with pytest.raises(AssertionError, match="y and a must have the same shape"):
        y_wrong = torch.randn(n_batch, n_train - 1)
        model(x, a, w, z, y_wrong)

    # Edge case: std=0 handling (no division by zero)
    x_const = torch.ones(n_batch, n_rows, n_cols_x)
    w_const = torch.ones(n_batch, n_rows, n_cols_w)
    z_const = torch.ones(n_batch, n_rows, n_cols_z)
    y_const = torch.ones(n_batch, n_train)

    out_const = model(x_const, a, w_const, z_const, y_const)
    assert out_const.shape == (n_batch, n_rows - n_train, 6)
    assert not torch.isnan(out_const).any()

def test_proximal_nanotab_icl_v2_zero_blocks():
    # Test boundary case with minimal blocks
    config = ModelConfig(
        embed_dim=16,
        col_num_blocks=0,
        row_num_blocks=1, # Must be >=1 as row_blocks[-1] is accessed
        icl_num_blocks=1, # Must be >=1 as icl_blocks[-1] is accessed
        feature_group_size_x=1,
        feature_group_size_w=1,
        feature_group_size_z=1,
        n_gmm_components=2
    )
    model = ProximalGMM(config)
    x = torch.randn(3, 10, 5)
    w = torch.randn(3, 10, 4)
    z = torch.randn(3, 10, 6)
    a = torch.randint(0, 2, (3, 8)).float()
    y = torch.randn(3, 8)

    out = model(x, a, w, z, y)
    assert out.shape == (3, 2, 6)

def test_estimate_cate_success():
    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
        n_cls_cols=1,
        n_cls_rows=1,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
    )
    model = ProximalGMM(config)

    batch_size = 2
    n_train = 5
    n_test = 3

    x_train = torch.randn(batch_size, n_train, 4)
    w_train = torch.randn(batch_size, n_train, 4)
    z_train = torch.randn(batch_size, n_train, 4)
    a_train = torch.randn(batch_size, n_train)
    y_train = torch.randn(batch_size, n_train)

    x_test = torch.randn(batch_size, n_test, 4)
    w_test = torch.randn(batch_size, n_test, 4)
    z_test = torch.randn(batch_size, n_test, 4)

    # Test without variance
    mean = model.estimate_CATE(
        x=x_train,
        a_train=a_train,
        w=w_train,
        z=z_train,
        y_train=y_train,
        x_test=x_test,
        w_test=w_test,
        z_test=z_test,
        return_log_variance=False
    )
    assert mean.shape == (batch_size, n_test, 1)

    # Test with variance
    mean, log_var = model.estimate_CATE(
        x=x_train,
        a_train=a_train,
        w=w_train,
        z=z_train,
        y_train=y_train,
        x_test=x_test,
        w_test=w_test,
        z_test=z_test,
        return_log_variance=True
    )
    assert mean.shape == (batch_size, n_test, 1)
    assert log_var.shape == (batch_size, n_test, 1)

def test_estimate_cate_full_sequence_mode():
    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
    )
    model = ProximalGMM(config)

    batch_size = 2
    n_train = 5
    n_test = 3
    n_full = n_train + n_test

    x_full = torch.randn(batch_size, n_full, 4)
    w_full = torch.randn(batch_size, n_full, 4)
    z_full = torch.randn(batch_size, n_full, 4)
    a_train = torch.randn(batch_size, n_train)
    y_train = torch.randn(batch_size, n_train)

    mean = model.estimate_CATE(
        x=x_full,
        a_train=a_train,
        w=w_full,
        z=z_full,
        y_train=y_train,
        return_log_variance=False
    )
    assert mean.shape == (batch_size, n_test, 1)

    # Test with variance
    mean, log_var = model.estimate_CATE(
        x=x_full,
        a_train=a_train,
        w=w_full,
        z=z_full,
        y_train=y_train,
        return_log_variance=True
    )
    assert mean.shape == (batch_size, n_test, 1)
    assert log_var.shape == (batch_size, n_test, 1)




def test_estimate_cate_math_verification():
    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
        n_gmm_components=2,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
    )
    model = ProximalGMM(config)

    batch_size = 1
    n_train = 2
    n_test = 1
    n_gmm = config.n_gmm_components

    x = torch.randn(batch_size, n_train, 4)
    w = torch.randn(batch_size, n_train, 4)
    z = torch.randn(batch_size, n_train, 4)
    a = torch.randn(batch_size, n_train)
    y = torch.randn(batch_size, n_train)

    x_test = torch.randn(batch_size, n_test, 4)
    w_test = torch.randn(batch_size, n_test, 4)
    z_test = torch.randn(batch_size, n_test, 4)

    # Mock forward to return specific values
    # logits = [0, 0] -> pi = [0.5, 0.5]
    # means = [2.0, 4.0]
    # log_vars = [log(1.0), log(1.0)] -> vars = [1.0, 1.0]

    out_tensor = torch.zeros(batch_size, n_test, 3 * n_gmm)
    # Logits
    out_tensor[0, 0, 0] = 0.0
    out_tensor[0, 0, 1] = 0.0
    # Means
    out_tensor[0, 0, 2] = 2.0
    out_tensor[0, 0, 3] = 4.0
    # Log vars
    out_tensor[0, 0, 4] = 0.0 # log(1) = 0
    out_tensor[0, 0, 5] = 0.0 # log(1) = 0

    with unittest.mock.patch.object(model, 'forward', return_value=out_tensor):
        mean, log_var = model.estimate_CATE(
            x=x, a_train=a, w=w, z=z, y_train=y,
            x_test=x_test, w_test=w_test, z_test=z_test,
            return_log_variance=True
        )

        # Expected mean = 0.5 * 2.0 + 0.5 * 4.0 = 3.0
        assert torch.allclose(mean, torch.tensor([[[3.0]]]))

        # Expected x_squared = 0.5 * (1.0 + 4.0) + 0.5 * (1.0 + 16.0) = 0.5 * 5.0 + 0.5 * 17.0 = 2.5 + 8.5 = 11.0
        # Expected var = 11.0 - 3.0**2 = 11.0 - 9.0 = 2.0
        # Expected log_var = log(2.0) = 0.693147
        expected_log_var = math.log(2.0)
        assert torch.allclose(log_var, torch.tensor([[[expected_log_var]]]), atol=1e-5)

def test_estimate_cate_variance_clamping():
    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
        n_gmm_components=1,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
    )
    model = ProximalGMM(config)

    batch_size = 1
    n_train = 2
    n_test = 1
    n_gmm = config.n_gmm_components

    x = torch.randn(batch_size, n_train, 4)
    w = torch.randn(batch_size, n_train, 4)
    z = torch.randn(batch_size, n_train, 4)
    a = torch.randn(batch_size, n_train)
    y = torch.randn(batch_size, n_train)

    x_test = torch.randn(batch_size, n_test, 4)
    w_test = torch.randn(batch_size, n_test, 4)
    z_test = torch.randn(batch_size, n_test, 4)

    out_tensor = torch.zeros(batch_size, n_test, 3 * n_gmm)
    # Logits
    out_tensor[0, 0, 0] = 0.0
    # Means
    out_tensor[0, 0, 1] = 2.0
    # Log vars: extremely small, making calculated total_variance <= 0 without clamping
    # Here variance = exp(-50) ~ 0
    # Total variance formula: expected_x_squared - expected_mean**2
    # = (variance + mean^2) - mean^2 = variance ~ 0
    out_tensor[0, 0, 2] = -50.0

    with unittest.mock.patch.object(model, 'forward', return_value=out_tensor):
        mean, log_var = model.estimate_CATE(
            x=x, a_train=a, w=w, z=z, y_train=y,
            x_test=x_test, w_test=w_test, z_test=z_test,
            return_log_variance=True
        )

        # Expected log_var = log(1e-8) = -18.42068
        expected_log_var = math.log(1e-8)
        assert torch.allclose(log_var, torch.tensor([[[expected_log_var]]]), atol=1e-5)

def test_estimate_cate_partial_test_inputs():
    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
    )
    model = ProximalGMM(config)

    batch_size = 2
    n_train = 5
    n_test = 3
    n_full = n_train + n_test

    x_full = torch.randn(batch_size, n_full, 4)
    w_full = torch.randn(batch_size, n_full, 4)
    z_full = torch.randn(batch_size, n_full, 4)
    a_train = torch.randn(batch_size, n_train)
    y_train = torch.randn(batch_size, n_train)

    # Provide x_test and w_test, but not z_test
    # The mode should fallback to full_sequence, and the explicit_test cat should not be triggered
    # However, since x_full, w_full, z_full are already of full size, the model should process them
    # and return predictions for the last n_test elements

    mean = model.estimate_CATE(
        x=x_full,
        a_train=a_train,
        w=w_full,
        z=z_full,
        y_train=y_train,
        x_test=torch.randn(batch_size, n_test, 4), # Partial explicit test inputs
        w_test=torch.randn(batch_size, n_test, 4), # Partial explicit test inputs
        # z_test=None
    )

    assert mean.shape == (batch_size, n_test, 1)

def test_estimate_cate_multiple_gmm_components():
    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
        n_gmm_components=5,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
    )
    model = ProximalGMM(config)

    batch_size = 2
    n_train = 5
    n_test = 3

    x = torch.randn(batch_size, n_train, 4)
    w = torch.randn(batch_size, n_train, 4)
    z = torch.randn(batch_size, n_train, 4)
    a = torch.randn(batch_size, n_train)
    y = torch.randn(batch_size, n_train)

    x_test = torch.randn(batch_size, n_test, 4)
    w_test = torch.randn(batch_size, n_test, 4)
    z_test = torch.randn(batch_size, n_test, 4)

    mean, log_var = model.estimate_CATE(
        x=x, a_train=a, w=w, z=z, y_train=y,
        x_test=x_test, w_test=w_test, z_test=z_test,
        return_log_variance=True
    )

    assert mean.shape == (batch_size, n_test, 1)
    assert log_var.shape == (batch_size, n_test, 1)



def test_gmm_loss_math_verification():
    import torch.nn.functional as F

    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
        n_gmm_components=2,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
    )
    model = ProximalGMM(config)

    batch_size = 2
    n_test = 3
    n_gmm = config.n_gmm_components

    # Create dummy outputs and targets
    outputs = torch.randn(batch_size, n_test, 3 * n_gmm)
    target = torch.randn(batch_size, n_test, 1)

    # Compute loss using the model's loss function
    loss = model.loss(outputs, target)

    # Recompute loss manually to verify
    logits = outputs[..., : n_gmm]
    mu = outputs[..., n_gmm : 2 * n_gmm]
    log_var = outputs[..., 2 * n_gmm :]

    log_pi = F.log_softmax(logits, dim=-1)
    z_sq = ((target - mu) ** 2) * torch.exp(-log_var)
    log_prob = -0.5 * z_sq - 0.5 * log_var
    log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)
    expected_loss = -log_mix.mean()

    assert torch.allclose(loss, expected_loss)
    assert loss.dim() == 0  # Should be a scalar

def test_gmm_loss_target_shape_handling():
    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
        n_gmm_components=2,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
    )
    model = ProximalGMM(config)

    batch_size = 2
    n_test = 3
    n_gmm = config.n_gmm_components

    outputs = torch.randn(batch_size, n_test, 3 * n_gmm)
    target_3d = torch.randn(batch_size, n_test, 1)
    target_2d = target_3d.squeeze(-1)  # shape [batch, n_test]

    loss_3d = model.loss(outputs, target_3d)
    loss_2d = model.loss(outputs, target_2d)

    assert torch.allclose(loss_3d, loss_2d)

def test_gmm_loss_single_component_equivalence():
    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
        n_gmm_components=1,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
    )
    model = ProximalGMM(config)

    batch_size = 2
    n_test = 3
    n_gmm = config.n_gmm_components

    outputs = torch.randn(batch_size, n_test, 3 * n_gmm)
    target = torch.randn(batch_size, n_test, 1)

    loss = model.loss(outputs, target)

    # Standard Gaussian NLL manually
    mu = outputs[..., n_gmm : 2 * n_gmm]
    log_var = outputs[..., 2 * n_gmm :]
    var = torch.exp(log_var)

    gnll = torch.nn.GaussianNLLLoss(full=False, eps=0.0, reduction='mean')
    expected_gnll = gnll(mu, target, var)

    # Note: Our GMM loss might have a different constant factor.
    # Let's verify manually:
    # GMM Loss logic for 1 component:
    # log_pi = 0 (since log_softmax of a single value is 0)
    # log_prob = -0.5 * ((target - mu) ** 2) / exp(log_var) - 0.5 * log_var
    # GMM loss = -log_prob.mean()
    # GaussianNLLLoss(full=False): 0.5 * (log(max(var, eps)) + (target - mu)**2 / max(var, eps)).mean()
    # Thus, they should be exactly equal

    assert torch.allclose(loss, expected_gnll, atol=1e-6)

def test_gmm_loss_numerical_stability():
    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
        n_gmm_components=2,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
    )
    model = ProximalGMM(config)

    batch_size = 2
    n_test = 3
    n_gmm = config.n_gmm_components

    # Create outputs with extreme values
    outputs = torch.zeros(batch_size, n_test, 3 * n_gmm)

    # Extreme logits
    outputs[..., 0] = 1000.0  # Large logit
    outputs[..., 1] = -1000.0 # Small logit

    # Extreme means
    outputs[..., 2] = 1e6
    outputs[..., 3] = -1e6

    # Extreme log variances
    outputs[..., 4] = 50.0  # Large positive log variance (var -> inf)
    outputs[..., 5] = -50.0 # Large negative log variance (var -> 0)

    target = torch.ones(batch_size, n_test, 1)

    loss = model.loss(outputs, target)

    # The loss should be a valid number, not NaN or Inf
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()

def test_gmm_loss_gradient_flow():
    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
        n_gmm_components=2,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
    )
    model = ProximalGMM(config)

    batch_size = 2
    n_test = 3
    n_gmm = config.n_gmm_components

    outputs = torch.randn(batch_size, n_test, 3 * n_gmm, requires_grad=True)
    target = torch.randn(batch_size, n_test, 1)

    loss = model.loss(outputs, target)
    loss.backward()

    # Gradients should be populated and valid
    assert outputs.grad is not None
    assert not torch.isnan(outputs.grad).any()
    assert not torch.isinf(outputs.grad).any()
    # At least some gradients should be non-zero
    assert torch.abs(outputs.grad).sum() > 0
