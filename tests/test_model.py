import pytest
import torch
import torch.nn as nn
from proximalfm.Structure import ModelConfig
from proximalfm.Model import (
    ProximalNanoTabICLv2,
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
    )
    model = ProximalNanoTabICLv2(config)

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
    assert out.shape == (n_batch, n_rows - n_train, 2)

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
    assert out_const.shape == (n_batch, n_rows - n_train, 2)
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
    )
    model = ProximalNanoTabICLv2(config)
    x = torch.randn(3, 10, 5)
    w = torch.randn(3, 10, 4)
    z = torch.randn(3, 10, 6)
    a = torch.randint(0, 2, (3, 8)).float()
    y = torch.randn(3, 8)

    out = model(x, a, w, z, y)
    assert out.shape == (3, 2, 2)

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
    model = ProximalNanoTabICLv2(config)

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
        return_variance=False
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
        return_variance=True
    )
    assert mean.shape == (batch_size, n_test, 1)
    assert log_var.shape == (batch_size, n_test, 1)


def test_estimate_cate_batch_size_mismatch():
    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
    )
    model = ProximalNanoTabICLv2(config)

    batch_size = 2
    n_train = 5
    n_test = 3

    x_train = torch.randn(batch_size, n_train, 4)
    a_train = torch.randn(batch_size, n_train)
    w_train = torch.randn(batch_size, n_train, 4)
    z_train = torch.randn(batch_size, n_train, 4)
    y_train = torch.randn(batch_size, n_train)

    # Mismatched batch size for x_test
    x_test = torch.randn(batch_size + 1, n_test, 4)
    w_test = torch.randn(batch_size, n_test, 4)
    z_test = torch.randn(batch_size, n_test, 4)

    with pytest.raises(ValueError, match="All inputs must have the same batch size"):
        model.estimate_CATE(
            x=x_train,
            a_train=a_train,
            w=w_train,
            z=z_train,
            y_train=y_train,
            x_test=x_test,
            w_test=w_test,
            z_test=z_test
        )

def test_estimate_cate_train_length_mismatch():
    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
    )
    model = ProximalNanoTabICLv2(config)

    batch_size = 2
    n_train = 5
    n_test = 3

    x_train = torch.randn(batch_size, n_train, 4)
    a_train = torch.randn(batch_size, n_train + 1) # Mismatch
    w_train = torch.randn(batch_size, n_train, 4)
    z_train = torch.randn(batch_size, n_train, 4)
    y_train = torch.randn(batch_size, n_train)

    x_test = torch.randn(batch_size, n_test, 4)
    w_test = torch.randn(batch_size, n_test, 4)
    z_test = torch.randn(batch_size, n_test, 4)

    with pytest.raises(ValueError, match="All train inputs must have the same sequence length"):
        model.estimate_CATE(
            x=x_train,
            a_train=a_train,
            w=w_train,
            z=z_train,
            y_train=y_train,
            x_test=x_test,
            w_test=w_test,
            z_test=z_test
        )

def test_estimate_cate_test_length_mismatch():
    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
    )
    model = ProximalNanoTabICLv2(config)

    batch_size = 2
    n_train = 5
    n_test = 3

    x_train = torch.randn(batch_size, n_train, 4)
    a_train = torch.randn(batch_size, n_train)
    w_train = torch.randn(batch_size, n_train, 4)
    z_train = torch.randn(batch_size, n_train, 4)
    y_train = torch.randn(batch_size, n_train)

    x_test = torch.randn(batch_size, n_test, 4)
    w_test = torch.randn(batch_size, n_test + 1, 4) # Mismatch
    z_test = torch.randn(batch_size, n_test, 4)

    with pytest.raises(ValueError, match="All test inputs must have the same sequence length"):
        model.estimate_CATE(
            x=x_train,
            a_train=a_train,
            w=w_train,
            z=z_train,
            y_train=y_train,
            x_test=x_test,
            w_test=w_test,
            z_test=z_test
        )

def test_estimate_cate_full_sequence_mode():
    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
    )
    model = ProximalNanoTabICLv2(config)

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
        return_variance=False
    )
    assert mean.shape == (batch_size, n_test, 1)

    # Test with variance
    mean, log_var = model.estimate_CATE(
        x=x_full,
        a_train=a_train,
        w=w_full,
        z=z_full,
        y_train=y_train,
        return_variance=True
    )
    assert mean.shape == (batch_size, n_test, 1)
    assert log_var.shape == (batch_size, n_test, 1)

def test_estimate_cate_partial_test_args():
    config = ModelConfig(
        feature_group_size_x=2,
        feature_group_size_w=2,
        feature_group_size_z=2,
        embed_dim=16,
    )
    model = ProximalNanoTabICLv2(config)

    batch_size = 2
    n_train = 5

    x_full = torch.randn(batch_size, n_train + 3, 4)
    w_full = torch.randn(batch_size, n_train + 3, 4)
    z_full = torch.randn(batch_size, n_train + 3, 4)
    a_train = torch.randn(batch_size, n_train)
    y_train = torch.randn(batch_size, n_train)

    x_test = torch.randn(batch_size, 3, 4)

    # Missing w_test and z_test
    with pytest.raises(ValueError, match="Either provide all of x_test, w_test, z_test or none of them"):
        model.estimate_CATE(
            x=x_full,
            a_train=a_train,
            w=w_full,
            z=z_full,
            y_train=y_train,
            x_test=x_test
        )
