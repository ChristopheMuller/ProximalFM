import torch

from proximalfm.prior._dataset import SCMPrior, PriorDataset, DisablePrinting
from proximalfm.prior._genload import (
    SavePriorDataset,
    LoadPriorDataset,
    dense2sparse,
    sparse2dense,
    SliceNestedTensor,
    cat_slice_nested_tensors,
)


import pytest


class ArgsMock:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_scm_prior_initialization():
    prior = SCMPrior(batch_size=8, min_dim_U=2, max_dim_U=5)
    assert prior.batch_size == 8
    assert prior.min_dim_U == 2
    assert prior.max_dim_U == 5


def test_scm_prior_get_batch():
    prior = SCMPrior(
        batch_size=4,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=4,
        min_dim_Z=2,
        max_dim_Z=4,
        min_dim_X=3,
        max_dim_X=5,
        min_seq_len=50,
        max_seq_len=100,
        min_train_size=0.1,
        max_train_size=0.8,
        prior_type="mlp_scm",
    )
    (
        U,
        X,
        W,
        Z,
        A,
        Y_obs,
        Y0,
        Y1,
        d_U,
        d_X,
        d_W,
        d_Z,
        seq_lens,
        train_sizes,
    ) = prior.get_batch()

    assert X.shape[0] == 4
    assert W.shape[0] == 4
    assert Z.shape[0] == 4
    assert A.shape[0] == 4
    assert Y_obs.shape[0] == 4
    assert Y0.shape[0] == 4
    assert Y1.shape[0] == 4

    assert d_X.shape[0] == 4
    assert d_W.shape[0] == 4
    assert d_Z.shape[0] == 4

    assert seq_lens.shape[0] == 4
    assert train_sizes.shape[0] == 4


def test_scm_prior_mix_scm():
    prior = SCMPrior(
        batch_size=4,
        batch_size_per_gp=4,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=4,
        min_dim_Z=2,
        max_dim_Z=4,
        min_dim_X=3,
        max_dim_X=5,
        min_seq_len=50,
        max_seq_len=100,
        min_train_size=0.1,
        max_train_size=0.8,
        prior_type="mix_scm",
    )
    (
        U,
        X,
        W,
        Z,
        A,
        Y_obs,
        Y0,
        Y1,
        d_U,
        d_X,
        d_W,
        d_Z,
        seq_lens,
        train_sizes,
    ) = prior.get_batch()
    assert X.shape[0] == 4


def test_scm_prior_tree_scm():
    prior = SCMPrior(
        batch_size=2,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=4,
        min_dim_Z=2,
        max_dim_Z=4,
        min_dim_X=3,
        max_dim_X=5,
        min_seq_len=50,
        max_seq_len=100,
        min_train_size=0.1,
        max_train_size=0.8,
        prior_type="tree_scm",
    )
    (
        U,
        X,
        W,
        Z,
        A,
        Y_obs,
        Y0,
        Y1,
        d_U,
        d_X,
        d_W,
        d_Z,
        seq_lens,
        train_sizes,
    ) = prior.get_batch()
    assert X.shape[0] == 2


def test_scm_prior_seq_len_per_gp():
    prior = SCMPrior(
        batch_size=4,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=4,
        min_dim_Z=2,
        max_dim_Z=4,
        min_dim_X=3,
        max_dim_X=5,
        min_seq_len=50,
        max_seq_len=100,
        min_train_size=0.1,
        max_train_size=0.8,
        seq_len_per_gp=True,
        prior_type="mlp_scm",
    )
    (
        U,
        X,
        W,
        Z,
        A,
        Y_obs,
        Y0,
        Y1,
        d_U,
        d_X,
        d_W,
        d_Z,
        seq_lens,
        train_sizes,
    ) = prior.get_batch()

    # When seq_len_per_gp=True, the tensors should be nested tensors
    assert X.is_nested
    assert W.is_nested
    assert Z.is_nested
    assert A.is_nested
    assert Y_obs.is_nested
    assert Y0.is_nested
    assert Y1.is_nested

    assert X.size(0) == 4
    assert len(seq_lens) == 4


def test_prior_dataset():
    dataset = PriorDataset(
        batch_size=2,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=4,
        min_dim_Z=2,
        max_dim_Z=4,
        min_dim_X=3,
        max_dim_X=5,
        min_seq_len=50,
        max_seq_len=100,
        prior_type="mlp_scm",
    )

    U, X, W, Z, A, Y_obs, Y0, Y1, d_U, d_X, d_W, d_Z, seq_lens, train_sizes = next(
        iter(dataset)
    )

    assert X.shape[0] == 2


def test_save_and_load_prior_dataset_seq_len_per_gp(tmp_path):
    args = ArgsMock(
        save_dir=str(tmp_path),
        batch_size=4,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=4,
        max_dim_W=4,
        min_dim_Z=4,
        max_dim_Z=4,
        min_dim_X=5,
        max_dim_X=5,
        min_seq_len=50,
        max_seq_len=100,
        log_seq_len=False,
        seq_len_per_gp=True,
        min_train_size=10,
        max_train_size=40,
        replay_small=False,
        prior_type="mlp_scm",
        num_batches=2,
        resume_from=0,
        n_jobs=1,
        num_threads_per_generate=1,
        device="cpu",
    )

    saver = SavePriorDataset(args)
    saver.run()

    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "batch_000000.pt").exists()
    assert (tmp_path / "batch_000001.pt").exists()

    loader = LoadPriorDataset(data_dir=str(tmp_path), batch_size=3, max_batches=2)

    batches = []
    try:
        while True:
            batches.append(next(loader))
    except StopIteration:
        pass

    assert len(batches) == 3

    batch1 = batches[0]
    assert batch1[0].is_nested


def test_dense_sparse_conversions():
    dense = torch.tensor(
        [[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 0.0, 0.0], [6.0, 7.0, 8.0, 9.0]]
    )
    row_lengths = torch.tensor([3, 2, 4])

    sparse = dense2sparse(dense, row_lengths)
    expected_sparse = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert torch.allclose(sparse, expected_sparse)

    recovered_dense = sparse2dense(sparse, row_lengths, max_len=4)
    assert torch.allclose(recovered_dense, dense)

    # Test different data type
    dense_int = torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long)
    row_lengths_int = torch.tensor([2, 1])

    sparse_int = dense2sparse(dense_int, row_lengths_int, dtype=torch.long)
    expected_sparse_int = torch.tensor([1, 2, 3], dtype=torch.long)
    assert torch.equal(sparse_int, expected_sparse_int)

    recovered_dense_int = sparse2dense(
        sparse_int, row_lengths_int, max_len=3, dtype=torch.long
    )
    assert torch.equal(recovered_dense_int, dense_int)


def test_slice_nested_tensor():
    t1 = torch.ones(5, 3)
    t2 = torch.zeros(4, 3)
    nt = torch.nested.nested_tensor([t1, t2])

    snt = SliceNestedTensor(nt)

    # Test length
    assert len(snt) == 2

    # Test indexing
    assert torch.equal(snt[0], t1)
    assert torch.equal(snt[1], t2)

    # Test slicing
    sliced = snt[1:]
    assert isinstance(sliced, SliceNestedTensor)
    assert len(sliced) == 1
    assert torch.equal(sliced[0], t2)


def test_cat_slice_nested_tensors():
    t1 = torch.ones(5, 3)
    t2 = torch.zeros(4, 3)
    nt1 = torch.nested.nested_tensor([t1, t2])

    t3 = torch.full((6, 3), 2.0)
    nt2 = torch.nested.nested_tensor([t3])

    snt1 = SliceNestedTensor(nt1)
    snt2 = SliceNestedTensor(nt2)

    cat_snt = cat_slice_nested_tensors([snt1, snt2])

    assert len(cat_snt) == 3
    assert torch.equal(cat_snt[0], t1)
    assert torch.equal(cat_snt[1], t2)
    assert torch.equal(cat_snt[2], t3)


def test_disable_printing(capsys):
    with DisablePrinting():
        print("This should be hidden")
    print("This should be visible")

    captured = capsys.readouterr()
    assert "This should be hidden" not in captured.out
    assert "This should be visible" in captured.out


def test_sparse2dense():
    # 1. Happy path (max_len=None)
    sparse = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    row_lengths = torch.tensor([2, 3])
    dense = sparse2dense(sparse, row_lengths)
    expected = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]])
    assert torch.equal(dense, expected)

    # 2. max_len provided
    dense_padded = sparse2dense(sparse, row_lengths, max_len=4)
    expected_padded = torch.tensor([[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 5.0, 0.0]])
    assert torch.equal(dense_padded, expected_padded)

    # 3. Empty tensor
    sparse_empty = torch.tensor([], dtype=torch.float32)
    row_lengths_empty = torch.tensor([0, 0])
    dense_empty = sparse2dense(sparse_empty, row_lengths_empty)
    assert dense_empty.shape == (2, 0)

    # 4. Error conditions
    # Not 1D
    with pytest.raises(AssertionError, match="data must be 1D"):
        sparse2dense(torch.tensor([[1.0, 2.0]]), row_lengths)

    # Length mismatch
    with pytest.raises(
        AssertionError, match="data length must match sum of row_lengths"
    ):
        sparse2dense(sparse, torch.tensor([2, 2]))
