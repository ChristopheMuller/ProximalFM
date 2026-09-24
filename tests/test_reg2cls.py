import torch


import pytest


from proximalfm.prior._reg2cls import (
    num2cat,
    process_features,
    BlockProfile,
    block_profile,
    dominates,
)


def test_num2cat_enforced_profile_dominates_requirement():
    torch.manual_seed(0)
    T = 300
    max_categories = 8

    # A spread of U-profiles the proxy must dominate, all feasible given H cols.
    requirements = [
        BlockProfile(continuous=2, categorical=(3,)),
        BlockProfile(continuous=1, categorical=(4, 3)),
        BlockProfile(continuous=3, categorical=()),
        BlockProfile(continuous=0, categorical=(5, 2, 2)),
    ]

    for req in requirements:
        need = req.continuous + len(req.categorical)
        for H in (need, need + 2, need + 4):
            for _ in range(15):
                X = torch.randn(T, H)
                _, meta = num2cat(
                    X,
                    {"cat_col_prob": 1.0, "max_categories": max_categories},
                    require_profile=req,
                )
                prof = block_profile(H, meta)
                assert dominates(prof, req), (req, H, prof)


def test_process_features_enforced_profile_dominates_and_finite():
    torch.manual_seed(0)
    req = BlockProfile(continuous=2, categorical=(4, 3))

    for _ in range(20):
        X = torch.randn(256, 7)
        out, meta = process_features(
            X, {"cat_col_prob": 0.5, "max_categories": 10}, require_profile=req
        )
        assert out.shape == (256, 7)
        assert torch.isfinite(out).all()
        assert dominates(block_profile(out.shape[1], meta), req)


def test_num2cat_enforced_raises_when_infeasible():
    torch.manual_seed(0)
    # Requirement needs 4 columns (2 continuous + 2 categorical) but block has 3.
    req = BlockProfile(continuous=2, categorical=(3, 2))
    X = torch.randn(100, 3)
    with pytest.raises(ValueError):
        num2cat(X, {"cat_col_prob": 1.0, "max_categories": 5}, require_profile=req)


def test_num2cat_enforced_full_categorical_coverage_mirrors_u():
    torch.manual_seed(0)
    # proxy_cat_cover_prob=1 -> every requirement gets a categorical column, so
    # the proxy has at least as many categorical columns as U.
    req = BlockProfile(continuous=2, categorical=(5, 3))
    for _ in range(30):
        X = torch.randn(200, 6)
        _, meta = num2cat(
            X,
            {"cat_col_prob": 0.0, "max_categories": 10, "proxy_cat_cover_prob": 1.0},
            require_profile=req,
        )
        prof = block_profile(6, meta)
        assert dominates(prof, req)
        assert len(prof.categorical) >= len(req.categorical)
