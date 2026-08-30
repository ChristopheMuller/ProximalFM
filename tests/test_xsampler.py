"""Tests for ``proximalfm.prior._utils.XSampler``.

The sampler produces the root causes of ``X`` and every mechanism's exogenous
noise (eps) block, so a NaN here reaches ``process_features`` before any
mechanism has a chance to ``nan_to_num`` it.
"""

import random

import numpy as np
import pytest
import torch

from proximalfm.prior._utils import XSampler


def _seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _sampler(seq_len, num_features=3, sampling="mixed"):
    return XSampler(seq_len=seq_len, num_features=num_features, sampling=sampling)


def test_multinomial_column_is_standardised():
    _seed(0)
    x = _sampler(2048).sample_multinomial()

    assert torch.isfinite(x).all()
    assert abs(float(x.mean())) < 1e-4
    assert abs(float(x.std()) - 1.0) < 1e-4


def test_multinomial_survives_a_single_category_draw(monkeypatch):
    # torch.rand can hand one category nearly all the mass; every sample then
    # lands there, std is 0 and the column became all-NaN.
    monkeypatch.setattr(
        torch, "multinomial", lambda probs, n, replacement: torch.zeros(n).long()
    )
    _seed(0)
    x = _sampler(256).sample_multinomial()

    assert torch.isfinite(x).all()
    assert torch.equal(x, torch.zeros(256))


def test_multinomial_survives_seq_len_one():
    # A single sample has no unbiased std -- torch.std_mean returns NaN.
    _seed(0)
    x = _sampler(1).sample_multinomial()

    assert x.shape == (1,)
    assert torch.isfinite(x).all()


def test_multinomial_leaves_non_degenerate_draws_bit_identical():
    # The guard is a clamp, not an added epsilon, so a healthy draw must come
    # out exactly as `(x - mean) / std` did before.
    _seed(3)
    sampler = _sampler(1024)
    n_categories = random.randint(2, 20)
    probs = torch.rand(n_categories)
    raw = torch.multinomial(probs, 1024, replacement=True).float()
    std, mean = torch.std_mean(raw)
    expected = (raw - mean) / std

    _seed(3)
    got = sampler.sample_multinomial()

    assert float(std) > 1e-8
    assert torch.equal(got, expected)


def test_sample_never_emits_nan_across_strategies_and_sequence_sizes():
    for sampling, seq_len in (("normal", 1), ("mixed", 7), ("uniform", 128)):
        for seed in range(8):
            _seed(seed)
            X = _sampler(seq_len, num_features=6, sampling=sampling).sample()

            assert X.shape == (seq_len, 6), (sampling, seq_len, seed)
            assert torch.isfinite(X).all(), (sampling, seq_len, seed)


def test_sample_rejects_an_unknown_strategy():
    with pytest.raises(ValueError, match="Invalid sampling method"):
        _sampler(16, sampling="poisson").sample()
