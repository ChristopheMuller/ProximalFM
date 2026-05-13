import math
import numpy as np
import pytest
import torch

from proximalfm.prior._hp_sampling import (
    trunc_norm_sampler,
    beta_sampler,
    gamma_sampler,
    uniform_sampler,
    uniform_int_sampler,
    HpSampler,
    HpSamplerList,
)


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)


def test_trunc_norm_sampler():
    sampler = trunc_norm_sampler(mu=0.0, sigma=1.0)
    assert callable(sampler)
    val = sampler()
    assert isinstance(val, float)
    # The stats.truncnorm bounds in the code are (0-mu)/sigma to (1000000-mu)/sigma
    assert val >= 0.0


def test_beta_sampler():
    sampler = beta_sampler(a=2.0, b=5.0)
    assert callable(sampler)
    val = sampler()
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0


def test_gamma_sampler():
    sampler = gamma_sampler(a=2.0, b=2.0)
    assert callable(sampler)
    val = sampler()
    assert isinstance(val, float)
    assert val >= 0.0


def test_uniform_sampler():
    sampler = uniform_sampler(a=1.0, b=5.0)
    assert callable(sampler)
    val = sampler()
    assert isinstance(val, float)
    assert 1.0 <= val <= 5.0


def test_uniform_int_sampler():
    sampler = uniform_int_sampler(a=1.0, b=5.0)
    assert callable(sampler)
    val = sampler()
    assert isinstance(val, int)
    assert 1 <= val <= 5


def test_hp_sampler_uniform():
    sampler = HpSampler(distribution="uniform", device="cpu", min=1.0, max=5.0)
    val = sampler()
    assert isinstance(val, float)
    assert 1.0 <= val <= 5.0


def test_hp_sampler_beta():
    sampler = HpSampler(distribution="beta", device="cpu", a=2.0, b=5.0)
    val = sampler()
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0


def test_hp_sampler_uniform_int():
    sampler = HpSampler(distribution="uniform_int", device="cpu", min=1.0, max=5.0)
    val = sampler()
    assert isinstance(val, int)
    assert 1 <= val <= 5


def test_hp_sampler_unsupported_distribution():
    with pytest.raises(ValueError, match="Unsupported distribution: foo"):
        HpSampler(distribution="foo", device="cpu")


def test_hp_sampler_meta_beta():
    # meta_beta creates uniform samplers for b and k
    sampler = HpSampler(
        distribution="meta_beta", device="cpu", min=1.0, max=5.0, scale=2.0
    )
    sub_sampler = sampler()
    assert callable(sub_sampler)
    val = sub_sampler()
    assert isinstance(val, float)
    assert 0.0 <= val <= 2.0


def test_hp_sampler_meta_gamma():
    sampler = HpSampler(
        distribution="meta_gamma",
        device="cpu",
        max_alpha=2.0,
        max_scale=5.0,
        lower_bound=1.0,
        round=False,
    )
    sub_sampler = sampler()
    assert callable(sub_sampler)
    val = sub_sampler()
    assert isinstance(val, float)
    assert val >= 1.0


def test_hp_sampler_meta_gamma_round():
    sampler = HpSampler(
        distribution="meta_gamma",
        device="cpu",
        max_alpha=2.0,
        max_scale=5.0,
        lower_bound=1.0,
        round=True,
    )
    sub_sampler = sampler()
    assert callable(sub_sampler)
    val = sub_sampler()
    # When rounded, it returns a float that is a whole number, or int if float rounding gives whole number.
    # In python round() returns int but adding float lower_bound might make it float.
    assert isinstance(val, (int, float))
    assert val >= 1.0
    assert float(val).is_integer()


def test_hp_sampler_meta_trunc_norm():
    sampler = HpSampler(
        distribution="meta_trunc_norm",
        device="cpu",
        min_mean=0.0,
        max_mean=1.0,
        min_std=0.1,
        max_std=1.0,
        lower_bound=0.0,
        round=False,
    )
    sub_sampler = sampler()
    assert callable(sub_sampler)
    val = sub_sampler()
    assert isinstance(val, float)
    assert val >= 0.0


def test_hp_sampler_meta_trunc_norm_log_scaled():
    sampler = HpSampler(
        distribution="meta_trunc_norm_log_scaled",
        device="cpu",
        min_mean=0.1,
        max_mean=1.0,
        lower_bound=0.0,
        round=False,
    )
    sub_sampler = sampler()
    assert callable(sub_sampler)
    val = sub_sampler()
    assert isinstance(val, float)
    assert val >= 0.0


def test_hp_sampler_meta_choice():
    choice_values = ["A", "B", "C"]
    sampler = HpSampler(
        distribution="meta_choice", device="cpu", choice_values=choice_values
    )
    val = sampler()
    assert val in choice_values


def test_hp_sampler_meta_choice_mixed():
    def get_A():
        return "A"

    def get_B():
        return "B"

    choice_values = [get_A, get_B]
    sampler = HpSampler(
        distribution="meta_choice_mixed", device="cpu", choice_values=choice_values
    )
    sub_sampler_maker = sampler()
    assert callable(sub_sampler_maker)
    sub_sampler = sub_sampler_maker()
    assert callable(sub_sampler)
    val = sub_sampler()
    assert val in ["A", "B"]


def test_hp_sampler_unsupported_meta_distribution():
    with pytest.raises(ValueError, match="Unsupported meta distribution: meta_foo"):
        HpSampler(distribution="meta_foo", device="cpu")


def test_hp_sampler_list():
    hp_config = {
        "learning_rate": {
            "distribution": "meta_trunc_norm_log_scaled",
            "min_mean": 1e-4,
            "max_mean": 1e-1,
            "lower_bound": 0.0,
            "round": False,
        },
        "num_layers": {
            "distribution": "uniform_int",
            "min": 2,
            "max": 10,
        },
    }

    sampler_list = HpSamplerList(hyperparameters=hp_config, device="cpu")
    params = sampler_list.sample()

    assert "learning_rate" in params
    assert "num_layers" in params

    # learning_rate is a meta distribution, so it returns a closure
    assert callable(params["learning_rate"])
    lr = params["learning_rate"]()
    assert isinstance(lr, float)
    assert lr >= 0.0

    # num_layers is a basic distribution, so it directly returns an int
    assert isinstance(params["num_layers"], int)
    assert 2 <= params["num_layers"] <= 10
