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


def test_primitive_sampler_ranges_and_types():
    cases = [
        (trunc_norm_sampler(mu=0.0, sigma=1.0), float, lambda x: x >= 0.0),
        (beta_sampler(a=2.0, b=5.0), float, lambda x: 0.0 <= x <= 1.0),
        (gamma_sampler(a=2.0, b=2.0), float, lambda x: x >= 0.0),
        (uniform_sampler(a=1.0, b=5.0), float, lambda x: 1.0 <= x <= 5.0),
        (uniform_int_sampler(a=1.0, b=5.0), int, lambda x: 1 <= x <= 5),
    ]
    for sampler, expected_type, in_range in cases:
        value = sampler()
        assert isinstance(value, expected_type), sampler
        assert in_range(value), sampler


def test_hp_sampler_basic_distributions():
    cases = [
        ({"distribution": "uniform", "min": 1.0, "max": 5.0}, float, 1.0, 5.0),
        ({"distribution": "beta", "a": 2.0, "b": 5.0}, float, 0.0, 1.0),
        ({"distribution": "uniform_int", "min": 1, "max": 5}, int, 1, 5),
    ]
    for kwargs, expected_type, low, high in cases:
        value = HpSampler(device="cpu", **kwargs)()
        assert isinstance(value, expected_type), kwargs
        assert low <= value <= high, kwargs


def test_hp_sampler_meta_distributions():
    cases = [
        (
            {"distribution": "meta_beta", "min": 1.0, "max": 5.0, "scale": 2.0},
            0.0,
            2.0,
        ),
        (
            {
                "distribution": "meta_gamma",
                "max_alpha": 2.0,
                "max_scale": 5.0,
                "lower_bound": 1.0,
                "round": False,
            },
            1.0,
            math.inf,
        ),
        (
            {
                "distribution": "meta_trunc_norm",
                "min_mean": 0.0,
                "max_mean": 1.0,
                "min_std": 0.1,
                "max_std": 1.0,
                "lower_bound": 0.0,
                "round": False,
            },
            0.0,
            math.inf,
        ),
        (
            {
                "distribution": "meta_trunc_norm_log_scaled",
                "min_mean": 0.1,
                "max_mean": 1.0,
                "lower_bound": 0.0,
                "round": False,
            },
            0.0,
            math.inf,
        ),
    ]
    for kwargs, low, high in cases:
        sub_sampler = HpSampler(device="cpu", **kwargs)()
        value = sub_sampler()
        assert isinstance(value, float), kwargs
        assert low <= value <= high, kwargs

    rounded = HpSampler(
        distribution="meta_gamma",
        device="cpu",
        max_alpha=2.0,
        max_scale=5.0,
        lower_bound=1.0,
        round=True,
    )()()
    assert rounded >= 1.0 and float(rounded).is_integer()

    choice_values = ["A", "B", "C"]
    assert (
        HpSampler(
            distribution="meta_choice", device="cpu", choice_values=choice_values
        )()
        in choice_values
    )


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


def test_hp_sampler_rejects_unknown_distributions():
    for distribution, message in [
        ("foo", "Unsupported distribution: foo"),
        ("meta_foo", "Unsupported meta distribution: meta_foo"),
    ]:
        with pytest.raises(ValueError, match=message):
            HpSampler(distribution=distribution, device="cpu")


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
