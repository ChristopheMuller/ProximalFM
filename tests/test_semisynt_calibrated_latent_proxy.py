"""Focused contracts for native calibrated-latent-proxy random streams."""


import json


from pathlib import Path


import pytest


import torch


import yaml


from eval_datasets.semisynt_eval._config import load_case_configs


from eval_datasets.semisynt_eval.mechanisms.calibrated_latent_proxy import (
    CalibratedLatentProxyMechanism,
)


def _case(name: str):
    recipe = yaml.safe_load(
        (
            Path(__file__).parents[1]
            / "eval_datasets"
            / "semisynt_eval"
            / "config.yaml"
        ).read_text()
    )
    return load_case_configs(recipe["common"], recipe["cases"])[name]


def _data():
    generator = torch.Generator().manual_seed(91)
    return (
        torch.randn(96, 4, generator=generator, dtype=torch.float64),
        torch.randn(96, 3, generator=generator, dtype=torch.float64),
        torch.randn(20, 4, generator=generator, dtype=torch.float64),
        torch.randn(20, 3, generator=generator, dtype=torch.float64),
    )


def _fit(case, x_cal, u_cal):
    return CalibratedLatentProxyMechanism.fit_with_identity(
        x_cal,
        u_cal,
        case,
        master_seed=20260904,
        source_id="source.csv",
        replicate=2,
    )


@pytest.mark.parametrize("family", ("linear", "nonlinear"))
def test_latent_calibration_and_family_transform(family):
    x_cal, u_cal, _, _ = _data()
    case = _case(f"{family}_conf_low_proxy_low")
    mechanism = _fit(case, x_cal, u_cal)
    latent_cal = mechanism._augment_latent(
        u_cal, mechanism.streams.calibration_innovation()
    )
    expected = latent_cal if family == "linear" else torch.asinh(latent_cal)
    torch.testing.assert_close(mechanism.latent_transform(latent_cal), expected)

    coordinates = mechanism.latent_coordinates(latent_cal)
    torch.testing.assert_close(
        coordinates.mean(0),
        torch.zeros(mechanism.d_u, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    latent_index = coordinates @ mechanism.latent_direction
    torch.testing.assert_close(
        latent_index.std(unbiased=False),
        torch.tensor(case.latent.direction_sd, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


def test_proxy_reliability_rescales_only_named_measurement_noise():
    x_cal, u_cal, x_eval, u_eval = _data()
    low = _fit(_case("linear_conf_low_proxy_low"), x_cal, u_cal)
    high = _fit(_case("linear_conf_low_proxy_high"), x_cal, u_cal)
    low_output, high_output = low.generate(x_eval, u_eval), high.generate(
        x_eval, u_eval
    )

    for field in ("A", "A_probs", "Y_obs", "Y0", "Y1", "CATE"):
        torch.testing.assert_close(
            getattr(low_output, field), getattr(high_output, field)
        )
    assert not torch.equal(low_output.W, high_output.W)
    assert not torch.equal(low_output.Z, high_output.Z)

    noise = low.innovations(len(x_eval))
    latent = (
        1 - low.config.latent.innovation_sd**2
    ) ** 0.5 * u_eval + low.config.latent.innovation_sd * noise.u
    coordinates = low.latent_coordinates(latent)
    for name, observed_low, observed_high, rotation, signal_sd, innovation in (
        (
            "w",
            low_output.W,
            high_output.W,
            low.rotation_w,
            low.proxy_signal_sd_w,
            noise.w,
        ),
        (
            "z",
            low_output.Z,
            high_output.Z,
            low.rotation_z,
            low.proxy_signal_sd_z,
            noise.z,
        ),
    ):
        signal = coordinates @ rotation.T + low.config.proxies.x_scale * low.component(
            x_eval, name
        )
        low_multiplier = (
            (1 - low.default_case.reliability) / low.default_case.reliability
        ) ** 0.5
        high_multiplier = (
            (1 - high.default_case.reliability) / high.default_case.reliability
        ) ** 0.5
        torch.testing.assert_close(
            (observed_low - signal) / low_multiplier, signal_sd * innovation
        )
        torch.testing.assert_close(
            (observed_high - signal) / high_multiplier, signal_sd * innovation
        )


@pytest.mark.parametrize("family", ("linear", "nonlinear"))
def test_generation_causal_contract_precision_and_metadata(family):
    x_cal, u_cal, x_eval, u_eval = _data()
    case = _case(f"{family}_conf_high_proxy_high")
    mechanism = _fit(case, x_cal, u_cal)
    output = mechanism.generate(x_eval, u_eval)

    torch.testing.assert_close(output.Y1 - output.Y0, output.CATE)
    torch.testing.assert_close(
        output.Y_obs, torch.where(output.A.bool(), output.Y1, output.Y0)
    )
    bridge0 = mechanism.bridge(x_eval, output.W, torch.tensor(0), gamma=case.gamma)
    bridge1 = mechanism.bridge(x_eval, output.W, torch.tensor(1), gamma=case.gamma)
    torch.testing.assert_close(bridge1 - bridge0, output.CATE)
    torch.testing.assert_close(
        mechanism.unwrap_proxy(mechanism.wrap_proxy(x_eval[:, :3])), x_eval[:, :3]
    )

    for field in ("W", "Z", "A", "A_probs", "Y_obs", "Y0", "Y1", "CATE"):
        value = getattr(output, field)
        assert value.dtype == torch.float64
        assert bool(torch.isfinite(value).all())
        converted = value.float()
        assert converted.dtype == torch.float32
        torch.testing.assert_close(converted.double(), value, atol=1e-6, rtol=1e-6)
    assert bool(
        (
            (output.A_probs >= case.treatment.propensity_lower)
            & (output.A_probs <= case.treatment.propensity_upper)
        ).all()
    )

    metadata = mechanism.metadata()
    json.dumps(metadata)
    assert {
        "basis",
        "components",
        "whitener",
        "latent_direction",
        "B_W",
        "B_Z",
        "proxy_signal_sd_w",
        "proxy_signal_sd_z",
        "treatment_intercept",
    } <= metadata.keys()
