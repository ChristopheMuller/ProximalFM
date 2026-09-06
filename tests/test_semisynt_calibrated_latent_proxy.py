"""Focused contracts for native calibrated-latent-proxy random streams."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
import yaml

from semisynt_eval._config import load_case_configs
from semisynt_eval.mechanisms.calibrated_latent_proxy import (
    CalibratedLatentProxyMechanism,
    NamedRandomStreams,
    stable_seed,
)


def _case(name: str):
    recipe = yaml.safe_load(
        (Path(__file__).parents[1] / "semisynt_eval" / "config.yaml").read_text()
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


def test_stable_seed_has_pinned_reference_values():
    assert (
        stable_seed(20260904, "characterization", 2, "linear", "fit")
        == 8168745455118636734
    )
    assert (
        stable_seed(20260904, "characterization", 2, "cal_innovation")
        == 3988771038141720034
    )
    assert (
        stable_seed(20260904, "characterization", 2, "evaluation", "u")
        == 9220404442919492927
    )


def test_named_stream_family_scope_and_replay():
    linear = NamedRandomStreams(20260904, "source.csv", 3, "linear")
    nonlinear = NamedRandomStreams(20260904, "source.csv", 3, "nonlinear")

    linear_fit = torch.randn(12, generator=linear.fit_state())
    nonlinear_fit = torch.randn(12, generator=nonlinear.fit_state())
    assert not torch.equal(linear_fit, nonlinear_fit)
    torch.testing.assert_close(
        torch.randn(12, generator=linear.fit_state()), linear_fit
    )

    torch.testing.assert_close(
        torch.randn(12, generator=linear.calibration_innovation()),
        torch.randn(12, generator=nonlinear.calibration_innovation()),
    )
    for name in ("u", "w", "z", "y"):
        torch.testing.assert_close(
            torch.randn(12, generator=linear.evaluation(name)),
            torch.randn(12, generator=nonlinear.evaluation(name)),
        )
    torch.testing.assert_close(
        torch.rand(12, generator=linear.evaluation("a")),
        torch.rand(12, generator=nonlinear.evaluation("a")),
    )


def test_evaluation_components_use_distinct_streams():
    streams = NamedRandomStreams(71, "source.csv", 1, "linear")
    draws = {
        name: torch.randn(16, generator=streams.evaluation(name))
        for name in ("u", "w", "z", "y")
    }
    assert all(
        not torch.equal(draws[left], draws[right])
        for index, left in enumerate(draws)
        for right in tuple(draws)[index + 1 :]
    )
    with pytest.raises(ValueError, match="unknown evaluation innovation"):
        streams.evaluation("fit")


def test_explicit_identity_makes_fit_and_generation_order_independent():
    case = _case("nonlinear_conf_low_proxy_low")
    source_rng = torch.Generator().manual_seed(19)
    x_cal = torch.randn(40, 3, generator=source_rng, dtype=torch.float64)
    u_cal = torch.randn(40, 2, generator=source_rng, dtype=torch.float64)
    x_eval = torch.randn(8, 3, generator=source_rng, dtype=torch.float64)
    u_eval = torch.randn(8, 2, generator=source_rng, dtype=torch.float64)

    kwargs = {
        "master_seed": 20260904,
        "source_id": "source.csv",
        "replicate": 4,
    }
    first = CalibratedLatentProxyMechanism.fit_with_identity(
        x_cal, u_cal, case, **kwargs
    )
    first_output = first.generate(x_eval, u_eval)
    first.generate(10 * x_eval.flip(0), -10 * u_eval.flip(0))
    replay = CalibratedLatentProxyMechanism.fit_with_identity(
        x_cal, u_cal, case, **kwargs
    )
    replay_output = replay.generate(x_eval, u_eval)

    assert replay.metadata() == first.metadata()
    for field in ("W", "Z", "A", "A_probs", "Y_obs", "Y0", "Y1", "CATE"):
        torch.testing.assert_close(
            getattr(replay_output, field), getattr(first_output, field)
        )


@pytest.mark.parametrize("replicate", (0, -1))
def test_stream_identity_requires_one_based_replicate(replicate):
    with pytest.raises(ValueError, match="one-based"):
        NamedRandomStreams(1, "source.csv", replicate, "linear")


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


def test_qr_rotations_are_full_rank_and_have_positive_orientation():
    for seed in (13, 29):
        rotation_rng = torch.Generator().manual_seed(seed)
        draw_rng = torch.Generator().manual_seed(seed)
        rotation = CalibratedLatentProxyMechanism._orthogonal(4, rotation_rng)
        draw = torch.randn(4, 4, generator=draw_rng, dtype=torch.float64)
        factor = rotation.T @ draw
        torch.testing.assert_close(
            rotation.T @ rotation, torch.eye(4, dtype=torch.float64)
        )
        assert torch.linalg.matrix_rank(rotation) == 4
        assert bool((torch.diagonal(factor) >= 0).all())
        torch.testing.assert_close(factor.tril(-1), torch.zeros_like(factor))


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


def test_prepared_state_is_reused_across_factorial_cases():
    x_cal, u_cal, x_eval, u_eval = _data()
    cases = {
        (case.confounding, case.proxy): case
        for case in (
            _case("linear_conf_low_proxy_low"),
            _case("linear_conf_low_proxy_high"),
            _case("linear_conf_high_proxy_low"),
        )
    }
    mechanism = _fit(cases[("low", "low")], x_cal, u_cal)
    rows = torch.arange(len(x_eval))
    prepared = mechanism.prepare_evaluation(x_eval, u_eval, rows)
    outputs = {
        levels: mechanism.generate_case(prepared, case)
        for levels, case in cases.items()
    }
    low = outputs[("low", "low")]
    high_proxy = outputs[("low", "high")]
    high_conf = outputs[("high", "low")]

    for output in outputs.values():
        assert output.U is prepared.U
        assert output.U_source is prepared.U_source
        assert output.V is prepared.V
        assert output.CATE is prepared.CATE
    for field in ("U", "U_source", "V", "A", "A_probs", "Y0", "Y1", "Y_obs", "CATE"):
        torch.testing.assert_close(getattr(low, field), getattr(high_proxy, field))
    for field in ("U", "U_source", "V", "W", "Z", "CATE"):
        torch.testing.assert_close(getattr(low, field), getattr(high_conf, field))
    assert not torch.equal(low.W, high_proxy.W)
    assert not torch.equal(low.Z, high_proxy.Z)


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


def test_fit_rejects_invalid_calibration_inputs_and_reliability():
    x_cal, u_cal, _, _ = _data()
    case = _case("linear_conf_low_proxy_low")
    with pytest.raises(ValueError, match="matching rows"):
        _fit(case, x_cal[:-1], u_cal)
    invalid_x = x_cal.clone()
    invalid_x[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        _fit(case, invalid_x, u_cal)
    with pytest.raises(ValueError, match=r"reliability must lie in \(0, 1\)"):
        _fit(replace(case, reliability=0), x_cal, u_cal)
