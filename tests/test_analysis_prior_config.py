"""Tests for analysis_prior.config."""

import json

import pytest

from analysis_prior.config import PRESETS, PriorSpec, pinned
from proximalfm.prior._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


def test_defaults_are_valid():
    spec = PriorSpec()
    assert spec.name == "default"
    assert spec.n_datasets >= 1


def test_to_dict_from_dict_round_trip():
    spec = PriorSpec(
        name="rt",
        n_datasets=3,
        seq_len=50,
        fixed_hp_overrides={"cat_prob": 0.0},
        sampled_hp_overrides={"propensity_shift": pinned(1.5)},
    )
    assert PriorSpec.from_dict(spec.to_dict()) == spec


def test_to_dict_is_json_serialisable():
    spec = PriorSpec(sampled_hp_overrides={"propensity_shift": pinned(0.0)})
    # Must survive a real JSON round trip, not just json.dumps: the sidecar
    # written next to each cache is read back as plain JSON.
    assert PriorSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_from_dict_rejects_unknown_fields():
    data = PriorSpec().to_dict()
    data["not_a_field"] = 1
    # Silently ignoring the key would yield a spec that hashes differently from
    # the one that was saved.
    with pytest.raises(ValueError, match="Unknown PriorSpec fields"):
        PriorSpec.from_dict(data)


def test_digest_is_stable_across_calls():
    spec = PriorSpec(name="s", n_datasets=2)
    assert spec.digest() == PriorSpec(name="s", n_datasets=2).digest()


def test_digest_is_insensitive_to_override_key_order():
    a = PriorSpec(fixed_hp_overrides={"cat_prob": 0.0, "corrupt_U": True})
    b = PriorSpec(fixed_hp_overrides={"corrupt_U": True, "cat_prob": 0.0})
    assert a.digest() == b.digest()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "other"},
        {"n_datasets": 2},
        {"seq_len": 999},
        {"max_dim_U": 9},
        {"prior_type": "tree_scm"},
        {"mcmc_cate": 7},
        {"seed": 1},
        {"fixed_hp_overrides": {"cat_prob": 0.0}},
        {"sampled_hp_overrides": {"propensity_shift": pinned(0.0)}},
    ],
)
def test_digest_changes_with_every_field(kwargs):
    # Any field that changes the generated data must change the cache key,
    # otherwise a stale cache would be served for a different configuration.
    assert PriorSpec(**kwargs).digest() != PriorSpec().digest()


def test_cache_key_is_content_addressed():
    # The cache key is the generative digest: specs that produce identical data
    # must share one cache file regardless of their name or how many datasets
    # were requested.
    spec = PriorSpec(name="abc")
    assert spec.cache_key == spec.generative_digest()
    assert spec.cache_key == PriorSpec(name="xyz", n_datasets=999).cache_key
    # ...but a genuinely different configuration must not collide.
    assert spec.cache_key != PriorSpec(name="abc", seed=1).cache_key


def test_generative_dict_excludes_non_generative_fields():
    d = PriorSpec(name="abc", n_datasets=7).generative_dict()
    assert "name" not in d and "n_datasets" not in d
    assert d["seq_len"] == PriorSpec().seq_len


def test_same_data_as():
    assert PriorSpec(name="a", n_datasets=2).same_data_as(
        PriorSpec(name="b", n_datasets=9)
    )
    assert not PriorSpec(seed=0).same_data_as(PriorSpec(seed=1))


@pytest.mark.parametrize("name", ["", "has space", "has.dot", "has/slash"])
def test_invalid_names_rejected(name):
    with pytest.raises(ValueError, match="name must be"):
        PriorSpec(name=name)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_datasets": 0}, "n_datasets"),
        ({"seq_len": 1}, "seq_len"),
        ({"mcmc_cate": 0}, "mcmc_cate"),
        ({"prior_type": "nope"}, "Unknown prior_type"),
        ({"min_dim_U": 6, "max_dim_U": 5}, "min_dim_U must be <= max_dim_U"),
        ({"min_dim_W": 0}, "min_dim_W must be >= 1"),
    ],
)
def test_validation_rejects_bad_specs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PriorSpec(**kwargs)


def test_non_serialisable_overrides_rejected():
    # Activation classes live in DEFAULT_SAMPLED_HP and are not JSON-serialisable;
    # letting one into an override would produce an unhashable, unreproducible spec.
    with pytest.raises(ValueError, match="JSON-serialisable"):
        PriorSpec(sampled_hp_overrides={"mlp_activations": object()})


def test_resolved_hp_merges_over_defaults():
    spec = PriorSpec(
        fixed_hp_overrides={"cat_prob": 0.99},
        sampled_hp_overrides={"propensity_shift": pinned(2.0)},
    )
    fixed, sampled = spec.resolved_hp()

    assert fixed["cat_prob"] == 0.99
    # Untouched defaults must still be present.
    assert fixed["propensity_epsilon"] == DEFAULT_FIXED_HP["propensity_epsilon"]
    assert sampled["propensity_shift"] == pinned(2.0)
    assert "mlp_activations" in sampled


def test_resolved_hp_does_not_mutate_module_defaults():
    # resolved_hp returns deep copies; mutating the result must not leak into the
    # config shared with training.
    original = DEFAULT_FIXED_HP["cat_prob"]
    fixed, sampled = PriorSpec().resolved_hp()
    fixed["cat_prob"] = 123.0
    sampled["num_layers"]["max_mean"] = 999

    assert DEFAULT_FIXED_HP["cat_prob"] == original
    assert DEFAULT_SAMPLED_HP["num_layers"]["max_mean"] != 999


def test_dataset_seed_is_deterministic_and_distinct():
    spec = PriorSpec(n_datasets=5)
    seeds = [spec.dataset_seed(i) for i in range(5)]
    assert seeds == [spec.dataset_seed(i) for i in range(5)]
    assert len(set(seeds)) == 5


def test_dataset_seed_depends_on_master_seed():
    a = PriorSpec(name="s", seed=0)
    b = PriorSpec(name="s", seed=1)
    assert a.dataset_seed(0) != b.dataset_seed(0)


def test_dataset_seed_stable_when_n_datasets_grows():
    # Adding datasets must not perturb the ones already generated and cached.
    small = PriorSpec(name="s", n_datasets=2)
    large = PriorSpec(name="s", n_datasets=10)
    assert [small.dataset_seed(i) for i in range(2)] == [
        large.dataset_seed(i) for i in range(2)
    ]


def test_dataset_seed_stable_when_renamed():
    # name is a label for figures and filenames, not a generative parameter.
    a = PriorSpec(name="before", n_datasets=3)
    b = PriorSpec(name="after", n_datasets=3)
    assert [a.dataset_seed(i) for i in range(3)] == [
        b.dataset_seed(i) for i in range(3)
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seq_len": 999},
        {"max_dim_U": 9},
        {"prior_type": "tree_scm"},
        {"mcmc_cate": 7},
        {"seed": 1},
        {"fixed_hp_overrides": {"cat_prob": 0.0}},
        {"sampled_hp_overrides": {"propensity_shift": pinned(0.0)}},
    ],
)
def test_dataset_seed_changes_with_generative_fields(kwargs):
    # Anything that changes what a dataset looks like must reseed it, otherwise a
    # spec change could silently produce the old data.
    assert PriorSpec(**kwargs).dataset_seed(0) != PriorSpec().dataset_seed(0)


@pytest.mark.parametrize("kwargs", [{"name": "other"}, {"n_datasets": 128}])
def test_generative_digest_ignores_non_generative_fields(kwargs):
    assert PriorSpec(**kwargs).generative_digest() == PriorSpec().generative_digest()


def test_generative_digest_differs_from_full_digest():
    # They serve different purposes; collapsing them would reintroduce the bug
    # where growing n_datasets resamples every dataset.
    spec = PriorSpec()
    assert spec.generative_digest() != spec.digest()


def test_dataset_seed_in_valid_numpy_range():
    spec = PriorSpec(n_datasets=50)
    for i in range(50):
        assert 0 <= spec.dataset_seed(i) < 2**31 - 1


@pytest.mark.parametrize("index", [-1, 5])
def test_dataset_seed_rejects_out_of_range_index(index):
    with pytest.raises(IndexError):
        PriorSpec(n_datasets=5).dataset_seed(index)


def test_pinned_is_a_degenerate_distribution():
    assert pinned(1.5) == {"distribution": "uniform", "min": 1.5, "max": 1.5}


def test_presets_are_valid_and_self_named():
    assert PRESETS  # guard against an accidentally emptied registry
    for key, spec in PRESETS.items():
        assert spec.name == key, f"preset {key!r} is registered under a different name"


# --------------------------------------------------------------------------
# SpecFamily
# --------------------------------------------------------------------------


def _family(**kwargs):
    from analysis_prior.config import SpecFamily

    defaults = dict(
        name="fam",
        base=PriorSpec(name="base", fixed_hp_overrides={"cat_prob": 0.0}),
        interventions={
            "identity": {},
            "uniform": {"fixed_hp_overrides": {"dim_U_distribution": "uniform"}},
        },
    )
    return SpecFamily(**{**defaults, **kwargs})


def test_family_resolve_names_and_order():
    variants = _family().resolve()
    # Intervention order is preserved (renderers rely on it for series order)
    # and variant names are derived from the base name and the label.
    assert list(variants) == ["identity", "uniform"]
    assert variants["identity"].name == "base_identity"
    assert variants["uniform"].name == "base_uniform"


def test_family_identity_variant_generates_the_same_data():
    fam = _family()
    identity = fam.resolve()["identity"]
    assert identity.same_data_as(fam.base)
    # Content-addressed caching makes the identity variant free: same cache key.
    assert identity.cache_key == fam.base.cache_key


def test_family_overrides_deep_merge_within_hp_dicts():
    variants = _family().resolve()
    merged = variants["uniform"].fixed_hp_overrides
    # The base's own override survives the intervention on a sibling key...
    assert merged == {"cat_prob": 0.0, "dim_U_distribution": "uniform"}


def test_family_intervention_wins_on_conflicting_keys():
    fam = _family(interventions={"recat": {"fixed_hp_overrides": {"cat_prob": 0.7}}})
    assert fam.resolve()["recat"].fixed_hp_overrides == {"cat_prob": 0.7}


def test_family_non_dict_fields_replace():
    fam = _family(interventions={"short": {"seq_len": 64, "mcmc_cate": 3}})
    variant = fam.resolve()["short"]
    assert variant.seq_len == 64
    assert variant.mcmc_cate == 3
    # Untouched fields come from the base.
    assert variant.fixed_hp_overrides == {"cat_prob": 0.0}


def test_family_resolve_does_not_mutate_base():
    fam = _family()
    fam.resolve()
    assert fam.base.fixed_hp_overrides == {"cat_prob": 0.0}
    assert fam.base.name == "base"


def test_family_rejects_empty_interventions():
    with pytest.raises(ValueError, match="at least one intervention"):
        _family(interventions={})


@pytest.mark.parametrize("label", ["", "has space", "has.dot"])
def test_family_rejects_bad_labels(label):
    with pytest.raises(ValueError, match="label"):
        _family(interventions={label: {}})


def test_family_rejects_name_intervention():
    with pytest.raises(ValueError, match="derived from the base name"):
        _family(interventions={"renamed": {"name": "sneaky"}})


def test_family_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown PriorSpec fields"):
        _family(interventions={"typo": {"seq_length": 10}})


def test_family_rejects_invalid_override_values_at_construction():
    # Eager resolve: a bad value fails when the family is built, not mid-run.
    with pytest.raises(ValueError, match="seq_len"):
        _family(interventions={"bad": {"seq_len": 1}})


def test_family_to_dict_is_json_serialisable():
    payload = json.loads(json.dumps(_family().to_dict()))
    assert set(payload["variants"]) == {"identity", "uniform"}
    assert payload["base"]["name"] == "base"


def test_families_registry_is_self_consistent():
    from analysis_prior.config import FAMILIES

    assert "dim_ablation" in FAMILIES
    for key, family in FAMILIES.items():
        assert family.name == key, f"family {key!r} registered under another name"
    # The dim ablation carries the three variants the appendix compares.
    assert list(FAMILIES["dim_ablation"].interventions) == [
        "triangular",
        "uniform",
        "unenforced",
    ]
