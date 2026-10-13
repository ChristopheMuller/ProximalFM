"""Tests for analysis.analysis_prior.config."""


from analysis.analysis_prior.config import PriorSpec, pinned


def test_to_dict_from_dict_round_trip():
    spec = PriorSpec(
        name="rt",
        n_datasets=3,
        seq_len=50,
        fixed_hp_overrides={"proxy_cat_cover_prob": 0.0},
        sampled_hp_overrides={"propensity_shift": pinned(1.5)},
    )
    assert PriorSpec.from_dict(spec.to_dict()) == spec


def test_dataset_seed_stable_when_n_datasets_grows():
    # Adding datasets must not perturb the ones already generated and cached.
    small = PriorSpec(name="s", n_datasets=2)
    large = PriorSpec(name="s", n_datasets=10)
    assert [small.dataset_seed(i) for i in range(2)] == [
        large.dataset_seed(i) for i in range(2)
    ]


def test_dataset_seed_changes_with_generative_fields():
    for kwargs in [
        {"seq_len": 999},
        {"max_dim_U": 9},
        {"prior_type": "tree_scm"},
        {"mc_cate": 7},
        {"seed": 1},
        {"fixed_hp_overrides": {"proxy_cat_cover_prob": 0.0}},
        {"sampled_hp_overrides": {"propensity_shift": pinned(0.0)}},
    ]:
        assert PriorSpec(**kwargs).dataset_seed(0) != PriorSpec().dataset_seed(
            0
        ), kwargs


def _family(**kwargs):
    from analysis.analysis_prior.config import SpecFamily

    defaults = dict(
        name="fam",
        base=PriorSpec(name="base", fixed_hp_overrides={"proxy_cat_cover_prob": 0.0}),
        interventions={
            "identity": {},
            "uniform": {"fixed_hp_overrides": {"dim_U_distribution": "uniform"}},
        },
    )
    return SpecFamily(**{**defaults, **kwargs})
