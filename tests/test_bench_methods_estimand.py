"""Which conditional each method output claims to answer.

``method_estimand`` routes an output to the benchmark's tracks, and it is
*derived* from the existing config rather than declared — so these tests are
what guarantee that every config file written before the tracks existed still
means what it did. The consequential case is ``xw``: an output evaluated at the
query row's own proxies estimates E[Y1-Y0 | X, W], which must never be ranked
against the CATE target as if it were an estimate of tau(x).
"""

import pytest

from benchmarks.methods import method_estimand


@pytest.mark.parametrize(
    "strategy,expected",
    [
        ("paired", "xw"),  # evaluated at the query row's own W
        ("independent", "x"),  # W marginalised out -> back to tau(x)
        ("conditional", "x"),
        ("marginal", "ate"),  # one scalar per episode
    ],
)
def test_strategy_decides_the_estimand(strategy, expected):
    cfg = {"name": "PMMR", "cls": "PMMR", "strategy": strategy}
    assert method_estimand(cfg) == expected


def test_outputs_entry_wins_over_the_method_level_strategy():
    cfg = {"name": "PMMR", "cls": "PMMR", "strategy": "marginal"}
    out = {"name": "PMMR-XW", "strategy": "paired"}
    assert method_estimand(cfg, out) == "xw"


def test_shipped_pmmr_outputs_span_all_three_tracks():
    """The names in config_val.yaml, resolved."""
    cfg = {"name": "PMMR", "cls": "PMMR"}
    outputs = [
        {"name": "PMMR-XW", "strategy": "paired"},
        {"name": "PMMR-X-Wind", "strategy": "independent"},
        {"name": "PMMR-X-Wcond", "strategy": "conditional"},
        {"name": "PMMR-ATE", "strategy": "marginal"},
    ]
    assert [method_estimand(cfg, o) for o in outputs] == ["xw", "x", "x", "ate"]


def test_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="unknown strategy"):
        method_estimand({"name": "m", "cls": "PMMR", "strategy": "sideways"})


@pytest.mark.parametrize("cls", ["S_Learner", "T_Learner", "X_Learner"])
def test_learners_follow_their_proxy_flags(cls):
    assert method_estimand({"cls": cls, "use_W": False, "use_Z": False}) == "x"
    assert method_estimand({"cls": cls, "use_W": True, "use_Z": False}) == "xw"
    assert method_estimand({"cls": cls, "use_W": False, "use_Z": True}) == "xw"


@pytest.mark.parametrize(
    "cls", ["S_Learner", "T_Learner", "X_Learner", "CausalFM", "DoPFN", "CausalPFN"]
)
def test_proxy_flags_default_to_true_as_the_factories_do(cls):
    """build_method reads use_W/use_Z with default True for these classes, so an
    entry that says neither really is fitted on the proxies."""
    assert method_estimand({"cls": cls}) == "xw"


def test_p2sls_is_ate_only():
    assert method_estimand({"name": "P2SLS", "cls": "P2SLS"}) == "ate"


@pytest.mark.parametrize("cls", ["PLearner", "CausalForest"])
def test_methods_that_discard_proxies_at_predict_are_tau_x(cls):
    # Both accept W/Z at fit and neither uses them at predict time.
    assert method_estimand({"cls": cls, "use_W": True, "use_Z": True}) == "x"


def test_explicit_estimand_overrides_everything():
    cfg = {"name": "custom", "cls": "S_Learner", "use_W": True, "estimand": "x"}
    assert method_estimand(cfg) == "x"
    out = {"name": "o", "strategy": "paired", "estimand": "ate"}
    assert method_estimand({"cls": "PMMR"}, out) == "ate"


def test_explicit_estimand_is_validated():
    with pytest.raises(ValueError, match="estimand must be one of"):
        method_estimand({"cls": "S_Learner", "estimand": "tau_of_everything"})


def test_unknown_class_defaults_to_the_conservative_reading():
    """Scored as tau(x), a secretly-proxy-using method merely looks bad on the
    primary track; the reverse would hand it an unearned advantage."""
    assert method_estimand({"name": "?", "cls": "SomethingNew"}) == "x"
