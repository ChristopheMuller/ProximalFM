"""Which conditional each method output claims to answer.

``method_estimand`` routes an output to the benchmark's tracks, and it is
*derived* from the existing config rather than declared — so these tests are
what guarantee that every config file written before the tracks existed still
means what it did. The consequential case is ``cate(x,w)``: an output evaluated
at the query row's own proxies estimates E[Y1-Y0 | X, W], which must never be
ranked against the CATE target as if it were an estimate of tau(x).

The tag spells out the *predict-time* inputs, so it is also a readable label:
what routes is the kind (cate/ate) plus whether w or z is in the parenthesis.
"""

import pytest

from benchmarks.methods import method_estimand


@pytest.mark.parametrize(
    "strategy,expected",
    [
        ("paired", "cate(x,w)"),  # evaluated at the query row's own W
        ("independent", "cate(x)"),  # W marginalised out -> back to tau(x)
        ("conditional", "cate(x)"),
        ("marginal", "ate(x)"),  # one scalar per episode
    ],
)
def test_strategy_decides_the_estimand(strategy, expected):
    cfg = {"name": "PMMR", "cls": "PMMR", "strategy": strategy}
    assert method_estimand(cfg) == expected


def test_outputs_entry_wins_over_the_method_level_strategy():
    cfg = {"name": "PMMR", "cls": "PMMR", "strategy": "marginal"}
    out = {"name": "PMMR-XW", "strategy": "paired"}
    assert method_estimand(cfg, out) == "cate(x,w)"


def test_shipped_pmmr_outputs_span_all_three_tracks():
    """The names in config_val.yaml, resolved."""
    cfg = {"name": "PMMR", "cls": "PMMR"}
    outputs = [
        {"name": "PMMR-XW", "strategy": "paired"},
        {"name": "PMMR-X-Wind", "strategy": "independent"},
        {"name": "PMMR-X-Wcond", "strategy": "conditional"},
        {"name": "PMMR-ATE", "strategy": "marginal"},
    ]
    assert [method_estimand(cfg, o) for o in outputs] == [
        "cate(x,w)",
        "cate(x)",
        "cate(x)",
        "ate(x)",
    ]


def test_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="unknown strategy"):
        method_estimand({"name": "m", "cls": "PMMR", "strategy": "sideways"})


@pytest.mark.parametrize("cls", ["S_Learner", "T_Learner", "X_Learner"])
def test_learners_follow_their_proxy_flags(cls):
    base = {"cls": cls}
    assert method_estimand({**base, "use_W": False, "use_Z": False}) == "cate(x)"
    assert method_estimand({**base, "use_W": True, "use_Z": False}) == "cate(x,w)"
    assert method_estimand({**base, "use_W": False, "use_Z": True}) == "cate(x,z)"
    assert method_estimand({**base, "use_W": True, "use_Z": True}) == "cate(x,w,z)"


@pytest.mark.parametrize(
    "cls", ["S_Learner", "T_Learner", "X_Learner", "CausalFM", "DoPFN", "CausalPFN"]
)
def test_proxy_flags_default_to_true_as_the_factories_do(cls):
    """build_method reads use_W/use_Z with default True for these classes, so an
    entry that says neither really is fitted on the proxies."""
    assert method_estimand({"cls": cls}) == "cate(x,w,z)"


def test_p2sls_is_ate_only():
    assert method_estimand({"name": "P2SLS", "cls": "P2SLS"}) == "ate(x)"


@pytest.mark.parametrize("cls", ["PLearner", "CausalForest"])
def test_methods_that_discard_proxies_at_predict_are_tau_x(cls):
    # Both accept W/Z at fit and neither uses them at predict time.
    assert method_estimand({"cls": cls, "use_W": True, "use_Z": True}) == "cate(x)"


def test_is_oracle_puts_u_in_the_tag():
    """run._worker concatenates the true U onto X, so the prediction reads it."""
    cfg = {"cls": "S_Learner", "use_W": False, "use_Z": False, "is_oracle": True}
    assert method_estimand(cfg) == "cate(x,u)"
    assert method_estimand({**cfg, "use_W": True}) == "cate(x,w,u)"
    assert method_estimand({"cls": "P2SLS", "is_oracle": True}) == "ate(x,u)"


def test_explicit_estimand_overrides_everything():
    cfg = {"name": "custom", "cls": "S_Learner", "use_W": True, "estimand": "cate(x)"}
    assert method_estimand(cfg) == "cate(x)"
    out = {"name": "o", "strategy": "paired", "estimand": "ate(x)"}
    assert method_estimand({"cls": "PMMR"}, out) == "ate(x)"


def test_explicit_legacy_estimand_still_resolves():
    """`estimand: xw` in a config written before the tags were spelled out."""
    assert method_estimand({"cls": "PLearner", "estimand": "xw"}) == "cate(x,w)"


def test_explicit_estimand_is_validated():
    with pytest.raises(ValueError, match="estimand must be"):
        method_estimand({"cls": "S_Learner", "estimand": "tau_of_everything"})


def test_unknown_class_defaults_to_the_conservative_reading():
    """Scored as tau(x), a secretly-proxy-using method merely looks bad on the
    primary track; the reverse would hand it an unearned advantage."""
    assert method_estimand({"name": "?", "cls": "SomethingNew"}) == "cate(x)"
