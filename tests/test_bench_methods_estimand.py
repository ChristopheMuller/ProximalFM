"""Method labels must describe the conditional each estimator answers."""

from benchmarks.methods import method_estimand


def test_estimand_routing_distinguishes_cate_proxy_and_ate_methods():
    assert (
        method_estimand({"cls": "S_Learner", "use_W": False, "use_Z": False})
        == "cate(x)"
    )
    assert (
        method_estimand({"cls": "S_Learner", "use_W": True, "use_Z": False})
        == "cate(x,w)"
    )
    assert method_estimand({"cls": "P2SLS"}) == "ate(x)"
