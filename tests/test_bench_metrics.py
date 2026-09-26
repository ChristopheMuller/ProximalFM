"""Essential benchmark scoring semantics."""

import numpy as np
import pytest

from benchmarks.metrics import TASKS, score_episode


def test_cate_scores_have_the_oracle_and_constant_anchors():
    target = np.array([1.0, -2.0, 0.5, 3.0])
    oracle = score_episode(target, target, TASKS["cate"])
    constant = score_episode(np.full_like(target, target.mean()), target, TASKS["cate"])
    assert oracle["npehe"] == pytest.approx(0)
    assert constant["npehe"] == pytest.approx(1)


def test_noise_share_is_reported_but_never_changes_the_target_score():
    target = np.array([1.0, -2.0, 0.5, 3.0])
    pred = target + np.array([0.3, -0.1, 0.2, 0.4])
    plain = score_episode(pred, target, TASKS["cate"])
    noisy = score_episode(pred, target, TASKS["cate"], s2=np.full(len(target), 0.1))
    assert noisy["npehe"] == plain["npehe"]
    assert noisy["s2_over_var"] == pytest.approx(0.1 / target.var())


def test_task_admission_prevents_proxy_conditioned_methods_from_headline_cate_ranking():
    assert TASKS["cate"].admits("cate(x)")
    assert not TASKS["cate"].admits("cate(x,w)")
    assert TASKS["ite"].admits("cate(x,w)")
    assert TASKS["ate"].admits("ate(x)")
