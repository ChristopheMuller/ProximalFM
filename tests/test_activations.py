"""Tests for the prior's activation pool (``proximalfm.prior._activations``).

The pool is the ``choice_values`` of the ``mlp_activations`` sampled
hyperparameter, so its composition *is* the distribution over mechanism
nonlinearities.
"""

import numpy as np
import pytest
import torch

from proximalfm.prior._activations import (
    RandomChoiceActivation,
    RandomChoiceFactory,
    StdRandomScaleFactory,
    get_activations,
)


def _nesting_depth(module):
    """How many RandomChoiceActivation layers wrap the actual activation."""
    depth = 0
    while isinstance(module, RandomChoiceActivation):
        depth += 1
        module = module.act
    return depth


def test_choice_factory_pool_excludes_itself():
    # The factory is constructed from the list it is then appended to. Holding
    # a reference (rather than a copy) put the factory into its own choice
    # pool, so "pick a random activation" could pick "pick a random
    # activation", recursively.
    acts = get_activations(random=True, scale=True, diverse=True)
    factories = [a for a in acts if isinstance(a, RandomChoiceFactory)]

    assert factories, "diverse=True should contribute at least one factory"
    for factory in factories:
        assert not any(
            isinstance(choice, RandomChoiceFactory) for choice in factory.act_classes
        )


def test_choice_factory_copies_its_pool():
    pool = [StdRandomScaleFactory(torch.nn.Tanh)]
    factory = RandomChoiceFactory(pool)
    pool.append(factory)

    assert factory.act_classes is not pool
    assert len(factory.act_classes) == 1


def test_diverse_activations_never_nest():
    # A single level of indirection is the whole point of `diverse`; more than
    # one is the self-reference bug re-appearing.
    acts = get_activations(random=True, scale=True, diverse=True)
    np.random.seed(0)
    torch.manual_seed(0)

    depths = {_nesting_depth(acts[np.random.randint(len(acts))]()) for _ in range(300)}

    assert depths <= {0, 1}


def test_get_activations_does_not_mutate_its_base_list():
    # `activations = simple_activations` followed by `+=` mutated the base list
    # in place; harmless today only because it is rebuilt per call.
    first = get_activations(random=True, scale=True, diverse=True)
    second = get_activations(random=True, scale=True, diverse=True)

    assert len(first) == len(second)


@pytest.mark.parametrize("scale", [True, False])
def test_pool_is_half_direct_half_diverse(scale):
    # Composition the sampler relies on: `diverse` doubles the pool, so a
    # per-layer random choice is drawn with probability 1/2.
    base = get_activations(random=True, scale=scale, diverse=False)
    full = get_activations(random=True, scale=scale, diverse=True)

    assert len(full) == 2 * len(base)
    assert sum(isinstance(a, RandomChoiceFactory) for a in full) == len(base)


def test_every_pool_entry_builds_a_working_module():
    acts = get_activations(random=True, scale=True, diverse=True)
    np.random.seed(1)
    torch.manual_seed(1)
    x = torch.randn(64, 8)

    for factory in acts:
        out = factory()(x)
        assert out.shape == x.shape
