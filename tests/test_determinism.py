import hashlib
import torch
import pytest
from proximalfm.Prior import ProximalPrior

def test_seed_determinism():
    """Verify that seed generation is deterministic and independent of Python's hash() randomization."""
    seed = 42
    episode_seed = 123

    prior = ProximalPrior(seed=seed)

    # Manually calculate expected seed using the same logic as in Prior.py
    seed_data = f"{seed}-{episode_seed}".encode("utf-8")
    expected_hash = hashlib.sha256(seed_data).hexdigest()
    expected_combined = int(expected_hash, 16) % (2**32)

    # We can't directly access 'combined' from _make_generator without refactoring,
    # so we compare the output of a generator seeded with expected_combined.
    gen = prior._make_generator(episode_seed=episode_seed)

    gen_manual = torch.Generator(device=gen.device)
    gen_manual.manual_seed(expected_combined)

    # Generate random tensors and compare
    rand_t1 = torch.randn(10, generator=gen)
    rand_t2 = torch.randn(10, generator=gen_manual)

    assert torch.allclose(rand_t1, rand_t2), "Generator output does not match expected deterministic seed output"

def test_seed_none_handling():
    """Verify that seed generation handles None values correctly."""
    prior = ProximalPrior(seed=None)

    # When both are None, it should call gen.seed() which is non-deterministic
    # but we should ensure it doesn't crash.
    gen = prior._make_generator(episode_seed=None)
    assert isinstance(gen, torch.Generator)

    # When one is None, it should still be deterministic
    episode_seed = 456
    gen = prior._make_generator(episode_seed=episode_seed)

    seed_data = f"None-{episode_seed}".encode("utf-8")
    expected_combined = int(hashlib.sha256(seed_data).hexdigest(), 16) % (2**32)

    gen_manual = torch.Generator(device=gen.device)
    gen_manual.manual_seed(expected_combined)

    assert torch.allclose(torch.randn(10, generator=gen), torch.randn(10, generator=gen_manual))
