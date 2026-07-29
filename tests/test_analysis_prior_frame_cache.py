"""Tests for the assembled-frame cache.

The layer exists purely for speed, so the tests are about it never changing an
answer: same rows out, only the missing indices ever built, and a stale entry
never served after the builder's source changes.
"""

import pandas as pd
import pytest

from analysis_prior.frame_cache import _source_digest, cached_frame


def _build_calls(records=("a",)):
    """A builder that records which indices it was asked for."""
    calls = []

    def build(indices):
        calls.append(list(indices))
        return pd.DataFrame(
            {"index": list(indices), "value": [10 * i for i in indices]}
        )

    return build, calls


def test_builds_once_then_serves_from_disk(tmp_path):
    build, calls = _build_calls()
    first = cached_frame("t", [0, 1, 2], build, cache_dir=tmp_path)
    second = cached_frame("t", [0, 1, 2], build, cache_dir=tmp_path)

    assert calls == [[0, 1, 2]]  # the second call built nothing
    pd.testing.assert_frame_equal(first, second)
    assert list(first["value"]) == [0, 10, 20]


def test_subset_is_a_prefix_read_not_a_rebuild(tmp_path):
    build, calls = _build_calls()
    cached_frame("t", [0, 1, 2, 3], build, cache_dir=tmp_path)
    subset = cached_frame("t", [0, 1], build, cache_dir=tmp_path)

    assert calls == [[0, 1, 2, 3]]
    assert list(subset["index"]) == [0, 1]


def test_growing_builds_only_the_missing_indices(tmp_path):
    build, calls = _build_calls()
    cached_frame("t", [0, 1], build, cache_dir=tmp_path)
    grown = cached_frame("t", [0, 1, 2, 3], build, cache_dir=tmp_path)

    assert calls == [[0, 1], [2, 3]]
    assert list(grown["index"]) == [0, 1, 2, 3]
    # The appended rows carry the builder's values, not placeholders.
    assert list(grown["value"]) == [0, 10, 20, 30]


def test_rows_are_returned_sorted_regardless_of_build_order(tmp_path):
    build, _ = _build_calls()
    cached_frame("t", [2, 3], build, cache_dir=tmp_path)
    out = cached_frame("t", [0, 1, 2, 3], build, cache_dir=tmp_path)
    assert list(out["index"]) == [0, 1, 2, 3]


def test_a_different_tag_does_not_collide(tmp_path):
    build_a, calls_a = _build_calls()
    build_b, calls_b = _build_calls()
    cached_frame("model=rf", [0, 1], build_a, cache_dir=tmp_path)
    cached_frame("model=linear", [0, 1], build_b, cache_dir=tmp_path)
    assert calls_a == [[0, 1]] and calls_b == [[0, 1]]


def test_editing_the_builder_source_invalidates(tmp_path):
    """The whole point of ``depends_on``: a changed measurement must not be
    served from a frame computed under the old one."""

    def measure_v1(x):
        return x

    def measure_v2(x):
        return x + 1  # same signature, different result

    build, calls = _build_calls()
    cached_frame("t", [0, 1], build, depends_on=(measure_v1,), cache_dir=tmp_path)
    cached_frame("t", [0, 1], build, depends_on=(measure_v1,), cache_dir=tmp_path)
    assert calls == [[0, 1]]  # still cached under the same source

    cached_frame("t", [0, 1], build, depends_on=(measure_v2,), cache_dir=tmp_path)
    assert calls == [[0, 1], [0, 1]]  # rebuilt under the new source


def test_source_digest_is_stable_and_discriminating():
    def a(x):
        return x

    def b(x):
        return x + 1

    assert _source_digest([a]) == _source_digest([a])
    assert _source_digest([a]) != _source_digest([b])
    assert _source_digest([a, b]) != _source_digest([b, a])


def test_source_digest_tolerates_functions_without_source():
    # A C builtin has no retrievable source; the digest must degrade rather
    # than raise, or an audit could not use it at all.
    assert _source_digest([len])


def test_a_corrupt_entry_is_a_miss_not_a_crash(tmp_path):
    build, calls = _build_calls()
    cached_frame("t", [0, 1], build, cache_dir=tmp_path)
    for path in tmp_path.glob("*.pkl"):
        path.write_bytes(b"not a pickle")

    out = cached_frame("t", [0, 1], build, cache_dir=tmp_path)
    assert calls == [[0, 1], [0, 1]]
    assert list(out["index"]) == [0, 1]


@pytest.mark.parametrize("wanted", ([], [5]))
def test_handles_empty_and_single_requests(tmp_path, wanted):
    build, _ = _build_calls()
    out = cached_frame("t", wanted, build, cache_dir=tmp_path)
    assert list(out["index"]) == wanted
