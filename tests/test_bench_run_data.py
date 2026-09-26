"""One integration check for runner data preparation."""

import numpy as np

from benchmarks import run
from test_bench_data import build_from_pool, make_pool


def test_build_batches_keeps_query_rows_fixed_when_context_is_shortened():
    data = build_from_pool(make_pool(train_size=8))
    full, short = run.build_batches(data), run.build_batches(data, n_ctx=3)
    for all_rows, prefix in zip(full, short):
        for key in ("X", "W", "Z"):
            np.testing.assert_array_equal(
                prefix[f"{key}_ctx"], all_rows[f"{key}_ctx"][:3]
            )
            np.testing.assert_array_equal(prefix[f"{key}_q"], all_rows[f"{key}_q"])
