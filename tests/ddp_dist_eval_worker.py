"""One rank of the sharded ``score_frozen_set`` pass (EVAL_PLAN step 6d).

Not a test module -- ``tests/test_posterior_eval.py`` launches ``world_size``
copies of this script as separate processes, one per rank, with the usual
``RANK``/``WORLD_SIZE``/``MASTER_ADDR``/``MASTER_PORT`` environment.

Separate processes rather than ``mp.spawn`` on purpose: the thing under test is
that *every* rank reaches the collectives, so the launcher needs to be able to
time out and kill a rank that hangs. Nothing here is pickled across the
boundary; each rank writes its own metric dict to JSON and the launcher
compares them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_posterior_eval import _FakeCateModel  # noqa: E402

from proximalfm.train._dist_eval import score_frozen_set  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        metrics = score_frozen_set(
            _FakeCateModel(),
            Path(args.dir),
            device="cpu",
            max_episodes=args.max_episodes,
            rank=rank,
            world_size=world_size,
            sync_ddp=True,
        )
        # A barrier after the pass: if any rank returned early from the
        # reductions instead of entering them, this is where it shows up as a
        # timeout rather than as a plausible-looking number.
        dist.barrier()
    finally:
        dist.destroy_process_group()

    Path(args.out).write_text(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    torch.manual_seed(0)
    raise SystemExit(main())
