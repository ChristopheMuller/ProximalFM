"""Released-checkpoint graph checks used in the selected additions.

The included CSV summaries already contain completed results. This script
reruns only the requested controlled checks; it does not train a model.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from generators import generate_episode
from metrics import point_metrics, uncertainty_metrics
from released_adapter import ReleasedProximalFM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    model = ReleasedProximalFM(args.checkpoint, device=args.device, threads=2,
                               timing_repeats=1, vendor_path=args.repo)
    args.output.mkdir(parents=True, exist_ok=True)
    configurations = [dict(kind='gaussian', family=family, d_x=5, d_u=5,
                           confounding=1.5, proxy_noise=.6)
                      for family in ['dense', 'chain', 'sparse']]
    contexts = [2048]
    rows = []
    for settings in configurations:
        for n in contexts:
            for seed in range(900000000, 900000030):
                episode = generate_episode(seed=seed, n_context=n, n_query=256,
                                           **settings)
                prediction = model.predict(episode)
                interval_metrics, arrays = uncertainty_metrics(
                    prediction['weights'], prediction['edges'], episode.tau)
                row = {**settings, 'seed': seed, 'n_context': n,
                       **point_metrics(arrays['mean'], episode.tau),
                       **interval_metrics, **prediction['performance']}
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.output / 'task_results.csv', index=False)
    (args.output / 'model.json').write_text(json.dumps(model.provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
