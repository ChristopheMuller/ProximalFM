"""Regenerate the selected manuscript figure from the supplied result summary."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'tex' / 'images'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 240})
data = pd.read_csv(ROOT / 'results/selected_nano_summary.csv')
rows = data[(data.model_family == 'finite_raw_full_k16') &
            (data.metric == 'mse_amortization') &
            data.n_context.isin([64, 128, 256])].sort_values('n_context')
assert len(rows) == 3
fig, ax = plt.subplots(figsize=(5.6, 1.75), constrained_layout=True)
values = rows['mean'].to_numpy() * 1000
ax.plot([0, 1, 2], values, color='#225f93', marker='o', linewidth=2)
for x, value in enumerate(values):
    ax.annotate(f'{value:.3f}', (x, value), xytext=(0, 7),
                textcoords='offset points', ha='center', fontsize=9)
ax.set(xticks=[0, 1, 2], xticklabels=['64', '128', '256'],
       xlabel='Context size N', ylabel=r'MSE ($10^{-3}$)',
       ylim=(0, 3.1), xlim=(-.15, 2.15))
ax.grid(axis='y', alpha=.18)
fig.savefig(OUT / 'oracle_error.png', bbox_inches='tight')
plt.close(fig)
print(OUT / 'oracle_error.png')
