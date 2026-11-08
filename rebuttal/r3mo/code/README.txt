Code and results for the five selected r3mo responses

results/ contains the completed graph, oracle, calibration and benchmark summaries.
run_graph_checks.py reruns the released-checkpoint graph comparison.
metrics.py computes nPEHE, centered shape R^2 and the level-error decomposition.
make_figures.py regenerates the oracle-error figure from the supplied CSV.

Use the existing ProximalFM environment/repository at commit
ebf355a342e281be92e06ac04901bb18e9d23e8f:
https://github.com/ChristopheMuller/ProximalFM
Released checkpoint SHA256:
2dc76f3dbe001af9816ce90e642108cd9d41a0ca59060b15395e382a7582b5df

From this folder:
python run_graph_checks.py --repo /path/to/ProximalFM \
  --checkpoint /path/to/proximalfm.ckpt --output /path/to/new_graph_results
python make_figures.py

The selected Nano CSV contains the completed exact-oracle and chain-role
results used in the text. Main and extended metric CSVs supply both scores
for the five replacement benchmark tables.

run_dimension_pca.py runs the matched two-source k=5/10/20/30 study.
Example: python run_dimension_pca.py --repo /path/to/ProximalFM \
  --checkpoint /path/to/proximalfm.ckpt --data-root /path/to/source_csvs \
  --output /path/to/new_pca_results --device cuda:0
results/dimension_pca_v1/ includes all168 raw/PCA comparisons, PCA variance
ratios, full-rank paired comparisons and the saved-prediction audit receipt.
PCA fitting uses context rows only, separately for X/Z/W; all variants are
scored against the same original CATE within each task. Full prediction and
PCA-fit arrays are retained in the internal reviewer_r3mo/results directory.
