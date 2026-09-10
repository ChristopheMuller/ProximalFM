# Synthetic proxy sweep

300 datasets: 50,000 rows each, d ∈ {1,3,5}, and 100 equally spaced joint
population R² levels s ∈ [0.01,0.99]. Each of X, U, W and Z has d coordinates.
Seed: 20260908; train/test split: 40,000/10,000.

Draw independent standard Gaussian X, U, εW, εZ and scalar εY. For fixed
orthogonal matrices QW, QZ, set (column-vector notation)

    q = s / (2-s)
    W = √q QW U + √(1-q) εW
    Z = √q QZ U + √(1-q) εZ

Each proxy block has R²=q for U; jointly, R²=2q/(1+q)=s per U coordinate.
Define S=sum(X)/√d and L=sum(U)/√d, then

    p = clip(sigmoid(0.5 S + 1.25 L), 0.02, 0.98)
    A ~ Bernoulli(p)
    Y0 = 0.5 S + 1.5 L + εY
    CATE(X) = 0.5 + S
    ITE = CATE(X) + 0.5 L
    Y1 = Y0 + ITE
    Y = A Y1 + (1-A) Y0

An outcome bridge is h(W,a,X)=0.5 S+(1.5+0.5a)L̂(W)+a(0.5+S), with
L̂(W)=sum(QWᵀW)/√(qd). Within each dimension, only proxy strength varies;
underlying draws are paired across levels. Shared coordinates also pair dimensions.

Python 3.10+ with NumPy, SciPy and Matplotlib. The included scores reproduce
the figure by running `plot_results.py`. To regenerate the full study:

```sh
python generate_synthetic_sweep.py
python run_ablation.py --oracle
# Include an in-context checkpoint; 2,048 context rows are used from each
# 40,000-row training split and all 10,000 held-out rows are scored in chunks.
python run_ablation.py --proximalfm-checkpoint ../mc_run_7_NoiseAwareTuning/best_model.ckpt \\
  --proximalfm-device mps
python plot_results.py
```

Outputs are saved beside the scripts. Comparisons use linear proximal 2SLS,
OLS(X), OLS(X,W,Z), optionally ProximalFM, and oracle OLS(X,U), scored against
test CATE after averaging treatment contrasts over training nuisance covariates.
ProximalFM sees the observational context `(X,W,Z,A,Y)` but its held-out proxy
values are masked internally, so it estimates CATE(X), like its training task.
