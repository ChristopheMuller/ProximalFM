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
uv run python generate_synthetic_sweep.py --rows 8192 --levels 20
uv run python run_ablation.py --context-sizes 128 512 2048 8192 --proximalfm-device mps
uv run python plot_results.py
```

Outputs are saved beside the scripts. Comparisons use linear proximal 2SLS,
OLS(X), OLS(X,W,Z), ProximalFM, proxy-adjusted TabICL (default), optionally
proxy-adjusted CausalPFN, and optionally oracle OLS(X,U), scored against test
CATE after averaging treatment contrasts over context nuisance covariates.
Each figure row contains CATE RMSE curves for one context size; columns are proxy
dimensions and curves are methods. Override the shared prefixes with
`--context-sizes`; override the ProximalFM checkpoint with
`--proximalfm-checkpoint`.
TabICL and CausalPFN are fitted on `(X,W,Z)` as ordinary backdoor covariates.
Their proxy-conditional CATE predictions are evaluated at one seeded random
sample of aligned context `(W,Z)` rows and averaged, using the same proxy pairs
for every query and preserving the joint proxy law. This unconditional empirical
integration is appropriate here because the generator makes `(W,Z)` independent
of `X`. It is a backdoor benchmark, not a proximal estimator: noisy proxies need
not restore ignorability. The defaults use 16 proxy pairs and one TabICL view;
configure them with `--backdoor-proxy-samples`, `--backdoor-seed`, and
`--tabicl-estimators`. Select either implementation with
`--backdoor-methods tabicl` or `--backdoor-methods causalpfn`; pass
`--backdoor-methods tabicl causalpfn` for both, or pass `--backdoor-methods`
with no values to disable both. CausalPFN is opt-in because it is substantially
heavier and its FAISS dependency requires a one-thread OpenMP runtime on macOS;
the runner automatically relaunches itself once with that safe environment.
Successful dataset/method/context fits are persisted in `results/.fit_cache/`.
Later runs reuse every overlapping context size and compute only missing cells;
`method_results.csv` still contains exactly the sizes requested by the latest
run. The cache is invalidated when the dataset, checkpoint, numerical library,
or ProximalFM inference settings change. Pass `--force` to clear it explicitly.
ProximalFM sees the observational context `(X,W,Z,A,Y)` but its held-out proxy
values are masked internally, so it estimates CATE(X), like its training task.
