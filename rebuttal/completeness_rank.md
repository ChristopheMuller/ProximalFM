# Pilot rank test for latent completeness

Run this command from the repository root.

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m rebuttal.completeness_rank
```

The [script](completeness_rank.py) writes [per-test results](results/completeness_rank.csv), including the count tables with Z bins as rows and U bins as columns. Row-normalizing a count table gives the estimated conditional probability matrix. The CSV also records the seed, arm, pilot and evaluation sizes, smallest singular value, error bound, lower confidence bound, and rejection decision.

## Design

- **Prior draws.** Use ProximalFM's existing `general_ite` generator to sample twelve MLP SCMs and draw 200,000 observational rows per SCM. Choose the no-covariate case \(d_X=0\) and set \(d_U=d_Z=d_W=1\). The prior supplies the mechanisms for \(U,Z,W,A,Y\), including its usual categorical conversion. For each SCM, take \((U,Z)\) from the rows with observed \(A=0\) and observed \(A=1\), and analyze those arms separately.
- **Binning.** Use the first 20% of rows to fit arm-specific quantile cutoffs for each variable, then tabulate counts on the remaining rows. Test square tables with 2, 4, or 8 bins per variable. Flag an empty arm or bin separately from a test decision.
- **Test.** The null is that the population matrix \(P\) has deficient column rank. At a nominal 5% level per table, use Bonferroni-adjusted Clopper–Pearson intervals for the cell probabilities conditional on their row totals. Take the Euclidean norm of the maximum per-cell deviations allowed by those intervals. This bounds the Frobenius norm of the matrix error, and hence the change in the smallest singular value. Reject if the estimated smallest singular value exceeds that bound.
- **Controls.** Use one million observational rows per control, with assignment depending on \(U\). The two Gaussian measurements \(Z=U+0.2\epsilon\) and \(Z=0.2U+\epsilon\) have complete continuous operators, since Gaussian convolution is injective after reweighting by the arm propensity. The independent control has rank one. The dependent mixture control draws \(U\) and \(Z\) independently given a binary component. The conditional matrix in each arm factors through that component and has rank at most two at every resolution.

## Pilot results

Each cell gives the number of rejections out of the number of tested treatment-arm tables. The MLP evaluation samples ranged from 19,943 to 140,057 rows per arm. Quantile binning left empty bins in four MLP tables at four bins and five at eight bins. Those tables are recorded separately in the CSV.

| Simulator | 2 bins | 4 bins | 8 bins |
|---|---:|---:|---:|
| Scalar MLP prior | 21/24 | 7/20 | 0/19 |
| Strong Gaussian | 2/2 | 2/2 | 2/2 |
| Weak Gaussian | 2/2 | 0/2 | 0/2 |
| Independent | 0/2 | 0/2 | 0/2 |
| Rank-two mixture | 2/2 | 0/2 | 0/2 |

Across the tested MLP tables, the median smallest singular value was 0.1026 at two bins, 0.0114 at four, and 0.00385 at eight. The corresponding median error bounds were 0.0110, 0.0335, and 0.0823. At finer resolutions, the weak Gaussian control and the rank-two mixture both failed to reject despite having different completeness properties.

## Rejections by sample size

Run the [sweep script](completeness_rank_sweep.py) with the number of distinct SCMs, the number of bins per variable, and the evaluation sample sizes per treatment arm.

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m rebuttal.completeness_rank_sweep \
  --n-scm 6 --k 4 --n-samples 100 1000 10000 100000
```

Within a run, the same SCMs, treatment arms, and pilot cutoffs are used at every size. Each larger sample adds rows to the smaller one. The [plot](results/completeness_rank_sweep.png) shows the number of rejections divided by twice the number of SCMs. Tables with too few rows or empty bins stay in that denominator and are reported separately. By default the generated pool per SCM is 16 times the largest requested arm size, with a minimum of 10,000 rows. Use `--rows-per-scm` to set it explicitly. Keep it fixed when comparing sweeps that use different maximum sample sizes, since it also affects the SCM seeds.

## Interpretation

The finite-sample error guarantee applies to the controls with independent simulator rows and cutoffs fit on a separate sample. For the MLP draws, the existing generator fits feature transformations and treatment-logit standardization across the entire generated table before the audit split. The MLP decisions use the same threshold as a nominal Monte Carlo diagnostic. A formally calibrated version for that generator would freeze these transformations on a separate simulator sample before drawing the evaluation rows. The 5% level is per table, with no adjustment across SCMs, arms, or resolutions.

Rejecting the rank null supports injectivity on the chosen bin-constant functions of \(U\) in the tested stratum. Failure to reject is inconclusive. The experiment addresses the \(U\mid Z,A,X\) completeness condition at finite resolution. The manuscript’s second completeness condition and bridge existence require separate checks.
