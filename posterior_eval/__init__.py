"""Posterior-quality evaluation for ProximalFM.

Point-estimate evaluation lives in :mod:`proximalfm.train._eval`. This package
holds the *distributional* side: a frozen, deterministic, high-``K`` set of prior
episodes with a low-noise ``tau_star``, the per-row target-noise variance ``s2``,
a frozen comparison scale, and the per-episode prior hyperparameters the
stratified cuts are built from.

The modules, in the order you use them:

``build`` / ``build_prior_frozen_set``
    Generate a frozen set (ROADMAP A1). ``build_production_set.sh`` wraps it.
``verify``
    Integrity and sanity check, no checkpoint needed. Run it after copying a set
    between machines -- the sidecar-vs-disk check is the only thing that catches
    a short copy.
``report``
    The ``s/sigma_post`` read-off: is ``K`` large enough for *this* checkpoint.
``score``
    Pooled ``prior/point/*`` + ``prior/dist/*`` for a checkpoint.
``strata`` / ``score_strata``
    The stratified cuts (ROADMAP A2): a ``strata.pt`` membership manifest, and
    the per-cut scoring pass. Offline only, by design -- see
    ``proximalfm/train/metrics.md`` §5.
"""
