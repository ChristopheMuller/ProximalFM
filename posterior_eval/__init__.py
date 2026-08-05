"""Posterior-quality evaluation for ProximalFM.

Point-estimate evaluation lives in :mod:`proximalfm.train._eval`. This package
holds the *distributional* side described in ``POSTERIOR_EVAL_NOTES.md``,
starting with M1: a frozen, deterministic, high-``K`` set of prior episodes with
a low-noise ``tau_star``, the per-row target-noise variance ``s2``, a frozen
comparison scale, and the per-episode prior hyperparameters used for
stratification.
"""
