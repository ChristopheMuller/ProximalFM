"""Artifacts for the training section (Appendix C + the training figure).

Sibling of :mod:`analysis_prior` (Appendix B, the prior) and of ``prior_eval``/
``synt_eval`` (the evaluation sets): this package owns every figure and table
that describes the *training run* rather than the prior or the results. Nothing
here trains anything; each artifact is derived either from a training config
(``scripts/pretrain_prod.sh``) or from a run's logged history.

See ``analysis/analysis_training/README.md`` for the artifact -> command map.
"""
