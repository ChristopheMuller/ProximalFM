"""Entry point for ``python -m eval_datasets.prior_eval.build``.

The implementation lives in :mod:`eval_datasets.prior_eval.build_prior_frozen_set`; this
module only exists so the CLI reads the way ``POSTERIOR_EVAL_NOTES.md`` spells
it.
"""

from eval_datasets.prior_eval.build_prior_frozen_set import main

if __name__ == "__main__":
    raise SystemExit(main())
