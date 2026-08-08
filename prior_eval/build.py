"""Entry point for ``python -m prior_eval.build``.

The implementation lives in :mod:`prior_eval.build_prior_frozen_set`; this
module only exists so the CLI reads the way ``POSTERIOR_EVAL_NOTES.md`` spells
it.
"""

from prior_eval.build_prior_frozen_set import main

if __name__ == "__main__":
    raise SystemExit(main())
