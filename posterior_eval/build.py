"""Entry point for ``python -m posterior_eval.build``.

The implementation lives in :mod:`posterior_eval.build_prior_frozen_set`; this
module only exists so the CLI reads the way ``POSTERIOR_EVAL_NOTES.md`` spells
it.
"""

from posterior_eval.build_prior_frozen_set import main

if __name__ == "__main__":
    raise SystemExit(main())
