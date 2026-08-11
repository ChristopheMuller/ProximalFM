"""``lit_eval`` -- literature / external proximal benchmark datasets.

Third leg of the evaluation grid, beside ``prior_eval`` (frozen sets drawn from
the training prior) and ``synt_eval`` (hand-written "realistic" mechanisms).
This package adapts datasets that come from *outside* this project -- the
proximal literature's synthetic and semi-synthetic benchmarks -- into the one
on-disk format everything else already reads:

    <out_root>/<name>/
        val/batch_*.pt + metadata.json     # written by prior._genload.write_batch_file
        frozen_meta.pt                     # only when built with --mc_cate K > 1
        manifest.json, config.yaml

Consequently a build here is consumed exactly like a ``synt_eval`` build:

    benchmark:  data: {source: load, val_dir: <root>/val}
    scoring:    python -m prior_eval.score --dir <root> --ckpt <ckpt>

Adding a dataset is one module under :mod:`lit_eval.datasets` implementing
:class:`lit_eval.base.ProximalDataset` and registering itself; everything else
(replicate layout, Monte Carlo targets, on-disk writing, sidecar, digest) is
shared and lives here. :mod:`lit_eval.datasets.dummy_linear` is the reference
implementation and the thing the tests exercise -- it is a placeholder DGP, not
a literature dataset, and it exists so the plumbing can be built and verified
before any external code is vendored.

Two invariants inherited from the rest of the project, both of which the base
class enforces so an adapter cannot get them wrong:

* **The estimand is** ``tau*(x) = E[Y1 - Y0 | X = x]``, marginalizing over the
  latent, over ``W``, and over the outcome noise -- exactly what
  ``SCMPrior.generate_dataset`` supervises on. Not ``E[. | X, W, Z]``. Scoring a
  posterior against a different conditional makes a calibrated model read as
  over-wide.
* **A distributional score requires replicates.** At ``mc_cate = 1`` there is no
  Monte Carlo CEPO target, only realized potential outcomes, and no sidecar is
  written -- that absence is the gate that makes the trainer refuse the set
  rather than silently score a posterior against sampled ITEs.
"""

from lit_eval.base import (  # noqa: F401
    Dims,
    EpisodeSpec,
    ProximalDataset,
    RawEpisode,
)
from lit_eval.registry import (  # noqa: F401
    available_datasets,
    build_dataset,
    register_dataset,
)

# Importing the subpackage is what populates the registry.
import lit_eval.datasets  # noqa: F401,E402
