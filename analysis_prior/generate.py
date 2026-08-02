"""Generation and on-disk caching of prior datasets for the appendix audits.

Every dataset is an independent draw from the prior, generated from its own seed
derived from the spec. This buys three things:

1. **Reproducibility under parallelism.** The prior reads the global ``random``,
   ``numpy`` and ``torch`` RNGs. ``joblib`` workers do not inherit a seeded
   state, so ``PriorDataset.get_batch()`` with ``n_jobs > 1`` is *not*
   reproducible even under a fixed global seed. Seeding once per dataset, inside
   the worker, makes the result independent of ``n_jobs`` entirely.
2. **Stable datasets under growth.** Raising ``n_datasets`` re-uses the same
   draws for the existing indices.
3. **Independent SCMs.** ``batch_size_per_gp=1`` plus one ``get_batch`` call per
   dataset means no hyperparameter sharing between datasets.

The cache keeps the latent confounder ``U`` and the Monte Carlo targets
``MU0``/``MU1``, none of which the model ever sees. They are what make the audits
possible.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import torch
from tqdm.auto import tqdm

from proximalfm.prior import PriorDataset

from .config import CACHE_FORMAT_VERSION, PriorSpec

#: Default location for cache files. Overridable from the CLI.
DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"

#: Per-row fields of a dataset record, in the order returned by ``get_batch``.
_MATRIX_FIELDS = ("U", "X", "W", "Z")
_VECTOR_FIELDS = ("A", "A_probs", "Y_obs", "Y0", "Y1", "MU0", "MU1")


@dataclass(frozen=True)
class PriorDatasetRecord:
    """One dataset drawn from the prior, with latents retained.

    All arrays are trimmed to their *active* columns: the generator pads feature
    blocks up to ``max_dim_*`` and drops constant columns, so ``U.shape[1]`` is
    the number of columns that survived filtering, not the requested dimension.
    This is deliberate -- the surviving width is what the completeness heuristic
    has to hold for in the data actually delivered.

    Attributes
    ----------
    index : int
        Position of this dataset within its spec.

    seed : int
        The seed it was generated from. Regenerating with this seed reproduces
        the dataset exactly.

    U : ndarray, shape (T, d_U)
        Latent confounder. Never observed by the model; available here only
        because the prior is synthetic.

        ``d_U`` may be ``0``. The generator rejects datasets with an empty
        ``X``, ``W`` or ``Z`` but does not check ``U``, so a draw whose
        confounder columns are all constant is accepted and is simply
        unconfounded. Analyses must handle the empty case; the rate at which it
        occurs is an audit result in its own right.

    X, W, Z : ndarray, shape (T, d_*)
        Baseline covariates, outcome-inducing proxy, treatment-inducing proxy.

    A : ndarray, shape (T,)
        Binary treatment.

    A_probs : ndarray, shape (T,)
        True propensity P(A=1 | X, U), post clamping. Ground truth for the
        overlap audit.

    Y_obs : ndarray, shape (T,)
        Observed outcome, ``Y1`` where ``A == 1`` else ``Y0``.

    Y0, Y1 : ndarray, shape (T,)
        Both potential outcomes for the realised replicate. ``Y1 - Y0`` is the
        ITE.

    MU0, MU1 : ndarray, shape (T,)
        Monte Carlo estimates of E[Y(a) | X], averaged over ``mcmc_cate``
        replicates. ``MU1 - MU0`` is the CATE target.

    sampled_hp : dict, optional
        The scalar hyperparameters this dataset was actually generated with
        (sampled shares, dims, seq_len, ...). ``None`` in caches written before
        diagnostics existed; regenerating with ``--force`` reproduces the same
        data (generation is unchanged) and fills these in.

    mechanisms : dict, optional
        Input-composition diagnostics for ``f_A`` and ``f_Y``: per parent
        block, the input widths, first-layer injected variances and permutation
        total effects (see ``proximalfm.prior._diagnostics``). Same caveat.

    gamma_hetero : float, optional
        The realised heterogeneity mix of the outcome remix. Same caveat.

    a_logits : ndarray, shape (T,), optional
        Raw ``f_A`` logits of the observed replicate, *before* the
        standardise/shift-scale/sigmoid/clamp pipeline of ``sample_A`` -- the
        input to the treatment-assignment walkthrough (``app:prior_treatment``).
        Row-aligned with ``A``/``A_probs``. Same "regenerate with --force"
        caveat as the diagnostics above.

    a_logit_mean, a_logit_std : float, optional
        Mean and (unbiased) std of the raw logits, i.e. the constants
        ``sample_A`` standardises by. ``f_A`` now runs directly on the ``T``
        observed rows (Z/A are leaves w.r.t. MU0/MU1, so the generator no
        longer replicates them K-fold), so these coincide with ``a_logits``'s
        own moments for any current record. Stored explicitly for older
        records generated before that change, where ``sample_A`` standardised
        over the full ``T*K`` population and ``a_logits`` held only the
        observed-row slice of it -- reproducing the standardised logit from
        the slice alone would not have been exact for those.
    """

    index: int
    seed: int
    U: np.ndarray
    X: np.ndarray
    W: np.ndarray
    Z: np.ndarray
    A: np.ndarray
    A_probs: np.ndarray
    Y_obs: np.ndarray
    Y0: np.ndarray
    Y1: np.ndarray
    MU0: np.ndarray
    MU1: np.ndarray
    sampled_hp: Optional[Dict[str, Any]] = None
    mechanisms: Optional[Dict[str, Any]] = None
    gamma_hetero: Optional[float] = None
    a_logits: Optional[np.ndarray] = None
    a_logit_mean: Optional[float] = None
    a_logit_std: Optional[float] = None

    @property
    def n_rows(self) -> int:
        return self.X.shape[0]

    @property
    def dims(self) -> Dict[str, int]:
        """Active (post-filtering) width of each feature block."""
        return {f"d_{k}": getattr(self, k).shape[1] for k in _MATRIX_FIELDS}

    @property
    def ite(self) -> np.ndarray:
        """Individual treatment effect of the realised replicate."""
        return self.Y1 - self.Y0

    @property
    def cate(self) -> np.ndarray:
        """Monte Carlo CATE target, E[Y(1) - Y(0) | X]."""
        return self.MU1 - self.MU0

    @property
    def true_ate(self) -> float:
        """Ground-truth ATE, the mean of the Monte Carlo CATE."""
        return float(self.cate.mean())

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"index": self.index, "seed": self.seed}
        for name in _MATRIX_FIELDS + _VECTOR_FIELDS:
            payload[name] = getattr(self, name)
        for name in (
            "sampled_hp",
            "mechanisms",
            "gamma_hetero",
            "a_logits",
            "a_logit_mean",
            "a_logit_std",
        ):
            payload[name] = getattr(self, name)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PriorDatasetRecord":
        return cls(**data)


@dataclass(frozen=True)
class PriorCache:
    """A spec together with the datasets drawn from it."""

    spec: PriorSpec
    records: List[PriorDatasetRecord]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, idx: int) -> PriorDatasetRecord:
        return self.records[idx]


def _generate_one(spec: PriorSpec, index: int) -> PriorDatasetRecord:
    """Generate dataset ``index`` of ``spec``.

    Seeds every RNG the prior touches before generating. ``PriorDataset`` is
    constructed here rather than passed in because it is cheap to build (it holds
    configuration, not state) and because a fresh instance cannot leak sampler
    state across datasets.
    """
    seed = spec.dataset_seed(index)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    fixed_hp, sampled_hp = spec.resolved_hp()

    # Mechanism diagnostics ride along with every generated dataset. The flag
    # is set here, not in the spec's overrides, so it stays out of the cache
    # key and the per-dataset seeds: collection runs under fork_rng after the
    # accepting draw, so the generated data is bit-identical with or without it.
    fixed_hp = {**fixed_hp, "collect_mechanism_diagnostics": True}

    dataset = PriorDataset(
        batch_size=1,
        # One independent SCM per dataset: no hyperparameter sharing.
        batch_size_per_gp=1,
        batch_size_per_subgp=1,
        min_dim_U=spec.min_dim_U,
        max_dim_U=spec.max_dim_U,
        min_dim_W=spec.min_dim_W,
        max_dim_W=spec.max_dim_W,
        min_dim_Z=spec.min_dim_Z,
        max_dim_Z=spec.max_dim_Z,
        min_dim_X=spec.min_dim_X,
        max_dim_X=spec.max_dim_X,
        # min_seq_len=None makes the generator use max_seq_len exactly, so every
        # dataset has the same T and the cache stays a dense array.
        min_seq_len=None,
        max_seq_len=spec.seq_len,
        log_seq_len=False,
        seq_len_per_gp=False,
        replay_small=False,
        prior_type=spec.prior_type,
        mcmc_cate=spec.mcmc_cate,
        scm_fixed_hp=fixed_hp,
        scm_sampled_hp=sampled_hp,
        # Nested parallelism would reintroduce the RNG problem this function
        # exists to solve; parallelism happens across datasets instead.
        n_jobs=1,
        device="cpu",
    )

    (
        U,
        X,
        W,
        Z,
        A,
        A_probs,
        Y_obs,
        Y0,
        Y1,
        MU0,
        MU1,
        d_U,
        d_X,
        d_W,
        d_Z,
        _seq_lens,
        _train_sizes,
    ) = dataset.get_batch()

    widths = {"U": int(d_U[0]), "X": int(d_X[0]), "W": int(d_W[0]), "Z": int(d_Z[0])}
    blocks = {"U": U, "X": X, "W": W, "Z": Z}

    payload: Dict[str, Any] = {"index": index, "seed": seed}
    for name in _MATRIX_FIELDS:
        # Drop the batch dim, then trim padding: active columns are the leading
        # ``d`` of each block after filter_constant_features_and_pad.
        payload[name] = (
            blocks[name][0, :, : widths[name]].cpu().numpy().astype(np.float32)
        )
    for name, tensor in zip(_VECTOR_FIELDS, (A, A_probs, Y_obs, Y0, Y1, MU0, MU1)):
        payload[name] = tensor[0].cpu().numpy().astype(np.float32)

    # None for prior types without diagnostics support (e.g. tree_scm).
    diagnostics = (dataset.last_diagnostics or [None])[0]
    if diagnostics is not None:
        payload["sampled_hp"] = diagnostics["params"]
        payload["mechanisms"] = diagnostics["mechanisms"]
        payload["gamma_hetero"] = diagnostics["gamma_hetero"]
        payload["a_logits"] = diagnostics["A_logits"].numpy().astype(np.float32)
        payload["a_logit_mean"] = diagnostics["A_logit_mean"]
        payload["a_logit_std"] = diagnostics["A_logit_std"]

    return PriorDatasetRecord.from_dict(payload)


def generate(
    spec: PriorSpec,
    n_jobs: int = 1,
    progress: bool = True,
    progress_bar: bool = False,
    indices: Optional[Iterable[int]] = None,
) -> PriorCache:
    """Draw datasets for ``spec``.

    The result is identical for any ``n_jobs`` because each dataset is seeded
    from the spec rather than from inherited RNG state.

    Parameters
    ----------
    spec : PriorSpec
        What to generate.

    n_jobs : int, default=1
        Processes to use. ``-1`` uses all cores.

    progress : bool, default=True
        Print a line before starting. Generation of a large spec is slow enough
        that silence is confusing.

    progress_bar : bool, default=False
        Show a live ``tqdm`` progress bar over datasets as they complete, on
        top of the one-line ``progress`` announcement. Off by default so
        non-interactive runs (CI, logs piped to a file) stay clean.

    indices : iterable of int, optional
        Which dataset indices to draw. Defaults to ``range(spec.n_datasets)``.
        Used by :func:`load_or_generate` to fill only the gap when extending an
        existing cache.
    """
    indices = list(range(spec.n_datasets) if indices is None else indices)
    if progress and indices:
        print(
            f"Generating {len(indices)} datasets for cache {spec.cache_key} "
            f"(T={spec.seq_len}, K={spec.mcmc_cate}, n_jobs={n_jobs})"
        )

    desc = f"cache {spec.cache_key}"
    if n_jobs == 1:
        records = [
            _generate_one(spec, i)
            for i in tqdm(indices, desc=desc, disable=not progress_bar)
        ]
    else:
        # return_as="generator" preserves submission order (like the default
        # "list") but yields each result as it completes, so the bar advances
        # per-dataset instead of jumping to 100% once every worker returns.
        results = joblib.Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
            joblib.delayed(_generate_one)(spec, i) for i in indices
        )
        records = list(
            tqdm(results, total=len(indices), desc=desc, disable=not progress_bar)
        )

    return PriorCache(spec=spec, records=list(records))


def cache_path(spec: PriorSpec, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Path of the cache file for ``spec``.

    Content-addressed (see :attr:`PriorSpec.cache_key`): specs differing only by
    ``name`` or ``n_datasets`` map to the same file.
    """
    return Path(cache_dir) / f"{spec.cache_key}.pt"


def save(cache: PriorCache, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Write ``cache`` to disk, atomically.

    Stores only the *generative* spec fields: ``name`` and ``n_datasets`` are
    excluded, because the file is shared by every spec with the same generative
    configuration. The stored record count is what ``n_datasets`` would mean for
    this file, and it is simply ``len(cache)``.

    Also writes a ``.json`` sidecar, so a cache directory can be inspected
    without loading tensors.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache.spec, cache_dir)

    meta = {
        "version": CACHE_FORMAT_VERSION,
        "generative_spec": cache.spec.generative_dict(),
        "n_stored": len(cache),
    }
    payload = {**meta, "records": [r.to_dict() for r in cache.records]}

    # Write-then-rename: a cache file either exists complete or not at all, so an
    # interrupted run cannot leave a truncated file that later loads as valid.
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)

    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    return path


def load(spec: PriorSpec, cache_dir: Path = DEFAULT_CACHE_DIR) -> PriorCache:
    """Load the cached datasets for ``spec``'s generative configuration.

    Returns whatever is stored, which may hold *more or fewer* datasets than
    ``spec.n_datasets``; use :func:`load_or_generate` to get exactly
    ``n_datasets``. The returned cache carries ``spec`` itself, so the caller's
    name is preserved.

    Raises
    ------
    FileNotFoundError
        If no cache exists for this generative configuration.

    ValueError
        If the format version differs, or the stored generative spec does not
        match the request -- a hash collision. Both are worth failing loudly on
        rather than silently analysing the wrong data.
    """
    path = cache_path(spec, cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"No cache for {spec.cache_key} at {path}")

    payload = torch.load(path, weights_only=False)

    if payload.get("version") != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Cache {path} has format version {payload.get('version')!r}, "
            f"expected {CACHE_FORMAT_VERSION}. Regenerate with --force."
        )

    stored = payload.get("generative_spec")
    if stored != spec.generative_dict():
        raise ValueError(
            f"Cache {path} holds generative spec {stored!r}, which does not "
            f"match the requested {spec.generative_dict()!r}. This is a hash "
            "collision; regenerate with --force."
        )

    records = [PriorDatasetRecord.from_dict(r) for r in payload["records"]]
    return PriorCache(spec=spec, records=records)


def load_or_generate(
    spec: PriorSpec,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
    n_jobs: int = 1,
    progress: bool = True,
    progress_bar: bool = False,
) -> PriorCache:
    """Return exactly ``spec.n_datasets`` datasets, using the cache where possible.

    Because dataset ``i``'s content depends only on the generative configuration
    and ``i`` (never on how many were requested), a cache can be reused three
    ways:

    - **hit**: it holds at least ``n_datasets`` -- return the first ``n``.
    - **extend**: it holds fewer -- generate only the missing indices, append,
      and re-save. The existing datasets are untouched.
    - **miss**: no cache -- generate all of them.

    Parameters
    ----------
    force : bool, default=False
        Ignore any existing cache and regenerate from scratch.
    """
    n = spec.n_datasets
    existing: List[PriorDatasetRecord] = []

    if not force:
        try:
            existing = list(load(spec, cache_dir).records)
        except FileNotFoundError:
            pass

    if len(existing) >= n:
        if progress:
            print(
                f"Loaded {n} of {len(existing)} cached datasets from "
                f"{cache_path(spec, cache_dir)}"
            )
        return PriorCache(spec=spec, records=existing[:n])

    # Generate only what is missing; existing indices are reused as-is.
    if existing and progress:
        print(f"Extending cache {spec.cache_key} from {len(existing)} to {n} datasets")
    fresh = generate(
        spec,
        n_jobs=n_jobs,
        progress=progress,
        progress_bar=progress_bar,
        indices=range(len(existing), n),
    )

    cache = PriorCache(spec=spec, records=existing + list(fresh.records))
    path = save(cache, cache_dir)
    if progress:
        print(f"Cached {len(cache)} datasets to {path}")
    return cache


def list_caches(cache_dir: Path = DEFAULT_CACHE_DIR) -> List[Dict[str, Any]]:
    """Summarise the caches present in ``cache_dir``.

    Reads the JSON sidecars only, so this stays fast on a directory of large
    caches.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []

    entries = []
    for sidecar in sorted(cache_dir.glob("*.json")):
        if not sidecar.with_suffix(".pt").exists():
            continue
        with open(sidecar) as f:
            meta = json.load(f)
        entries.append(
            {
                "path": str(sidecar.with_suffix(".pt")),
                "version": meta.get("version"),
                "n_stored": meta.get("n_stored"),
                # Generative fields only: caches are content-addressed and shared
                # across every spec name that resolves to them.
                "generative_spec": meta.get("generative_spec", {}),
            }
        )
    return entries
