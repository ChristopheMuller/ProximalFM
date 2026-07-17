"""Base class and registry for prior audits.

Each audit is an :class:`Analysis`. The contract deliberately separates two
concerns:

- :meth:`Analysis.compute` -- pure, deterministic, returns a results object. No
  plotting, no file I/O. This is what tests exercise.
- :meth:`Analysis.render` -- turns a results object into figures and tables on
  disk. Not usually unit-tested beyond "it produces the promised files".

:meth:`Analysis.run` chains the two and writes a ``summary.json`` plus a copy of
the generating spec, so every output directory is self-describing and traceable
back to the exact prior configuration that produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Type

from ..generate import PriorCache


@dataclass
class AnalysisArtifacts:
    """What an analysis wrote, and the scalar findings worth surfacing.

    Attributes
    ----------
    summary : dict
        JSON-serialisable scalar results (the numbers that go in the text, e.g.
        "12% of datasets violate completeness"). Written to ``summary.json``.

    figures, tables : list of Path
        Files written, for logging and for tests to assert on.
    """

    summary: Dict[str, Any] = field(default_factory=dict)
    figures: List[Path] = field(default_factory=list)
    tables: List[Path] = field(default_factory=list)


class Analysis:
    """Base class for a single prior audit.

    Subclasses set :attr:`name` and :attr:`description` and implement
    :meth:`compute` and :meth:`render`.
    """

    #: Short identifier, used on the CLI and as the output subdirectory name.
    name: str = ""
    #: One-line human description, shown by ``--list-analyses``.
    description: str = ""

    def compute(self, cache: PriorCache) -> Any:
        """Return a results object from ``cache``. Pure and deterministic."""
        raise NotImplementedError

    def render(self, results: Any, outdir: Path) -> AnalysisArtifacts:
        """Write figures/tables for ``results`` into ``outdir``."""
        raise NotImplementedError

    def run(self, cache: PriorCache, outdir: Path) -> AnalysisArtifacts:
        """Compute, render, and write provenance files under ``outdir``.

        The output directory is created if needed. Two provenance files are
        always written: ``spec.json`` (the configuration the data came from) and
        ``summary.json`` (the scalar findings).
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        results = self.compute(cache)
        artifacts = self.render(results, outdir)

        with open(outdir / "spec.json", "w") as f:
            json.dump(cache.spec.to_dict(), f, indent=2)
        with open(outdir / "summary.json", "w") as f:
            json.dump(artifacts.summary, f, indent=2)

        return artifacts


class ComparisonAnalysis(Analysis):
    """An audit that contrasts several prior variants in one set of figures.

    Where a plain :class:`Analysis` sees one :class:`PriorCache` (and is run
    once per variant when driven by a family), a comparison analysis receives
    **all** the family's caches at once as ``{label: PriorCache}``, because its
    output is the contrast itself -- e.g. dimension marginals under triangular
    vs uniform ``d_U`` on shared axes.

    The dict preserves the family's intervention order; renderers can rely on
    it for consistent series ordering. Provenance is ``specs.json`` (one spec
    per variant) instead of ``spec.json``.
    """

    def compute(self, caches: Dict[str, PriorCache]) -> Any:
        """Return a results object from the labelled caches. Pure."""
        raise NotImplementedError

    def run(self, caches: Dict[str, PriorCache], outdir: Path) -> AnalysisArtifacts:
        """Compute, render, and write provenance for a multi-variant run."""
        if not caches:
            raise ValueError(f"{self.name}: no caches supplied")
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        results = self.compute(caches)
        artifacts = self.render(results, outdir)

        with open(outdir / "specs.json", "w") as f:
            json.dump(
                {label: c.spec.to_dict() for label, c in caches.items()}, f, indent=2
            )
        with open(outdir / "summary.json", "w") as f:
            json.dump(artifacts.summary, f, indent=2)

        return artifacts


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Analysis] = {}


def register(cls: Type[Analysis]) -> Type[Analysis]:
    """Class decorator that adds an :class:`Analysis` to the registry.

    Instantiates the class (analyses are stateless) and stores the instance.
    Rejects blank or duplicate names so a typo fails loudly at import time
    rather than silently shadowing another analysis.
    """
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty `name`")
    if cls.name in _REGISTRY:
        raise ValueError(f"Duplicate analysis name: {cls.name!r}")
    _REGISTRY[cls.name] = cls()
    return cls


def get_analysis(name: str) -> Analysis:
    """Return the registered analysis called ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown analysis {name!r}. Available: {sorted(_REGISTRY)}"
        ) from None


def all_analyses() -> Dict[str, Analysis]:
    """Return a copy of the registry, keyed by name."""
    return dict(_REGISTRY)
