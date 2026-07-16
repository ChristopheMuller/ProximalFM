"""Prior audit analyses.

Importing this package registers every analysis. Add a new analysis by creating
a module here that defines an :class:`~analysis_prior.analyses.base.Analysis`
subclass decorated with ``@register``, then importing it below.
"""

from .base import Analysis, AnalysisArtifacts, all_analyses, get_analysis, register

# Importing each analysis module runs its @register decorator. Keep this list in
# sync when adding analyses; the import is the registration.
from . import dimensions  # noqa: E402,F401

__all__ = [
    "Analysis",
    "AnalysisArtifacts",
    "all_analyses",
    "get_analysis",
    "register",
]
