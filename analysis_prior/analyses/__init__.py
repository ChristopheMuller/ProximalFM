"""Prior audit analyses.

Importing this package registers every analysis. Add a new analysis by creating
a module here that defines an :class:`~analysis_prior.analyses.base.Analysis`
subclass decorated with ``@register``, then importing it below.
"""

from .base import (
    Analysis,
    AnalysisArtifacts,
    ComparisonAnalysis,
    all_analyses,
    get_analysis,
    register,
)

# Importing each analysis module runs its @register decorator. Keep this list in
# sync when adding analyses; the import is the registration.
from . import categorical  # noqa: E402,F401
from . import dimensions  # noqa: E402,F401
from . import exclusion  # noqa: E402,F401
from . import hyperparameters  # noqa: E402,F401
from . import mc_targets  # noqa: E402,F401
from . import mechanism_inputs  # noqa: E402,F401
from . import mechanism_outputs  # noqa: E402,F401
from . import outcome  # noqa: E402,F401
from . import overview  # noqa: E402,F401
from . import treatment  # noqa: E402,F401

__all__ = [
    "Analysis",
    "AnalysisArtifacts",
    "ComparisonAnalysis",
    "all_analyses",
    "get_analysis",
    "register",
]
