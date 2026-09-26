"""Name -> adapter registry.

One flat namespace, populated by importing :mod:`lit_eval.datasets`. A dataset
registers itself at import time so adding one is a single new module plus a line
in ``datasets/__init__.py`` -- the same shape as ``analysis_prior.analyses``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from lit_eval.base import ProximalDataset

_REGISTRY: Dict[str, Type[ProximalDataset]] = {}


def register_dataset(cls: Type[ProximalDataset]) -> Type[ProximalDataset]:
    """Class decorator: register ``cls`` under its ``name``."""
    name = getattr(cls, "name", None)
    if not name or name == "unnamed":
        raise ValueError(f"{cls.__name__} must set a class-level `name`")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"dataset name {name!r} is already registered")
    _REGISTRY[name] = cls
    return cls


def available_datasets() -> List[str]:
    return sorted(_REGISTRY)


def dataset_class(name: str) -> Type[ProximalDataset]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}; available: {', '.join(available_datasets())}"
        ) from None


def build_dataset(name: str, **options: Any) -> ProximalDataset:
    return dataset_class(name)(**options)
