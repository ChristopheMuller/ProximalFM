# models/__init__.py
from ..Structure import ModelConfig

from .registry import MODEL_REGISTRY
from . import gmm


def build_model(model_config: ModelConfig):
    model_type = model_config.type
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Model type '{model_type}' not found. Available: {list(MODEL_REGISTRY.keys())}"
        )

    ModelClass = MODEL_REGISTRY[model_type]
    return ModelClass(model_config)
