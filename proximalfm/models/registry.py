# models/registry.py
MODEL_REGISTRY = {}

def register_model(name: str):
    """Decorator to register a model class in the registry."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator