"""Core modules for Multi-REAL benchmark."""

from .registry import MultiRealTask, MultiRealRegistry, registry

# Lazy imports for modules with external dependencies
def __getattr__(name):
    if name == "MultiRealHarness":
        from .harness import MultiRealHarness
        return MultiRealHarness
    elif name == "MultiRealResult":
        from .harness import MultiRealResult
        return MultiRealResult
    elif name == "HybridValidator":
        from .validator import HybridValidator
        return HybridValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MultiRealTask",
    "MultiRealRegistry",
    "registry",
    "MultiRealHarness",
    "MultiRealResult",
    "HybridValidator",
]
