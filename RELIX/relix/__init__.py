"""RELIX diffraction indexing from measured reflections."""

__version__ = "1.0.0-core"


def load(checkpoint, device="cpu"):
    """Load the unified RELIX checkpoint and its fixed inference configuration."""
    from .runtime import load as _load
    return _load(checkpoint, device=device)
