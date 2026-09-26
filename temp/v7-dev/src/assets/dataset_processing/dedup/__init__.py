"""Source and reference Dedup namespace."""
from .src.bundle import deduplicate_bundle
from .src.runner import main
__all__ = ["deduplicate_bundle", "main"]
