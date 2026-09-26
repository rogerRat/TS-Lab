"""Shared shard-data schemas retained for the RAGS data pipeline.

Legacy selection policies, transfer solvers, grid/feature indexes and training
schedulers have been removed. RAGS algorithms live in the rags package.
"""
from .spec.errors import TS3ProtocolError

__version__ = "0.1.0"
__all__ = ["TS3ProtocolError", "__version__"]
