"""Shared shard-data schemas retained for the RAGE data pipeline.

Legacy selection policies, transfer solvers, grid/feature indexes and training
schedulers have been removed. RAGE algorithms live in the rage package.
"""
from .spec.errors import TslProtocolError

__version__ = "0.1.0"
__all__ = ["TslProtocolError", "__version__"]
