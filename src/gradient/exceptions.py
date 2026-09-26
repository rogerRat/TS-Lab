"""Exceptions raised by gradient model loading."""

from __future__ import annotations


class ModelNotFoundError(FileNotFoundError):
    """Required warm-up model is missing or cannot be loaded."""


__all__ = ["ModelNotFoundError"]
