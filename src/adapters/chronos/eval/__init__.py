"""Explicit lazy exports for benchmark-specific Chronos evaluation adapters."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Load only the benchmark adapter requested by the active environment.

    @input: name is one public package attribute.
    @output: Requested adapter class or configuration class.
    @scenario: Let the TIME environment resolve its adapter without installing FEV.
    @logic: Imports the FEV module only for FEV symbols and the TIME module only
            for the TIME symbol; all other names raise AttributeError.
    @param name: Public attribute name requested by an importer.
    """
    if name in {"ChronosFEVEvaluationAdapter", "ChronosFEVEvaluationConfig"}:
        from .fev import ChronosFEVEvaluationAdapter, ChronosFEVEvaluationConfig
        return {"ChronosFEVEvaluationAdapter": ChronosFEVEvaluationAdapter, "ChronosFEVEvaluationConfig": ChronosFEVEvaluationConfig}[name]
    if name == "ChronosTIMEEvaluationAdapter":
        from .time import ChronosTIMEEvaluationAdapter
        return ChronosTIMEEvaluationAdapter
    raise AttributeError(name)


__all__ = ["ChronosFEVEvaluationAdapter", "ChronosFEVEvaluationConfig", "ChronosTIMEEvaluationAdapter"]
