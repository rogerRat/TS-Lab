"""Real-data RAGE experiments with lazy compatibility exports.

Importing a bottom-layer experiment component must not eagerly import the
PyArrow-backed data layer. Existing public data helpers remain available via
PEP 562 lazy attribute resolution.
"""

_DATA_EXPORTS = {
    "load_bundle_index",
    "reference_load_dirs",
    "build_reference_fragments",
    "build_candidate_fragments",
}


def __getattr__(name: str):
    if name in _DATA_EXPORTS:
        from . import data
        return getattr(data, name)
    raise AttributeError(name)

__all__ = [
    "load_bundle_index",
    "reference_load_dirs",
    "build_reference_fragments",
    "build_candidate_fragments",
]
