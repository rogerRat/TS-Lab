"""Fetch adapter for Time Series PILE."""

from assets.dataset_processing.configs.pile import SPEC

def fetch(output_dir, subsets=None, force=False, sample_probability=None, seed=42, revision=None, num_workers=1):
    """Fetch Time Series PILE raw .ts files."""
    from .hub_files import fetch_adapter
    if subsets == ["all"]:
        subsets = [""]
    return fetch_adapter(output_dir, "pile", SPEC, (".ts",), "pile", subsets, force, sample_probability, seed, revision, num_workers)
