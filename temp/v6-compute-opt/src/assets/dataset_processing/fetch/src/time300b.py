"""Fetch adapter for Time-300B."""

from assets.dataset_processing.configs.time300b import SPEC

def fetch(output_dir, subsets=None, force=False, sample_probability=None, seed=42, revision=None, num_workers=1):
    """Fetch Time-300B raw BIN plus metadata files."""
    from .hub_files import fetch_adapter
    if subsets == ["all"]:
        subsets = [""]
    return fetch_adapter(output_dir, "time300b", SPEC, (".bin", ".json"), "time300b", subsets, force, sample_probability, seed, revision, num_workers)
