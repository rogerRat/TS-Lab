"""Fetch adapter for LOTSA."""

from assets.dataset_processing.configs.lotsa import SPEC

def fetch(output_dir, subsets=None, force=False, sample_probability=None, seed=42, revision=None, num_workers=1):
    """Fetch LOTSA raw Arrow or Parquet files."""
    from .hub_files import fetch_adapter
    if subsets == ["all"]:
        subsets = [""]
    return fetch_adapter(output_dir, "lotsa", SPEC, (".arrow", ".parquet"), "row", subsets, force, sample_probability, seed, revision, num_workers)
