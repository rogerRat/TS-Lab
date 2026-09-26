"""Fetch adapter for RMISC."""

from assets.dataset_processing.configs.rmisc import SPEC

def fetch(output_dir, subsets=None, force=False, sample_probability=None, seed=42, revision=None, num_workers=1):
    """Fetch RMISC raw Parquet files."""
    from .hub_files import fetch_adapter
    if subsets == ["all"]:
        subsets = [""]
    return fetch_adapter(output_dir, "rmisc", SPEC, (".arrow", ".parquet"), "grouped", subsets, force, sample_probability, seed, revision, num_workers)
