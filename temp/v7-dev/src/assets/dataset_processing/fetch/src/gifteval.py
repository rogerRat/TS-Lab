"""Fetch adapter for GiftEval."""

from assets.dataset_processing.configs.gifteval import SPEC

def fetch(output_dir, subsets=None, force=False, sample_probability=None, seed=42, revision=None, num_workers=1):
    """Fetch GiftEval raw Arrow or Parquet files."""
    from .hub_files import fetch_adapter
    if subsets == ["all"]:
        subsets = [""]
    return fetch_adapter(output_dir, "gifteval", SPEC, (".arrow", ".parquet"), "row", subsets, force, sample_probability, seed, revision, num_workers)
