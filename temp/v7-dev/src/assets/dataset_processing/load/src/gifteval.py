"""Load adapter for GiftEval."""

def load(input_dir, runtime_writer, subset, max_series=float("inf"), seed=42,
         preserve_invalid_values=False):
    """Convert GiftEval Arrow or Parquet directly into RuntimeDataset rows."""
    from .extended import load_adapter
    return load_adapter(input_dir, runtime_writer, "gifteval", subset, _records, max_series, seed, preserve_invalid_values)

def _records(path):
    """Yield GiftEval record trajectories."""
    from .extended import _generic_records
    yield from _generic_records(path, "gifteval")
