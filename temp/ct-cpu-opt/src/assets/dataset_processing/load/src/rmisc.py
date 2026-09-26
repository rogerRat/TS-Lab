"""Load adapter for RMISC."""

def load(input_dir, runtime_writer, subset, max_series=float("inf"), seed=42,
         preserve_invalid_values=False):
    """Convert RMISC raw Parquet directly into RuntimeDataset rows."""
    from .extended import load_adapter
    return load_adapter(input_dir, runtime_writer, "rmisc", subset, _records, max_series, seed, preserve_invalid_values)

def _records(path):
    """Yield RMISC grouped or file-scoped trajectories."""
    from .extended import _iter_contiguous_groups
    import pyarrow as pa
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    excluded = {"_original_filename", "date", "date_time", "datetime", "timestamp", "time"}
    columns = [name for name, field in zip(names, parquet.schema_arrow) if name not in excluded and (pa.types.is_integer(field.type) or pa.types.is_floating(field.type))]
    if not columns:
        raise ValueError("RMISC file %s has no numeric value fields" % path)
    if "_original_filename" not in names:
        for name in columns:
            values = []
            for batch in parquet.iter_batches(columns=[name], batch_size=4096):
                values.extend(value for value in batch.column(0).to_pylist() if value is not None)
            yield path.stem, name, values
        return
    for group in _iter_contiguous_groups(parquet, "_original_filename"):
        for name in columns:
            yield group[0]["_original_filename"], name, [row[name] for row in group if row[name] is not None]
