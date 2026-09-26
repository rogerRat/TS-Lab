"""Load adapter for Quito Corpus."""

def load(input_dir, runtime_writer, subset, max_series=float("inf"), seed=42,
         preserve_invalid_values=False):
    """Convert Quito raw Parquet directly into RuntimeDataset rows.

    @input: input_dir is the raw source root; runtime_writer is the destination;
            subset is the explicit Fetch selection; max_series and seed are the
            shared Load controls.
    @output: Appended row count, or None when the source selection is unavailable.
    @scenario: Loads Quito files whose item rows may be split into non-
               contiguous blocks.
    """
    from .extended import load_adapter
    return load_adapter(input_dir, runtime_writer, "quito", subset, _records, max_series, seed, preserve_invalid_values)

def _records(path):
    """Yield globally grouped Quito item-variable trajectories.

    @input: path is one Quito Parquet file containing item_id, date_time, and
            five numeric ind_* columns.
    @output: Iterator of ``(item_id, variable_name, values)`` tuples, with one
             tuple per item and variable and values ordered by date_time.
    @scenario: Quito repeats item_id values in separate source blocks. A
               disk-backed index combines those blocks without materializing
               the whole Parquet file in memory, so each Canonical ID is
               emitted exactly once.
    """
    import numpy as np
    import pyarrow.parquet as pq
    import pyarrow as pa

    required = {"item_id", "date_time", "ind_1", "ind_2", "ind_3", "ind_4", "ind_5"}
    parquet = pq.ParquetFile(path)
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError("Quito file %s is missing fields %s" % (path, missing))
    date_type = parquet.schema_arrow.field("date_time").type
    if not pa.types.is_timestamp(date_type):
        raise ValueError("Quito file %s date_time must be a timestamp" % path)

    columns = ["item_id", "date_time", "ind_1", "ind_2", "ind_3", "ind_4", "ind_5"]
    variable_names = ("ind_1", "ind_2", "ind_3", "ind_4", "ind_5")
    # Quito repeats item_id blocks.  The former implementation inserted every
    # row into SQLite, built a secondary index, then scanned that index once per
    # item and variable.  That is effectively several full-size disk passes.
    # Read the columns in parallel and perform one vectorized stable sort.  The
    # input ordinal is the final key so equal timestamps retain source order.
    table = parquet.read(columns=columns, use_threads=True)
    if table["item_id"].null_count:
        raise ValueError("Quito file %s contains null item_id" % path)
    if table["date_time"].null_count:
        raise ValueError("Quito file %s contains null date_time" % path)

    item_ids = table["item_id"].combine_chunks().to_numpy(zero_copy_only=False)
    time_keys = table["date_time"].combine_chunks().cast(pa.int64()).to_numpy(
        zero_copy_only=False
    )
    source_ordinal = np.arange(len(item_ids), dtype=np.int64)
    order = np.lexsort((source_ordinal, time_keys, item_ids))
    ordered_item_ids = item_ids[order]
    boundaries = np.flatnonzero(ordered_item_ids[1:] != ordered_item_ids[:-1]) + 1
    starts = np.concatenate((np.asarray([0], dtype=np.int64), boundaries))
    stops = np.concatenate((boundaries, np.asarray([len(order)], dtype=np.int64)))
    value_arrays = {
        name: table[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in variable_names
    }

    for start, stop in zip(starts, stops):
        group_order = order[int(start):int(stop)]
        item_id = str(ordered_item_ids[int(start)])
        for name in variable_names:
            # A per-item list is the existing decoder contract; only this small
            # group is gathered, so the complete file never gets duplicated.
            yield item_id, name, value_arrays[name][group_order].tolist()
