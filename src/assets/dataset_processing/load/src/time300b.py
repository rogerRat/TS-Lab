"""Load adapter for Time-300B."""


def load(input_dir, runtime_writer, subset, max_series=float("inf"), seed=42,
         preserve_invalid_values=False):
    """Convert Time-300B BIN shards plus JSON directly into source rows."""
    from pathlib import Path
    from .extended import _raw_files, _write_records

    paths = _raw_files(Path(input_dir), subset)
    if paths is None:
        return None
    metadata_paths = [path for path in paths if path.name == "meta.json"]
    binary_paths = [path for path in paths if path.suffix.lower() == ".bin"]
    if len(metadata_paths) != 1 or not binary_paths:
        raise ValueError("Time-300B selection must contain BIN shards and one meta.json: %s" % subset)
    for path in binary_paths:
        runtime_writer.record_input(path)
    runtime_writer.record_input(metadata_paths[0], "raw_metadata")
    return _write_records(
        _records(binary_paths, metadata_paths[0]), runtime_writer, "time300b", subset,
        max_series, seed, metadata_paths[0], preserve_invalid_values,
    )


def _records(binary_paths, meta_path):
    """Yield float32 trajectories using metadata offsets across ordered BIN shards."""
    import json
    from bisect import bisect_right
    import numpy as np

    try:
        metadata = json.loads(meta_path.read_text(encoding="ascii"))
        scales = metadata["scales"]
        declared_files = metadata["files"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("Invalid Time-300B metadata: %s" % meta_path) from exc
    if metadata.get("dtype") != "float32" or not isinstance(scales, list):
        raise ValueError("Time-300B metadata must declare float32 scales: %s" % meta_path)
    ordered_paths = sorted(binary_paths, key=lambda path: int(path.stem.split("-")[1]))
    if not isinstance(declared_files, dict) or set(declared_files) != {path.name for path in ordered_paths}:
        raise ValueError("Time-300B BIN shards do not match metadata: %s" % meta_path)
    mapped = []
    boundaries = []
    total_values = 0
    for path in ordered_paths:
        count = declared_files[path.name]
        if not isinstance(count, int) or count <= 0 or path.stat().st_size != count * 4:
            raise ValueError("Time-300B BIN shard size is invalid: %s" % path)
        boundaries.append(total_values)
        total_values += count
        mapped.append(np.memmap(path, dtype=np.float32, mode="r"))
    try:
        for ordinal, scale in enumerate(scales):
            if not isinstance(scale, dict):
                raise ValueError("Time-300B scale is invalid: %s" % scale)
            offset, length = scale.get("offset"), scale.get("length")
            if (not isinstance(offset, int) or not isinstance(length, int)
                    or offset < 0 or length <= 0 or offset + length > total_values):
                raise ValueError("Time-300B scale is invalid: %s" % scale)
            position, remaining, parts = offset, length, []
            while remaining:
                shard_index = bisect_right(boundaries, position) - 1
                local_offset = position - boundaries[shard_index]
                count = min(remaining, len(mapped[shard_index]) - local_offset)
                parts.append(mapped[shard_index][local_offset:local_offset + count])
                position += count
                remaining -= count
            values = parts[0] if len(parts) == 1 else np.concatenate(parts)
            yield "%s_%d" % (meta_path.parent.name, ordinal), "", values.tolist()
    finally:
        del mapped
