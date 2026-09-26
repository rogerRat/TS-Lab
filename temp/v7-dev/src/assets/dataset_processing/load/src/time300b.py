"""Load adapter for Time-300B."""

def load(input_dir, runtime_writer, subset, max_series=float("inf"), seed=42,
         preserve_invalid_values=False):
    """Convert Time-300B BIN plus JSON directly into RuntimeDataset rows."""
    from pathlib import Path
    from .extended import _raw_files, _write_records
    paths = _raw_files(Path(input_dir), subset)
    if paths is None:
        return None
    meta = [path for path in paths if path.name == "meta.json"]
    binary = [path for path in paths if path.suffix.lower() == ".bin"]
    if len(meta) != 1 or len(binary) != 1:
        raise ValueError("Time-300B selection must contain one .bin and one meta.json: %s" % subset)
    runtime_writer.record_input(binary[0])
    runtime_writer.record_input(meta[0], "raw_metadata")
    return _write_records(
        _records(binary[0], meta[0]), runtime_writer, "time300b", subset,
        max_series, seed, binary[0], preserve_invalid_values,
    )

def _records(binary_path, meta_path):
    """Yield Time-300B metadata-scoped float32 trajectories."""
    import json
    import numpy as np
    try:
        metadata = json.loads(meta_path.read_text(encoding="ascii"))
        scales = metadata["scales"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("Invalid Time-300B metadata: %s" % meta_path) from exc
    if metadata.get("dtype") != "float32" or not isinstance(scales, list):
        raise ValueError("Time-300B metadata must declare float32 scales: %s" % meta_path)
    values = np.memmap(binary_path, dtype=np.float32, mode="r")
    try:
        for ordinal, scale in enumerate(scales):
            if not isinstance(scale, dict):
                raise ValueError("Time-300B scale is invalid: %s" % scale)
            offset, length = scale.get("offset"), scale.get("length")
            if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length <= 0 or offset + length > len(values):
                raise ValueError("Time-300B scale is invalid: %s" % scale)
            yield "%s_%d" % (binary_path.stem, ordinal), "", values[offset:offset + length].tolist()
    finally:
        del values
