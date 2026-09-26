"""Load adapter for Time Series PILE."""

def load(input_dir, runtime_writer, subset, max_series=float("inf"), seed=42,
         preserve_invalid_values=False):
    """Convert PILE .ts files directly into RuntimeDataset rows."""
    from .extended import load_adapter
    return load_adapter(input_dir, runtime_writer, "pile", subset, _records, max_series, seed, preserve_invalid_values)

def _records(path):
    """Yield PILE .ts trajectories."""
    lines = path.read_text(encoding="utf-8").splitlines()
    data_index = next((index for index, line in enumerate(lines) if line.strip().lower() == "@data"), None)
    if data_index is None:
        raise ValueError("Time Series PILE .ts file has no @data section: %s" % path)
    class_label = any(line.strip().lower().startswith("@classlabel true") for line in lines[:data_index])
    ordinal = 0
    for line in lines[data_index + 1:]:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        fields = text.split(":")
        if class_label:
            if len(fields) < 2:
                raise ValueError("Time Series PILE class-labelled row has no values: %s" % path)
            fields = fields[:-1]
        for dimension, field in enumerate(fields):
            try:
                values = [float("nan") if value.strip() in ("", "?") else float(value.strip()) for value in field.split(",")]
            except ValueError as exc:
                raise ValueError("Time Series PILE only supports non-timestamp numeric .ts values: %s" % path) from exc
            if not values:
                raise ValueError("Time Series PILE row has an empty dimension: %s" % path)
            yield "%s_%d" % (path.stem, ordinal), "" if len(fields) == 1 else "dimension_%d" % dimension, values
        ordinal += 1
    if ordinal == 0:
        raise ValueError("Time Series PILE .ts file contains no trajectories: %s" % path)
