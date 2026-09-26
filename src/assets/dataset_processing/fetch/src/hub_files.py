"""Shared Hugging Face file-tree Fetch workflow.

The workflow preserves the upstream file tree. Source adapter modules provide
their repository configuration, leaf-selection policy, supported raw formats,
and format-specific sampling mode. When sampling is enabled, one complete
source trajectory is the minimum unit: synchronized multivariate variables are
never sampled independently in Fetch.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .utils import DownloadFailureStack, RetryingHuggingFaceDownloader
from assets.dataset_processing.progress import ProcessingProgress
from .sampling import (
    FileTrajectorySampler,
    atomic_replace,
    cache_path,
    discard_cache_file,
    file_rng,
    remove_empty_cache,
    parallel_map,
    validate_sample_probability,
)

log = logging.getLogger("fetch")


def _split_time300b_leaf_files(files: list[str], selection: str) -> list[tuple[str, list[str]]]:
    """Split a Time-300B parent selection into complete leaf selections.

    @input: files is the supported repository-relative file list returned for
            one requested selection; selection is the requested Hub path.
    @output: Sorted ``(leaf_path, files)`` pairs, each containing exactly one
             ``meta.json`` and one ``.bin`` file.
    @scenario: Keep every Time-300B BIN/metadata pair independently loadable
               when a Profile or explicit request names a parent directory.
    @logic: Group files by their immediate parent directory, reject unrelated
            JSON files and incomplete pairs, then return one stable leaf task
            per complete group. No files are silently discarded.
    """
    grouped: dict[str, list[str]] = {}
    for name in files:
        path = Path(name)
        if path.name != "meta.json" and path.suffix.lower() != ".bin":
            raise ValueError(
                "Time-300B selection contains unsupported metadata file %s: %s"
                % (name, selection or "<root>")
            )
        grouped.setdefault(path.parent.as_posix(), []).append(name)
    if not grouped:
        raise ValueError("Time-300B selection has no leaf files: %s" % (selection or "<root>"))

    leaves = []
    for leaf, leaf_files in sorted(grouped.items()):
        meta_files = [name for name in leaf_files if Path(name).name == "meta.json"]
        bin_files = [name for name in leaf_files if Path(name).suffix.lower() == ".bin"]
        if len(meta_files) != 1 or not bin_files:
            raise ValueError(
                "Time-300B leaf must contain one .bin and one meta.json, with additional .bin files allowed: %s"
                % leaf
            )
        leaves.append((leaf, sorted(leaf_files)))
    return leaves


def _safe_marker_name(value: str) -> str:
    """Return an ASCII marker filename for a source selection.

    @input: value is a repository-relative subset or file path.
    @output: A stable filename-safe string.
    @scenario: Completion markers must not create ambiguous nested paths.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("._") or "root"


def _list_files(spec: dict, supported_suffixes: tuple[str, ...], subset: str) -> list[str]:
    """List supported raw files below one explicit Hub selection.

    @input: spec identifies one repository; supported_suffixes are explicit
            raw formats; subset is a Hub path prefix.
    @output: Sorted repository-relative raw file paths.
    @scenario: A source adapter lists only its explicitly supported formats.
    """
    from huggingface_hub import list_repo_tree

    if subset.lower().endswith(supported_suffixes):
        return [subset]
    entries = list_repo_tree(
        spec["repo"], repo_type="dataset", revision=spec["revision"],
        path_in_repo=subset or None, recursive=True,
    )
    files = sorted(
        entry.path for entry in entries
        if entry.path.lower().endswith(supported_suffixes)
    )
    unsupported = [entry.path for entry in entries if entry.path.lower().endswith((".7z", ".zip"))]
    if unsupported:
        raise ValueError(
            "Source selection contains compressed files that this adapter does not decode: %s"
            % ", ".join(unsupported[:3])
        )
    if not files:
        raise ValueError("No supported raw files below %s/%s" % (spec["repo"], subset))
    return files


def _sample_time300b_files(cache_root: Path, raw_root: Path, files: list[str],
                           probability: float, seed: int) -> tuple[int, int]:
    """Sample Time-300B sequences while preserving its BIN plus JSON format.

    @input: cache_root contains downloaded source files; raw_root is the final
            raw tree; files is one selection's repository-relative file list;
            probability and seed control deterministic sequence sampling.
    @output: ``(candidate_count, selected_count)``.
    @scenario: Time-300B stores contiguous float32 sequences in one or more
               BIN files and describes their global offsets in sibling metadata.
               Sampled output is normalized to one BIN for the existing loader.
    """
    import numpy as np

    meta_names = [name for name in files if Path(name).name == "meta.json"]
    bin_names = [name for name in files if name.lower().endswith(".bin")]
    if len(meta_names) != 1 or not bin_names:
        raise ValueError("Time-300B selection must contain one meta.json and one or more .bin files")
    meta_path = cache_root / meta_names[0]
    try:
        metadata = json.loads(meta_path.read_text(encoding="ascii"))
        scales = metadata["scales"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("Invalid Time-300B metadata: %s" % meta_path) from exc
    source_files = metadata.get("files")
    if metadata.get("dtype") != "float32" or not isinstance(scales, list) or not scales:
        raise ValueError("Time-300B metadata must declare non-empty float32 scales: %s" % meta_path)
    if not isinstance(source_files, dict) or set(source_files) != {Path(name).name for name in bin_names}:
        raise ValueError("Time-300B metadata BIN file map is invalid: %s" % meta_path)
    ordered_names = sorted(bin_names, key=lambda name: int(Path(name).name.split("-")[1]))
    bin_paths = [cache_root / name for name in ordered_names]
    boundaries = []
    total_values = 0
    for path, name in zip(bin_paths, ordered_names):
        expected_values = source_files[Path(name).name]
        expected_bytes = expected_values * 4 if isinstance(expected_values, int) else None
        if expected_bytes is None or expected_values <= 0 or path.stat().st_size != expected_bytes:
            raise ValueError("Time-300B BIN size is invalid: %s" % path)
        total_values += expected_values
        boundaries.append(total_values)
    sampler = FileTrajectorySampler(probability, file_rng(seed, str(raw_root / bin_names[0])))
    selected = []
    with ProcessingProgress(
        log, "FETCH_TIME300B_SELECT", str(raw_root), len(scales),
        "metadata=%s source_bins=%d" % (meta_path, len(bin_paths)),
    ) as progress:
        for scale in scales:
            if not isinstance(scale, dict):
                raise ValueError("Time-300B scale must be an object: %s" % meta_path)
            offset, length = scale.get("offset"), scale.get("length")
            if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length <= 0:
                raise ValueError("Time-300B scale has invalid offset or length: %s" % scale)
            if offset + length > total_values:
                raise ValueError("Time-300B scale exceeds BIN contents: %s" % scale)
            if length >= 10 and sampler.keep():
                selected.append((offset, length))
            progress.advance(detail="retained=%d candidates=%d" % (len(selected), sampler.candidates))
    if sampler.candidates == 0:
        raise ValueError(
            "Time-300B source contains no valid trajectory with length at least 10: %s" % meta_path
        )
    published_name = bin_names[0] if len(bin_names) == 1 else str(Path(meta_names[0]).with_name("sampled.bin"))
    bin_destination = raw_root / published_name
    meta_destination = raw_root / meta_names[0]
    bin_destination.parent.mkdir(parents=True, exist_ok=True)
    meta_destination.parent.mkdir(parents=True, exist_ok=True)
    bin_partial = bin_destination.with_name(bin_destination.name + ".partial")
    meta_partial = meta_destination.with_name(meta_destination.name + ".partial")
    bin_partial.unlink(missing_ok=True)
    meta_partial.unlink(missing_ok=True)
    try:
        sources = [np.memmap(path, dtype=np.float32, mode="r") for path in bin_paths]
        next_offset = 0
        sampled_scales = []
        with bin_partial.open("wb") as handle:
            with ProcessingProgress(
                log, "FETCH_TIME300B_WRITE", str(bin_destination), len(selected),
                "partial=%s" % bin_partial,
                detail_provider=lambda: "selected_values_written=%d partial_bytes=%d" % (
                    next_offset, bin_partial.stat().st_size if bin_partial.exists() else 0,
                ),
            ) as progress:
                for offset, length in selected:
                    remaining = length
                    cursor = offset
                    while remaining:
                        source_index = next(index for index, boundary in enumerate(boundaries) if cursor < boundary)
                        source_start = 0 if source_index == 0 else boundaries[source_index - 1]
                        take = min(remaining, boundaries[source_index] - cursor)
                        sources[source_index][cursor - source_start:cursor - source_start + take].tofile(handle)
                        cursor += take
                        remaining -= take
                    sampled_scales.append({"offset": next_offset, "length": length})
                    next_offset += length
                    progress.advance()
        del sources
        sampled_metadata = dict(metadata)
        sampled_metadata["num_sequences"] = len(sampled_scales)
        sampled_metadata["files"] = {Path(published_name).name: next_offset}
        sampled_metadata["scales"] = sampled_scales
        meta_partial.write_text(json.dumps(sampled_metadata, sort_keys=True) + "\n", encoding="ascii")
        atomic_replace(bin_partial, bin_destination)
        atomic_replace(meta_partial, meta_destination)
    except Exception:
        bin_partial.unlink(missing_ok=True)
        meta_partial.unlink(missing_ok=True)
        raise
    return sampler.candidates, sampler.selected


def _has_valid_sequence(values) -> bool:
    """Return whether values produce a Canonical trajectory of length at least 10.

    @input: values is one raw sequence field, possibly one- or two-dimensional.
    @output: bool indicating whether at least one valid univariate trajectory exists.
    @scenario: Keep Fetch sampling aligned with Load's minimum series-length rule.
    """
    import numpy as np
    from assets.dataset_processing.series import split_univariate_series

    array = np.asarray(values)
    candidates = [values] if array.ndim == 1 else array.tolist() if array.ndim == 2 else []
    for candidate in candidates:
        split_records, skipped = split_univariate_series(
            "fetch", candidate, preserve_invalid_values=True
        )
        if skipped is None and any(len(series) >= 10 for _series_id, series in split_records):
            return True
    return False


def _row_has_valid_trajectory(row: dict) -> bool:
    """Return whether one generic Arrow or Parquet row has a valid sequence.

    @input: row is one decoded raw record.
    @output: bool indicating whether Load can emit at least one length-10 series.
    @scenario: Exclude metadata-only and short records before Fetch publication.
    """
    excluded = {
        "id", "item_id", "start", "freq", "date", "date_time",
        "timestamp", "time", "label", "labels", "target_class",
        "_original_filename",
    }
    return any(
        name not in excluded and isinstance(value, (list, tuple)) and _has_valid_sequence(value)
        for name, value in row.items() if value is not None
    )


def _group_has_valid_trajectory(rows: list[dict], source: str) -> bool:
    """Return whether one Quito or RMISC group can produce a length-10 series.

    @input: rows contains one complete grouped raw trajectory; source is quito or rmisc.
    @output: bool indicating whether the group is valid for Load.
    @scenario: Prevent sampled grouped files from retaining metadata-only groups.
    """
    if not rows:
        return False
    if source == "quito":
        return len(rows) >= 10 and any(
            _has_valid_sequence([row.get(name) for row in rows])
            for name in ("ind_1", "ind_2", "ind_3", "ind_4", "ind_5")
        )
    excluded = {"_original_filename", "date", "date_time", "datetime", "timestamp", "time"}
    return any(
        _has_valid_sequence([row.get(name) for row in rows if row.get(name) is not None])
        for name in rows[0] if name not in excluded
    )


def _marker(output_dir: Path, subset: str) -> Path:
    """Return the completion marker for one extended source selection.

    @input: output_dir is the source raw root; subset is the selection label.
    @output: Marker path under ``.complete``.
    @scenario: Load must reject partially fetched selections.
    """
    return output_dir / ".complete" / (_safe_marker_name(subset) + ".json")


def _sampled_marker_matches(payload: dict, files: list[str], probability: float,
                            seed: int, raw_root: Path) -> bool:
    """Return whether one sampled selection is complete and has valid output.

    @input: payload is one completion-marker object; files is the requested
            source-file list; probability and seed are the requested policy.
    @output: bool -- True only for a matching marker with retained trajectories.
    @scenario: Apply the same sampled-fetch completion contract to every
            extended dataset adapter.
    """
    counts = payload.get("valid_trajectories")
    return (
        payload.get("source_files", payload.get("files")) == files
        and payload.get("sampling") == {"probability": probability, "seed": int(seed)}
        and isinstance(counts, dict)
        and bool(counts)
        and all(int(count) > 0 for count in counts.values())
        and all((raw_root / name).is_file() for name in payload.get("files", []))
    )


def _sample_row_file(source: str, source_path: Path, destination: Path,
                     probability: float, seed: int) -> tuple[int, int]:
    """Sample complete Arrow/Parquet records without changing their schema.

    @input: source identifies the adapter; source_path is cached raw input;
            destination is the original-tree output; probability and seed are
            trajectory-level sampling controls.
    @output: ``(candidate_count, selected_count)``.
    @scenario: Arrow-record datasets store a whole synchronized trajectory in
               one record.  Fetch never samples nested variables separately.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    sampler = FileTrajectorySampler(probability, file_rng(seed, str(destination)))
    try:
        if source_path.suffix.lower() == ".parquet":
            parquet = pq.ParquetFile(source_path)
            writer = pq.ParquetWriter(str(partial), parquet.schema_arrow)
            try:
                for batch in parquet.iter_batches(batch_size=1024):
                    selected = [
                        row for row in batch.to_pylist()
                        if _row_has_valid_trajectory(row) and sampler.keep()
                    ]
                    if selected:
                        writer.write_table(pa.Table.from_pylist(selected, schema=parquet.schema_arrow))
            finally:
                writer.close()
        else:
            source_map = pa.memory_map(str(source_path), "r")
            sink = None
            try:
                try:
                    reader = ipc.open_stream(source_map)
                    batches = reader
                    sink = pa.OSFile(str(partial), "wb")
                    writer = ipc.new_stream(sink, reader.schema)
                except pa.ArrowInvalid:
                    source_map.seek(0)
                    reader = ipc.open_file(source_map)
                    batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
                    sink = pa.OSFile(str(partial), "wb")
                    writer = ipc.new_file(sink, reader.schema)
                try:
                    for batch in batches:
                        selected = [
                            row for row in batch.to_pylist()
                            if _row_has_valid_trajectory(row) and sampler.keep()
                        ]
                        if selected:
                            writer.write_batch(pa.RecordBatch.from_pylist(selected, schema=reader.schema))
                finally:
                    writer.close()
                    sink.close()
            finally:
                source_map.close()
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    if sampler.candidates == 0:
        partial.unlink(missing_ok=True)
        raise ValueError(
            "Source file contains no valid trajectory with length at least 10: %s"
            % source_path
        )
    atomic_replace(partial, destination)
    return sampler.candidates, sampler.selected


def _sample_pile_ts_file(source_path: Path, destination: Path,
                         probability: float, seed: int) -> tuple[int, int]:
    """Sample complete UCR ``.ts`` records while retaining its source format.

    @input: source_path is a Time Series PILE UCR file; destination is its raw
            output path; probability and seed select complete data records.
    @output: ``(candidate_count, selected_count)``.
    @scenario: Time Series PILE stores UCR trajectories as text rather than
               Arrow/Parquet; Fetch must preserve headers and selected records.
    """
    lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
    data_index = next((index for index, line in enumerate(lines)
                       if line.strip().lower() == "@data"), None)
    if data_index is None:
        raise ValueError("Time Series PILE .ts file has no @data section: %s" % source_path)
    class_label = any(
        line.strip().lower().startswith("@classlabel true")
        for line in lines[:data_index]
    )
    sampler = FileTrajectorySampler(probability, file_rng(seed, str(destination)))
    selected = []
    for line in lines[data_index + 1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            selected.append(line)
            continue
        fields = line.strip().split(":")
        if class_label:
            if len(fields) < 2:
                continue
            fields = fields[:-1]
        valid = False
        for field in fields:
            try:
                values = [
                    float("nan") if value.strip() in ("", "?") else float(value.strip())
                    for value in field.split(",")
                ]
            except ValueError:
                continue
            if len(values) >= 10:
                valid = True
                break
        if valid and sampler.keep():
            selected.append(line)
    if sampler.candidates == 0:
        raise ValueError(
            "Time Series PILE .ts file contains no valid trajectory with length at least 10: %s"
            % source_path
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        partial.write_text("".join(lines[:data_index + 1] + selected), encoding="utf-8")
        atomic_replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return sampler.candidates, sampler.selected


def _sample_grouped_parquet(source_path: Path, destination: Path, source: str,
                            probability: float, seed: int) -> tuple[int, int]:
    """Sample complete grouped trajectories from Quito or RMISC Parquet.

    @input: source_path and destination are one raw file pair; source is ``quito``
            or ``rmisc``; probability and seed control deterministic sampling.
    @output: ``(trajectory_count, selected_count)``.
    @scenario: Long/tidy sources contain many timestamps per trajectory.  The
               complete item/group, including every synchronized variable, is
               the indivisible Fetch sampling unit.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    group_key = "item_id" if source == "quito" else "_original_filename"
    parquet = pq.ParquetFile(source_path)
    names = parquet.schema_arrow.names
    if group_key not in names:
        if source != "rmisc":
            raise ValueError("%s source file is missing grouping field %s: %s" % (source, group_key, source_path))
        # RMISC contains both tidy multi-source Parquet files and one-series
        # Parquet files such as ACSF1/f1. The latter has no source-id field,
        # so the complete file remains its single indivisible trajectory.
        # This is an intentional warning boundary, not a global ratio fix:
        # FileTrajectorySampler also forces the first candidate in each file.
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        partial.unlink(missing_ok=True)
        numeric_columns = [
            field.name for field in parquet.schema_arrow
            if pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
        ]
        valid = any(
            sum(batch.num_rows for batch in parquet.iter_batches(columns=[name], batch_size=4096)) >= 10
            for name in numeric_columns
        )
        if not valid:
            raise ValueError(
                "RMISC source file contains no valid trajectory with length at least 10: %s"
                % source_path
            )
        sampler = FileTrajectorySampler(probability, file_rng(seed, str(destination)))
        try:
            if sampler.keep():
                pq.write_table(parquet.read(), partial)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        atomic_replace(partial, destination)
        return sampler.candidates, sampler.selected
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    sampler = FileTrajectorySampler(probability, file_rng(seed, str(destination)))
    writer = pq.ParquetWriter(str(partial), parquet.schema_arrow)
    selected_keys = set()
    seen_keys = set()
    counts = {}
    value_columns = (
        ("ind_1", "ind_2", "ind_3", "ind_4", "ind_5")
        if source == "quito" else tuple(
            field.name for field in parquet.schema_arrow
            if field.name not in {group_key, "date", "date_time", "datetime", "timestamp", "time"}
            and (pa.types.is_integer(field.type) or pa.types.is_floating(field.type))
        )
    )
    try:
        # First pass decides once per complete trajectory, even when a source
        # file interleaves rows from different trajectories.
        for batch in parquet.iter_batches(columns=[group_key, *value_columns], batch_size=4096):
            for row in batch.to_pylist():
                key = row[group_key]
                counts.setdefault(key, 0)
                if any(row.get(name) is not None for name in value_columns):
                    counts[key] += 1
        valid_keys = {key for key, count in counts.items() if count >= 10}
        if not valid_keys:
            raise ValueError(
                "Source file contains no valid grouped trajectory with length at least 10: %s"
                % source_path
            )
        for batch in parquet.iter_batches(columns=[group_key], batch_size=4096):
            for key in batch.column(0).to_pylist():
                if key in valid_keys and key not in seen_keys:
                    seen_keys.add(key)
                    if sampler.keep():
                        selected_keys.add(key)
        for batch in parquet.iter_batches(batch_size=4096):
            rows = [row for row in batch.to_pylist() if row[group_key] in selected_keys]
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=parquet.schema_arrow))
    finally:
        writer.close()
    if sampler.candidates == 0:
        partial.unlink(missing_ok=True)
        raise ValueError(
            "Source file contains no valid grouped trajectory with length at least 10: %s"
            % source_path
        )
    atomic_replace(partial, destination)
    return sampler.candidates, sampler.selected


def fetch_adapter(output_dir, source: str, source_spec: dict,
                  supported_suffixes: tuple[str, ...], sampling_mode: str,
                  subsets=None, force=False, sample_probability=None, seed=42,
                  revision=None, num_workers=1) -> int:
    """Fetch one source adapter with optional bounded sampling.

    @input: output_dir is the raw source root; source identifies one adapter;
            subsets is a list of Hub paths or None; force refreshes files;
            sample_probability is None or in (0, 1].
    @output: 0 on success, 1 when any requested selection fails.
    @scenario: Adds raw-source acquisition for six heterogeneous corpora while
               preserving paths and limiting sampled-fetch storage to one file.
    """
    probability = validate_sample_probability(sample_probability)
    spec = dict(source_spec)
    if revision:
        spec["revision"] = revision
    selections = list(spec["defaults"] if subsets is None else subsets)
    if source == "rmisc" and probability is not None:
        log.warning(
            "RMISC sampling keeps the existing file-scoped fallback for files "
            "without _original_filename; each such file is treated as one "
            "trajectory and its first candidate is forced to remain."
        )
    raw_root = Path(output_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    cache_root = raw_root / ".cache"
    failures = DownloadFailureStack()
    downloader = RetryingHuggingFaceDownloader(failure_stack=failures)
    failed = []
    completed = []
    if sampling_mode == "time300b":
        expanded_selections = []
        try:
            for selection in selections:
                listed_files = _list_files(spec, supported_suffixes, selection)
                expanded_selections.extend(
                    _split_time300b_leaf_files(listed_files, selection)
                )
        except Exception as exc:
            failed.extend(selections)
            log.warning("Extended source %s leaf expansion failed: %s", source, exc)
            expanded_selections = []
    else:
        expanded_selections = [(selection, None) for selection in selections]
    for subset, prelisted_files in expanded_selections:
        marker = _marker(raw_root, subset)
        try:
            files = prelisted_files if prelisted_files is not None else _list_files(
                spec, supported_suffixes, subset
            )
            published_files = files
            log.info(
                "Fetch selection starting: source=%s subset=%s files=%d sampling_mode=%s probability=%s raw_root=%s",
                source, subset, len(files), sampling_mode, probability, raw_root,
            )
            if not force and marker.is_file():
                payload = json.loads(marker.read_text(encoding="ascii"))
                sampling = payload.get("sampling")
                if probability is None and payload.get("files") == files and sampling is None and all(
                    (raw_root / name).is_file() for name in files
                ):
                    completed.append(subset)
                    continue
                if probability is not None and _sampled_marker_matches(
                    payload, files, probability, seed, raw_root
                ):
                    completed.append(subset)
                    continue
            if probability is not None and sampling_mode == "time300b":
                def cache_one(filename):
                    cached = cache_path(cache_root, filename)
                    downloaded = downloader.download_file(
                        spec["repo"], filename, repo_type="dataset",
                        revision=spec["revision"], local_dir=cache_root,
                        force_download=True,
                    )
                    if downloaded is None or not cached.is_file():
                        raise RuntimeError("Invalid cached raw file: %s" % filename)
                    return filename
                parallel_map(cache_one, files, num_workers)
                counts = _sample_time300b_files(cache_root, raw_root, files, probability, seed)
                if len([name for name in files if name.lower().endswith(".bin")]) > 1:
                    published_files = [str(Path(next(name for name in files if Path(name).name == "meta.json")).with_name("sampled.bin")), next(name for name in files if Path(name).name == "meta.json")]
                valid_trajectories = {"selection": counts[1]}
                for filename in files:
                    discard_cache_file(cache_path(cache_root, filename), cache_root)
                log.info("Sampled %s %s: retained %d/%d trajectories", source, subset, counts[1], counts[0])
            else:
                def fetch_one(filename):
                    destination = raw_root / filename
                    if probability is None:
                        downloaded = downloader.download_file(
                            spec["repo"], filename, repo_type="dataset",
                            revision=spec["revision"], local_dir=raw_root,
                            force_download=force,
                        )
                        if downloaded is None or not destination.is_file():
                            raise RuntimeError("Invalid raw file after download: %s" % filename)
                    else:
                        cached = cache_path(cache_root, filename)
                        downloaded = downloader.download_file(
                            spec["repo"], filename, repo_type="dataset",
                            revision=spec["revision"], local_dir=cache_root,
                            force_download=True,
                        )
                        if downloaded is None or not cached.is_file():
                            raise RuntimeError("Invalid cached raw file: %s" % filename)
                        if sampling_mode == "grouped":
                            counts = _sample_grouped_parquet(cached, destination, source, probability, seed)
                        elif sampling_mode == "pile":
                            counts = _sample_pile_ts_file(cached, destination, probability, seed)
                        else:
                            counts = _sample_row_file(source, cached, destination, probability, seed)
                        discard_cache_file(cached, cache_root)
                        log.info("Sampled %s %s: retained %d/%d trajectories", source, filename, counts[1], counts[0])
                    return filename, counts[1] if probability is not None else None
                sampled_files = parallel_map(fetch_one, files, num_workers)
                valid_trajectories = (
                    dict(sampled_files) if probability is not None else None
                )
            marker.parent.mkdir(parents=True, exist_ok=True)
            payload = {"dataset": spec["repo"], "revision": spec["revision"],
                       "subset": subset, "files": published_files}
            if published_files != files:
                payload["source_files"] = files
            if probability is not None:
                payload["sampling"] = {"probability": probability, "seed": int(seed)}
                if not valid_trajectories or any(int(count) <= 0 for count in valid_trajectories.values()):
                    raise ValueError("Sampled selection has no valid retained trajectory: %s" % subset)
                payload["valid_trajectories"] = valid_trajectories
            partial = marker.with_name(marker.name + ".partial")
            partial.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
            partial.replace(marker)
            completed.append(subset)
            log.info(
                "Fetch selection completed: source=%s subset=%s marker=%s files=%d",
                source, subset, marker, len(published_files),
            )
        except Exception as exc:
            failed.append(subset)
            log.exception("Extended source %s selection %s failed: %s", source, subset, exc)
    remove_empty_cache(cache_root)
    failures.log(log)
    manifest = raw_root / "fetch_manifest.json"
    manifest.write_text(json.dumps({"dataset": spec["repo"], "revision": spec["revision"],
                                    "selections": sorted(completed)}, indent=2, sort_keys=True) + "\n", encoding="ascii")
    log.info(
        "Fetch source completed: source=%s completed_selections=%d failed_selections=%d manifest=%s",
        source, len(completed), len(failed), manifest,
    )
    return 1 if failed else 0
