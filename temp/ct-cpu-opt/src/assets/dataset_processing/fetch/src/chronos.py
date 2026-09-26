"""Dataset Fetch -- Chronos raw HuggingFace Parquet acquisition.

@input: Chronos config names and a destination directory.
@output: Original HuggingFace Parquet files under <config>/ with completion markers.
@scenario: Acquire source files without decoding, sampling, or canonical conversion.
"""

import json
import logging
import math
from pathlib import Path

from assets.dataset_processing.configs.chronos import (
    CHRONOS_CONFIG_PATHS,
    CHRONOS_HF_DATASET,
    CHRONOS_REPRESENTATIVE_SUBSETS,
)
from .utils import DownloadFailureStack, RetryingHuggingFaceDownloader
from .sources import SOURCE_REVISIONS
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
from assets.dataset_processing.series import is_valid_parquet_file

log = logging.getLogger("fetch")

_CHRONOS_HF_DATASET = CHRONOS_HF_DATASET
_RAW_DIRECTORY = "."
_COMPLETE_FILENAME = ".complete"


def _complete_marker(raw_root, config_name):
    """Return one raw-config completion marker path.

    @input: raw_root is the Chronos raw directory; config_name is one HF config.
    @output: Path to raw_root/config_name/.complete.
    @scenario: Keeps raw acquisition completion separate from source Parquet files.
    """
    return Path(raw_root) / config_name / _COMPLETE_FILENAME


def _has_sampled_marker(raw_root, config_name):
    """Return whether a raw config marker describes sampled content."""
    try:
        payload = json.loads(_complete_marker(raw_root, config_name).read_text(encoding="ascii"))
        return "sampling" in payload
    except (OSError, ValueError, TypeError):
        return False


def _completed_raw_files(raw_root, config_name):
    """Return valid source paths named by a raw-config marker, otherwise None.

    @input: raw_root is the local raw root; config_name is one Chronos config.
    @output: list[Path] for a complete valid config, otherwise None.
    @scenario: Makes raw fetch resumable without inferring source filenames.
    """
    marker = _complete_marker(raw_root, config_name)
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="ascii"))
        if "sampling" in payload:
            return None
        filenames = payload["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(filenames, list) or not filenames:
        return None
    paths = [Path(raw_root) / str(filename) for filename in filenames]
    if any(not path.is_file() or not is_valid_parquet_file(path) for path in paths):
        return None
    return paths


def _list_config_parquet_files(config_name):
    """List the exact original Parquet paths for one Chronos configuration.

    @input: config_name is a HuggingFace dataset configuration name.
    @output: Sorted repository-relative Parquet paths.
    @scenario: Preserves every upstream shard name instead of inventing local names.
    """
    from huggingface_hub import list_repo_tree

    try:
        path_in_repo = CHRONOS_CONFIG_PATHS[config_name]
    except KeyError as exc:
        raise ValueError("Unsupported Chronos config: %s" % config_name) from exc
    entries = list_repo_tree(
        _CHRONOS_HF_DATASET,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        recursive=True,
    )
    filenames = sorted(
        entry.path for entry in entries
        if entry.path.endswith(".parquet")
    )
    if not filenames:
        raise ValueError("Chronos config has no Parquet files: %s" % config_name)
    return filenames


def _write_complete_marker(raw_root, config_name, filenames):
    """Atomically record one fully validated raw Chronos configuration.

    @input: raw_root is the destination root; config_name identifies one config;
            filenames are repository-relative original Parquet names.
    @output: Path to the final completion marker.
    @scenario: Prevents load from consuming a partial multi-file source config.
    """
    marker = _complete_marker(raw_root, config_name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    partial = marker.with_name(marker.name + ".partial")
    partial.write_text(
        json.dumps({"dataset": _CHRONOS_HF_DATASET, "files": filenames}, sort_keys=True) + "\n",
        encoding="ascii",
    )
    partial.replace(marker)
    return marker


def _remove_legacy_config_shards(output_dir, config_name):
    """Delete replaced near-canonical shards for one successfully fetched config.

    @input: output_dir is data/chronos; config_name has a valid raw completion marker.
    @output: int number of removed legacy files.
    @scenario: Finalizes the approved migration away from fetch-time conversion.
    """
    subset_dir = Path(output_dir) / "subsets"
    candidates = [
        subset_dir / ("chronos__%s.parquet" % config_name),
        subset_dir / ("chronos__%s.parquet.complete" % config_name),
    ]
    candidates.extend(sorted(subset_dir.glob("chronos__%s_chunk_*.parquet" % config_name)))
    candidates.extend(sorted(subset_dir.glob("chronos__%s_chunk_*.parquet.complete" % config_name)))
    removed = 0
    for path in candidates:
        if path.is_file():
            path.unlink()
            removed += 1
    if removed:
        log.info("  Removed %d replaced legacy shard file(s) for '%s'.", removed, config_name)
    return removed


def _write_fetch_manifest(output_dir, configs):
    """Write the explicit raw-layout manifest for this Chronos fetch invocation.

    @input: output_dir is data/chronos; configs are successfully completed names.
    @output: Path to fetch_manifest.json.
    @scenario: Documents that fetch output is raw source data, not canonical data.
    """
    output_dir = Path(output_dir)
    payload = {
        "dataset": _CHRONOS_HF_DATASET,
        "layout": "raw_huggingface_parquet",
        "configs": sorted(configs),
        "raw_directory": _RAW_DIRECTORY,
    }
    path = output_dir / "fetch_manifest.json"
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    partial.replace(path)
    return path


def _sample_chronos_file(source_path, destination_path, subset, probability, seed):
    """Sample complete Chronos source rows while preserving Parquet schema.

    @input: source_path is one downloaded Parquet file; destination_path is its
            original relative output path; subset identifies the config; probability
            is the trajectory retention probability; seed controls reproducibility.
    @output: Number of source rows and retained rows.
    @scenario: Publish a row-sampled Chronos file without converting its fields.
            One source row is the minimum unit.  If a row contains multiple
            synchronized variable fields, every field remains together; this
            function never samples Chronos columns independently.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from assets.dataset_processing.configs.chronos import CHRONOS_SEQUENCE_COLUMNS
    from assets.dataset_processing.series import split_univariate_series

    source = pq.ParquetFile(source_path)
    fields = CHRONOS_SEQUENCE_COLUMNS.get(subset)
    if fields is None:
        raise ValueError("No explicit Chronos field mapping for subset: %s" % subset)
    sampler = FileTrajectorySampler(
        probability, file_rng(seed, str(destination_path))
    )
    partial = destination_path.with_name(destination_path.name + ".partial")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)
    writer = None
    try:
        writer = pq.ParquetWriter(str(partial), source.schema_arrow)
        for batch in source.iter_batches(batch_size=1024):
            # A complete source record is the sampling unit.  Keep all fields
            # in a selected record, including every synchronized variable.
            selected = []
            for ordinal, row in enumerate(batch.to_pylist()):
                source_id = str(row.get("id", "%s_%d" % (subset, ordinal)))
                valid = False
                for field in fields:
                    if field not in row:
                        raise ValueError(
                            "Chronos source record %s is missing field %s" % (source_id, field)
                        )
                    split_records, skipped = split_univariate_series(
                        source_id, row[field], preserve_invalid_values=True
                    )
                    if skipped is None and any(
                            len(values) >= 10 for _series_id, values in split_records
                    ):
                        valid = True
                        break
                if valid and sampler.keep():
                    selected.append(row)
            if selected:
                table = pa.Table.from_pylist(selected, schema=source.schema_arrow)
                writer.write_table(table)
        if sampler.candidates == 0:
            raise ValueError(
                "Chronos source file contains no valid trajectory with length at least 10: %s"
                % source_path
            )
        writer.close()
        writer = None
        if not is_valid_parquet_file(partial):
            raise ValueError("Sampled Chronos output is invalid: %s" % partial)
        atomic_replace(partial, destination_path)
        return sampler.candidates, sampler.selected
    finally:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)


def fetch_chronos(
    output_dir,
    max_series=float("inf"),
    seed=42,
    force=False,
    subsets=None,
    streaming=False,
    delete_sources=False,
    reservoir_dir=None,
    parallel=False,
    sample_probability=None, num_workers=1,
):
    """Download original Chronos Parquet files without data conversion.

    @input: output_dir is data/chronos; max_series must be infinity; seed and
            reservoir_dir are unused compatibility arguments; force refreshes raw
            files; subsets selects configs; streaming and parallel must be false;
            delete_sources must be false.
    @output: 0 when all selected raw configs are complete, otherwise 1.
    @scenario: Separates acquisition from Loader-owned schema adaptation.
    """
    sample_probability = validate_sample_probability(sample_probability)
    if sample_probability is not None:
        if not math.isinf(max_series) or streaming or parallel or delete_sources:
            log.error("Chronos sampling mode cannot combine with legacy sampling or conversion flags.")
            return 1
        return _fetch_chronos_sampled(output_dir, force, subsets, sample_probability, seed, num_workers)
    if not math.isinf(max_series):
        log.error("Chronos fetch downloads complete raw files; use load --max-series for sampling.")
        return 1
    if streaming or parallel:
        log.error("Chronos raw fetch does not support streaming or parallel conversion.")
        return 1
    if delete_sources:
        log.error("Chronos raw files are Loader inputs and cannot be deleted by --del-src.")
        return 1
    if reservoir_dir is not None:
        log.warning("Chronos raw fetch ignores reservoir_dir.")
    if seed != 42:
        log.warning("Chronos raw fetch ignores seed; use load --seed for sampling.")

    configs = list(CHRONOS_REPRESENTATIVE_SUBSETS if subsets is None else subsets)
    output_dir = Path(output_dir)
    raw_root = output_dir
    raw_root.mkdir(parents=True, exist_ok=True)
    failure_stack = DownloadFailureStack()
    downloader = RetryingHuggingFaceDownloader(failure_stack=failure_stack)
    completed = []
    failed = []

    for config_name in configs:
        existing = None if force else _completed_raw_files(raw_root, config_name)
        if existing is not None:
            log.info("Config '%s' raw files are complete (%d file(s)); skipping.", config_name, len(existing))
            completed.append(config_name)
            continue
        log.info("Downloading raw Chronos config '%s'...", config_name)
        try:
            filenames = _list_config_parquet_files(config_name)
            def download_one(filename):
                downloaded = downloader.download_file(
                    _CHRONOS_HF_DATASET,
                    filename,
                    repo_type="dataset",
                    local_dir=raw_root,
                    force_download=force or _has_sampled_marker(raw_root, config_name),
                )
                local_path = raw_root / filename
                if downloaded is None or not is_valid_parquet_file(local_path):
                    raise RuntimeError("Invalid raw Parquet after download: %s" % filename)
                return filename
            parallel_map(download_one, filenames, num_workers)
            _write_complete_marker(raw_root, config_name, filenames)
            _remove_legacy_config_shards(output_dir, config_name)
            completed.append(config_name)
            log.info("Config '%s' raw download is complete (%d file(s)).", config_name, len(filenames))
        except Exception as exc:
            failed.append(config_name)
            log.warning("Config '%s' raw download failed: %s", config_name, exc)

    failure_stack.log(log)
    _write_fetch_manifest(output_dir, completed)
    if failed:
        log.warning("Chronos raw fetch incomplete; resumable configs failed: %s", failed)
        return 1
    log.info("Chronos raw fetch complete; source files are available in %s.", raw_root)
    return 0


def _fetch_chronos_sampled(output_dir, force, subsets, probability, seed, num_workers):
    """Download one Chronos file to raw/.cache, sample rows, and publish it.

    @input: output_dir is the raw Chronos root; force refreshes files; subsets
            selects configs; probability is in (0, 1]; seed is deterministic.
    @output: 0 when every selected config is sampled and marked complete, else 1.
    @scenario: Bound disk usage to one source file while keeping raw file names.
    """
    configs = list(CHRONOS_REPRESENTATIVE_SUBSETS if subsets is None else subsets)
    raw_root = Path(output_dir)
    cache_root = raw_root / ".cache"
    raw_root.mkdir(parents=True, exist_ok=True)
    failures = DownloadFailureStack()
    downloader = RetryingHuggingFaceDownloader(failure_stack=failures)
    completed = []
    failed = []
    for config_name in configs:
        marker = _complete_marker(raw_root, config_name)
        if not force and _sampled_marker_matches(raw_root, marker, probability, seed):
            completed.append(config_name)
            continue
        try:
            filenames = _list_config_parquet_files(config_name)
            def sample_one(filename):
                relative = str(Path(filename))
                destination = raw_root / relative
                cached = cache_path(cache_root, relative)
                downloaded = downloader.download_file(
                    _CHRONOS_HF_DATASET, filename, repo_type="dataset",
                    # Always refresh the cache source. If interrupted after
                    # publication, cache must never contain an already sampled
                    # file that could be sampled a second time.
                    local_dir=cache_root, force_download=True,
                )
                if downloaded is None or not is_valid_parquet_file(cached):
                    raise RuntimeError("Invalid cached raw Parquet: %s" % filename)
                candidates, selected = _sample_chronos_file(
                    cached, destination, config_name, probability, seed
                )
                discard_cache_file(cached, cache_root)
                log.info("Sampled Chronos %s: retained %d/%d trajectories", filename, selected, candidates)
                return filename, selected
            counts = dict(parallel_map(sample_one, filenames, num_workers))
            _write_sampled_marker(
                marker, _CHRONOS_HF_DATASET, filenames, probability, seed, counts
            )
            completed.append(config_name)
        except Exception as exc:
            failed.append(config_name)
            log.warning("Sampled Chronos config '%s' failed: %s", config_name, exc)
    remove_empty_cache(cache_root)
    failures.log(log)
    _write_fetch_manifest(raw_root, completed)
    if failed:
        return 1
    return 0


def _sampled_marker_matches(raw_root, marker, probability, seed):
    """Return whether a sampled config marker matches the requested run."""
    try:
        payload = json.loads(Path(marker).read_text(encoding="ascii"))
        sampling = payload["sampling"]
        return (
            float(sampling["probability"]) == float(probability)
            and int(sampling["seed"]) == int(seed)
            and all(int(count) > 0 for count in payload["valid_trajectories"].values())
            and all((Path(raw_root) / str(name)).is_file() for name in payload["files"])
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _write_sampled_marker(marker, dataset, filenames, probability, seed, valid_trajectories):
    """Atomically record sampled raw files and their explicit sampling policy."""
    marker = Path(marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    partial = marker.with_name(marker.name + ".partial")
    partial.write_text(json.dumps({
        "dataset": dataset,
        "files": filenames,
        "sampling": {"probability": probability, "seed": int(seed)},
        "valid_trajectories": valid_trajectories,
    }, sort_keys=True) + "\n", encoding="ascii")
    partial.replace(marker)


def fetch(output_dir, subsets=None, force=False, sample_probability=None, seed=42,
          revision=None, num_workers=1):
    """Fetch Chronos leaves through the common source-adapter entry point.

    @input: output_dir is the raw root; subsets is None, explicit config names,
            or ``["all"]``; revision must be None; remaining arguments control
            raw refresh and optional trajectory sampling.
    @output: Integer Fetch exit status.
    @scenario: Give Chronos the same adapter boundary as every active source.
    @logic: Rejects an unsupported revision, resolves the explicit all token to
            all supported configs, then delegates format-specific work to fetch_chronos.
    """
    if revision not in (None, SOURCE_REVISIONS["chronos"]):
        raise ValueError("Chronos revision must be %s" % SOURCE_REVISIONS["chronos"])
    if subsets == ["all"]:
        from assets.dataset_processing.configs.chronos import CHRONOS_SUPPORTED_SUBSETS
        subsets = list(CHRONOS_SUPPORTED_SUBSETS)
    return fetch_chronos(
        output_dir, force=force, subsets=subsets,
        sample_probability=sample_probability, seed=seed, num_workers=num_workers,
    )
