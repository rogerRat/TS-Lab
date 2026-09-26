"""Dataset Fetch -- UTSD raw HuggingFace Arrow acquisition.

@input: UTSD configuration and destination directory.
@output: Original Arrow files under <config>/ with a completion marker.
@scenario: Acquire UTSD source files without streaming records or converting data.
"""

import json
import logging
import math
from pathlib import Path

from assets.dataset_processing.configs.utsd import UTSD_DEFAULT_CONFIG, UTSD_HF_DATASET

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

log = logging.getLogger("fetch")

_UTSD_HF_DATASET = UTSD_HF_DATASET


def _sample_utsd_file(source_path, destination_path, probability, seed):
    """Sample complete UTSD Arrow records while preserving Arrow IPC format.

    @input: source_path is one cached Arrow shard; destination_path is the same
            raw-tree path; probability is in (0, 1]; seed is deterministic.
    @output: Source-record and retained-record counts.
    @scenario: Publish one trajectory-sampled UTSD shard without canonical conversion.
            One Arrow record is the minimum unit.  A record containing multiple
            synchronized variables is retained as one complete record; fields
            are never sampled independently.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc
    from assets.dataset_processing.series import split_univariate_series

    source = pa.memory_map(str(source_path), "r")
    reader = None
    mode = "stream"
    try:
        try:
            reader = ipc.open_stream(source)
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.open_file(source)
            mode = "file"
        schema = reader.schema
        partial = destination_path.with_name(destination_path.name + ".partial")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        partial.unlink(missing_ok=True)
        sink = pa.OSFile(str(partial), "wb")
        writer = ipc.new_file(sink, schema) if mode == "file" else ipc.new_stream(sink, schema)
        sampler = FileTrajectorySampler(
            probability, file_rng(seed, str(destination_path))
        )
        source_ordinal = 0
        try:
            batches = reader if mode == "stream" else (
                reader.get_batch(index) for index in range(reader.num_record_batches)
            )
            for batch in batches:
                # Do not select target/metadata fields independently: one full
                # source record, including all synchronized variables, survives.
                selected = []
                for row in batch.to_pylist():
                    values = row.get("target")
                    source_id = row.get("item_id", "%s_%d" % (destination_path.stem, source_ordinal))
                    source_ordinal += 1
                    split_records, skipped = split_univariate_series(
                        source_id, values, preserve_invalid_values=True
                    )
                    if skipped is not None or not any(
                            len(series) >= 10 for _series_id, series in split_records
                    ):
                        continue
                    if sampler.keep():
                        selected.append(row)
                if selected:
                    writer.write_batch(pa.RecordBatch.from_pylist(selected, schema=schema))
            if sampler.candidates == 0:
                raise ValueError(
                    "UTSD source file contains no valid trajectory with length at least 10: %s"
                    % source_path
                )
        finally:
            writer.close()
            sink.close()
        atomic_replace(partial, destination_path)
        return sampler.candidates, sampler.selected
    finally:
        source.close()


def _raw_marker(raw_root, config):
    """Return the UTSD raw completion marker for one configuration.

    @input: raw_root is the raw destination; config is a UTSD configuration name.
    @output: Path to the config completion marker.
    @scenario: Prevents Loader from reading an incomplete Arrow shard collection.
    """
    return Path(raw_root) / config / ".complete"


def _has_sampled_marker(raw_root, config):
    """Return whether a UTSD marker describes sampled content."""
    try:
        return "sampling" in json.loads(_raw_marker(raw_root, config).read_text(encoding="ascii"))
    except (OSError, ValueError, TypeError):
        return False


def _completed_files(raw_root, config):
    """Return completed original UTSD Arrow paths, otherwise None.

    @input: raw_root and config identify one local raw UTSD collection.
    @output: list[Path] or None.
    @scenario: Provides idempotent raw download resumption without decoding Arrow rows.
    """
    marker = _raw_marker(raw_root, config)
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="ascii"))
        if "sampling" in payload:
            return None
        files = payload["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    paths = [Path(raw_root) / str(name) for name in files] if isinstance(files, list) else []
    return paths if paths and all(path.is_file() and path.stat().st_size > 0 for path in paths) else None


def _list_arrow_files(config):
    """List exact repository-relative Arrow source paths for one UTSD config.

    @input: config is one UTSD HuggingFace configuration name.
    @output: Sorted list of original Arrow filenames.
    @scenario: Retains every upstream shard name during fetch-only acquisition.
    """
    from huggingface_hub import list_repo_tree

    files = sorted(
        item.path for item in list_repo_tree(
            _UTSD_HF_DATASET, repo_type="dataset", path_in_repo=config, recursive=True
        ) if item.path.endswith(".arrow")
    )
    if not files:
        raise ValueError("UTSD config has no Arrow files: %s" % config)
    return files


def fetch(output_dir, subsets=None, force=False, sample_probability=None, seed=42,
          revision=None, num_workers=1):
    """Fetch UTSD config leaves through the common source-adapter entry point.

    @input: output_dir is raw output; subsets is None, one config, or ["all"];
            revision must be None; remaining arguments control refresh/sampling.
    @output: Integer Fetch exit status.
    @scenario: Keep UTSD config resolution and Arrow handling in its adapter.
    @logic: Resolves the default/all/explicit config list and runs each config in order.
    """
    if revision not in (None, SOURCE_REVISIONS["utsd"]):
        raise ValueError("UTSD revision must be %s" % SOURCE_REVISIONS["utsd"])
    if subsets is None:
        configs = [UTSD_DEFAULT_CONFIG]
    elif subsets == ["all"]:
        from assets.dataset_processing.configs.utsd import UTSD_CONFIGS
        configs = list(UTSD_CONFIGS)
    else:
        configs = list(subsets)
    invalid = [config for config in configs if config not in {
        "UTSD-1G", "UTSD-2G", "UTSD-4G", "UTSD-12G"
    }]
    if invalid:
        raise ValueError("Unsupported UTSD config: %s" % ", ".join(invalid))
    return max(
        fetch_utsd(output_dir, config=config, force=force,
                   sample_probability=sample_probability, seed=seed,
                   num_workers=num_workers)
        for config in configs
    )


def fetch_utsd(output_dir, config=UTSD_DEFAULT_CONFIG, max_series=float("inf"), seed=42,
               force=False, streaming=False, delete_sources=False,
               sample_probability=None, num_workers=1):
    """Download original UTSD Arrow shards without record conversion.

    @input: output_dir is data/utsd; config selects a Hub configuration; max_series
            must be infinity; seed is unused; force refreshes source files.
    @output: 0 when all raw files are complete, otherwise 1.
    @scenario: Moves UTSD parsing, filtering, and sampling exclusively to Loader.
    """
    sample_probability = validate_sample_probability(sample_probability)
    if sample_probability is not None:
        if not math.isinf(max_series) or streaming or delete_sources:
            log.error("UTSD sampling mode cannot combine with legacy sampling flags.")
            return 1
        return _fetch_utsd_sampled(output_dir, config, force, sample_probability, seed, num_workers)
    if not math.isinf(max_series):
        log.error("UTSD fetch downloads complete raw files; use load --max-series for sampling.")
        return 1
    if streaming:
        log.error("UTSD raw fetch does not support streaming conversion.")
        return 1
    if delete_sources:
        log.error("UTSD raw files are Loader inputs and cannot be deleted by --del-src.")
        return 1
    if seed != 42:
        log.warning("UTSD raw fetch ignores seed; use load --seed for sampling.")
    output_dir = Path(output_dir)
    raw_root = output_dir
    raw_root.mkdir(parents=True, exist_ok=True)
    existing = None if force else _completed_files(raw_root, config)
    if existing is not None:
        log.info("UTSD raw config '%s' is complete (%d file(s)); skipping.", config, len(existing))
        return 0
    failures = DownloadFailureStack()
    downloader = RetryingHuggingFaceDownloader(failure_stack=failures)
    try:
        files = _list_arrow_files(config)
        log.info("Downloading UTSD raw config '%s' (%d Arrow file(s))...", config, len(files))
        def download_one(filename):
            downloaded = downloader.download_file(
                _UTSD_HF_DATASET, filename, repo_type="dataset", local_dir=raw_root,
                force_download=force or _has_sampled_marker(raw_root, config),
            )
            local_path = raw_root / filename
            if downloaded is None or not local_path.is_file() or local_path.stat().st_size == 0:
                raise RuntimeError("Invalid raw UTSD file after download: %s" % filename)
            return filename
        parallel_map(download_one, files, num_workers)
        marker = _raw_marker(raw_root, config)
        partial = marker.with_name(marker.name + ".partial")
        partial.write_text(json.dumps({"dataset": _UTSD_HF_DATASET, "files": files}) + "\n", encoding="ascii")
        partial.replace(marker)
    except Exception as exc:
        log.warning("UTSD raw fetch failed for '%s': %s", config, exc)
        failures.log(log)
        return 1
    failures.log(log)
    log.info("UTSD raw files are available in %s.", raw_root)
    return 0


def _fetch_utsd_sampled(output_dir, config, force, probability, seed, num_workers):
    """Download UTSD shards one at a time into raw/.cache and sample records."""
    raw_root = Path(output_dir)
    cache_root = raw_root / ".cache"
    raw_root.mkdir(parents=True, exist_ok=True)
    failures = DownloadFailureStack()
    downloader = RetryingHuggingFaceDownloader(failure_stack=failures)
    marker = _raw_marker(raw_root, config)
    try:
        payload = json.loads(marker.read_text(encoding="ascii"))
        sampling = payload["sampling"]
        if (
            not force
            and float(sampling["probability"]) == float(probability)
            and int(sampling["seed"]) == int(seed)
            and all(int(count) > 0 for count in payload["valid_trajectories"].values())
            and all((raw_root / str(name)).is_file() for name in payload["files"])
        ):
            log.info("UTSD sampled config '%s' is complete; skipping.", config)
            return 0
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        files = _list_arrow_files(config)
        def sample_one(filename):
            relative = str(Path(filename))
            cached = cache_path(cache_root, relative)
            destination = raw_root / relative
            downloaded = downloader.download_file(
                _UTSD_HF_DATASET, filename, repo_type="dataset",
                # Never reuse a possibly published sampled shard as the source.
                local_dir=cache_root, force_download=True,
            )
            if downloaded is None or not cached.is_file() or cached.stat().st_size == 0:
                raise RuntimeError("Invalid cached raw UTSD file: %s" % filename)
            candidates, selected = _sample_utsd_file(cached, destination, probability, seed)
            discard_cache_file(cached, cache_root)
            log.info("Sampled UTSD %s: retained %d/%d trajectories", filename, selected, candidates)
            return filename, selected
        counts = dict(parallel_map(sample_one, files, num_workers))
        marker.parent.mkdir(parents=True, exist_ok=True)
        partial = marker.with_name(marker.name + ".partial")
        partial.write_text(json.dumps({
            "dataset": _UTSD_HF_DATASET,
            "files": files,
            "sampling": {"probability": probability, "seed": int(seed)},
            "valid_trajectories": counts,
        }, sort_keys=True) + "\n", encoding="ascii")
        partial.replace(marker)
    except Exception as exc:
        log.warning("UTSD sampled config '%s' failed: %s", config, exc)
        failures.log(log)
        return 1
    remove_empty_cache(cache_root)
    failures.log(log)
    return 0
