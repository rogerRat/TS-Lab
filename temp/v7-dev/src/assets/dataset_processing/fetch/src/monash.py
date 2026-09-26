"""Dataset Fetch -- Monash raw HuggingFace zip acquisition.

@input: Monash configuration names and destination directory.
@output: Original Hub zip files using their repository-relative names.
@scenario: Acquire source archives without extracting TSF records or writing Parquet.
"""

import json
import logging
import math
import zipfile
from pathlib import Path

from assets.dataset_processing.configs.monash import (
    MONASH_DEFAULT_CONFIGS, MONASH_HF_DATASET, MONASH_HF_CONFIGS,
    MONASH_HF_ZIP_MAP, MONASH_NAME_MAP,
)
from .utils import DownloadFailureStack, RetryingHuggingFaceDownloader
from .sources import SOURCE_REVISIONS
from .sampling import (
    FileTrajectorySampler,
    atomic_replace,
    cache_path,
    discard_cache_file,
    file_rng,
    parallel_map,
    remove_empty_cache,
    validate_sample_probability,
)

log = logging.getLogger("fetch")


def _resolve_monash_subsets(subset_str):
    """Resolve explicit Monash names to HuggingFace configuration names.

    @input: subset_str is a comma-separated name list or ``all``.
    @output: List of known HuggingFace configuration names.
    @scenario: Retains the documented Monash short-name aliases at the CLI boundary.
    """
    if subset_str.strip().lower() == "all":
        return list(MONASH_DEFAULT_CONFIGS)
    resolved = []
    for name in (part.strip() for part in subset_str.split(",")):
        if not name:
            return None
        config = MONASH_NAME_MAP.get(name, name)
        if config not in MONASH_HF_CONFIGS:
            log.warning("Unknown Monash subset: '%s' -- skipping", name)
            return None
        resolved.append(config)
    return resolved


def fetch(output_dir, subsets=None, force=False, sample_probability=None, seed=42,
          revision=None, num_workers=1):
    """Fetch Monash leaves through the common source-adapter entry point.

    @input: output_dir is raw output; subsets is None, config names, or ["all"];
            revision must be None; remaining parameters control refresh/sampling.
    @output: Integer Fetch exit status.
    @scenario: Keep Monash selection rules inside its source adapter, not runner.
    @logic: Rejects revisions, resolves aliases and all, then calls fetch_monash.
    """
    if revision not in (None, SOURCE_REVISIONS["monash"]):
        raise ValueError("Monash revision must be %s" % SOURCE_REVISIONS["monash"])
    if subsets is None:
        resolved = None
    elif subsets == ["all"]:
        resolved = list(MONASH_DEFAULT_CONFIGS)
    else:
        resolved = []
        for name in subsets:
            config = MONASH_NAME_MAP.get(name, name)
            if config not in MONASH_HF_CONFIGS:
                raise ValueError("Unknown Monash subset: %s" % name)
            resolved.append(config)
    return fetch_monash(
        output_dir, force=force, subsets=resolved,
        sample_probability=sample_probability, seed=seed, num_workers=num_workers,
    )


def _marker(raw_root, config):
    """Return one Monash raw-config completion marker path.

    @input: raw_root is the local raw root; config is a Monash config name.
    @output: Path below .complete/.
    @scenario: Tracks config completion without altering the upstream zip filename.
    """
    return Path(raw_root) / ".complete" / (config + ".json")


def _has_sampled_marker(raw_root, config):
    """Return whether a Monash marker describes sampled content."""
    try:
        return "sampling" in json.loads(_marker(raw_root, config).read_text(encoding="ascii"))
    except (OSError, ValueError, TypeError):
        return False


def _completed_zip(raw_root, config):
    """Return a valid original Monash zip named by a completion marker.

    @input: raw_root and config identify one local source archive.
    @output: Path to a valid zip, or None.
    @scenario: Lets fetch resume while Loader never sees partial source archives.
    """
    marker = _marker(raw_root, config)
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="ascii"))
        if "sampling" in payload:
            return None
        filename = payload["file"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    path = Path(raw_root) / str(filename)
    return path if path.is_file() and zipfile.is_zipfile(path) else None


def _sample_monash_zip(source_path, destination_path, probability, seed, config):
    """Sample TSF records inside a Monash archive and preserve archive members.

    @input: source_path is one cached Monash zip; destination_path is its original
            raw path; probability and seed control complete TSF-record sampling.
    @output: Number of candidate and retained TSF trajectory records.
    @scenario: Publish a structurally compatible sampled archive without conversion.
            A complete source record/line is the minimum unit.  If a source
            format encodes synchronized variables in one record, that complete
            record is copied; this function never samples variable fields.
    """
    from assets.dataset_processing.tsf_utils import parse_tsf_record
    from assets.dataset_processing.series import split_univariate_series

    sampler = FileTrajectorySampler(
        probability, file_rng(seed, str(destination_path))
    )
    partial = destination_path.with_name(destination_path.name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(source_path, "r") as source, zipfile.ZipFile(
            partial, "w", compression=zipfile.ZIP_DEFLATED
        ) as output:
            tsf_names = [name for name in source.namelist() if name.lower().endswith(".tsf")]
            if not tsf_names:
                raise ValueError("Monash archive has no TSF file: %s" % source_path)
            for info in source.infolist():
                data = source.read(info.filename)
                if not info.filename.lower().endswith(".tsf"):
                    output.writestr(info, data)
                    continue
                kept_lines = []
                source_ordinal = 0
                for raw_line in data.splitlines(keepends=True):
                    text = raw_line.decode("utf-8", errors="replace")
                    stripped = text.strip()
                    if not stripped or stripped.startswith("@") or stripped.startswith("#"):
                        kept_lines.append(raw_line)
                        continue
                    source_id, values = parse_tsf_record(stripped, config, source_ordinal)
                    source_ordinal += 1
                    split_records, skipped = split_univariate_series(
                        source_id, values, preserve_invalid_values=True
                    )
                    if skipped is not None or not any(
                            len(series) >= 10 for _series_id, series in split_records
                    ):
                        continue
                    # Copy the complete source record.  Never split a multivariate
                    # record into independently sampled variable fields.
                    if sampler.keep():
                        kept_lines.append(raw_line)
                output.writestr(info, b"".join(kept_lines))
            if sampler.candidates == 0:
                raise ValueError(
                    "Monash archive contains no valid TSF trajectory with length at least 10: %s"
                    % source_path
                )
        atomic_replace(partial, destination_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return sampler.candidates, sampler.selected


def fetch_monash(output_dir, max_series=float("inf"), seed=42, force=False,
                 subsets=None, streaming=False, delete_sources=False, parallel=False,
                 sample_probability=None, num_workers=1):
    """Download original Monash zip archives without parsing TSF records.

    @input: output_dir is data/monash; max_series must be infinity; seed is unused;
            force refreshes selected source files; subsets selects Hub configs.
    @output: 0 when selected archives are valid and marked complete, otherwise 1.
    @scenario: Separates source acquisition from Loader-owned TSF conversion.
    """
    sample_probability = validate_sample_probability(sample_probability)
    if sample_probability is not None:
        if not math.isinf(max_series) or streaming or delete_sources:
            log.error("Monash sampling mode cannot combine with legacy sampling flags.")
            return 1
        return _fetch_monash_sampled(output_dir, force, subsets, sample_probability, seed, num_workers)
    if not math.isinf(max_series):
        log.error("Monash fetch downloads complete raw archives; use load --max-series for sampling.")
        return 1
    if streaming:
        log.error("Monash raw fetch does not support streaming conversion.")
        return 1
    if delete_sources:
        log.error("Monash raw archives are Loader inputs and cannot be deleted by --del-src.")
        return 1
    if seed != 42:
        log.warning("Monash raw fetch ignores seed; use load --seed for sampling.")
    configs = list(MONASH_DEFAULT_CONFIGS if subsets is None else subsets)
    output_dir = Path(output_dir)
    raw_root = output_dir
    raw_root.mkdir(parents=True, exist_ok=True)
    failures = DownloadFailureStack()
    downloader = RetryingHuggingFaceDownloader(failure_stack=failures)
    failed = []

    def fetch_one(config):
        existing = None if force else _completed_zip(raw_root, config)
        if existing is not None:
            log.info("Monash raw config '%s' is complete; skipping.", config)
            return None
        zip_stem = MONASH_HF_ZIP_MAP.get(config)
        if zip_stem is None:
            log.warning("No source zip mapping for Monash config '%s'.", config)
            return config
        filename = "data/%s.zip" % zip_stem
        log.info("Downloading Monash raw config '%s' (%s)...", config, filename)
        downloaded = downloader.download_file(
            MONASH_HF_DATASET, filename, repo_type="dataset", local_dir=raw_root,
            force_download=force or _has_sampled_marker(raw_root, config),
        )
        local_path = raw_root / filename
        if downloaded is None or not zipfile.is_zipfile(local_path):
            return config
        marker = _marker(raw_root, config)
        marker.parent.mkdir(parents=True, exist_ok=True)
        partial = marker.with_name(marker.name + ".partial")
        partial.write_text(json.dumps({"dataset": MONASH_HF_DATASET, "file": filename}) + "\n", encoding="ascii")
        partial.replace(marker)
        log.info("Monash raw config '%s' is complete.", config)
        return None

    failed.extend(item for item in parallel_map(fetch_one, configs, num_workers) if item)
    failures.log(log)
    if failed:
        log.warning("Monash raw fetch incomplete; failed configs: %s", failed)
        return 1
    manifest = output_dir / "fetch_manifest.json"
    manifest.write_text(json.dumps({"dataset": MONASH_HF_DATASET, "layout": "raw_huggingface_zip", "configs": configs}, indent=2) + "\n", encoding="ascii")
    log.info("Monash raw fetch complete; source archives are available in %s.", raw_root)
    return 0


def _fetch_monash_sampled(output_dir, force, subsets, probability, seed, num_workers):
    """Download Monash archives to raw/.cache, sample TSF records, and publish."""
    configs = list(MONASH_DEFAULT_CONFIGS if subsets is None else subsets)
    raw_root = Path(output_dir)
    cache_root = raw_root / ".cache"
    raw_root.mkdir(parents=True, exist_ok=True)
    failures = DownloadFailureStack()
    downloader = RetryingHuggingFaceDownloader(failure_stack=failures)
    failed = []

    def sample_one(config):
        zip_stem = MONASH_HF_ZIP_MAP.get(config)
        if zip_stem is None:
            log.warning("No source zip mapping for Monash config '%s'.", config)
            return config
        filename = "data/%s.zip" % zip_stem
        destination = raw_root / filename
        marker = _marker(raw_root, config)
        try:
            if not force:
                payload = json.loads(marker.read_text(encoding="ascii"))
                sampling = payload["sampling"]
                if (
                    float(sampling["probability"]) == float(probability)
                    and int(sampling["seed"]) == int(seed)
                    and int(payload["valid_trajectories"]) > 0
                    and destination.is_file() and zipfile.is_zipfile(destination)
                ):
                    log.info("Monash sampled config '%s' is complete; skipping.", config)
                    return None
        except (OSError, ValueError, KeyError, TypeError):
            pass
        cached = cache_path(cache_root, filename)
        log.info("Downloading Monash raw config '%s' to cache ...", config)
        downloaded = downloader.download_file(
            MONASH_HF_DATASET, filename, repo_type="dataset", local_dir=cache_root,
            # Never reuse a possibly published sampled archive as the source.
            force_download=True,
        )
        if downloaded is None or not zipfile.is_zipfile(cached):
            return config
        try:
            candidates, selected = _sample_monash_zip(
                cached, destination, probability, seed, config
            )
            discard_cache_file(cached, cache_root)
            marker.parent.mkdir(parents=True, exist_ok=True)
            partial = marker.with_name(marker.name + ".partial")
            partial.write_text(json.dumps({
                "dataset": MONASH_HF_DATASET,
                "file": filename,
                "sampling": {"probability": probability, "seed": int(seed)},
                "valid_trajectories": selected,
            }, sort_keys=True) + "\n", encoding="ascii")
            partial.replace(marker)
            log.info("Sampled Monash %s: retained %d/%d trajectories", filename, selected, candidates)
        except Exception as exc:
            log.warning("Failed to sample Monash config '%s': %s", config, exc)
            return config
        return None

    failed.extend(item for item in parallel_map(sample_one, configs, num_workers) if item)
    remove_empty_cache(cache_root)
    failures.log(log)
    if failed:
        log.warning("Monash sampled fetch incomplete; failed configs: %s", failed)
        return 1
    return 0
