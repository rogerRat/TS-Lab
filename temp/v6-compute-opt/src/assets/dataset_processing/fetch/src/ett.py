"""Explicit-only legacy ETT raw CSV acquisition."""

import json
import logging
from pathlib import Path

from assets.dataset_processing.configs.ett import ETT_FILES, ETT_REPO_URL
from .sampling import (
    FileTrajectorySampler, atomic_replace, cache_path, discard_cache_file,
    file_rng, parallel_map, remove_empty_cache, validate_sample_probability,
)
from .utils import DownloadFailureStack, RetryingHuggingFaceDownloader

log = logging.getLogger("fetch")


def _marker(output_dir, filename):
    """Return the explicit sampled ETT completion marker path.

    @input: output_dir is the ETT raw root; filename is one CSV name.
    @output: Path under ``.complete``.
    @scenario: Make sampled legacy files resumable only by matching policy.
    @logic: Preserves the source filename in a JSON marker name.
    """
    return Path(output_dir) / ".complete" / (filename + ".json")


def _sampled_marker_matches(output_dir, filename, probability, seed):
    """Return whether a sampled ETT file satisfies the explicit request.

    @input: raw root, filename, probability, and seed identify the request.
    @output: bool.
    @scenario: Avoid reusing an incompatible sampled compatibility artifact.
    @logic: Validates marker fields, positive retained count, and published CSV presence.
    """
    try:
        payload = json.loads(_marker(output_dir, filename).read_text(encoding="ascii"))
        return (payload["file"] == filename and
                payload["sampling"] == {"probability": probability, "seed": int(seed)} and
                int(payload["valid_trajectories"]) > 0 and
                (Path(output_dir) / filename).is_file())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _sample_ett_file(source_path, destination_path, probability, seed, subset):
    """Sample ETT as one complete multivariate CSV trajectory.

    @input: Cached source/destination paths, Bernoulli probability, seed, and subset label.
    @output: Candidate and retained counts.
    @scenario: Preserve date and synchronized columns together for legacy ETT sampling.
    @logic: Validates one numeric value column of length at least ten and publishes the whole CSV.
    """
    import csv

    with Path(source_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 11 or len(rows[0]) < 2:
        raise ValueError("ETT source contains no valid trajectory with length at least 10: %s" % source_path)
    sampler = FileTrajectorySampler(probability, file_rng(seed, str(destination_path)))
    if not sampler.keep():
        raise ValueError("ETT sampling selected no trajectory: %s" % source_path)
    partial = Path(destination_path).with_name(Path(destination_path).name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    atomic_replace(partial, Path(destination_path))
    return sampler.candidates, sampler.selected


def fetch(output_dir, subsets=None, force=False, sample_probability=None, seed=42,
          revision=None, num_workers=1):
    """Fetch legacy ETT only when users explicitly select ``--source ett``.

    @input: Standard source-adapter arguments; subsets is None or ETT subset names; revision is unsupported.
    @output: 0 when all requested files complete, otherwise 1.
    @scenario: Preserve explicit historical compatibility while excluding ETT from all defaults.
    @logic: Downloads source CSVs directly or cache-samples whole files, then writes matching markers.
    """
    if revision is not None:
        raise ValueError("ETT does not accept an explicit revision")
    probability = validate_sample_probability(sample_probability)
    filenames = list(ETT_FILES if subsets in (None, ["all"]) else tuple(name + ".csv" if not name.endswith(".csv") else name for name in subsets))
    invalid = [name for name in filenames if name not in ETT_FILES]
    if invalid:
        raise ValueError("Unsupported ETT subset: %s" % ", ".join(invalid))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cache_root = root / ".cache"
    failures = DownloadFailureStack()
    downloader = RetryingHuggingFaceDownloader(failure_stack=failures)
    def fetch_one(filename):
        destination = root / filename
        if probability is None:
            if destination.is_file() and not force:
                return True
            return downloader.download_url(ETT_REPO_URL + "/" + filename, destination) is not None
        if not force and _sampled_marker_matches(root, filename, probability, seed):
            return True
        cached = cache_path(cache_root, filename)
        if downloader.download_url(ETT_REPO_URL + "/" + filename, cached) is None:
            return False
        try:
            _candidates, selected = _sample_ett_file(cached, destination, probability, seed, destination.stem)
            marker = _marker(root, filename)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.with_name(marker.name + ".partial").write_text(json.dumps({"file": filename, "sampling": {"probability": probability, "seed": int(seed)}, "valid_trajectories": selected}, sort_keys=True) + "\n", encoding="ascii")
            marker.with_name(marker.name + ".partial").replace(marker)
            return True
        finally:
            discard_cache_file(cached, cache_root)
    success = all(parallel_map(fetch_one, filenames, num_workers))
    remove_empty_cache(cache_root)
    failures.log(log)
    return 0 if success else 1
