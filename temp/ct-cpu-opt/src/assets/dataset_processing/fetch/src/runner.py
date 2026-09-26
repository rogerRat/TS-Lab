"""
Dataset Sources Stage 1 -- Dataset Fetcher (Download).

Fetches raw datasets from official sources into the local data directory.
Uses source-owned raw acquisition adapters.

@input:  CLI arguments (--source, --output-dir, --subsets, --force)
@output: Raw data files under data/src/raw/<source>/ (parquet for HF sources, csv/dir for others)
@scenario: One-shot data acquisition step before loading/feature extraction.

Idempotent: checks for existing files and skips download unless --force is set.

CLI Usage:
    # Monash -- stable archives without declared missing values
    python -m assets.dataset_processing.fetch --source monash

    # Monash -- specific configs (supports old short names)
    python -m assets.dataset_processing.fetch --source monash --subsets weather,traffic_hourly

    # Monash -- old short names auto-mapped to HF configs
    python -m assets.dataset_processing.fetch --source monash --subsets electricity,traffic

    # Chronos (original HuggingFace Parquet files, no conversion)
    python -m assets.dataset_processing.fetch --source chronos

    # UTSD original Arrow shards
    python -m assets.dataset_processing.fetch --source utsd

    # Force re-download
    python -m assets.dataset_processing.fetch --source monash --force
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

# -- Project root for imports ------------------------------------------------
from configs.paths import PROJECT_ROOT as _PROJECT_ROOT
sys.path.insert(0, str(_PROJECT_ROOT))

from configs.logger import configure_logger  # noqa: E402
from configs.paths import output_path  # noqa: E402
from assets.dataset_processing.configs.config import DEPRECATED_SOURCES  # noqa: E402
from assets.dataset_processing.configs.profiles import (  # noqa: E402
    PROFILE_SCHEMA_VERSION, PROFILE_SPECS,
)
from assets.dataset_processing.configs.utsd import UTSD_CONFIGS  # noqa: E402
from assets.dataset_processing.fetch.src.sources import (  # noqa: E402
    FETCH_SOURCE_NAMES, SOURCE_REVISIONS,
)
from assets.dataset_processing.progress import ProcessingProgress  # noqa: E402

FETCH_SOURCE_CHOICES = FETCH_SOURCE_NAMES
FETCH_NON_DEPRECATED_SOURCES = tuple(
    source for source in FETCH_SOURCE_CHOICES if source not in DEPRECATED_SOURCES
)

# ===========================================================================
# Logging Setup
# ===========================================================================

log = logging.getLogger("fetch")

_PROFILE_SOURCE_ORDER = (
    "chronos", "gifteval", "lotsa", "monash", "pile", "quito", "rmisc",
    "time300b", "utsd",
)


# ===========================================================================
# Shared Utilities
# ===========================================================================

def _positive_int(value):
    """Parse a positive worker count for Fetch.

    @input: value is a CLI string containing an integer greater than zero.
    @output: Positive int, or argparse.ArgumentTypeError on invalid input.
    @scenario: Rejects invalid parallelism before any Fetch side effects occur.
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def parse_args(argv=None):
    """Parse command-line arguments for the fetch stage."""
    parser = argparse.ArgumentParser(
        description="Dataset Sources Stage 1 -- Dataset Fetcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --source monash\n"
            "  %(prog)s --source monash --subsets weather,traffic_hourly\n"
            "  %(prog)s --source chronos\n"
            "  %(prog)s --source chronos --subsets exchange_rate,m4_hourly\n"
            "  %(prog)s --source utsd"
        ),
    )
    parser.add_argument(
        "--source",
        choices=["all"] + list(FETCH_SOURCE_CHOICES),
        help=(
            "Dataset source to download. ``all`` or omission downloads every "
            "supported source."
        ),
    )
    parser.add_argument(
        "--profile", choices=tuple(PROFILE_SPECS), default=None,
        help="Frozen cross-source download Profile.",
    )
    parser.add_argument(
        "--output-dir",
        help="Target directory for downloaded raw data. Default: data/src/raw/<source>.",
    )
    parser.add_argument(
        "--subsets",
        help=(
            "Comma-separated subset names, or ``all`` for stable default subsets. "
            "Applies to Monash (HF config names; old short names like 'electricity' "
            "are auto-mapped) and Chronos (HF config names). For extended sources, "
            "``all`` recursively scans that source repository and is never implied "
            "by ``--source all``. Default: quality-filtered configs for Monash; "
            "representative subsets for Chronos; one bounded selection per extended source."
        ),
    )
    parser.add_argument(
        "--utsd-config", type=str, default=None,
        choices=["UTSD-1G", "UTSD-2G", "UTSD-4G", "UTSD-12G"],
        help="UTSD source collection. Smaller collections download fewer raw Arrow shards. Default: UTSD-1G.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if target files already exist.",
    )
    parser.add_argument(
        "--sample-probability", type=float, default=None,
        help=("DEPRECATED: trajectory-level Bernoulli sampling probability. "
              "Retained only for legacy profiles and compatibility; omit it "
              "to fetch complete selected subsets."),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Deterministic seed for explicit fetch-stage trajectory sampling.",
    )
    parser.add_argument(
        "--revision", default=None,
        help="Optional Hugging Face revision for extended sources.",
    )
    parser.add_argument(
        "--num-workers", type=_positive_int, default=4,
        help="Number of parallel Fetch download/parse workers. Default: 4.",
    )
    return parser.parse_args(argv)


def _validate_profile_spec(profile, spec):
    """Validate one frozen Profile mapping before creating its output root.

    @input: profile is one of the CLI Profile names; spec is its mapping from
            PROFILE_SPECS and contains estimate fields plus source mappings.
    @output: None on success; raises ValueError for a malformed development
             mapping.
    @scenario: Prevents a Profile typo from falling back to source defaults or
               an implicit repository-wide selection.
    @param profile: Profile name used in validation errors.
    @param spec: Frozen Profile mapping to validate.
    @logic: Checks the top-level schema, source order, allowed fields, literal
            subset ordering, fixed revisions, UTSD-only configuration, and
            finite shared sampling probability.
    """
    if PROFILE_SCHEMA_VERSION <= 0:
        raise ValueError("Profile schema version must be positive")
    if not isinstance(spec, dict):
        raise ValueError("Profile %s must be a mapping" % profile)
    if set(spec) != {
        "estimate_method", "estimated_observations", "sample_probability", "sources",
    }:
        raise ValueError("Profile %s has invalid top-level fields" % profile)
    probability = spec["sample_probability"]
    if probability is not None and (
            not isinstance(probability, (int, float)) or not math.isfinite(probability)
    ):
        raise ValueError("Profile %s probability must be finite" % profile)
    if probability is not None and (probability <= 0 or probability > 1):
        raise ValueError("Profile %s probability must be in (0, 1]" % profile)
    if not isinstance(spec["estimated_observations"], int) or spec["estimated_observations"] <= 0:
        raise ValueError("Profile %s estimated_observations must be positive" % profile)
    if not isinstance(spec["estimate_method"], str):
        raise ValueError("Profile %s estimate_method must be a string" % profile)
    sources = spec["sources"]
    if not isinstance(sources, tuple) or not sources:
        raise ValueError("Profile %s sources must be a non-empty tuple" % profile)
    if any(not isinstance(item, dict) for item in sources):
        raise ValueError("Profile %s has a non-mapping source" % profile)
    source_names = tuple(item.get("source") for item in sources)
    expected_order = tuple(
        source for source in _PROFILE_SOURCE_ORDER if source in source_names
    )
    if source_names != expected_order:
        raise ValueError("Profile %s sources are not in the frozen order" % profile)

    allowed_sources = set(FETCH_NON_DEPRECATED_SOURCES)
    for source_spec in sources:
        source = source_spec.get("source")
        if source not in allowed_sources or source == "all":
            raise ValueError("Profile %s has invalid source %r" % (profile, source))
        if source == "utsd":
            expected_fields = {"source", "revision", "utsd_config"}
            if set(source_spec) != expected_fields:
                raise ValueError("Profile %s UTSD mapping has invalid fields" % profile)
            if source_spec["utsd_config"] not in UTSD_CONFIGS:
                raise ValueError("Profile %s has invalid UTSD config" % profile)
        else:
            expected_fields = {"source", "revision", "subsets"}
            if set(source_spec) != expected_fields:
                raise ValueError("Profile %s source %s has invalid revision fields" % (profile, source))
        if source_spec["revision"] != SOURCE_REVISIONS[source]:
            raise ValueError("Profile %s source %s revision must equal %s" % (
                profile, source, SOURCE_REVISIONS[source],
            ))
        if source != "utsd":
            subsets = source_spec["subsets"]
            if not isinstance(subsets, tuple) or not subsets:
                raise ValueError("Profile %s source %s subsets must be a non-empty tuple" % (profile, source))
            if subsets != tuple(sorted(set(subsets))):
                raise ValueError("Profile %s source %s subsets are not sorted and unique" % (profile, source))
            for subset in subsets:
                if not isinstance(subset, str) or not subset:
                    raise ValueError("Profile %s source %s has an invalid subset" % (profile, source))
                try:
                    subset.encode("ascii")
                except UnicodeEncodeError as exc:
                    raise ValueError("Profile %s source %s has a non-ASCII subset" % (profile, source)) from exc


def _profile_child_args(profile_root, source_spec, probability, seed, num_workers, force):
    """Build the existing single-source Fetch argv for one Profile source.

    @input: profile_root is a Path; source_spec is one validated source mapping;
            probability is a finite float in (0, 1]; seed and num_workers are
            CLI integers; force is a boolean.
    @output: List[str] containing only existing Fetch CLI options.
    @scenario: Reuses the current single-source dispatch without exposing the
               Profile name to any adapter.
    @param profile_root: Raw root under which the source directory is placed.
    @param source_spec: Frozen source, revision, and selection mapping.
    @param probability: Profile-wide trajectory sampling probability.
    @param seed: Deterministic sampling seed.
    @param num_workers: Fetch worker count.
    @param force: Whether to forward the existing force flag.
    @logic: Adds source-independent options first, then the fixed revision and
            either UTSD config or comma-joined source subsets.
    """
    source = source_spec["source"]
    child_args = [
        "--source", source,
        "--output-dir", str(profile_root / source),
        "--seed", str(seed),
        "--num-workers", str(num_workers),
    ]
    if probability is not None:
        child_args[child_args.index("--seed"):child_args.index("--seed")] = ["--sample-probability", str(probability)]
    child_args.extend(["--revision", source_spec["revision"]])
    if source == "utsd":
        child_args.extend(["--utsd-config", source_spec["utsd_config"]])
    else:
        child_args.extend(["--subsets", ",".join(source_spec["subsets"])])
    if force:
        child_args.append("--force")
    return child_args


def _write_profile_manifest(profile_root, profile, spec, seed, status, exit_codes):
    """Write the Profile manifest using a partial file and atomic replace.

    @input: profile_root is a Path; profile is a Profile name; spec is a
            validated Profile mapping; seed is an integer; status is running,
            completed, or failed; exit_codes maps source names to integers.
    @output: None; writes profile_manifest.json below profile_root.
    @scenario: Makes Profile selection, provenance, sampling, and child status
               inspectable without changing adapter completion markers.
    @param profile_root: Profile raw root.
    @param profile: Profile name.
    @param spec: Frozen Profile mapping.
    @param seed: Sampling seed persisted for reproducibility.
    @param status: Current Profile execution status.
    @param exit_codes: Source child exit statuses collected so far.
    @logic: Converts source tuples to JSON arrays, serializes ASCII sorted JSON,
            and replaces the target only after the partial file is complete.
    """
    manifest = {
        "exit_codes": dict(exit_codes),
        "estimate_method": spec["estimate_method"],
        "estimated_observations": spec["estimated_observations"],
        "profile": profile,
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "sample_probability": spec["sample_probability"],
        "seed": seed,
        "sources": [dict(source_spec) for source_spec in spec["sources"]],
        "status": status,
    }
    profile_root.mkdir(parents=True, exist_ok=True)
    target = profile_root / "profile_manifest.json"
    partial = profile_root / "profile_manifest.json.partial"
    partial.write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="ascii",
    )
    partial.replace(target)


def _run_profile(args):
    """Execute all static source tasks in one Profile and publish its manifest.

    @input: args is the parsed Fetch namespace with profile, output, seed,
            worker, and force fields.
    @output: Maximum child exit code, or zero when every child succeeds.
    @scenario: Expands a frozen cross-source Profile while allowing later
               sources to run after an earlier source returns nonzero.
    @param args: Parsed Fetch CLI namespace.
    @logic: Validates the mapping, writes a running manifest, recursively calls
            the current single-source main path, then atomically publishes the
            completed or failed manifest with all child exit codes.
    """
    spec = PROFILE_SPECS[args.profile]
    _validate_profile_spec(args.profile, spec)
    profile_root = output_path("source_raw", args.output_dir) if args.output_dir else output_path("source_raw", Path("data/src/raw/profiles") / args.profile)
    _write_profile_manifest(
        profile_root, args.profile, spec, args.seed, "running", {}
    )
    exit_codes = {}
    with ProcessingProgress(
        log, "FETCH_PROFILE", args.profile, len(spec["sources"]),
        "profile_root=%s workers=%d" % (profile_root, args.num_workers),
    ) as progress:
        for source_spec in spec["sources"]:
            source = source_spec["source"]
            child_args = _profile_child_args(
                profile_root, source_spec, spec["sample_probability"], args.seed,
                args.num_workers, args.force,
            )
            log.info("Profile %s dispatch source %s args=%s", args.profile, source, child_args)
            try:
                exit_codes[source] = int(main(child_args))
            except Exception:
                log.exception("Profile %s source %s raised an unhandled exception", args.profile, source)
                raise
            progress.advance(detail="source=%s exit_code=%d" % (source, exit_codes[source]))
    status = "completed" if all(code == 0 for code in exit_codes.values()) else "failed"
    _write_profile_manifest(
        profile_root, args.profile, spec, args.seed, status, exit_codes
    )
    return max(exit_codes.values(), default=0)


def main(argv=None):
    """Entry point for assets.dataset_processing.fetch."""
    args = parse_args(argv)
    configure_logger("fetch", "data/fetch", "fetch", "FETCH")

    if args.profile is not None:
        conflicts = []
        for option, value in (
            ("--source", args.source),
            ("--subsets", args.subsets),
            ("--utsd-config", args.utsd_config),
            ("--revision", args.revision),
            ("--sample-probability", args.sample_probability),
        ):
            if value is not None:
                conflicts.append(option)
        if conflicts:
            log.error(
                "--profile cannot be combined with: %s", ", ".join(conflicts)
            )
            return 2
        return _run_profile(args)

    if args.source is None or args.source == "all":
        if args.output_dir is not None:
            raise SystemExit("--output-dir requires one explicit --source; omit it for default output directories")
        exit_codes = []
        for source in FETCH_NON_DEPRECATED_SOURCES:
            # Repository-wide selection is never implicit: recursive source
            # dispatch deliberately forwards no ``--subsets all`` argument.
            child_args = [
                "--source", source,
                "--output-dir", str(Path("data/src/raw") / source),
                "--seed", str(args.seed),
                "--num-workers", str(args.num_workers),
            ]
            if args.utsd_config is not None:
                child_args.extend(["--utsd-config", args.utsd_config])
            if args.force:
                child_args.append("--force")
            if args.sample_probability is not None:
                child_args.extend(["--sample-probability", str(args.sample_probability)])
            if args.revision is not None:
                child_args.extend(["--revision", args.revision])
            exit_codes.append(main(child_args))
        return max(exit_codes)
    if args.output_dir is None:
        args.output_dir = str(Path("data/src/raw") / args.source)

    output_dir = output_path("source_raw", args.output_dir)

    subsets = None if not args.subsets else [
        item.strip() for item in args.subsets.split(",") if item.strip()
    ]
    if args.source == "utsd" and args.utsd_config is not None:
        if subsets is not None:
            log.error("UTSD accepts either --subsets or --utsd-config, not both.")
            return 2
        subsets = [args.utsd_config]
    module = __import__("assets.dataset_processing.fetch.src.%s" % args.source, fromlist=["fetch"])
    log.info(
        "Fetch source starting: source=%s output_dir=%s subsets=%s workers=%d sampling_probability=%s",
        args.source, output_dir, subsets, args.num_workers, args.sample_probability,
    )
    try:
        code = module.fetch(
            output_dir, subsets=subsets, force=args.force,
            sample_probability=args.sample_probability, seed=args.seed,
            revision=args.revision, num_workers=args.num_workers,
        )
    except Exception:
        log.exception(
            "Fetch source failed with an unhandled exception: source=%s output_dir=%s subsets=%s",
            args.source, output_dir, subsets,
        )
        raise
    log.info("Fetch source exited: source=%s exit_code=%s output_dir=%s", args.source, code, output_dir)

    return code
