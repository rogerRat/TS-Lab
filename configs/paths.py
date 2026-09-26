# Canonical layout: data/{bench,src/{raw,shard_dataset},ref/shard_dataset},
# outputs/{train,eval,vis,cache}, logs/, configs/. Resolve via these helpers; validate writes with output_path().
from __future__ import annotations

from pathlib import Path
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_EXPERIMENT_CONFIGS = Path("/root/autodl-tmp/TS3/profile-10b/experiments/configs")
CONFIGS = PROJECT_ROOT / "configs"
LOGS = PROJECT_ROOT / "logs"
BENCH_DATA = PROJECT_ROOT / "data" / "bench"
SOURCE_RAW_DATA = PROJECT_ROOT / "data" / "src" / "raw"
SOURCE_SHARD_DATA = PROJECT_ROOT / "data" / "src" / "shard_dataset"
REFERENCE_SHARD_DATA = PROJECT_ROOT / "data" / "ref" / "shard_dataset"
DATA_DIR = PROJECT_ROOT / "data"
PRETRAINED_MODELS_DIR = PROJECT_ROOT / "cache" / "pretrained_models"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
TRAIN_OUTPUTS = PROJECT_ROOT / "outputs" / "train"
EVAL_OUTPUTS = PROJECT_ROOT / "outputs" / "eval"
VIS_OUTPUTS = PROJECT_ROOT / "outputs" / "vis"

# Project-wide behavior for a requested path outside its declared category.
# Set to "error" after the historical migration is complete.
PATH_POLICY = "warn"

# A writing CLI declares one of these categories when calling output_path().
OUTPUT_ROOTS = {
    "benchmark": BENCH_DATA,
    "source_raw": SOURCE_RAW_DATA,
    "source_shard": SOURCE_SHARD_DATA,
    "reference_shard": REFERENCE_SHARD_DATA,
    "train": TRAIN_OUTPUTS,
    "eval": EVAL_OUTPUTS,
    "vis": VIS_OUTPUTS,
}


class PathPolicyWarning(UserWarning):
    """A requested output is outside the directory assigned to its category."""


def project_path(value: str | Path) -> Path:
    """Resolve an absolute or project-relative path without creating it."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def output_path(category: str, value: str | Path) -> Path:
    """Resolve a CLI output path and apply the project-wide path policy.

    ``category`` must be one of OUTPUT_ROOTS. In ``warn`` mode the requested
    path is retained after a visible warning; in ``error`` mode it is rejected
    before the caller can create a directory or write a file.
    """
    if PATH_POLICY not in {"warn", "error"}:
        raise ValueError("PATH_POLICY must be 'warn' or 'error', got %r" % PATH_POLICY)
    try:
        root = OUTPUT_ROOTS[category].resolve()
    except KeyError as exc:
        raise ValueError("unknown output category: %s" % category) from exc
    path = project_path(value)
    if _is_within(path, root):
        return path
    message = "%s output should be below %s, got %s" % (category, root, path)
    if PATH_POLICY == "error":
        raise ValueError(message)
    warnings.warn(message, PathPolicyWarning, stacklevel=2)
    return path
def config_file(value: str | Path) -> Path:
    path = project_path(value)
    allowed_roots = (CONFIGS.resolve(), DATA_EXPERIMENT_CONFIGS.resolve())
    if not any(_is_within(path, root) for root in allowed_roots):
        raise ValueError(
            "config must be below one of %s: %s"
            % (", ".join(str(root) for root in allowed_roots), path)
        )
    return path


def pretrained_model_path(value: str | Path) -> Path:
    """Resolve a read-only pretrained checkpoint below ``cache/pretrained_models``."""
    path = project_path(value)
    if not _is_within(path, PRETRAINED_MODELS_DIR.resolve()):
        raise ValueError("pretrained model must be below %s: %s" % (PRETRAINED_MODELS_DIR, path))
    return path
