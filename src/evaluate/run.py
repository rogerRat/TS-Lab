"""Unified explicit CLI for FEV, TIME and QuitoBench evaluation runs."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from configs.logger import configure_logger
from configs.paths import config_file, output_path, project_path
from evaluate.schemas import EvaluationRunConfig


from configs.paths import PROJECT_ROOT
UTC8 = timezone(timedelta(hours=8))
log = logging.getLogger("evaluate")


def _resolve_project_path(value: Path) -> Path:
    """Resolve an explicit project-relative path.

    @input: value is an absolute or project-relative filesystem path.
    @output: Absolute resolved path.
    @scenario: Make CLI paths independent of a caller's shell cwd.
    @logic: Relative values are rooted at the repository; absolute values are retained.
    @param value: CLI path value.
    """
    return project_path(value)


def validate_checkpoint(path: Path, require_manifest: bool = True) -> Path:
    """Require a local Chronos checkpoint and optionally its TS3 manifest.

    @input: path points to a candidate checkpoint directory.
    @output: Resolved checkpoint directory.
    @scenario: Reject incomplete training output before benchmark-specific work.
    @logic: Requires config/weights; FEV also requires the TS3 manifest.
    @param path: Candidate checkpoint path.
    """
    checkpoint = _resolve_project_path(path)
    required = ("config.json", "model.safetensors")
    if require_manifest:
        required += ("ts3_checkpoint_manifest.json",)
    if not checkpoint.is_dir() or any(not (checkpoint / name).is_file() for name in required):
        raise FileNotFoundError("checkpoint is incomplete: %s" % checkpoint)
    return checkpoint


def _metadata(config: EvaluationRunConfig, checkpoint: Path) -> dict[str, Any]:
    """Build common root provenance.

    @input: Validated run config and complete checkpoint directory.
    @output: JSON-compatible common metadata mapping.
    @scenario: Provide a benchmark-independent resume identity.
    @logic: Hashes the exact YAML and checkpoint before backend execution.
    """
    return {
        "schema_version": 1,
        "benchmark": config.benchmark,
        "model_id": config.model_id,
        "adapter_class": config.adapter_class,
        "checkpoint": str(checkpoint),
        "config": str(config.source_path),
        "config_values": config.as_mapping(),
        "created_at": datetime.now(UTC8).isoformat(),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write one ASCII JSON document.

    @input: path is a destination; value is JSON-compatible.
    @output: A complete JSON file at path.
    @scenario: Avoid a partially written root provenance file.
    @logic: Writes a sibling partial file then replaces the destination.
    @param path: JSON destination.
    @param value: Metadata mapping.
    """
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    temporary.replace(path)


def prepare_output(config: EvaluationRunConfig, checkpoint: Path, output_dir: Path, resume: bool) -> Path:
    """Create or validate the shared output directory contract.

    @input: Common config/checkpoint, requested result path, and resume flag.
    @output: Resolved output directory available to the selected backend.
    @scenario: Prevent accidental result replacement while allowing explicit resume.
    @logic: New runs require a nonexistent directory; explicit resume reuses the
            existing directory without requiring historical metadata to match.
    @param config: Parsed common evaluation configuration.
    @param checkpoint: Complete checkpoint directory.
    @param output_dir: Requested output directory.
    @param resume: Whether existing output is explicitly permitted.
    """
    destination = output_path("eval", output_dir)
    expected = _metadata(config, checkpoint)
    metadata_path = destination / "run_metadata.json"
    if destination.exists():
        if not resume:
            raise FileExistsError("output directory already exists; pass --resume only for this exact run: %s" % destination)
        return destination
    if resume:
        raise FileNotFoundError("--resume requires an existing output directory: %s" % destination)
    destination.mkdir(parents=True)
    _write_json(metadata_path, expected)
    return destination


def run(config: EvaluationRunConfig, checkpoint: Path, output_dir: Path, resume: bool = False) -> Path:
    """Dispatch one common evaluation run to its explicit backend.

    @input: Parsed config, checkpoint, output directory, and explicit resume flag.
    @output: Backend primary result path.
    @scenario: Use one public run contract for FEV, TIME and QuitoBench.
    @logic: Validates common IO first, then uses a fixed if/elif dispatch only.
    @param config: Common evaluation run config.
    @param checkpoint: CLI checkpoint path.
    @param output_dir: CLI output directory path.
    @param resume: Explicit resume permission.
    """
    checked_checkpoint = validate_checkpoint(checkpoint, require_manifest=config.benchmark == "fev")
    log.info("Evaluation requested: benchmark=%s checkpoint=%s output_dir=%s resume=%s", config.benchmark, checked_checkpoint, output_dir, resume)
    destination = prepare_output(config, checked_checkpoint, output_dir, resume)
    log.info("Evaluation output prepared: %s", destination)
    if config.benchmark == "fev":
        from evaluate.fev import run_fev
        return run_fev(config, checked_checkpoint, destination, resume)
    if config.benchmark == "time":
        from evaluate.time import run_time
        return run_time(config, checked_checkpoint, destination, resume)
    if config.benchmark == "quito":
        from evaluate.quito import run_quito
        return run_quito(config, checked_checkpoint, destination, resume)
    raise AssertionError("validated benchmark cannot be dispatched")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute the common evaluation CLI.

    @input: Optional command-line argument sequence.
    @output: Process exit status 0 after a backend completes.
    @scenario: Provide the only supported project evaluation command.
    @logic: Parses three required paths plus explicit --resume, then dispatches.
    @param argv: Optional argument sequence for tests.
    """
    parser = argparse.ArgumentParser(description="Run one FEV, TIME or QuitoBench evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config_path = config_file(args.config)
    config = EvaluationRunConfig.from_file(config_path)
    configure_logger("evaluate", "evaluate", "evaluate_run", "EVALUATE")
    log.info("Evaluation configuration loaded: benchmark=%s model_id=%s config=%s", config.benchmark, config.model_id, config.source_path)
    result = run(config, args.checkpoint, args.output_dir, args.resume)
    log.info("Evaluation completed: result=%s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
