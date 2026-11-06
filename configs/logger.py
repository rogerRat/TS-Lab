"""Project-wide logging configuration and human-readable log formatting."""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from configs.paths import LOGS


LOG_DIR = LOGS
HEARTBEAT_INTERVAL_SECONDS = 30
_TZ = timezone(timedelta(hours=8))
_CLI_SUMMARY_REGISTERED = False
_CLI_STARTED_AT = None
_CLI_LOGGER = None


def format_duration(seconds: float | None) -> str:
    """Format a duration using compact human-readable units."""
    if seconds is None:
        return "unknown"
    value = max(0.0, float(seconds))
    if value < 60:
        return "%.1fs" % value
    minutes = value / 60
    if minutes < 60:
        return "%.1fm" % minutes
    hours = minutes / 60
    if hours < 24:
        return "%.1fh" % hours
    return "%.1fd" % (hours / 24)


def format_bytes(value: int | None) -> str:
    """Format a byte count using binary human-readable units."""
    if value is None:
        return "unknown"
    amount = float(max(0, int(value)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return "%.1f %s" % (amount, unit) if unit != "B" else "%d B" % amount
        amount /= 1024


def _cli_summary() -> None:
    if _CLI_LOGGER is None or _CLI_STARTED_AT is None:
        return
    _CLI_LOGGER.info("===== Summary =====")
    _CLI_LOGGER.info("result=completed_or_terminated")
    outputs = []
    for index, value in enumerate(sys.argv[1:]):
        if "/" not in value and not value.startswith("."):
            continue
        option = sys.argv[index] if index < len(sys.argv) else ""
        label = option.lstrip("-").replace("-", " ") or "path"
        if "output" in label or "dir" in label:
            kind = "output directory"
        elif "checkpoint" in label:
            kind = "checkpoint"
        elif "log" in label:
            kind = "audit/log file"
        elif "config" in label:
            kind = "configuration"
        else:
            kind = "path"
        outputs.append(f"{value} ({kind})")
    _CLI_LOGGER.info("outputs=%s", outputs or ["none declared"])
    _CLI_LOGGER.info("duration=%s", format_duration(time.monotonic() - _CLI_STARTED_AT))


def _register_cli_summary(logger: logging.Logger) -> None:
    global _CLI_LOGGER, _CLI_STARTED_AT, _CLI_SUMMARY_REGISTERED
    if _CLI_SUMMARY_REGISTERED:
        return
    _CLI_LOGGER = logger
    _CLI_STARTED_AT = time.monotonic()
    _CLI_SUMMARY_REGISTERED = True
    atexit.register(_cli_summary)


class _Heartbeat:
    """Emit liveness records only after a period of log silence."""

    def __init__(self, logger: logging.Logger, stage: str) -> None:
        self.logger = logger
        self.current_stage = stage
        self._started_at = time.monotonic()
        self._last_activity_at = self._started_at
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"{logger.name}-heartbeat", daemon=True
        )
        self._thread.start()

    def observe(self, record: logging.LogRecord) -> None:
        if getattr(record, "rage_event", None) == "heartbeat":
            return
        with self._lock:
            self._last_activity_at = time.monotonic()
            record_stage = getattr(record, "rage_stage", None)
            if record_stage:
                self.current_stage = str(record_stage)

    def _run(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            with self._lock:
                stage = self.current_stage
                idle_for = time.monotonic() - self._last_activity_at
            if idle_for < HEARTBEAT_INTERVAL_SECONDS:
                continue
            self.logger.info(
                "Still running: %s; no log output for %s; uptime %s",
                stage, format_duration(idle_for),
                format_duration(time.monotonic() - self._started_at),
                extra={"rage_event": "heartbeat"},
            )


class _RAGEFormatter(logging.Formatter):
    """Render all configured sinks with one readable project format."""

    converter = staticmethod(
        lambda timestamp: datetime.fromtimestamp(timestamp, _TZ).timetuple()
    )

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record)


def configure_logger(
    name: str,
    log_group: str,
    log_basename: str,
    stage: str,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    enable_heartbeat: bool = True,
    log_directory: Path | None = None,
) -> logging.Logger:
    """Configure one CLI logger with console, file, and heartbeat output."""
    if not name or not log_group or not log_basename or not stage:
        raise ValueError("name, log_group, log_basename, and stage must be non-empty")
    group_path = Path(log_group)
    if group_path.is_absolute() or ".." in group_path.parts:
        raise ValueError("log_group must be a relative path below logs")

    logger = logging.getLogger(name)
    if any(getattr(handler, "_rage_cli_handler", False) for handler in logger.handlers):
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    logs_root = LOG_DIR.resolve()
    destination = (Path(log_directory).resolve() if log_directory is not None else logs_root / group_path).resolve()
    if destination != logs_root and logs_root not in destination.parents:
        raise ValueError("log directory must be below %s, got %s" % (logs_root, destination))
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(_TZ).strftime("%Y%m%d%H%M%S")
    log_file = destination / f"{log_basename}_{timestamp}.log"
    formatter = _RAGEFormatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        str(log_file), encoding="ascii", errors="backslashreplace"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    file_handler._rage_cli_handler = True
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    console_handler._rage_cli_handler = True
    logger.addHandler(console_handler)

    heartbeat = _Heartbeat(logger, stage) if enable_heartbeat else None
    logger._rage_heartbeat = heartbeat

    class _StageFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if heartbeat is not None:
                heartbeat.observe(record)
            return True

    file_handler.addFilter(_StageFilter())
    console_handler.addFilter(_StageFilter())

    logger.info("Log file: %s", log_file)
    logger.info("===== Config =====")
    logger.info("command=%s", " ".join(sys.argv))
    logger.info("cwd=%s", os.getcwd())
    logger.info("arguments=%s", sys.argv[1:])
    _register_cli_summary(logger)
    return logger


__all__ = ["HEARTBEAT_INTERVAL_SECONDS", "configure_logger", "format_bytes", "format_duration"]
