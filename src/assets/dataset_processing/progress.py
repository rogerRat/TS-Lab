"""Periodic progress logging for long-running data-processing operations."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from configs.logger import format_duration


DEFAULT_PROGRESS_INTERVAL_SECONDS = 30.0


class ProcessingProgress:
    """Emit start, heartbeat, checkpoint, completion, and failure logs.

    @input: logger receives records; stage and operation identify one operation;
        total is an optional positive work-item count; detail_provider optionally
        returns ASCII-safe current-state text; interval_seconds is positive.
    @output: Context-compatible reporter whose methods only write log records.
    @scenario: Make a long data-processing operation observable even when its
        current work item does not naturally emit progress.
    @logic: A daemon thread snapshots protected counters at a fixed interval.
        Callers advance counters or replace details without changing processing
        control flow, inputs, outputs, timing decisions, or error behavior.
    """

    def __init__(self, logger: logging.Logger, stage: str, operation: str,
                 total: Optional[int] = None, detail: str = "",
                 detail_provider: Optional[Callable[[], str]] = None,
                 interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.logger = logger
        self.stage = str(stage)
        self.operation = str(operation)
        self.total = None if total is None else max(0, int(total))
        self.detail = str(detail)
        self.detail_provider = detail_provider
        self.interval_seconds = float(interval_seconds)
        self.completed = 0
        self.started_at = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._emit_lock = threading.Lock()
        self._last_heartbeat_snapshot = (self.completed, self.total, self.detail)
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="progress-%s" % self.stage.lower(), daemon=True,
        )
        self.logger.info(
            "Started %s (%s): total %s, detail %s",
            self.stage, self.operation, self.total if self.total is not None else "unknown",
            self.detail or "none",
        )
        self._thread.start()

    def _snapshot(self) -> tuple[int, Optional[int], str, float]:
        with self._lock:
            completed = self.completed
            total = self.total
            detail = self.detail
        if self.detail_provider is not None:
            try:
                provided = self.detail_provider()
            except Exception as exc:
                provided = "detail_provider_error=%s" % exc
            if provided:
                detail = str(provided)
        return completed, total, detail, time.monotonic() - self.started_at

    def _emit(self, event: str, level: int = logging.INFO) -> None:
        completed, total, detail, elapsed = self._snapshot()
        snapshot = (completed, total, detail)
        with self._emit_lock:
            if event == "heartbeat" and snapshot == self._last_heartbeat_snapshot:
                return
            self._last_heartbeat_snapshot = snapshot
        if total is None:
            progress = "unknown"
            eta = "unknown"
            bar = "?"
        elif total == 0:
            progress = "100.0%%"
            eta = format_duration(0)
            bar = "=" * 20
        else:
            ratio = min(completed, total) / total
            progress = "%.1f%%" % (100.0 * ratio)
            rate = completed / elapsed if completed > 0 and elapsed > 0 else 0.0
            eta = format_duration((total - completed) / rate) if rate > 0 else "unknown"
            filled = min(20, int(20 * ratio))
            bar = "=" * filled + "." * (20 - filled)
        self.logger.log(
            level,
            "%s %s: [%s] completed %d of %s, %s, elapsed %s, eta %s, detail %s",
            self.stage, event, bar, completed,
            total if total is not None else "unknown", progress, format_duration(elapsed), eta,
            detail or "none",
            extra={
                "ts3_event": "progress",
                "ts3_progress_event": event,
            },
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._emit("heartbeat")

    def advance(self, count: int = 1, detail: Optional[str] = None) -> None:
        """Record completed work without changing processing behavior.

        @input: count is a non-negative completed-item increment; detail replaces
            the current operation detail when supplied.
        @output: None.
        @scenario: Report deterministic loop progress from a processing caller.
        @logic: Updates only in-memory telemetry counters guarded by a lock.
        """
        if count < 0:
            raise ValueError("count must be non-negative")
        with self._lock:
            self.completed += int(count)
            if detail is not None:
                self.detail = str(detail)

    def checkpoint(self, detail: Optional[str] = None) -> None:
        """Immediately emit an informational progress checkpoint.

        @input: detail optionally replaces the current operation detail.
        @output: None.
        @scenario: Mark a phase boundary that operators need to distinguish.
        @logic: Updates the telemetry detail then writes one snapshot log record.
        """
        if detail is not None:
            with self._lock:
                self.detail = str(detail)
        self._emit("checkpoint")

    def close(self, success: bool = True, detail: Optional[str] = None) -> None:
        """Stop heartbeats and emit the final status once.

        @input: success selects completion or failure status; detail optionally
            replaces the final operation detail.
        @output: None.
        @scenario: Leave a durable terminal progress record before returning or
            propagating an exception.
        @logic: Stops and joins the daemon briefly, then emits a final snapshot.
        """
        if detail is not None:
            with self._lock:
                self.detail = str(detail)
        self._stop.set()
        self._thread.join(timeout=0.2)
        self._emit("complete" if success else "failed", logging.INFO if success else logging.ERROR)

    def __enter__(self) -> "ProcessingProgress":
        """Return this reporter for a context-managed processing scope.

        @input: No arguments.
        @output: This reporter.
        @scenario: Ensure terminal telemetry follows an operation scope.
        @logic: Delegates final state emission to ``__exit__``.
        """
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        """Close telemetry without suppressing a processing exception.

        @input: Standard context-manager exception triplet.
        @output: False, so the original exception behavior is unchanged.
        @scenario: Report failures while preserving the caller's error semantics.
        @logic: Emits success only when no exception escaped the context.
        """
        self.close(success=exc_type is None)
        return False
