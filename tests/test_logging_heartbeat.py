import logging
import time

from configs import logger
from assets.dataset_processing.progress import ProcessingProgress


class _RecordHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_processing_progress_marks_events_and_deduplicates_snapshots():
    logger = logging.getLogger("test.progress.heartbeat")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    handler = _RecordHandler()
    logger.addHandler(handler)
    progress = ProcessingProgress(logger, "TEST", "items", total=2, interval_seconds=60)
    try:
        progress._emit("heartbeat")
        assert [record for record in handler.records if record.getMessage().startswith("TEST heartbeat")] == []

        progress.advance(detail="item=1")
        progress._emit("heartbeat")
        progress._emit("heartbeat")
        heartbeats = [
            record for record in handler.records
            if record.getMessage().startswith("TEST heartbeat")
        ]
        assert len(heartbeats) == 1
        assert heartbeats[0].ts3_event == "progress"
        assert heartbeats[0].ts3_progress_event == "heartbeat"
    finally:
        progress.close()
        logger.removeHandler(handler)


def test_global_heartbeat_is_concise_and_does_not_replay_progress(monkeypatch, tmp_path):
    monkeypatch.setattr(logger, "HEARTBEAT_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(logger, "LOG_DIR", tmp_path)
    configured_logger = logger.configure_logger(
        "test.runtime.heartbeat", "tests", "heartbeat", "TEST",
        console_level=logging.CRITICAL,
    )
    try:
        configured_logger.info(
            "Progress heartbeat: [=====...............]",
            extra={"ts3_event": "progress", "ts3_progress_event": "heartbeat"},
        )
        time.sleep(0.08)
        log_file = next((tmp_path / "tests").glob("heartbeat_*.log"))
        contents = log_file.read_text(encoding="ascii")
        assert contents.count("Progress heartbeat:") == 1
        assert "Still running: TEST; no log output for" in contents
        assert "current stage: Progress heartbeat" not in contents
        assert "[=====...............]" not in contents.split("Still running: TEST;", 1)[1]
    finally:
        heartbeat = configured_logger._ts3_heartbeat
        heartbeat._stop.set()
        heartbeat._thread.join(timeout=1)
        for handler in list(configured_logger.handlers):
            configured_logger.removeHandler(handler)
            handler.close()


def test_configure_logger_rejects_log_directory_outside_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(logger, "LOG_DIR", tmp_path / "logs")

    try:
        logger.configure_logger(
            "test.logging.outside",
            "tests",
            "outside",
            "TEST",
            log_directory=tmp_path / "other",
        )
    except ValueError as exc:
        assert "log directory must be below" in str(exc)
    else:
        raise AssertionError("log directory outside logs must be rejected")
