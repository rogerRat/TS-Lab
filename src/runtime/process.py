"""Shared process lifecycle controls for bounded project workers.

@scenario: Keep child processes from surviving an unexpected parent exit while
           allowing Load and feature extraction to own their task protocols.
           This is project runtime infrastructure, not a TS3 algorithm.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_STOP_TIMEOUT_SECONDS = 3.0
_NUMERIC_THREAD_LIMITER = None


def resolve_worker_count(
        requested_jobs: int,
        cgroup_root: Path = Path("/sys/fs/cgroup"),
        ) -> int:
    """Resolve a safe worker count from an explicit request or CPU limits.

    @input: requested_jobs is a non-negative integer; zero enables automatic
            resolution. cgroup_root is a Linux cgroup root used for v2 or v1
            quota discovery and is injectable for tests.
    @output: Positive worker count. Explicit positive requests are returned
             unchanged; automatic requests respect cgroup quota and affinity.
    @scenario: Give all CPU-bound project stages one Docker-aware worker-count
               policy without coupling CPU concurrency to memory admission.
    @param requested_jobs: Positive explicit count, or zero for automatic mode.
    @param cgroup_root: Root containing cgroup v2 cpu.max or v1 cpu controls.
    """
    requested = int(requested_jobs)
    if requested < 0:
        raise ValueError("requested_jobs must be non-negative")
    if requested > 0:
        return requested

    try:
        affinity_jobs = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_jobs = mp.cpu_count()

    root = Path(cgroup_root)
    quota_jobs = None
    try:
        v2_path = root / "cpu.max"
        if v2_path.is_file():
            quota_text, period_text = v2_path.read_text(encoding="ascii").split()
            if quota_text != "max":
                quota_jobs = max(1, math.ceil(int(quota_text) / int(period_text)))
        else:
            for quota_path in (
                    root / "cpu" / "cpu.cfs_quota_us",
                    root / "cpu,cpuacct" / "cpu.cfs_quota_us",
            ):
                if not quota_path.is_file():
                    continue
                period_path = quota_path.with_name("cpu.cfs_period_us")
                quota = int(quota_path.read_text(encoding="ascii").strip())
                period = int(period_path.read_text(encoding="ascii").strip())
                if quota > 0 and period > 0:
                    quota_jobs = max(1, math.ceil(quota / period))
                break
    except (OSError, ValueError):
        quota_jobs = None

    return affinity_jobs if quota_jobs is None else min(affinity_jobs, quota_jobs)


def configure_worker_lifetime(parent_pid: int, logger: Optional[logging.Logger] = None) -> None:
    """Configure a worker to terminate when its coordinating parent disappears.

    @input: parent_pid - PID of the coordinating parent process.
    @output: None; kernel/watchdog parent-death handling is installed.
    @scenario: Prevent orphan workers after parent termination, SIGKILL, or OOM.
    """
    logger = logger or logging.getLogger("load")
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            result = libc.prctl(1, signal.SIGKILL, 0, 0, 0)
            if result != 0:
                error_number = ctypes.get_errno()
                logger.warning("Unable to configure Linux parent-death signal: %s", os.strerror(error_number))
        except Exception as exc:
            logger.warning("Unable to configure Linux parent-death signal: %s", exc)

    if os.getppid() != parent_pid:
        os._exit(1)

    try:
        os.setsid()
    except OSError:
        pass

    def parent_watchdog() -> None:
        while True:
            if os.getppid() != parent_pid:
                os._exit(1)
            time.sleep(0.2)

    threading.Thread(target=parent_watchdog, name="project-parent-watchdog", daemon=True).start()


def configure_pool_worker(parent_pid: int, limit_numeric_threads: bool = False) -> None:
    """Initialize a ProcessPoolExecutor worker with parent-death protection.

    @input: parent_pid - PID of the process that owns the executor.
    @input: parent_pid identifies the executor owner; limit_numeric_threads
        selects a persistent one-thread limit for BLAS/OpenMP libraries.
    @output: None; worker lifetime protection and an optional numeric thread
        limit are installed.
    @scenario: Apply the shared parent-death rule and avoid nested native
        thread pools in CPU-bound executor workers.
    """
    configure_worker_lifetime(parent_pid)
    if not limit_numeric_threads:
        return
    global _NUMERIC_THREAD_LIMITER
    try:
        from threadpoolctl import threadpool_limits

        _NUMERIC_THREAD_LIMITER = threadpool_limits(limits=1)
        _NUMERIC_THREAD_LIMITER.__enter__()
    except Exception:
        # The project supports NumPy builds without threadpoolctl.  The worker
        # remains correct; this only avoids avoidable nested parallelism.
        _NUMERIC_THREAD_LIMITER = None


def signal_process(process: Any, signum: int) -> None:
    """Send a signal to a process and its private process group when available.

    @input: process - process-like object with a PID; signum - OS signal number.
    @output: None; already-exited processes are ignored.
    @scenario: Stop worker descendants without signaling the coordinating parent.
    """
    pid = getattr(process, "pid", None)
    if pid is None:
        return
    try:
        process_group = os.getpgid(pid)
        if process_group == pid:
            os.killpg(process_group, signum)
        else:
            os.kill(pid, signum)
    except (ProcessLookupError, OSError):
        pass


def shutdown_processes(
    processes: Iterable[Any],
    graceful: bool,
    timeout_seconds: float = DEFAULT_STOP_TIMEOUT_SECONDS,
) -> None:
    """Stop processes with bounded graceful-then-forced escalation.

    @input: processes - process-like objects with ``is_alive`` and ``join``;
            graceful - whether the caller already requested normal completion;
            timeout_seconds - positive escalation deadline.
    @output: None; reachable processes are joined or sent SIGKILL.
    @scenario: Provide one shutdown policy for custom queues and executors.
    """
    process_list = list(processes)
    if not graceful:
        for process in process_list:
            if process.is_alive():
                signal_process(process, signal.SIGTERM)

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not any(process.is_alive() for process in process_list):
            break
        time.sleep(0.05)

    for process in process_list:
        if process.is_alive():
            signal_process(process, signal.SIGKILL)
    for process in process_list:
        process.join(timeout=0.2)
