"""Resume the paused V7 extension supervisor after two early selectors exit.

One-shot pidfd wait; no polling and no interruption of the Chronos asset worker.
"""
import os
from pathlib import Path
import selectors
import signal
import sys
import time

from assets.gradient.cache import atomic_json
from scripts.supervise_v7_night import pidfd

PARENT = 403235
SELECTORS = {485517: "timerxl-84m", 485518: "timesfm-2.5", 490168: "chronos-base"}
RUN = Path("/root/autodl-tmp/TS3/outputs/main-v7-20260923/extensions/time_coastal_ts_h-20260925")
RECEIPT = RUN / "logs/coastal_early_select_resume.json"


def words(pid):
    return Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")


def check_parent():
    args = words(PARENT)
    if "scripts.run_v7_extension_queue" not in args or "--parent" not in args:
        raise RuntimeError("Supervisor PID identity differs; refusing SIGCONT")
    if args[args.index("--parent") + 1] != str(RUN.parent.parent):
        raise RuntimeError("Supervisor parent run differs; refusing SIGCONT")
    if "T" not in Path(f"/proc/{PARENT}/status").read_text().split("State:", 1)[1].splitlines()[0]:
        raise RuntimeError("Supervisor is not stopped")


def main():
    check_parent()
    with selectors.DefaultSelector() as watcher:
        for pid, model in SELECTORS.items():
            args = words(pid)
            if ("scripts.run_v7" not in args or "--phase" not in args
                    or args[args.index("--phase") + 1] != "select"
                    or args[args.index("--student") + 1] != model
                    or args[args.index("--run") + 1] != str(RUN)):
                raise RuntimeError(f"Selector PID {pid} identity differs")
            fd = pidfd(pid)
            watcher.register(fd, selectors.EVENT_READ, (pid, model))
        atomic_json(RECEIPT, dict(state="waiting", parent=PARENT, selectors=SELECTORS, time=time.time()))
        while watcher.get_map():
            for key, _ in watcher.select():
                watcher.unregister(key.fd)
                os.close(key.fd)
    check_parent()
    completed = {model: all((RUN / f"seeds/seed{seed}/{model}/select/time_coastal_ts_h.json").is_file()
                            for seed in (0, 1)) for model in SELECTORS.values()}
    atomic_json(RECEIPT, dict(state="resuming", parent=PARENT, selectors=SELECTORS,
                              completed=completed, time=time.time()))
    os.kill(PARENT, signal.SIGCONT)  # Parent may idempotently retry an incomplete selector.
    print("V7_COASTAL_SUPERVISOR_RESUMED", completed, flush=True)


if __name__ == "__main__":
    sys.exit(main())
