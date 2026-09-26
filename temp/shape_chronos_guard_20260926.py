"""Event-only collision guard for scoped Chronos extension fill."""
import ctypes
import os
from pathlib import Path
import selectors
import signal
import sys

from scripts.supervise_v7_night import pidfd

ROOT = Path('/root/autodl-tmp/TS3/outputs/main-v7-20260923')
LOG = ROOT/'shape_backfill_20260925/logs/chronos-base.log'
SENTINEL = 'V7_SHAPE_ARM_TASK_COMPLETE rags-shape-linearSum chronos-base 1 time_solar_h'
IN_MODIFY = 0x00000002


def main(pid):
    handle = pidfd(pid)
    words = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    if 'temp/shape_chronos_helper_20260926.py' not in words:
        raise RuntimeError('Unexpected scoped helper PID identity')
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    event_fd = libc.inotify_init1(os.O_NONBLOCK)
    if event_fd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1')
    watch = libc.inotify_add_watch(event_fd, os.fsencode(LOG), IN_MODIFY)
    if watch < 0:
        raise OSError(ctypes.get_errno(), 'inotify_add_watch')
    with LOG.open() as stream, selectors.DefaultSelector() as selector:
        stream.seek(0, os.SEEK_END)
        selector.register(handle, selectors.EVENT_READ, 'helper')
        selector.register(event_fd, selectors.EVENT_READ, 'log')
        carry = ''
        while True:
            for key, _ in selector.select():
                if key.data == 'helper':
                    print('SCOPED_HELPER_EXITED', pid, flush=True)
                    return 0
                os.read(event_fd, 65536)
                chunk = stream.read()
                if SENTINEL in carry + chunk:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    print('SCOPED_HELPER_STOPPED_BEFORE_EXTENSION_COLLISION', pid, flush=True)
                    selector.unregister(event_fd)
                    while selector.select():
                        break
                    return 0
                carry = (carry + chunk)[-len(SENTINEL):]


if __name__ == '__main__':
    sys.exit(main(int(sys.argv[1])))
