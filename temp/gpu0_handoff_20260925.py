"""One-shot cell-boundary handoff; block on inotify, never poll."""
import ctypes
import json
import os
from pathlib import Path
import selectors
import signal
import struct
import time

ROOT = Path('/root/autodl-tmp/TS3/outputs/main-v7-20260923')
HELPER, MAIN = 267504, 222664
TASK = 'time_nem_price_5t'
CELLS = ROOT/'seeds/seed1/chronos-base/cells'
RECEIPT = ROOT/'logs/gpu0_handoff_20260925.json'

def verify(pid, helper=False):
    words = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    assert 'scripts.run_v7' in words
    for flag, value in [('--student', 'chronos-base'), ('--seed', '1'),
                        ('--phase', 'train'), ('--run', str(ROOT))]:
        assert flag in words and words[words.index(flag)+1] == value
    if helper:
        assert words[words.index('--train-task')+1] == TASK

def record(**fields):
    temporary = RECEIPT.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(time=time.time(), helper=HELPER,
        main=MAIN, task=TASK, **fields), indent=2))
    os.replace(temporary, RECEIPT)

verify(HELPER, True)
verify(MAIN)
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.inotify_init1(os.O_CLOEXEC)
if fd < 0:
    raise OSError(ctypes.get_errno(), 'inotify_init1')
assert libc.inotify_add_watch(fd, os.fsencode(CELLS), 0x80) >= 0  # IN_MOVED_TO
baseline = {p.name for p in CELLS.glob(TASK+'__*.json')}
record(state='armed', completed_cells=len(baseline))
print('ARMED', flush=True)
with selectors.DefaultSelector() as selector:
    selector.register(fd, selectors.EVENT_READ)
    deadline = time.monotonic()+900
    done = False
    while not done:
        if not selector.select(max(0, deadline-time.monotonic())):
            record(state='aborted_timeout_no_signal')
            raise RuntimeError('No cell boundary in 15 minutes; no process changed')
        data = os.read(fd, 65536)
        offset = 0
        while offset < len(data):
            wd, mask, cookie, length = struct.unpack_from('iIII', data, offset)
            name = data[offset+16:offset+16+length].split(b'\0')[0].decode()
            offset += 16+length
            if name in baseline or not (name.startswith(TASK+'__') and name.endswith('.json')):
                continue
            row = json.loads((CELLS/name).read_text())
            assert (row['student'], row['seed'], row['task']) == ('chronos-base', 1, TASK)
            verify(HELPER, True)
            verify(MAIN)
            locks = Path('/proc/locks').read_text().splitlines()
            assert any('->' in line and f' {MAIN} ' in line for line in locks)
            os.kill(HELPER, signal.SIGTERM)
            record(state='helper_term_at_durable_cell_main_resumes_via_flock', cell=name)
            print('HANDOFF', name, flush=True)
            done = True
            break
os.close(fd)
