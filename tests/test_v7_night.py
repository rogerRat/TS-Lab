import subprocess
import sys
from scripts.supervise_v7_night import wait_all


def test_second_exits_first_without_lost_pid():
    a=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.read()'],stdin=subprocess.PIPE)
    b=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.read()'],stdin=subprocess.PIPE)
    checked=[]
    def verify(pid):
        checked.append(pid)
        if pid==b.pid:
            # Force the second exit before the first, after descriptors acquired.
            b.stdin.close();b.wait(timeout=5)
            a.stdin.close()
    try:
        wait_all([a.pid,b.pid],verify)
        assert checked==[a.pid,b.pid]
        assert a.wait(timeout=5)==0
    finally:
        for p in (a,b):
            if p.poll() is None:p.kill();p.wait()
