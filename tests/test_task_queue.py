"""Queue safety tests: all jobs and databases live under pytest tmp_path."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from runtime.task_queue import Queue, alive, lock, locked, process_identity


def eventually(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.03)
    raise AssertionError("Timed out waiting for queue progress")


@pytest.fixture
def queue(tmp_path):
    q = Queue(tmp_path / "queue.sqlite")
    yield q
    q.close()


def command(q, tmp_path, code, *args):
    return q.add("test", "command", dict(cwd=str(tmp_path), steps=[
        [sys.executable, "-c", code, *map(str, args)]]))["id"]


def completed(q, job):
    q.tick()
    return q.get(job)["status"] == "succeeded"


def stop(process):
    if process.poll() is None:
        process.terminate()
    process.wait(timeout=5)


def worker(q):
    return subprocess.Popen([sys.executable, "-m", "runtime.task_queue", "--db",
        str(q.path), "work", "--interval", ".03"], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)


def test_append_during_execution_preserves_fifo(queue, tmp_path):
    started, release, output = [tmp_path / name for name in ("started", "release", "order")]
    first = command(queue, tmp_path,
        "import sys,time; from pathlib import Path; "
        "s,r,o=map(Path,sys.argv[1:]); s.touch(); "
        "exec('while not r.exists(): time.sleep(.02)'); o.write_text('first\\n')",
        started, release, output)
    queue.tick()
    try:
        eventually(started.exists)
        # A separate client appends while the actual child is running.
        client = Queue(queue.path)
        try:
            second = command(client, tmp_path,
                "import sys; from pathlib import Path; "
                "p=Path(sys.argv[1]); p.write_text(p.read_text()+'second\\n')", output)
            third = command(client, tmp_path,
                "import sys; from pathlib import Path; "
                "p=Path(sys.argv[1]); p.write_text(p.read_text()+'third\\n')", output)
        finally:
            client.close()
        assert queue.get(second)["status"] == "queued"
        assert queue.get(first)["status"] == "running"
    finally:
        release.touch()
    eventually(lambda: completed(queue, third))
    assert output.read_text().splitlines() == ["first", "second", "third"]


def test_failure_blocks_until_explicit_retry_or_skip(queue, tmp_path):
    flag = tmp_path / "retry-allowed"
    first = command(queue, tmp_path,
        "import sys; from pathlib import Path; sys.exit(0 if Path(sys.argv[1]).exists() else 7)", flag)
    second = command(queue, tmp_path, "pass")
    queue.tick()
    eventually(lambda: queue.get(first)["status"] == "failed")
    assert queue.get(first)["exit_code"] == 7
    assert queue.tick() == "failed"
    assert queue.get(second)["status"] == "queued"
    eventually(lambda: not alive(queue.get(first)["runner"]))
    flag.touch()
    queue.control(first, "retry")
    eventually(lambda: completed(queue, second))
    assert queue.get(first)["attempt"] == 2
    failed = command(queue, tmp_path, "raise SystemExit(3)")
    after = command(queue, tmp_path, "pass")
    queue.tick()
    eventually(lambda: queue.get(failed)["status"] == "failed")
    eventually(lambda: not alive(queue.get(failed)["runner"]))
    queue.control(failed, "skip")
    eventually(lambda: completed(queue, after))
    assert queue.get(failed)["status"] == "skipped"


def test_watch_process_exit_requires_success_evidence(queue, tmp_path):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        identity = process_identity(process.pid)
        job = queue.add("watch", "watch", dict(identity=identity,
            success=dict(path=str(tmp_path / "missing.json"), equals={"complete": True})))["id"]
        assert queue.tick() == "waiting"
    finally:
        stop(process)
    assert queue.tick() == "failed"
    assert "without success evidence" in queue.get(job)["message"]


def test_pid_reuse_and_worker_disappearance_are_not_success(queue, tmp_path):
    identity = process_identity(os.getpid())
    assert alive(identity)
    assert not alive(dict(identity, start_ticks=str(int(identity["start_ticks"]) + 1)))
    assert not alive(dict(identity, boot_id="different-boot"))
    job = command(queue, tmp_path, "pass")
    queue.conn.execute("UPDATE jobs SET status='running',runner=?,updated=? WHERE id=?",
        (json.dumps(dict(identity, start_ticks="-1")), time.time() - 20, job))
    queue.tick()
    assert queue.get(job)["status"] == "needs_attention"


def test_single_worker_lock(queue):
    with lock(str(queue.path) + ".worker.lock"):
        result = subprocess.run([sys.executable, "-m", "runtime.task_queue", "--db",
            str(queue.path), "work", "--once"], capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert "Already locked" in result.stderr


def test_worker_restart_adopts_live_runner_without_duplicate(queue, tmp_path):
    starts, release = tmp_path / "starts", tmp_path / "release"
    job = command(queue, tmp_path,
        "import sys,time; from pathlib import Path; p,r=map(Path,sys.argv[1:]); "
        "f=p.open('a'); f.write('started\\n'); f.close(); "
        "exec('while not r.exists(): time.sleep(.02)')", starts, release)
    first = worker(queue)
    second = None
    try:
        eventually(starts.exists)
        first.kill()  # A daemon crash must not terminate the detached runner.
        first.wait(timeout=5)
        second = worker(queue)
        time.sleep(.2)
        assert starts.read_text().splitlines() == ["started"]
        assert queue.get(job)["attempt"] == 1
        release.touch()
        eventually(lambda: queue.get(job)["status"] == "succeeded")
    finally:
        release.touch()
        stop(first)
        if second is not None:
            stop(second)


def test_argv_shell_characters_are_literal(queue, tmp_path):
    output = tmp_path / "arguments.json"
    arguments = ["$(touch SHOULD_NOT_EXIST)", "`echo injected`", "a;b", "$HOME", "a\nb", "'\"\\"]
    job = command(queue, tmp_path,
        "import sys,json; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))",
        output, *arguments)
    eventually(lambda: completed(queue, job))
    assert json.loads(output.read_text()) == arguments
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


def test_pause_finishes_current_job_but_holds_next(queue, tmp_path):
    started, release = tmp_path / "started", tmp_path / "release"
    first = command(queue, tmp_path,
        "import sys,time; from pathlib import Path; s,r=map(Path,sys.argv[1:]); s.touch(); "
        "exec('while not r.exists(): time.sleep(.02)')", started, release)
    second = command(queue, tmp_path, "pass")
    queue.tick()
    try:
        eventually(started.exists)
        queue.pause(True)
        queue.tick()
        assert queue.get(first)["status"] == "running"
        assert alive(queue.get(first)["runner"])
    finally:
        release.touch()
    eventually(lambda: queue.get(first)["status"] == "succeeded")
    assert queue.tick() == "paused"
    assert queue.get(second)["status"] == "queued"
    queue.pause(False)
    eventually(lambda: completed(queue, second))


def test_agent_claim_is_exclusive_and_finishing_requires_owner(queue):
    job = queue.add("agent", "agent", dict(prompt="inspect", cwd="/tmp"))["id"]
    assert queue.tick() == "needs_agent"
    other = Queue(queue.path)
    try:
        queue.claim_agent(job, "owner-one")
        with pytest.raises(ValueError):
            other.claim_agent(job, "owner-two")
        with pytest.raises(ValueError):
            other.finish_agent(job, "owner-two", True)
        queue.finish_agent(job, "owner-one", True)
    finally:
        other.close()
    assert queue.get(job)["status"] == "succeeded"


def test_killed_runner_child_holds_lock_until_exit(queue, tmp_path):
    started, release = tmp_path / "child.pid", tmp_path / "release-child"
    job = command(queue, tmp_path,
        "import os,sys,time; from pathlib import Path; p,r=map(Path,sys.argv[1:]); "
        "p.write_text(str(os.getpid())); "
        "exec('while not r.exists(): time.sleep(.02)')", started, release)
    queue.tick()
    child_identity = None
    try:
        eventually(started.exists)
        eventually(lambda: bool(started.read_text()))
        child_identity = process_identity(int(started.read_text()))
        runner = queue.get(job)["runner"]
        assert alive(child_identity)
        os.kill(runner["pid"], signal.SIGKILL)
        eventually(lambda: not alive(runner))
        queue.conn.execute("UPDATE jobs SET updated=? WHERE id=?", (time.time() - 20, job))
        queue.tick()
        assert queue.get(job)["status"] == "needs_attention"
        assert locked(queue.directory(job) / "run.lock")
        assert alive(child_identity)
        for action in ("retry", "skip"):
            with pytest.raises(ValueError, match="live runner or child"):
                queue.control(job, action)
        # Merely re-entering the scheduler cannot start another runner.
        queue.tick()
        assert queue.get(job)["attempt"] == 1
    finally:
        release.touch()
        if child_identity:
            eventually(lambda: not alive(child_identity))
        eventually(lambda: not locked(queue.directory(job) / "run.lock"))
    # Exit without the runner's committed completion remains ambiguous.
    assert queue.get(job)["status"] == "needs_attention"
    queue.control(job, "skip")
    assert queue.get(job)["status"] == "skipped"


def test_simultaneous_append_clients_do_not_lose_jobs(queue, tmp_path):
    code = (
        "import sys,time; from pathlib import Path; from runtime.task_queue import Queue; "
        "db,root,label,gate=sys.argv[1:]; "
        "exec('while not Path(gate).exists(): time.sleep(.01)'); "
        "q=Queue(db); "
        "exec('for i in range(12): q.add(label+str(i), \\\"command\\\", "
        "dict(cwd=root,steps=[[sys.executable,\\\"-c\\\",\\\"pass\\\"]]))'); q.close()"
    )
    gate = tmp_path / "append-gate"
    clients = [subprocess.Popen([sys.executable, "-c", code, str(queue.path),
        str(tmp_path), f"client-{i}-", str(gate)], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True) for i in range(4)]
    try:
        gate.touch()
        for client in clients:
            stdout, stderr = client.communicate(timeout=10)
            assert client.returncode == 0, (stdout, stderr)
    finally:
        for client in clients:
            stop(client)
    jobs = queue.jobs()
    assert len(jobs) == 48
    assert len({job["id"] for job in jobs}) == 48
    assert len({job["title"] for job in jobs}) == 48
    for client in range(4):
        suffixes = [int(job["title"].split("-")[-1]) for job in jobs
                    if job["title"].startswith(f"client-{client}-")]
        assert suffixes == list(range(12))
