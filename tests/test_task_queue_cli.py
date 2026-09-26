"""Exercise the user-facing CLI outside the repository and without PYTHONPATH."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from runtime.task_queue import Queue


CLI = Path(__file__).resolve().parents[1] / 'scripts/queue.py'


def test_standalone_cli_snapshots_script_and_runs_without_pythonpath(tmp_path):
    db = tmp_path / 'queue.sqlite'
    script = tmp_path / 'job.sh'
    script.write_text("printf '%s' 'original $literal' > output.txt\n")
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    def cli(*args):
        return subprocess.run([sys.executable, str(CLI), '--db', str(db), *args],
                              env=env, cwd=tmp_path, check=True, text=True, capture_output=True)
    cli('append', '--title', 'script', '--cwd', str(tmp_path), '--script', str(script))
    script.write_text('exit 17\n')
    cli('work', '--once')
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = json.loads(cli('get', '1').stdout)
        if job['status'] == 'succeeded':
            break
        time.sleep(.05)
    assert job['status'] == 'succeeded'
    assert (tmp_path / 'output.txt').read_text() == 'original $literal'
    assert 'succeeded' in cli('status').stdout


def test_finish_agent_receipt_is_idempotent(tmp_path):
    q = Queue(tmp_path / 'queue.sqlite')
    try:
        job = q.add('review', 'agent', dict(prompt='review'))['id']
        q.tick()
        q.claim_agent(job, 'receipt-1')
        q.finish_agent(job, 'receipt-1', True)
        q.finish_agent(job, 'receipt-1', True)
        with pytest.raises(ValueError):
            q.finish_agent(job, 'receipt-1', False, 'different result')
        with pytest.raises(ValueError):
            q.finish_agent(job, 'wrong-claim', True)
    finally:
        q.close()


def test_stale_transition_cannot_overwrite_completed_or_skipped_job(tmp_path):
    q = Queue(tmp_path / 'queue.sqlite')
    try:
        job = q.add('review', 'agent', dict(prompt='review'))['id']
        q.control(job, 'skip')
        assert not q.transition(job, 'needs_agent', expected={'queued'})
        assert q.get(job)['status'] == 'skipped'
        job = q.add('review2', 'agent', dict(prompt='review'))['id']
        q.transition(job, 'succeeded')
        assert not q.transition(job, 'needs_attention', expected={'running', 'starting'})
        assert q.get(job)['status'] == 'succeeded'
    finally:
        q.close()


def test_zero_exit_requires_declared_result(tmp_path):
    q = Queue(tmp_path / 'queue.sqlite')
    try:
        job = q.add('verify', 'command', dict(cwd=str(tmp_path), steps=[[sys.executable, '-c', 'pass']],
            success=dict(path=str(tmp_path / 'missing.json'), equals=dict(complete=True))))['id']
        q.tick()
        deadline = time.monotonic() + 10
        while q.get(job)['status'] not in {'failed', 'succeeded'} and time.monotonic() < deadline:
            time.sleep(.03)
        assert q.get(job)['status'] == 'failed'
        assert q.get(job)['exit_code'] == 0
    finally:
        q.close()
