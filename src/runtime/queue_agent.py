"""Optional WSL Codex consumer for runtime.task_queue agent jobs.

State and reports are local, atomic JSON records (no content hashes). An
interrupted claim is deliberately never replayed as another model invocation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import uuid

DEFAULT_STATE = Path.home() / 'Projects/codex-zcode-handoff/task-queue-20260914/bridge'


@contextmanager
def bridge_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / 'bridge.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another bridge is running for this state directory') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_record(path: Path, record: dict):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Bridge:
    def __init__(self, args):
        self.args = args
        self.directory = Path(args.state_dir).expanduser().resolve()
        self.identity = {'host': args.host, 'db': args.db, 'remote_cwd': args.remote_cwd}

    def remote(self, *arguments):
        command = 'cd ' + shlex.quote(self.args.remote_cwd) + ' && ' + shlex.join([
            'env', 'PYTHONPATH=src', self.args.remote_python, '-m',
            'runtime.task_queue', '--db', self.args.db, *map(str, arguments)])
        result = subprocess.run(
            ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
             self.args.host, command], capture_output=True, text=True,
            timeout=self.args.ssh_timeout)
        if result.returncode:
            # stderr may contain sensitive remote output; do not relay it.
            raise RuntimeError(f'remote {arguments[0]} failed (exit {result.returncode})')
        return json.loads(result.stdout) if result.stdout.strip() else None

    def finish(self, path, record):
        arguments = ['finish-agent', record['job_id'], '--claim', record['claim']]
        arguments += ['--success'] if record['success'] else ['--error', record['error']]
        self.remote(*arguments)
        record['state'] = 'finished'
        write_record(path, record)
        return {'status': 'finished', 'job_id': record['job_id'], 'success': record['success']}

    def tick(self):
        # Recover acknowledgements only. A claim without a durable result requires
        # inspection, even if the crash might have happened before Codex started.
        for path in sorted(self.directory.glob('run-*.json')):
            record = json.loads(path.read_text())
            if record['queue'] != self.identity:
                raise RuntimeError('state directory belongs to another queue; use a separate --state-dir')
            if record['state'] == 'result_ready':
                return self.finish(path, record)
            if record['state'] in ('claiming', 'claimed', 'needs_attention'):
                record['state'] = 'needs_attention'
                write_record(path, record)
                return {'status': 'needs_attention', 'job_id': record['job_id'], 'record': str(path)}
        job = self.remote('agent-next')
        if job is None:
            return {'status': 'idle'}
        claim = str(uuid.uuid4())
        path = self.directory / f'run-{claim}.json'
        record = {'queue': self.identity, 'job_id': job['id'], 'claim': claim,
                  'state': 'claiming', 'created_at': time.time(), 'job': job}
        write_record(path, record)
        try:
            self.remote('claim-agent', job['id'], '--claim', claim)
        except (RuntimeError, subprocess.TimeoutExpired):
            # Network failure may mean the claim committed but its reply was lost.
            record['state'] = 'needs_attention'
            write_record(path, record)
            raise
        record['state'] = 'claimed'
        write_record(path, record)
        report = path.with_suffix('.report.md')
        events = path.with_suffix('.events.jsonl')
        errors = path.with_suffix('.stderr.log')
        record.update(report=str(report), events=str(events), stderr=str(errors))
        write_record(path, record)
        try:
            spec = job['spec']
            cwd = str(Path(spec.get('cwd') or self.args.cwd).expanduser().resolve())
            prompt = str(spec['prompt'])
            docs = spec.get('docs', [])
            prompt += '\n\nQueue context: ' + json.dumps({
                'job_id': job['id'], 'title': job['title'], 'remote': self.identity,
                'reference_documents': docs}, ensure_ascii=False)
            prompt += ('\nRead applicable AGENTS.md and the referenced documents before acting. '
                       'Work only within this job scope. Report actual changes, validation, '
                       'remaining blockers and next steps. Do not complete this queue job yourself; '
                       'the bridge records your result.')
            command = [self.args.codex, 'exec', '--json', '-s', 'danger-full-access',
                       '-c', 'approval_policy="never"', '--skip-git-repo-check',
                       '-C', cwd, '-o', str(report), '-']
            with events.open('w') as stdout, errors.open('w') as stderr:
                result = subprocess.run(command, input=prompt, text=True,
                                        stdout=stdout, stderr=stderr)
            completed = False
            failed = False
            with events.open() as event_stream:
                for line in event_stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    completed |= event.get('type') == 'turn.completed'
                    failed |= event.get('type') == 'turn.failed'
            success = result.returncode == 0 and completed and not failed and report.is_file() and report.stat().st_size > 0
            error = '' if success else f'Codex did not produce a completed turn and report (exit {result.returncode}); inspect local logs'
        except Exception as exc:
            success, error = False, f'Agent invocation failed ({type(exc).__name__}); inspect local state'
        record.update(state='result_ready', success=success, error=error, completed_at=time.time())
        write_record(path, record)
        return self.finish(path, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='exp')
    parser.add_argument('--db', required=True)
    parser.add_argument('--remote-cwd', required=True)
    parser.add_argument('--remote-python', default='python')
    parser.add_argument('--cwd', default=str(Path.cwd()), help='default local WSL Agent cwd')
    parser.add_argument('--state-dir', default=str(DEFAULT_STATE))
    parser.add_argument('--codex', default='codex')
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--ssh-timeout', type=float, default=60)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.ssh_timeout <= 0:
        parser.error('interval and ssh-timeout must be positive')
    bridge = Bridge(args)
    try:
        with bridge_lock(bridge.directory):
            while True:
                try:
                    result = bridge.tick()
                except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
                    result = {'status': 'error', 'error': type(exc).__name__}
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if args.once or result['status'] == 'needs_attention':
                    return 0 if result['status'] in ('idle', 'finished') else 1
                if result['status'] != 'finished':
                    time.sleep(args.interval)
    except RuntimeError as exc:
        print(json.dumps({'status': 'error', 'error': str(exc)}), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
