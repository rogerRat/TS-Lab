"""Persistent FIFO task queue; append and control jobs without stopping workers.

Only the Python standard library is required. Run the CLI on the machine that
owns the database (never open a SQLite database over SSHFS).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

TERMINAL = {'succeeded', 'skipped'}
RETRYABLE = {'failed', 'needs_attention'}


def process_identity(pid):
    try:
        text = Path(f'/proc/{int(pid)}/stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=int(pid), start_ticks=fields[19],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (OSError, ValueError, IndexError):
        return None


def alive(identity):
    return bool(identity) and process_identity(identity['pid']) == identity


@contextmanager
def lock(path):
    stream = Path(path).open('a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError('Already locked: ' + str(path)) from None
    try:
        yield stream
    finally:
        stream.close()


def locked(path):
    try:
        with lock(path):
            return False
    except RuntimeError:
        return True


def field(value, dotted):
    for part in dotted.split('.'):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def check_json(rule):
    path = Path(rule['path'])
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    try:
        return all(field(value, key) == expected for key, expected in rule['equals'].items())
    except (KeyError, IndexError, TypeError):
        return False


class Queue:
    def __init__(self, db):
        self.path = Path(db).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts = self.path.parent / (self.path.stem + '.jobs')
        self.artifacts.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA synchronous=FULL')
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS jobs (
              id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL,
              kind TEXT NOT NULL, spec TEXT NOT NULL, status TEXT NOT NULL,
              attempt INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL,
              updated REAL NOT NULL, runner TEXT, claim TEXT, message TEXT,
              exit_code INTEGER);
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
            INSERT OR IGNORE INTO settings VALUES ('paused', 'false');
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER,
              time REAL NOT NULL, event TEXT NOT NULL, detail TEXT);
        ''')

    def close(self):
        self.conn.close()

    @contextmanager
    def transaction(self):
        self.conn.execute('BEGIN IMMEDIATE')
        try:
            yield
        except BaseException:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()

    def event(self, job, name, detail=''):
        self.conn.execute('INSERT INTO events(job_id,time,event,detail) VALUES(?,?,?,?)',
                          (job, time.time(), name, detail))

    def get(self, job_id):
        row = self.conn.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None:
            raise ValueError('Unknown job: ' + str(job_id))
        value = dict(row)
        value['spec'] = json.loads(value['spec'])
        value['runner'] = json.loads(value['runner']) if value['runner'] else None
        return value

    def jobs(self):
        return [self.get(row[0]) for row in self.conn.execute('SELECT id FROM jobs ORDER BY id')]

    def directory(self, job_id):
        path = self.artifacts / str(job_id)
        path.mkdir(exist_ok=True)
        return path

    def add(self, title, kind, spec):
        if kind not in {'command', 'watch', 'agent'}:
            raise ValueError('Unknown job kind')
        if kind == 'command':
            steps = spec.get('steps')
            if not isinstance(steps, list) or not steps:
                raise ValueError('Command requires nonempty steps: list of argv lists')
            for argv in steps:
                if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
                    raise ValueError('Each step must be a nonempty list of strings')
            if not Path(spec.get('cwd', '')).is_dir():
                raise ValueError('Command cwd must be an existing directory')
        if kind == 'watch':
            if not spec.get('success', {}).get('equals'):
                raise ValueError('Watch requires explicit JSON success evidence')
            if not spec.get('identity') and not spec.get('file_only'):
                raise ValueError('Watch requires a live process identity or explicit file_only')
        if kind == 'agent' and not spec.get('prompt'):
            raise ValueError('Agent requires a prompt')
        now = time.time()
        with self.transaction():
            cur = self.conn.execute('INSERT INTO jobs(title,kind,spec,status,created,updated) VALUES(?,?,?,?,?,?)',
                                    (title, kind, json.dumps(spec), 'queued', now, now))
            job_id = cur.lastrowid
            self.event(job_id, 'appended', title)
        return self.get(job_id)

    def transition(self, job_id, status, message='', exit_code=None, expected=None):
        with self.transaction():
            if expected is not None and self.get(job_id)['status'] not in expected:
                return False
            self.conn.execute('UPDATE jobs SET status=?,updated=?,message=?,exit_code=? WHERE id=?',
                              (status, time.time(), message, exit_code, job_id))
            self.event(job_id, status, message)
        return True

    def pause(self, paused):
        with self.transaction():
            self.conn.execute("UPDATE settings SET value=? WHERE key='paused'", (json.dumps(paused),))
            self.event(None, 'paused' if paused else 'resumed')

    @property
    def paused(self):
        return json.loads(self.conn.execute("SELECT value FROM settings WHERE key='paused'").fetchone()[0])

    def control(self, job_id, action):
        with self.transaction():
            job = self.get(job_id)
            if locked(self.directory(job_id) / 'run.lock') or alive(job['runner']):
                raise ValueError('Job still has a live runner or child; it will not be interrupted')
            if action == 'retry' and job['status'] not in RETRYABLE:
                raise ValueError('Retry requires failed or needs_attention status')
            if action == 'skip' and job['status'] not in RETRYABLE | {'queued', 'needs_agent', 'waiting'}:
                raise ValueError('Cannot skip active or successful job')
            if action == 'skip' and job['kind'] == 'watch' and alive(job['spec'].get('identity')):
                raise ValueError('Observed external process is still running; skip would violate ordering')
            status = 'queued' if action == 'retry' else 'skipped'
            self.conn.execute('UPDATE jobs SET status=?,runner=NULL,claim=NULL,message=NULL,updated=? WHERE id=?',
                              (status, time.time(), job_id))
            self.event(job_id, action)

    def agent_next(self):
        row = self.conn.execute("SELECT id FROM jobs WHERE status='needs_agent' ORDER BY id LIMIT 1").fetchone()
        return self.get(row[0]) if row else None

    def claim_agent(self, job_id, claim):
        with self.transaction():
            job = self.get(job_id)
            if job['status'] != 'needs_agent' or self.paused:
                raise ValueError('Agent job not claimable')
            self.conn.execute("UPDATE jobs SET status='agent_running',claim=?,updated=? WHERE id=?",
                              (claim, time.time(), job_id))
            self.event(job_id, 'agent_claimed', claim)

    def finish_agent(self, job_id, claim, success, error=''):
        with self.transaction():
            job = self.get(job_id)
            status = 'succeeded' if success else 'failed'
            if job['claim'] == claim and job['status'] == status and (job['message'] or '') == error:
                return  # A lost SSH reply may repeat the same completion receipt.
            if job['status'] != 'agent_running' or job['claim'] != claim:
                raise ValueError('Agent claim does not match active job')
            self.conn.execute('UPDATE jobs SET status=?,message=?,updated=? WHERE id=?',
                              (status, error, time.time(), job_id))
            self.event(job_id, status, error)

    def tick(self):
        # Called only while holding the daemon lock. Appenders and controls use
        # short SQLite transactions and never wait for the running command.
        pending = [j for j in self.jobs() if j['status'] not in TERMINAL]
        if not pending:
            return 'idle'
        job = pending[0]
        job_id, spec, status = job['id'], job['spec'], job['status']
        if status in {'starting', 'running'}:
            busy = locked(self.directory(job_id) / 'run.lock')
            if not alive(job['runner']) and time.time() - job['updated'] > 10:
                self.transition(job_id, 'needs_attention',
                                'Runner vanished; child may still be active' if busy else
                                'Runner vanished without an exit record; inspect before retry',
                                expected={'starting', 'running'})
            return status
        if status not in {'queued', 'waiting'}:
            return status
        if self.paused:
            return 'paused'
        if job['kind'] == 'watch':
            try:
                completed = check_json(spec['success'])
                identity = spec.get('identity')
                is_alive = alive(identity)
                if completed and (not spec.get('require_exit', True) or not is_alive):
                    self.transition(job_id, 'succeeded', 'External result verified', expected={'queued', 'waiting'})
                elif spec.get('error_file') and Path(spec['error_file']).exists():
                    self.transition(job_id, 'failed', 'External error file: ' + spec['error_file'], expected={'queued', 'waiting'})
                elif identity and not is_alive and not completed:
                    self.transition(job_id, 'failed', 'External process exited without success evidence', expected={'queued', 'waiting'})
                elif spec.get('stale_seconds') and Path(spec['success']['path']).exists() and \
                        time.time() - Path(spec['success']['path']).stat().st_mtime > spec['stale_seconds']:
                    self.transition(job_id, 'needs_attention', 'External progress file is stale', expected={'queued', 'waiting'})
                elif status != 'waiting':
                    self.transition(job_id, 'waiting', 'Observing existing job; no restart', expected={'queued'})
            except (ValueError, OSError, TypeError) as exc:
                self.transition(job_id, 'needs_attention', 'Cannot validate external result: ' + str(exc), expected={'queued', 'waiting'})
            return self.get(job_id)['status']
        if job['kind'] == 'agent':
            self.transition(job_id, 'needs_agent', 'Waiting for optional WSL agent bridge', expected={'queued'})
            return 'needs_agent'
        with self.transaction():
            # Persist launch intent before spawning: a crash in this gap is
            # ambiguous and requires inspection, never an automatic rerun.
            if self.get(job_id)['status'] != 'queued' or self.paused:
                return 'changed'
            self.conn.execute("UPDATE jobs SET status='starting',attempt=attempt+1,updated=? WHERE id=?",
                              (time.time(), job_id))
            self.event(job_id, 'starting')
        try:
            runner_env = dict(os.environ)
            runner_env['PYTHONPATH'] = os.pathsep.join(filter(None, (
                str(Path(__file__).resolve().parents[1]), os.environ.get('PYTHONPATH'))))
            with (self.directory(job_id) / 'runner.log').open('ab') as log:
                subprocess.Popen([sys.executable, '-m', 'runtime.task_queue', '--db', str(self.path),
                                          '_run', str(job_id)], start_new_session=True,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                         env=runner_env)
            # Runner writes its own identity before launching any user command.
            return 'starting'
        except OSError as exc:
            self.transition(job_id, 'failed', 'Could not launch runner: ' + str(exc))
            return 'failed'

    def run_job(self, job_id):
        directory = self.directory(job_id)
        with lock(directory / 'run.lock') as guard:
            with self.transaction():
                job = self.get(job_id)
                if job['status'] != 'starting':
                    raise ValueError('Job was not claimed by the worker')
                self.conn.execute("UPDATE jobs SET status='running',runner=?,updated=? WHERE id=?",
                                  (json.dumps(process_identity(os.getpid())), time.time(), job_id))
                self.event(job_id, 'running')
            spec = job['spec']
            env = dict(os.environ, **spec.get('env', {}))
            env['TASK_QUEUE_JOB_ID'] = str(job_id)
            env['TASK_QUEUE_DB'] = str(self.path)
            try:
                with (directory / f"attempt-{job['attempt']}.log").open('ab', buffering=0) as log:
                    for number, argv in enumerate(spec['steps']):
                        self.event(job_id, 'step_started', str(number))
                        # Child inherits the lock so an orphan cannot overlap a retry.
                        child = subprocess.Popen(argv, cwd=spec['cwd'], env=env,
                                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                                 pass_fds=(guard.fileno(),))
                        code = child.wait()
                        if code != 0:
                            self.transition(job_id, 'failed', f'Step {number} exited {code}', code)
                            return
                if spec.get('success') and not check_json(spec['success']):
                    self.transition(job_id, 'failed', 'Command exited zero but result check failed', 0)
                else:
                    self.transition(job_id, 'succeeded', 'All steps completed', 0)
            except BaseException as exc:
                self.transition(job_id, 'needs_attention', repr(exc))
                raise


def parse_env(values):
    env = {}
    for value in values:
        key, sep, text = value.partition('=')
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError('Environment must be NAME=value')
        env[key] = text
    return env


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', required=True, type=Path)
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('init')
    add = sub.add_parser('append', help='Append argv, a snapshotted shell script, or a step sequence')
    add.add_argument('--title', required=True)
    add.add_argument('--cwd', default=os.getcwd())
    add.add_argument('--env', action='append', default=[])
    add.add_argument('--script', type=Path)
    add.add_argument('--sequence', type=Path, help='JSON array of argv arrays')
    add.add_argument('--success-json', type=Path, help='JSON rule {path, equals}')
    add.add_argument('argv', nargs=argparse.REMAINDER)
    watch = sub.add_parser('attach', help='Observe existing process without restarting it')
    watch.add_argument('--title', required=True)
    watch.add_argument('--pid', type=int)
    watch.add_argument('--file-only', action='store_true')
    watch.add_argument('--status-file', type=Path, required=True)
    watch.add_argument('--equals', required=True, help='JSON object of required success field values')
    watch.add_argument('--error-file', type=Path)
    watch.add_argument('--stale-seconds', type=float, default=0)
    agent = sub.add_parser('append-agent')
    agent.add_argument('--title', required=True)
    agent.add_argument('--prompt-file', required=True, type=Path)
    agent.add_argument('--cwd', required=True, help='Working directory on WSL')
    agent.add_argument('--doc', action='append', default=[])
    sub.add_parser('list')
    sub.add_parser('status', help='Human-readable queue status')
    log_cmd = sub.add_parser('logs', help='Print the latest attempt log')
    log_cmd.add_argument('id', type=int)
    log_cmd.add_argument('--lines', type=int, default=30)
    for action in ('get', 'retry', 'skip', '_run'):
        sub.add_parser(action).add_argument('id', type=int)
    sub.add_parser('pause')
    sub.add_parser('resume')
    worker = sub.add_parser('work')
    worker.add_argument('--interval', type=float, default=1)
    worker.add_argument('--once', action='store_true')
    sub.add_parser('agent-next')
    claim = sub.add_parser('claim-agent')
    claim.add_argument('id', type=int)
    claim.add_argument('--claim', required=True)
    finish = sub.add_parser('finish-agent')
    finish.add_argument('id', type=int)
    finish.add_argument('--claim', required=True)
    outcome = finish.add_mutually_exclusive_group(required=True)
    outcome.add_argument('--success', action='store_true')
    outcome.add_argument('--error')
    sub.add_parser('events')
    args = p.parse_args(argv)
    q = Queue(args.db)
    result = None
    try:
        if args.action == 'append':
            command = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
            if sum(bool(x) for x in (command, args.script, args.sequence)) != 1:
                raise ValueError('Provide exactly one of argv, --script, --sequence')
            steps = (json.loads(args.sequence.read_text()) if args.sequence else
                     [['/bin/bash', '-euo', 'pipefail', '-c', args.script.read_text()]] if args.script else [command])
            spec = dict(steps=steps, cwd=str(Path(args.cwd).resolve()), env=parse_env(args.env))
            if args.success_json:
                spec['success'] = json.loads(args.success_json.read_text())
            result = q.add(args.title, 'command', spec)
        elif args.action == 'attach':
            identity = process_identity(args.pid) if args.pid else None
            if bool(args.pid) == bool(args.file_only) or (args.pid and not identity):
                raise ValueError('Choose a live --pid or explicit --file-only')
            result = q.add(args.title, 'watch', dict(identity=identity, file_only=args.file_only,
                success=dict(path=str(args.status_file.resolve()), equals=json.loads(args.equals)),
                error_file=str(args.error_file.resolve()) if args.error_file else None,
                stale_seconds=args.stale_seconds, require_exit=True))
        elif args.action == 'append-agent':
            result = q.add(args.title, 'agent', dict(prompt=args.prompt_file.read_text(), cwd=args.cwd, docs=args.doc))
        elif args.action == 'get':
            result = q.get(args.id)
        elif args.action in {'skip', 'retry'}:
            q.control(args.id, args.action)
            result = q.get(args.id)
        elif args.action == 'list':
            result = dict(paused=q.paused, jobs=q.jobs())
        elif args.action == 'status':
            print('Queue: ' + str(q.path) + (' [PAUSED]' if q.paused else ''))
            print(f"{'ID':>5}  {'STATE':<17} {'TYPE':<8} {'TRY':>3}  TITLE")
            for job in q.jobs():
                print(f"{job['id']:>5}  {job['status']:<17} {job['kind']:<8} {job['attempt']:>3}  {job['title']}")
                if job['message'] and job['status'] in RETRYABLE:
                    print('       ' + job['message'])
            return 0
        elif args.action == 'logs':
            from collections import deque
            job = q.get(args.id)
            log_path = q.directory(args.id) / f"attempt-{job['attempt']}.log"
            if not log_path.exists():
                print('No command log; use get for external/Agent job metadata.')
            else:
                with log_path.open(errors='replace') as stream:
                    print(''.join(deque(stream, maxlen=max(0, args.lines))), end='')
            return 0
        elif args.action in {'pause', 'resume'}:
            q.pause(args.action == 'pause')
            result = dict(paused=q.paused)
        elif args.action == 'work':
            if args.interval <= 0:
                raise ValueError('Interval must be positive')
            with lock(str(q.path) + '.worker.lock'):
                while True:
                    result = q.tick()
                    if args.once:
                        break
                    time.sleep(args.interval)
        elif args.action == '_run':
            q.run_job(args.id)
        elif args.action == 'agent-next':
            result = q.agent_next()
        elif args.action == 'claim-agent':
            q.claim_agent(args.id, args.claim)
            result = q.get(args.id)
        elif args.action == 'finish-agent':
            q.finish_agent(args.id, args.claim, args.success, args.error or '')
            result = q.get(args.id)
        elif args.action == 'events':
            result = [dict(row) for row in q.conn.execute('SELECT * FROM events ORDER BY id')]
        elif args.action == 'init':
            result = dict(db=str(q.path), initialized=True)
        if args.action != '_run':
            print(json.dumps(result, ensure_ascii=False))
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        q.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
