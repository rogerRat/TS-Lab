import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from runtime.queue_agent import Bridge, bridge_lock, write_record


class QueueAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.bridge = Bridge(argparse.Namespace(
            state_dir=self.temp.name, host='exp', db='/tmp/queue space;$(bad).db',
            remote_cwd='/tmp/repo space;$(bad)', remote_python='/venv/bin/python',
            ssh_timeout=10, cwd=self.temp.name, codex='fake-codex'))

    def fake_process(self, *, code=0, completed=True, report=True, finish_failure=False):
        calls = []
        job = {'id': 7, 'kind': 'agent', 'status': 'needs_agent', 'title': 'check results',
               'spec': {'prompt': 'Inspect results; $(do not run)', 'docs': ['docs/TODO.md']}}

        def run(argv, **kwargs):
            calls.append((argv, kwargs))
            if argv[0] == 'ssh':
                remote = shlex.split(argv[-1])
                command = remote[remote.index('--db') + 2]
                if command == 'finish-agent' and finish_failure:
                    raise subprocess.TimeoutExpired(argv, 10)
                return subprocess.CompletedProcess(argv, 0, json.dumps(job if command == 'agent-next' else {}), '')
            self.assertEqual(argv[0], 'fake-codex')
            self.assertEqual(argv[-1], '-')
            for disallowed in ('-m', '--model', 'resume'):
                self.assertNotIn(disallowed, argv)
            self.assertIn('Inspect results; $(do not run)', kwargs['input'])
            if completed:
                kwargs['stdout'].write('{"type":"turn.completed"}\n')
            if report:
                Path(argv[argv.index('-o') + 1]).write_text('Completed inspection.')
            return subprocess.CompletedProcess(argv, code)

        return run, calls

    def test_mutex(self):
        with bridge_lock(self.directory):
            with self.assertRaisesRegex(RuntimeError, 'another bridge'):
                with bridge_lock(self.directory):
                    self.fail('second bridge acquired lock')
        with bridge_lock(self.directory):
            pass

    def test_success_and_shell_quoting(self):
        run, calls = self.fake_process()
        with patch('runtime.queue_agent.subprocess.run', run), bridge_lock(self.directory):
            self.assertEqual(self.bridge.tick(), {'status': 'finished', 'job_id': 7, 'success': True})
        remote = shlex.split(calls[0][0][-1])
        self.assertEqual(remote[:3], ['cd', self.bridge.args.remote_cwd, '&&'])
        self.assertEqual(remote[remote.index('--db') + 1], self.bridge.args.db)
        self.assertEqual(calls[0][0][0:3], ['ssh', '-o', 'BatchMode=yes'])
        self.assertIn('--success', shlex.split(calls[-1][0][-1]))
        record = json.loads(next(self.directory.glob('run-*.json')).read_text())
        self.assertEqual(record['state'], 'finished')
        self.assertEqual(Path(record['report']).read_text(), 'Completed inspection.')

    def test_failure_requires_exit_event_and_report(self):
        for code, completed, report in [(1, True, True), (0, False, True), (0, True, False)]:
            with self.subTest(code=code, completed=completed, report=report):
                run, calls = self.fake_process(code=code, completed=completed, report=report)
                with patch('runtime.queue_agent.subprocess.run', run), bridge_lock(self.directory):
                    self.assertFalse(self.bridge.tick()['success'])
                self.assertIn('--error', shlex.split(calls[-1][0][-1]))

    def test_interrupted_claim_never_reinvokes_model(self):
        for state in ['claiming', 'claimed', 'needs_attention']:
            with self.subTest(state=state):
                write_record(self.directory / 'run-test.json', {
                    'queue': self.bridge.identity, 'state': state, 'claim': 'token', 'job_id': 7})
                with patch('runtime.queue_agent.subprocess.run', side_effect=AssertionError('must not invoke external process')), bridge_lock(self.directory):
                    self.assertEqual(self.bridge.tick()['status'], 'needs_attention')
                    self.assertEqual(self.bridge.tick()['status'], 'needs_attention')

    def test_lost_finish_reply_only_replays_ack(self):
        run, calls = self.fake_process(finish_failure=True)
        with bridge_lock(self.directory):
            with patch('runtime.queue_agent.subprocess.run', run), self.assertRaises(subprocess.TimeoutExpired):
                self.bridge.tick()
            record = json.loads(next(self.directory.glob('run-*.json')).read_text())
            self.assertEqual(record['state'], 'result_ready')
            run, calls = self.fake_process()
            with patch('runtime.queue_agent.subprocess.run', run):
                self.assertTrue(self.bridge.tick()['success'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], 'ssh')
        self.assertIn('finish-agent', shlex.split(calls[0][0][-1]))

    def test_uncertain_claim_blocks_replay(self):
        def remote(command, *args):
            if command == 'agent-next':
                return {'id': 8, 'title': 'job', 'spec': {}}
            raise subprocess.TimeoutExpired('ssh', 10)
        with patch.object(self.bridge, 'remote', remote), bridge_lock(self.directory):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.bridge.tick()
            self.assertEqual(self.bridge.tick()['status'], 'needs_attention')

    def test_state_directory_cannot_mix_queues(self):
        write_record(self.directory / 'run-old.json', {'queue': {'host': 'other'}, 'state': 'finished'})
        with bridge_lock(self.directory), self.assertRaisesRegex(RuntimeError, 'another queue'):
            self.bridge.tick()


if __name__ == '__main__':
    unittest.main()
