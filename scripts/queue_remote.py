#!/usr/bin/env python3
"""Control a queue over outbound SSH; no public client IP is needed.

Examples: queue_remote.py --host exp --db /data/queue.sqlite status
          queue_remote.py --host exp --db /data/queue.sqlite append --title job -- python job.py
Script/sequence/prompt paths are on the remote host. Upload local files first.
"""
import argparse
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='exp')
    parser.add_argument('--db', required=True)
    parser.add_argument('--repo', default='/root/TS3')
    parser.add_argument('--python', default='/root/miniconda3/envs/ts3/bin/python')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error('provide a queue command such as status or append')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    return subprocess.call(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', args.host,
                            shlex.join([args.python, args.repo.rstrip('/') + '/scripts/queue.py',
                                        '--db', args.db, *command])])


if __name__ == '__main__':
    raise SystemExit(main())
