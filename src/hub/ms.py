#!/usr/bin/env python3
# Purpose:
#   Upload/download ModelScope Hub repos through the `ms` or `modelscope` CLI.
#
# Dependencies:
#   pip install -U modelscope-hub
#   # or, if you still use old ModelScope package:
#   pip install -U modelscope
#
# Usage:
#   # Run these commands from the project root.
#   export MODELSCOPE_API_TOKEN=<your_token>
#   # or export MS_TOKEN=<your_token>
#
#   Upload:
#     python -m hub.ms upload /path/to/folder \
#       --namespace rogerspyke
#
#   By default the repository name is derived from the folder basename and the
#   repository is created as a dataset. Pass --repo-name and/or
#   --repo-type model to override those defaults.
#
#   Download:
#     python -m hub.ms download rogerspyke/my_model \
#       --repo-type model \
#       --local-dir ./my_model

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Iterable, List, Optional, TypeVar

from configs.logger import configure_logger


from configs.paths import LOGS as LOG_DIR

DEFAULT_NAMESPACE = "rogerspyke"

# New official env name in modelscope-hub is MODELSCOPE_API_TOKEN.
# Keep MS_TOKEN for compatibility with your old script.
ENV_TOKEN_NAMES = ("MODELSCOPE_API_TOKEN", "MS_TOKEN")


def _utc8_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=8)))


def _setup_logging(task: str = "ms") -> logging.Logger:
    return configure_logger("hub.ms", "hub", task, "HUB/MS")


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token and cli_token.strip():
        return cli_token.strip()

    for name in ENV_TOKEN_NAMES:
        t = os.environ.get(name, "").strip()
        if t:
            return t

    return None


def sanitize_repo_name(name: str) -> str:
    s = name.strip().strip("/")
    if not s:
        return "upload-repo"

    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-._")

    return s or "upload-repo"


def make_repo_id(namespace: str, repo_name: str) -> str:
    ns = sanitize_repo_name(namespace)
    name = sanitize_repo_name(repo_name)
    return f"{ns}/{name}"


def normalize_remote_path(value: Optional[str]) -> Optional[str]:
    """Return a safe repository-relative destination path, if supplied."""
    if value is None:
        return None
    candidate = value.strip().replace("\\", "/")
    if not candidate:
        return None
    path = PurePosixPath(candidate)
    if not path.parts or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("remote path must be a non-empty repository-relative path")
    return str(path)


def find_modelscope_cli(logger: logging.Logger) -> Optional[str]:
    for cmd in ("ms", "modelscope"):
        p = shutil.which(cmd)
        if p:
            logger.debug("Found CLI: %s", p)
            return p

    logger.error(
        "'ms' or 'modelscope' CLI not found. "
        "Install: pip install -U modelscope-hub"
    )
    return None


def _argv_for_log(argv: List[str]) -> str:
    out: List[str] = []
    skip_next = False

    for a in argv:
        if skip_next:
            out.append("<redacted>")
            skip_next = False
            continue

        if a in ("--token", "-t"):
            out.append(a)
            skip_next = True
        elif a.startswith("--token=") or a.startswith("-t="):
            out.append(a.split("=", 1)[0] + "=<redacted>")
        else:
            out.append(a)

    return " ".join(out)


_T = TypeVar("_T")


def _chunks(items: List[_T], size: int) -> Iterable[List[_T]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_cmd(
    logger: logging.Logger,
    argv: List[str],
    cwd: Optional[str] = None,
) -> None:
    logger.info("RUN: %s", _argv_for_log(argv))
    r = subprocess.run(argv, cwd=cwd)

    if r.returncode != 0:
        logger.error("Command failed, exit=%s", r.returncode)
        r.check_returncode()


def run_cmd_with_retry(
    logger: logging.Logger,
    argv: List[str],
    cwd: Optional[str] = None,
    *,
    max_retries: int = -1,
    retry_base_seconds: float = 5.0,
    retry_max_seconds: float = 300.0,
) -> None:
    failures = 0

    while True:
        try:
            run_cmd(logger, argv, cwd)
            return
        except subprocess.CalledProcessError:
            if max_retries >= 0 and failures >= max_retries:
                logger.error("Giving up after %s failed attempt(s)", failures + 1)
                raise

            delay = min(retry_base_seconds * (2**failures), retry_max_seconds)
            logger.warning(
                "Command failed, attempt=%s; retrying in %.1fs",
                failures + 1,
                delay,
            )
            time.sleep(delay)
            failures += 1


def add_token_args(argv: List[str], token: Optional[str]) -> List[str]:
    if token:
        return argv + ["--token", token]
    return argv


def ensure_repo_and_upload(
    *,
    cli: str,
    token: Optional[str],
    repo_id: str,
    folder: Path,
    repo_type: str,
    visibility: Optional[str],
    logger: logging.Logger,
    commit_message: Optional[str],
    batch_size: int,
    remote_path: Optional[str],
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
) -> None:
    create_argv = [
        cli,
        "create",
        repo_id,
        "--repo-type",
        repo_type,
        "--exist-ok",
    ]

    if visibility:
        create_argv.extend(["--visibility", visibility])

    create_argv = add_token_args(create_argv, token)

    run_cmd_with_retry(
        logger,
        create_argv,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
    )

    if batch_size <= 0:
        upload_argv = [
            cli,
            "upload",
            repo_id,
            str(folder.resolve()),
        ]
        if remote_path:
            upload_argv.append(remote_path)
        upload_argv.extend(["--repo-type", repo_type])

        if commit_message:
            upload_argv.extend(["--commit-message", commit_message])

        upload_argv = add_token_args(upload_argv, token)

        run_cmd_with_retry(
            logger,
            upload_argv,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
        )

        logger.info("Done: uploaded %s -> %s", folder, repo_id)
        return

    entries = sorted(folder.iterdir(), key=lambda p: p.name)

    if not entries:
        logger.warning("No top-level files or directories under %s; repo created only", folder)
        return

    total_batches = (len(entries) + batch_size - 1) // batch_size

    for batch_idx, batch in enumerate(_chunks(entries, batch_size), start=1):
        logger.info(
            "Upload batch %s/%s, item_count=%s",
            batch_idx,
            total_batches,
            len(batch),
        )

        for entry in batch:
            destination = entry.name if remote_path is None else "%s/%s" % (remote_path, entry.name)
            upload_argv = [
                cli,
                "upload",
                repo_id,
                str(entry.resolve()),
                destination,
                "--repo-type",
                repo_type,
            ]

            if commit_message:
                upload_argv.extend(["--commit-message", commit_message])

            upload_argv = add_token_args(upload_argv, token)

            run_cmd_with_retry(
                logger,
                upload_argv,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
            )

    logger.info("Done: uploaded %s -> %s", folder, repo_id)


def download_repo(
    *,
    cli: str,
    token: Optional[str],
    repo_id: str,
    repo_type: str,
    local_dir: Optional[Path],
    revision: Optional[str],
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    files: Optional[List[str]],
    max_workers: Optional[int],
    dry_run: bool,
    force: bool,
    logger: logging.Logger,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
) -> None:
    argv = [
        cli,
        "download",
        repo_id,
    ]

    if files:
        argv.extend(files)

    argv.extend(["--repo-type", repo_type])

    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
        argv.extend(["--local-dir", str(local_dir.resolve())])

    if revision:
        argv.extend(["--revision", revision])

    if include:
        for pat in include:
            argv.extend(["--include", pat])

    if exclude:
        for pat in exclude:
            argv.extend(["--exclude", pat])

    if max_workers is not None:
        argv.extend(["--max-workers", str(max_workers)])

    if dry_run:
        argv.append("--dry-run")
    if force:
        argv.append("--force")

    argv = add_token_args(argv, token)

    run_cmd_with_retry(
        logger,
        argv,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
    )

    if local_dir:
        logger.info("Done: downloaded %s -> %s", repo_id, local_dir)
    else:
        logger.info("Done: downloaded %s to ModelScope cache", repo_id)


def add_common_retry_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--max-retries",
        type=int,
        default=-1,
        help="Max retries per CLI failure (-1=until success, 0=fail fast).",
    )
    p.add_argument(
        "--retry-base-seconds",
        type=float,
        default=5.0,
        help="Initial retry backoff seconds. Default: 5.",
    )
    p.add_argument(
        "--retry-max-seconds",
        type=float,
        default=300.0,
        help="Max retry backoff seconds. Default: 300.",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload/download ModelScope Hub repos using ms/modelscope CLI.",
    )

    sub = p.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="Upload a local folder to ModelScope Hub.")
    up.add_argument(
        "folder",
        type=str,
        help="Local directory to upload. Repo name defaults to folder basename.",
    )
    up.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help=f"ModelScope namespace/user. Default: {DEFAULT_NAMESPACE}.",
    )
    up.add_argument(
        "--repo-name",
        default=None,
        help="Remote repo name. Default: sanitized basename of folder.",
    )
    up.add_argument(
        "--token",
        default=None,
        help=(
            "ModelScope API token. Else read from MODELSCOPE_API_TOKEN or MS_TOKEN; "
            "else use local ms login state."
        ),
    )
    up.add_argument(
        "--repo-type",
        choices=("model", "dataset"),
        default="dataset",
        help="Repository type. Default: dataset.",
    )
    up.add_argument(
        "--visibility",
        choices=("public", "private", "internal"),
        default=None,
        help="Optional repo visibility.",
    )
    up.add_argument(
        "--commit-message",
        default=None,
        help="Optional commit message.",
    )
    up.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help=(
            "Top-level files/dirs per upload pass. "
            "Use 0 for one upload of the whole folder. Default: 20."
        ),
    )
    up.add_argument(
        "--remote-path",
        default=None,
        help="Optional repository-relative destination directory for the upload.",
    )
    add_common_retry_args(up)

    down = sub.add_parser("download", help="Download a ModelScope Hub repo.")
    down.add_argument(
        "repo_id",
        type=str,
        help="Remote repo id, e.g. rogerspyke/my_model.",
    )
    down.add_argument(
        "files",
        nargs="*",
        help="Optional specific file paths inside repo. Omit for full snapshot.",
    )
    down.add_argument(
        "--repo-type",
        choices=("model", "dataset"),
        default="model",
        help="Repository type. Default: model.",
    )
    down.add_argument(
        "--local-dir",
        default=None,
        help="Local output directory. If omitted, downloads to ModelScope cache.",
    )
    down.add_argument(
        "--revision",
        default=None,
        help="Branch, tag, or commit hash. Default is decided by ModelScope CLI.",
    )
    down.add_argument(
        "--include",
        nargs="+",
        default=None,
        help='Allow patterns, e.g. --include "*.safetensors" "*.json".',
    )
    down.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        help='Ignore patterns, e.g. --exclude "*.bin".',
    )
    down.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel download workers.",
    )
    down.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading, if supported by your CLI version.",
    )
    down.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh remote download when verifying an uploaded file.",
    )
    down.add_argument(
        "--token",
        default=None,
        help=(
            "ModelScope API token. Else read from MODELSCOPE_API_TOKEN or MS_TOKEN; "
            "else use local ms login state."
        ),
    )
    add_common_retry_args(down)

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logger = _setup_logging(f"ms_{args.command}")

    cli = find_modelscope_cli(logger)
    if not cli:
        return 127

    token = resolve_token(getattr(args, "token", None))

    try:
        if args.command == "upload":
            folder = Path(args.folder).expanduser().resolve()

            if not folder.is_dir():
                logger.error("Not a directory: %s", folder)
                return 1

            base = args.repo_name or folder.name
            repo_id = make_repo_id(args.namespace, base)

            logger.info(
                "Upload config: repo_id=%s folder=%s repo_type=%s",
                repo_id,
                folder,
                args.repo_type,
            )

            ensure_repo_and_upload(
                cli=cli,
                token=token,
                repo_id=repo_id,
                folder=folder,
                repo_type=args.repo_type,
                visibility=args.visibility,
                logger=logger,
                commit_message=args.commit_message,
                batch_size=args.batch_size,
                remote_path=normalize_remote_path(args.remote_path),
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                retry_max_seconds=args.retry_max_seconds,
            )
            return 0

        if args.command == "download":
            repo_id = args.repo_id.strip()

            if not repo_id or "/" not in repo_id:
                logger.error("Invalid repo_id: %s. Expected format: namespace/name", repo_id)
                return 1

            local_dir = Path(args.local_dir).expanduser().resolve() if args.local_dir else None

            logger.info(
                "Download config: repo_id=%s local_dir=%s repo_type=%s revision=%s",
                repo_id,
                local_dir,
                args.repo_type,
                args.revision,
            )

            download_repo(
                cli=cli,
                token=token,
                repo_id=repo_id,
                repo_type=args.repo_type,
                local_dir=local_dir,
                revision=args.revision,
                include=args.include,
                exclude=args.exclude,
                files=args.files,
                max_workers=args.max_workers,
                dry_run=args.dry_run,
                force=args.force,
                logger=logger,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                retry_max_seconds=args.retry_max_seconds,
            )
            return 0

        logger.error("Unknown command: %s", args.command)
        return 2

    except subprocess.CalledProcessError:
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
