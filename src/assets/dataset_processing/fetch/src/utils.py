"""Shared Fetch download utilities.

This module owns the source-independent network boundary: Hugging Face and
ordinary HTTP downloads, retry policy, and exhausted-operation reporting.
Source adapters own their leaf selections, raw-format validation, sampling,
completion markers, and manifests.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from assets.dataset_processing.configs.config import (
    HF_HTTP_TIMEOUT_SECONDS,
    HF_MAX_RETRIES,
    HF_RETRY_BASE_DELAY_SECONDS,
    HF_RETRY_MAX_DELAY_SECONDS,
)
from assets.dataset_processing.progress import ProcessingProgress

log = logging.getLogger("fetch")


class DownloadFailureStack:
    """Collect Fetch download operations that exhausted their retry budget.

    @input: No construction arguments.
    @output: ``items`` is a list of operation/error mappings.
    @scenario: Let every source finish its independent selections before one
               common report describes downloads that could not be recovered.
    @logic: ``push`` appends terminal failures and ``log`` emits either an
            explicit empty record or each collected operation in insertion order.
    """

    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def push(self, operation: str, error: Exception | str) -> None:
        """Record one exhausted download operation.

        @input: operation identifies the remote action; error is its terminal exception.
        @output: None.
        @scenario: Preserve source-level failure context for the final Fetch log.
        @logic: Converts the error to ASCII-safe text and appends one mapping.
        """
        self.items.append({"operation": operation, "error": str(error)})

    def log(self, logger: logging.Logger) -> None:
        """Write the complete terminal download-failure report.

        @input: logger is the Fetch logger used by the calling source adapter.
        @output: None.
        @scenario: Make partial acquisition explicit after all selected leaves run.
        @logic: Emits an empty-stack info record or numbered error records.
        """
        if not self.items:
            logger.info("DOWNLOAD_FAILURE_STACK empty.")
            return
        logger.error(
            "DOWNLOAD_FAILURE_STACK size=%d: some data was skipped after retries.",
            len(self.items),
        )
        for index, item in enumerate(self.items, start=1):
            logger.error(
                "DOWNLOAD_FAILURE_STACK[%d] operation=%s error=%s",
                index, item["operation"], item["error"],
            )


class RetryingHuggingFaceDownloader:
    """Download Hub or HTTP raw files with the project retry policy.

    @input: Timeout/retry parameters default to ``configs.config`` constants;
            failure_stack optionally records terminal failures.
    @output: Download methods return a local path string or ``None`` on exhaustion.
    @scenario: Share network behavior without deciding any source's selection,
               file format, cache layout, parsing, or publication contract.
    @logic: Sets Hub timeout variables once, executes an action, and applies
            randomized exponential backoff until success or the retry budget ends.
    """

    def __init__(
        self,
        *,
        timeout_seconds: int = HF_HTTP_TIMEOUT_SECONDS,
        max_retries: int = HF_MAX_RETRIES,
        base_delay_seconds: float = HF_RETRY_BASE_DELAY_SECONDS,
        max_delay_seconds: float = HF_RETRY_MAX_DELAY_SECONDS,
        failure_stack: DownloadFailureStack | None = None,
    ) -> None:
        self.timeout_seconds = int(timeout_seconds)
        self.max_retries = max(1, int(max_retries))
        self.base_delay_seconds = float(base_delay_seconds)
        self.max_delay_seconds = float(max_delay_seconds)
        self.failure_stack = failure_stack
        self._configure_environment()

    def _configure_environment(self) -> None:
        """Set process-local Hugging Face timeout environment variables.

        @input: No arguments; uses the instance timeout.
        @output: None.
        @scenario: Apply one explicit timeout policy to all Hub client calls.
        @logic: Writes all supported Hub and datasets timeout variable names.
        """
        timeout = str(self.timeout_seconds)
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = timeout
        os.environ["HF_HUB_ETAG_TIMEOUT"] = timeout
        os.environ["HF_DATASETS_DOWNLOAD_TIMEOUT"] = timeout

    def _backoff(self, operation: str, attempt: int, exc: Exception) -> bool:
        """Record or delay after one failed network attempt.

        @input: operation names the request; attempt is one-based; exc is the failure.
        @output: True when the caller must retry, otherwise False.
        @scenario: Bound transient network recovery and retain terminal failures.
        @logic: Pushes exhausted failures; otherwise sleeps a jittered exponential delay.
        """
        if attempt >= self.max_retries:
            if self.failure_stack is not None:
                self.failure_stack.push(operation, exc)
            log.error("Download operation '%s' exhausted %d retries; skipping.", operation, self.max_retries)
            return False
        ceiling = min(self.max_delay_seconds, self.base_delay_seconds * (2 ** (attempt - 1)))
        delay = random.uniform(0.5 * ceiling, 1.5 * ceiling)
        log.warning(
            "Download operation '%s' failed on attempt %d/%d: %s; retrying in %.1fs.",
            operation, attempt, self.max_retries, exc, delay,
        )
        time.sleep(delay)
        return True

    def _retry(self, operation: str, action: Callable[[int], Any]) -> Any:
        """Execute one action through the configured retry policy.

        @input: operation labels the action; action receives the one-based attempt number.
        @output: Action result on success, otherwise None.
        @scenario: Use one failure behavior for every downloader public method.
        @logic: Calls the action up to max_retries and delegates each exception to _backoff.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                log.info(
                    "Download attempt starting: operation=%s attempt=%d/%d timeout_seconds=%d",
                    operation, attempt, self.max_retries, self.timeout_seconds,
                )
                with ProcessingProgress(
                    log, "FETCH_DOWNLOAD", operation, detail=(
                        "attempt=%d/%d timeout_seconds=%d" % (
                            attempt, self.max_retries, self.timeout_seconds,
                        )
                    ),
                ):
                    result = action(attempt)
                log.info(
                    "Download attempt completed: operation=%s attempt=%d/%d result=%s",
                    operation, attempt, self.max_retries, result,
                )
                return result
            except Exception as exc:
                if not self._backoff(operation, attempt, exc):
                    return None
        return None

    def download_file(self, repo_id: str, filename: str, *, repo_type: str = "dataset", revision: str | None = None, local_dir: Any = None, force_download: bool = False) -> str | None:
        """Download one repository file while preserving its relative filename.

        @input: Hub repo/file identity, optional revision/local root, and force flag.
        @output: Local path string on success, otherwise None.
        @scenario: Let a source adapter obtain its declared raw file before validation.
        @logic: Lazily imports the Hub client and retries its download call unchanged.
        """
        from huggingface_hub import hf_hub_download

        def action(attempt: int) -> str:
            # A failed proxy transfer can leave a corrupt destination or
            # .incomplete file. Remove both before a retry so the CDN cannot
            # resume from a stale offset (which commonly returns HTTP 416).
            if attempt > 1 and local_dir is not None:
                target = Path(local_dir) / filename
                target.unlink(missing_ok=True)
                target.with_name(target.name + ".incomplete").unlink(missing_ok=True)
                from huggingface_hub import hf_hub_url
                from urllib.request import urlopen
                url = hf_hub_url(repo_id, filename, repo_type=repo_type, revision=revision)
                partial = target.with_name(target.name + ".partial")
                with urlopen(url, timeout=self.timeout_seconds) as response:
                    expected = response.headers.get("Content-Length")
                    written = 0
                    with partial.open("wb") as output:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                            written += len(chunk)
                if expected is not None and written != int(expected):
                    partial.unlink(missing_ok=True)
                    raise IOError("short direct download: expected %s bytes, got %s" % (expected, written))
                partial.replace(target)
                return str(target)
            return hf_hub_download(
                repo_id, filename, repo_type=repo_type, revision=revision,
                etag_timeout=self.timeout_seconds, local_dir=local_dir,
                force_download=force_download,
            )

        return self._retry(f"{repo_id}/{filename}", action)

    def download_url(self, url: str, destination: Any) -> str | None:
        """Download one HTTP URL atomically with the same retry policy.

        @input: url is an HTTP resource; destination is its final local path.
        @output: Destination string on success, otherwise None.
        @scenario: Support non-Hub Fetch sources such as deprecated ETT.
        @logic: Streams into a sibling .partial path and replaces destination only on success.
        """
        from urllib.request import urlopen

        destination = Path(destination)
        partial = destination.with_name(destination.name + ".partial")

        def action(_attempt: int) -> str:
            with urlopen(url, timeout=self.timeout_seconds) as response:
                with partial.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            partial.replace(destination)
            return str(destination)

        return self._retry(url, action)

    def load_streaming_dataset(self, repo_id: str, config_name: str, *, split: str = "train") -> Any:
        """Open one Hugging Face streaming dataset through the retry policy.

        @input: repo_id/config_name identify one dataset configuration; split selects its split.
        @output: Dataset stream or None after retry exhaustion.
        @scenario: Preserve the existing downloader API for Fetch callers that stream data.
        @logic: Lazily imports datasets and retries load_dataset(streaming=True).
        """
        from datasets import load_dataset

        return self._retry(
            f"{repo_id}:{config_name}",
            lambda _attempt: load_dataset(repo_id, config_name, split=split, streaming=True),
        )

    def iter_streaming_dataset(self, repo_id: str, config_name: str, *, split: str = "train") -> Iterator[Any]:
        """Yield a streaming dataset and resume at the last yielded item after errors.

        @input: Hub dataset identity and split.
        @output: Iterator of source items; stops after a terminal failure.
        @scenario: Keep the legacy resumable streaming facility without source semantics.
        @logic: Reopens the stream after an exception, skips consumed ordinals, and retries.
        """
        seen = 0
        stream_attempt = 0
        while True:
            dataset = self.load_streaming_dataset(repo_id, config_name, split=split)
            if dataset is None:
                return
            try:
                for index, item in enumerate(dataset):
                    if index < seen:
                        continue
                    seen = index + 1
                    yield item
                return
            except Exception as exc:
                stream_attempt += 1
                if not self._backoff(f"{repo_id}:{config_name} stream at item {seen}", stream_attempt, exc):
                    return
