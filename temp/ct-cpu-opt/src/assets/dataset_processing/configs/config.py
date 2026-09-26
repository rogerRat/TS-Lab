"""Shared configuration for the dataset-processing pipeline.

This module owns settings shared by Fetch and Load infrastructure. Dataset
specific values live in sibling modules under this package.
"""

HF_HTTP_TIMEOUT_SECONDS = 300
HF_MAX_RETRIES = 5
HF_RETRY_BASE_DELAY_SECONDS = 5.0
HF_RETRY_MAX_DELAY_SECONDS = 300.0

# ETT remains an explicit legacy compatibility source only. Fetch/Load default
# selection and Profiles must never include it.
DEPRECATED_SOURCES: set[str] = {"ett"}
