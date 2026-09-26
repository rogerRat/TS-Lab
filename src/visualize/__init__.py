"""TS3 visualization package boundary.

This package retains generic loss/performance plotting and runtime setup.
Legacy feature-space, grid and transfer-method reports have been removed.
"""

from __future__ import annotations

import os

from configs.paths import PROJECT_ROOT


def configure_runtime_environment() -> None:
    """Configure runtime-only environment defaults for visualization jobs.

    @input: None.
    @output: None.
    @scenario: Ensure UMAP import and execution have a writable numba cache.
    @logic: Set ``NUMBA_CACHE_DIR`` only when the caller has not supplied one.
    """
    if "NUMBA_CACHE_DIR" not in os.environ:
        cache_dir = PROJECT_ROOT / "output" / ".numba_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
