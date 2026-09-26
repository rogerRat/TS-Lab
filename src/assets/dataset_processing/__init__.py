"""Dataset asset production workflow.

This layer fetches source/reference data, writes numeric shard bundles and
deduplicates them. RAGS reads these trajectories directly. Legacy descriptor,
catalog, grid and transfer-selection production stages have been removed.

Subpackages:
  assets.dataset_processing.fetch  — Download raw datasets from official sources.
  assets.dataset_processing.load   — Convert raw data to canonical Parquet format.
  assets.dataset_processing.series — Explicitly split source arrays into univariate series.
  assets.dataset_processing.dedup  — Globally deduplicate canonical Parquet tables.
"""
