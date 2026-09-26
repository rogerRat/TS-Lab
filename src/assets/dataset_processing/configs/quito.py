"""Shared Quito source configuration for Fetch and Load.

The default keeps one pretraining Parquet selection. Only an explicit
``--source quito --subsets all`` scans the repository root, which is currently
approximately 4.18 GiB. ``--source all`` keeps this default and never implies
the repository-wide selection.
"""

SPEC = {
    "repo": "hq-bench/quito-corpus",
    "revision": "main",
    "defaults": ["v20260315/pretrain_hour-00001-of-00001.parquet"],
}
