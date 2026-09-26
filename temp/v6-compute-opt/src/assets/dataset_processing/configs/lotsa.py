"""Shared LOTSA source configuration for Fetch and Load.

The default keeps one bounded PEMS03 selection. Only an explicit
``--source lotsa --subsets all`` scans the repository root; this can enumerate
the approximately 1.80 TiB upstream corpus. ``--source all`` keeps this
default and never implies the repository-wide selection.
"""

SPEC = {"repo": "Salesforce/lotsa_data", "revision": "main", "defaults": ["PEMS03"]}
