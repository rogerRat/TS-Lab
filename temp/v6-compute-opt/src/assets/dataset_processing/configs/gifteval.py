"""Shared GiftEval source configuration for Fetch and Load.

The default keeps one bounded PEMS03 selection. Only an explicit
``--source gifteval --subsets all`` scans the repository root; this can
enumerate the approximately 908 GiB upstream corpus. ``--source all`` keeps
this default and never implies the repository-wide selection.
"""

SPEC = {"repo": "Salesforce/GiftEvalPretrain", "revision": "main", "defaults": ["PEMS03"]}
