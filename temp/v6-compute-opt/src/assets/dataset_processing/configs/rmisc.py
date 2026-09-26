"""Shared RMISC source configuration for Fetch and Load.

The default keeps one bounded ACSF1 selection on the smaller_version revision.
Only an explicit ``--source rmisc --subsets all`` scans that revision root; the
upstream RMISC repository is multi-terabyte scale. ``--source all`` keeps this
default and never implies the repository-wide selection.
"""

SPEC = {"repo": "nju-zhangsq/RMISC", "revision": "smaller_version", "defaults": ["ACSF1"]}
