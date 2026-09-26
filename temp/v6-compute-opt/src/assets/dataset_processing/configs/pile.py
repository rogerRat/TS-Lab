"""Shared Time Series PILE source configuration for Fetch and Load.

The default keeps one bounded UCR classification selection. Only an explicit
``--source pile --subsets all`` scans the repository root, which is currently
approximately 6.37 GiB. ``--source all`` keeps this default and never implies
the repository-wide selection.
"""

SPEC = {
    "repo": "AutonLab/Timeseries-PILE",
    "revision": "main",
    "defaults": ["classification/UCR/ACSF1"],
}
