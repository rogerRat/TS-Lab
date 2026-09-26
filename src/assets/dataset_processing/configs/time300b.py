"""Shared Time-300B source configuration for Fetch and Load.

The default keeps one bounded Australian electricity selection. Only an
explicit ``--source time300b --subsets all`` scans the repository root; this
can enumerate the approximately 1.15 TiB upstream corpus. ``--source all``
keeps this default and never implies the repository-wide selection. Sampling
and Load require one BIN plus one meta.json per selection, so a root-wide
Time-300B selection is raw-acquisition only until it is split by leaf path.
"""

SPEC = {
    "repo": "Maple728/Time-300B",
    "revision": "main",
    "defaults": ["energy/australian_electricity_demand"],
}
