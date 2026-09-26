"""Shared UTSD source configuration for Fetch and Load.

The default is UTSD-1G. UTSD-2G, UTSD-4G, and UTSD-12G require an explicit
``--utsd-config`` value; ``--source all`` preserves UTSD-1G rather than
implicitly choosing a larger collection. The full upstream repository is
currently approximately 12.23 GiB.
"""

UTSD_HF_DATASET = "thuml/UTSD"
UTSD_CONFIGS = ("UTSD-1G", "UTSD-2G", "UTSD-4G", "UTSD-12G")
UTSD_DEFAULT_CONFIG = "UTSD-1G"
UTSD_SUBSETS = ["sample"]
