"""Shared Fetch source catalog and explicit orchestration categories.

Every source owns its ``SPEC`` (where applicable), leaf selection, raw-format
validation, and sampling publication in its own adapter module. This catalog
only gives Fetch, Load, and Profile validation the same source names.
"""

FETCH_SOURCE_NAMES = (
    "chronos",
    "ett",
    "gifteval",
    "lotsa",
    "monash",
    "pile",
    "quito",
    "rmisc",
    "time300b",
    "utsd",
)

# Every Hugging Face source has one explicit default revision. ETT is a plain
# HTTP source and therefore has no Hub revision.
SOURCE_REVISIONS = {
    "chronos": "main",
    "gifteval": "main",
    "lotsa": "main",
    "monash": "main",
    "pile": "main",
    "quito": "main",
    "rmisc": "smaller_version",
    "time300b": "main",
    "utsd": "main",
}
