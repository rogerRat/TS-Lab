"""less external baseline: algorithms, runner, and provenance."""
from .contract import CONTRACT


def select(*args, **kwargs):
    """Run this baseline; model dependencies are imported only on execution."""
    from .runner import less_select
    return less_select(*args, **kwargs)


__all__ = ["CONTRACT", "select"]
