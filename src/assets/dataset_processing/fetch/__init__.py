"""Dataset fetch package public entry points.

The source implementation lives in :mod:`assets.dataset_processing.fetch.src.runner` so importing
the package does not configure logging or load CLI dependencies.
"""

def main(argv=None):
    """Run the fetch CLI while preserving the historical package entry point.

    @input: argv is None or a list of CLI argument strings.
    @output: int -- Fetch process exit status.
    @scenario: Compatibility entry point for ``python -m`` and callers that
               previously imported ``assets.dataset_processing.fetch.main``.
    """
    from .src.runner import main as _main

    return _main(argv)


def parse_args(argv=None):
    """Parse fetch CLI arguments through the implementation module."""
    from .src.runner import parse_args as _parse_args

    return _parse_args(argv)


__all__ = ["main", "parse_args"]
