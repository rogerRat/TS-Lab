"""Source Load entry point for ShardDatasetBundle publication."""
from assets.gradient.dataset import GradientDataset

def main(argv=None):
    """Run Source Load."""
    from .src.runner import main as _main

    return _main(argv)


def parse_args(argv=None):
    """Parse load CLI arguments through the implementation module."""
    from .src.runner import parse_args as _parse_args

    return _parse_args(argv)


__all__ = ["main", "parse_args", "GradientDataset"]
