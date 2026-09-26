"""timesfm pretrained-model resource entry."""
import sys
from .fetch import main as fetch_main


def main(argv=None):
    return fetch_main(["timesfm", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
