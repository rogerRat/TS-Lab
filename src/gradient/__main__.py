"""Compatibility CLI; resource orchestration lives in assets.gradient."""
from assets.gradient.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
