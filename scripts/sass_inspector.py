#!/usr/bin/env python3
"""Compatibility entry point for :mod:`matmul_inspector.sass`."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from matmul_inspector.sass import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    raise SystemExit(main())
