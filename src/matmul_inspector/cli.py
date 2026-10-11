"""Installed command-line dispatcher for the Matmul Inspector Python toolkit."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import __version__
from . import nki, ptx, reporting, sass


COMMANDS = {
    "ptx": (ptx.main, "Inspect or compare NVIDIA PTX text"),
    "sass": (sass.main, "Inspect or compare NVIDIA nvdisasm text"),
    "nki": (nki.main, "Compare saved AWS Neuron NKI compiler artifacts"),
    "report": (reporting.main, "Combine saved compiler and benchmark evidence"),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="matmul-inspector",
        description="CPU-only compiler-artifact analysis toolkit",
        epilog=("The native C++ benchmark executable has the same primary command name. "
                "Use matmul-inspector-python to select this installed Python entry point."),
    )
    result.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    result.add_argument("command", nargs="?", choices=sorted(COMMANDS), help="analysis command")
    result.add_argument("arguments", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return result


def print_help() -> None:
    parser().print_help()
    print("\ncommands:")
    for name, (_entry, description) in COMMANDS.items():
        print(f"  {name:<8} {description}")


def main(arguments: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if arguments is None else arguments)
    if not values or values[0] in ("-h", "--help"):
        print_help()
        return 0
    if values[0] in ("-V", "--version"):
        print(f"matmul-inspector {__version__}")
        return 0
    command = values.pop(0)
    selected = COMMANDS.get(command)
    if selected is None:
        parser().error(f"invalid command: {command!r}")
    return selected[0](values)


if __name__ == "__main__":
    raise SystemExit(main())
