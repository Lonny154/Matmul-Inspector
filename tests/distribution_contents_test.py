#!/usr/bin/env python3
"""Audit wheel and sdist archives for intended Python-release contents."""

import argparse
from pathlib import Path
import tarfile
import zipfile


FORBIDDEN_PARTS = {
    "AGENTS.md", "project_context.md", "results", "artifacts", ".github", "scripts", "tests",
    "CMakeFiles", "CMakeCache.txt",
}
FORBIDDEN_SUFFIXES = {
    ".o", ".a", ".so", ".exe", ".ptx", ".cubin", ".neff", ".ntff", ".pftrace", ".ncu-rep",
}
PACKAGE_FILES = {
    "matmul_inspector/__init__.py", "matmul_inspector/cli.py", "matmul_inspector/ptx.py",
    "matmul_inspector/sass.py", "matmul_inspector/nki.py", "matmul_inspector/reporting.py",
}


def audit(names, archive_type):
    normalized = [name.lstrip("./") for name in names if name and not name.endswith("/")]
    relative = ["/".join(name.split("/")[1:]) if archive_type == "sdist" else name for name in normalized]
    bad = []
    for name in relative:
        path = Path(name)
        if any(part in FORBIDDEN_PARTS or part.startswith("build-") for part in path.parts):
            bad.append(name)
        if path.suffix in FORBIDDEN_SUFFIXES or path.name.endswith("~"):
            bad.append(name)
        if (archive_type == "sdist" and name.startswith("src/")
                and not name.startswith(("src/matmul_inspector/", "src/matmul_inspector.egg-info/"))):
            bad.append(name)
    if bad:
        raise AssertionError(f"forbidden archive entries: {sorted(set(bad))}")
    package_names = {name.removeprefix("src/") for name in relative}
    missing = PACKAGE_FILES - package_names
    if missing:
        raise AssertionError(f"missing package files: {sorted(missing)}")
    if archive_type == "wheel" and not any(name.endswith(".dist-info/entry_points.txt") for name in relative):
        raise AssertionError("wheel is missing console entry points")
    if archive_type == "wheel":
        for required in ("LICENSE", "NOTICE"):
            if not any(name.endswith(f".dist-info/licenses/{required}") for name in relative):
                raise AssertionError(f"wheel is missing {required}")
    if archive_type == "sdist":
        for required in ("README.md", "pyproject.toml", "LICENSE", "NOTICE"):
            if required not in relative:
                raise AssertionError(f"sdist is missing {required}")
    print(f"{archive_type}: {len(relative)} files; content audit PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.wheel) as archive:
        audit(archive.namelist(), "wheel")
    with tarfile.open(args.sdist, "r:gz") as archive:
        audit([member.name for member in archive.getmembers() if member.isfile()], "sdist")


if __name__ == "__main__":
    main()
