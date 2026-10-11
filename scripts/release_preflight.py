#!/usr/bin/env python3
"""Fail-closed release preflight and distribution metadata validation."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from typing import Sequence
import zipfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility for the CPU test matrix.
    tomllib = None


PROJECT_NAME = "matmul-inspector"
LICENSE_EXPRESSION = "Apache-2.0"
TAG_PATTERN = re.compile(r"^refs/tags/v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def project_metadata(path: Path) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    if tomllib is not None:
        data = tomllib.loads(source)
        project = data.get("project")
        if not isinstance(project, dict):
            raise ValueError("pyproject.toml has no [project] table")
    else:
        project = fallback_project_table(source)
    values = {}
    for field in ("name", "version", "license"):
        value = project.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"project.{field} must contain exactly one nonempty string")
        values[field] = value
    if normalized_name(values["name"]) != PROJECT_NAME:
        raise ValueError(f"unexpected project name: {values['name']!r}")
    if values["license"] != LICENSE_EXPRESSION:
        raise ValueError(f"unexpected license expression: {values['license']!r}")
    return values


def fallback_project_table(source: str) -> dict[str, str]:
    """Read required scalar fields on Python 3.10; reject ambiguity."""
    in_project = False
    values = {}
    field_pattern = re.compile(r'^(name|version|license)\s*=\s*"([^"\r\n]+)"\s*(?:#.*)?$')
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if line == "[project]":
            if in_project:
                raise ValueError("pyproject.toml has conflicting [project] tables")
            in_project = True
            continue
        if in_project and line.startswith("["):
            break
        if not in_project or not line or line.startswith("#"):
            continue
        match = field_pattern.fullmatch(line)
        if match:
            field, value = match.groups()
            if field in values:
                raise ValueError(f"pyproject.toml has conflicting project.{field} values")
            values[field] = value
    if not in_project:
        raise ValueError("pyproject.toml has no [project] table")
    return values


def git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=False
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout.strip()


def release_preflight(
    *, event_name: str, ref: str, sha: str, root: Path, main_ref: str
) -> dict[str, str]:
    metadata = project_metadata(root / "pyproject.toml")
    result = {
        "expected_version": metadata["version"],
        "publish_allowed": "false",
        "validation_mode": "build-only",
    }
    if event_name == "workflow_dispatch":
        return result
    if event_name != "push":
        raise ValueError(f"unsupported release event: {event_name!r}")
    match = TAG_PATTERN.fullmatch(ref)
    if not match:
        raise ValueError("tag must have the exact form vMAJOR.MINOR.PATCH without leading zeros")
    tag_version = ".".join(match.groups())
    if tag_version != metadata["version"]:
        raise ValueError(
            f"tag version {tag_version!r} does not match project version {metadata['version']!r}"
        )
    tagged_commit = git_output(root, "rev-parse", f"{ref}^{{commit}}")
    supplied_commit = git_output(root, "rev-parse", f"{sha}^{{commit}}")
    if tagged_commit != supplied_commit:
        raise ValueError(
            f"tag resolves to {tagged_commit}, not workflow commit {supplied_commit}"
        )
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", tagged_commit, main_ref],
        cwd=root,
        check=False,
    )
    if ancestry.returncode == 1:
        raise ValueError(f"tagged commit {tagged_commit} is not an ancestor of {main_ref}")
    if ancestry.returncode != 0:
        raise ValueError(f"could not verify ancestry against {main_ref}")
    result.update(
        publish_allowed="true", validation_mode="production", tagged_commit=tagged_commit
    )
    return result


def one_metadata(messages: list[bytes], source: str) -> dict[str, str]:
    if len(messages) != 1:
        raise ValueError(f"{source} must contain exactly one core metadata file")
    message = BytesParser().parsebytes(messages[0])
    result = {}
    for field in ("Name", "Version", "License-Expression"):
        values = message.get_all(field, [])
        if len(values) != 1 or not values[0].strip():
            raise ValueError(f"{source} has missing or conflicting {field} metadata")
        result[field] = values[0]
    if normalized_name(result["Name"]) != PROJECT_NAME:
        raise ValueError(f"{source} has unexpected project name {result['Name']!r}")
    if result["License-Expression"] != LICENSE_EXPRESSION:
        raise ValueError(f"{source} has unexpected license {result['License-Expression']!r}")
    return result


def wheel_metadata(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata = [archive.read(name) for name in names if name.endswith(".dist-info/METADATA")]
        for filename in ("LICENSE", "NOTICE"):
            matches = [name for name in names if name.endswith(f".dist-info/licenses/{filename}")]
            if len(matches) != 1:
                raise ValueError(f"wheel must contain exactly one {filename}")
    return one_metadata(metadata, "wheel")


def sdist_metadata(path: Path) -> dict[str, str]:
    with tarfile.open(path, "r:gz") as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        metadata = [
            archive.extractfile(member).read()
            for member in files
            if len(Path(member.name).parts) == 2 and member.name.endswith("/PKG-INFO")
        ]
        root_files = {
            Path(member.name).parts[-1]
            for member in files
            if len(Path(member.name).parts) == 2
        }
        for filename in ("LICENSE", "NOTICE"):
            if filename not in root_files:
                raise ValueError(f"source distribution is missing {filename}")
    return one_metadata(metadata, "source distribution")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_artifacts(directory: Path, expected_version: str) -> list[dict[str, str | int]]:
    if not directory.is_dir():
        raise ValueError(f"distribution directory does not exist: {directory}")
    entries = sorted(directory.iterdir())
    if any(not entry.is_file() for entry in entries):
        raise ValueError("distribution directory contains a non-file entry")
    wheels = [path for path in entries if path.suffix == ".whl"]
    sdists = [path for path in entries if path.name.endswith(".tar.gz")]
    expected = set(wheels + sdists)
    unexpected = [path.name for path in entries if path not in expected]
    if len(wheels) != 1 or len(sdists) != 1 or unexpected:
        raise ValueError(
            "expected exactly one wheel and one .tar.gz source distribution; "
            f"unexpected={unexpected}, wheels={len(wheels)}, sdists={len(sdists)}"
        )
    wheel_pattern = re.compile(
        rf"^matmul_inspector-{re.escape(expected_version)}-(?:[^-]+-)?[^-]+-[^-]+-[^-]+\.whl$"
    )
    expected_sdist = f"matmul_inspector-{expected_version}.tar.gz"
    if not wheel_pattern.fullmatch(wheels[0].name) or sdists[0].name != expected_sdist:
        raise ValueError(
            "distribution filenames do not match the expected project name and version: "
            f"{wheels[0].name!r}, {sdists[0].name!r}"
        )
    records = []
    for path, reader in ((wheels[0], wheel_metadata), (sdists[0], sdist_metadata)):
        metadata = reader(path)
        if metadata["Version"] != expected_version:
            raise ValueError(
                f"{path.name} version {metadata['Version']!r} does not match {expected_version!r}"
            )
        records.append({
            "filename": path.name,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        })
    return records


def write_outputs(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for key, value in values.items():
            stream.write(f"{key}={value}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--event-name", required=True)
    preflight.add_argument("--ref", required=True)
    preflight.add_argument("--sha", required=True)
    preflight.add_argument("--root", type=Path, default=Path.cwd())
    preflight.add_argument("--main-ref", default="origin/main")
    preflight.add_argument("--github-output", type=Path)
    artifacts = commands.add_parser("artifacts")
    artifacts.add_argument("--directory", type=Path, required=True)
    artifacts.add_argument("--expected-version", required=True)
    artifacts.add_argument("--github-step-summary", type=Path)
    return parser


def run(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    if args.command == "preflight":
        result = release_preflight(
            event_name=args.event_name,
            ref=args.ref,
            sha=args.sha,
            root=args.root.resolve(),
            main_ref=args.main_ref,
        )
        if args.github_output:
            write_outputs(args.github_output, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    records = validate_artifacts(args.directory, args.expected_version)
    lines = [f"{record['sha256']}  {record['filename']}" for record in records]
    print("\n".join(lines))
    if args.github_step_summary:
        with args.github_step_summary.open("a", encoding="utf-8") as stream:
            stream.write("## Validated release distributions\n\n")
            stream.write("| File | Bytes | SHA-256 |\n|---|---:|---|\n")
            for record in records:
                stream.write(
                    f"| '{record['filename']}' | {record['size_bytes']} | {record['sha256']} |\n"
                )
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        return run(arguments)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"release-preflight: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
