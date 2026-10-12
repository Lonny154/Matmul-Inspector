#!/usr/bin/env python3
"""Exercise an installed wheel or sdist from outside the repository."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile


PTX = """.version 8.0
.target sm_80
.address_size 64
.visible .entry smoke() {
  .reg .b32 %r<2>;
  mov.u32 %r1, 1;
  ret;
}
"""

SASS = """.target sm_89
.section .text.smoke,"ax",@progbits
.sectioninfo @"SHI_REGISTERS=4"
.type smoke,@function
smoke:
.text.smoke:
/*0000*/ MOV R1, c[0x0][0x28] ;
/*0010*/ EXIT ;
"""


def execute(command, arguments, cwd):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    result = subprocess.run([str(command), *arguments], cwd=cwd, env=environment,
                            text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {command} {' '.join(arguments)}\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def execute_failure(command, arguments, cwd):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    result = subprocess.run([str(command), *arguments], cwd=cwd, env=environment,
                            text=True, capture_output=True, check=False)
    if result.returncode == 0:
        raise RuntimeError(f"command unexpectedly succeeded: {command} {' '.join(arguments)}")
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nki_capture(path, symbol):
    path.mkdir()
    (path / "module.mlir").write_text(
        f"module {{ func.func @{symbol}() attributes {{kernel}} {{ return }} }}\n", encoding="utf-8")
    (path / "kernel.neff").write_bytes(symbol.encode())


def benchmark(path):
    path.mkdir()
    (path / "metadata.json").write_text(json.dumps({
        "schema_version": 3, "status": "complete",
        "config": {"reference": "cpu", "candidate": "cpu"},
    }), encoding="utf-8")
    write_csv(path / "summary.csv", [{
        "row_id": 0, "M": 1, "N": 1, "K": 1, "kernel": "cpu", "backend": "cpu",
        "timing_mode": "host_matmul", "median_ms": 1, "gflops": 0.000002,
        "median_ci_low_ms": "", "median_ci_high_ms": "", "median_speedup": 1,
        "stability_warnings": "", "bitwise_equal": "true", "divergent_count": 0,
        "tolerance_pass": "true", "tolerance_failures": 0, "max_ulp": 0,
        "mean_divergent_ulp": 0, "max_absolute_error": 0, "max_relative_error": 0,
    }])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--alias", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="matmul-inspector-installed-") as directory:
        root = Path(directory).resolve()
        location = execute(args.python, ["-c", "import matmul_inspector, pathlib; print(pathlib.Path(matmul_inspector.__file__).resolve())"], root)
        if "site-packages" not in location:
            raise RuntimeError(f"package did not import from an isolated installation: {location}")
        license_expression = execute(
            args.python,
            ["-c", "from importlib.metadata import metadata; print(metadata('matmul-inspector')['License-Expression'])"],
            root,
        )
        if license_expression != "Apache-2.0":
            raise RuntimeError(f"unexpected license expression: {license_expression!r}")
        version = execute(args.command, ["--version"], root)
        alias_version = execute(args.alias, ["--version"], root)
        if version != "matmul-inspector 0.2.0" or alias_version != version:
            raise RuntimeError(f"unexpected versions: {version!r}, {alias_version!r}")
        help_text = execute(args.command, ["--help"], root)
        if "Compiler-artifact analysis and optional GPU experiment toolkit" not in help_text:
            raise RuntimeError("top-level help was not installed")
        for command_name in ("experiment", "experiment-batch", "session"):
            if command_name not in help_text:
                raise RuntimeError(f"installed help omitted {command_name!r}")
        execute(args.command, ["experiment", "--print-default-config"], root)
        missing_gpu = execute_failure(
            args.command, ["experiment", "--output-root", str(root / "gpu-missing")], root)
        combined = missing_gpu.stdout + missing_gpu.stderr
        if "PyTorch is required; install a CUDA-enabled PyTorch build" not in combined:
            raise RuntimeError(f"missing GPU dependencies were not actionable:\n{combined}")

        ptx = root / "smoke.ptx"
        ptx.write_text(PTX, encoding="utf-8")
        execute(args.command, ["ptx", "inspect", "--input", str(ptx),
                               "--output", str(root / "ptx-output")], root)
        sass = root / "smoke.sass"
        sass.write_text(SASS, encoding="utf-8")
        execute(args.command, ["sass", "inspect", "--input", str(sass),
                               "--output", str(root / "sass-output")], root)
        left, right = root / "nki-left", root / "nki-right"
        nki_capture(left, "left")
        nki_capture(right, "right")
        execute(args.command, ["nki", str(left), str(right),
                               "--output", str(root / "nki-output")], root)
        run = root / "benchmark"
        benchmark(run)
        execute(args.command, ["report", "--benchmark", str(run),
                               "--output", str(root / "report-output")], root)
        for expected in (root / "ptx-output" / "analysis.json",
                         root / "sass-output" / "analysis.json",
                         root / "nki-output" / "comparison.json",
                         root / "report-output" / "experiment.json"):
            if not expected.is_file():
                raise RuntimeError(f"missing installed CLI output: {expected}")
        print(f"installed package: {location}")
        print(f"license expression: {license_expression}")
        print(f"primary CLI: {version}")
        print("PTX, SASS, NKI, report, and optional-GPU boundary smoke checks: PASS")


if __name__ == "__main__":
    main()
