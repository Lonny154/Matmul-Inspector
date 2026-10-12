#!/usr/bin/env python3
"""Source-tree tests for package imports, dispatch, and compatibility wrappers."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import compiler_performance_report
import nki_compare_artifacts
import ptx_inspector
import sass_inspector
from matmul_inspector import __version__
from matmul_inspector import cli, experiment, nki, ptx, reporting, sass


class PythonPackagingTests(unittest.TestCase):
    def run_module(self, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "matmul_inspector.cli", *arguments],
            cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
            text=True, capture_output=True, check=False,
        )

    def test_version_and_top_level_help(self):
        self.assertEqual(__version__, "0.1.0")
        version = self.run_module("--version")
        self.assertEqual(version.returncode, 0)
        self.assertEqual(version.stdout.strip(), "matmul-inspector 0.1.0")
        help_result = self.run_module("--help")
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("native C++ benchmark executable", help_result.stdout)

    def test_nested_help_reaches_existing_parsers(self):
        for arguments, expected in (
                (("ptx", "inspect", "--help"), "--kernel"),
                (("sass", "compare", "--help"), "--baseline-kernel"),
                (("nki", "--help"), "--subgraph"),
                (("report", "--help"), "--reference-map"),
                (("experiment", "--help"), "--print-default-config"),
                (("experiment-batch", "--help"), "--output-root")):
            result = self.run_module(*arguments)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(expected, result.stdout)

    def test_argument_forwarding_and_return_code(self):
        called = []
        with mock.patch.dict(cli.COMMANDS, {"ptx": (lambda args: called.append(args) or 7, "test")}):
            self.assertEqual(cli.main(["ptx", "inspect", "--input", "x"]), 7)
        self.assertEqual(called, [["inspect", "--input", "x"]])

    def test_invalid_command_has_argparse_error_exit(self):
        result = self.run_module("unknown")
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid command", result.stderr)

    def test_compatibility_wrappers_reexport_public_functions(self):
        self.assertIs(ptx_inspector.run, ptx.run)
        self.assertIs(sass_inspector.run, sass.run)
        self.assertIs(nki_compare_artifacts.run, nki.run)
        self.assertIs(compiler_performance_report.run, reporting.run)
        for module in (ptx_inspector, sass_inspector, nki_compare_artifacts,
                       compiler_performance_report):
            self.assertTrue(callable(module.main))

    def test_empty_invocation_prints_commands(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli.main([]), 0)
        for name in ("ptx", "sass", "nki", "report", "experiment", "experiment-batch"):
            self.assertIn(name, output.getvalue())


if __name__ == "__main__":
    unittest.main()
