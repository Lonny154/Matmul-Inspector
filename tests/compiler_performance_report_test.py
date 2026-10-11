#!/usr/bin/env python3
"""CPU-only tests for unified compiler-to-performance reporting."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compiler_performance_report as report


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ptx_fixture():
    def symbol(name, fma):
        mnemonics = {"fma.rn.f32": 5} if fma else {"mul.rn.f32": 5, "add.rn.f32": 5}
        return {"name": name, "kind": "entry", "instruction_count": 10 if fma else 15,
                "instruction_categories": {"arithmetic": 5 if fma else 10},
                "instruction_mnemonics": mnemonics, "ptx_declared_registers_by_type": {"b32": 4},
                "ptx_declared_static_shared_bytes": 0, "has_dynamic_shared_memory": False}
    left, right = symbol("mi_naive_fma", True), symbol("mi_naive_no_fma", False)
    file = {"path": "capture.ptx", "size_bytes": 123, "sha256": "a" * 64}
    return {"schema_version": 1, "tool": "ptx-inspector", "mode": "compare",
            "baseline": {"file": file, "analysis": {"target": "sm_89", "ptx_version": "9.3", "address_size": 64}},
            "candidate": {"file": file, "analysis": {"target": "sm_89", "ptx_version": "9.3", "address_size": 64}},
            "comparison": {"raw": {"scope": "full_module", "identical": False, "changed_lines": 4},
                           "normalized": {"scope": "full_module", "identical": False, "changed_lines": 4},
                           "compatibility": {"target": True},
                           "kernel": {"scope": "selected_entry_only", "baseline": left, "candidate": right,
                                      "instruction_categories": [], "instruction_mnemonics": [],
                                      "ptx_declared_registers": []}}}


def sass_fixture(target="sm_89"):
    def function(name, fma):
        operations = {"FFMA": 5} if fma else {"FMUL": 5, "FADD": 5}
        return {"name": name, "role": "primary", "instruction_count": 145 if fma else 151,
                "instruction_categories": {"floating_point_arithmetic": 5 if fma else 10},
                "instruction_mnemonics": operations, "base_operations": operations,
                "allocated_registers_per_thread": 28,
                "calls": [{"address": "0130", "mnemonic": "CALL.REL.NOINC", "target": "$div"}]}
    left, right = function("mi_naive_fma", True), function("mi_naive_no_fma", False)
    file = {"path": "capture.sass", "size_bytes": 456, "sha256": "b" * 64}
    return {"schema_version": 1, "tool": "sass-inspector", "mode": "compare",
            "baseline": {"file": file, "analysis": {"target": target, "ambiguities": []}},
            "candidate": {"file": file, "analysis": {"target": target, "ambiguities": []}},
            "comparison": {"raw": {"scope": "full_file", "identical": True, "changed_lines": 0},
                           "module_normalized": {"scope": "full_file", "identical": True, "changed_lines": 0},
                           "function": {"scope": "selected_function_only", "baseline": left, "candidate": right,
                                        "normalized": {"identical": False}, "instruction_categories": [],
                                        "instruction_mnemonics": [], "base_operations": []}}}


def create_benchmark(root, status="tolerance_failed", omit=()):
    root.mkdir()
    metadata = {"schema_version": 3, "status": status, "git_commit": "deadbeef", "git_dirty": True,
                "gpu_name": "Example GPU", "gpu_compute_capability": "8.9", "cuda_compiler": "NVIDIA 13.3",
                "cuda_architectures": "89", "cuda_flags": "-O3", "timestamp_utc": "2026-01-02T03:04:05Z",
                "fp_verification_library_sha256": "c" * 64, "fp_verified": True,
                "timing_methodology": "CUDA events; kernel only", "trial_method": "reference then candidate",
                "config": {"reference": "cuda-naive-fma", "candidate": "cuda-naive-no-fma", "dtype": "float32",
                           "warmups": 10, "iterations": 100, "trials": 5, "atol": 1e-6, "rtol": 1e-5}}
    rows = []
    for row_id, kernel, median, passed, failures in (
            (0, "cuda-naive-fma", ".26", "true", "0"),
            (1, "cuda-naive-no-fma", ".25", "false", "25")):
        rows.append({"row_id": row_id, "M": 512, "N": 512, "K": 512, "kernel": kernel,
                     "backend": "cuda", "timing_mode": "kernel_only", "median_ms": median,
                     "gflops": "1000", "median_ci_low_ms": ".24", "median_ci_high_ms": ".27",
                     "median_speedup": "1" if row_id == 0 else "1.04", "stability_warnings": "",
                     "bitwise_equal": "true" if row_id == 0 else "false", "divergent_count": "25",
                     "tolerance_pass": passed, "tolerance_failures": failures, "max_ulp": "4",
                     "mean_divergent_ulp": "2", "max_absolute_error": "2e-6", "max_relative_error": "3e-6"})
    pair = [{"M": 512, "N": 512, "K": 512, "reference_kernel": "cuda-naive-fma",
             "candidate_kernel": "cuda-naive-no-fma", "reference_median_ms": ".26",
             "candidate_median_ms": ".25", "latency_ratio": ".9615", "percent_difference": "-3.85",
             "speedup": "1.04", "median_ci_overlap": "true", "tolerance_pass": "false",
             "reference_timing_mode": "kernel_only", "candidate_timing_mode": "kernel_only"}]
    stats = [{"row_id": value, "scope": "overall", "measurement_count": 500,
              "warmup_count": 50, "trial_count": 5} for value in (0, 1)]
    values = {"metadata.json": metadata, "summary.csv": rows, "pairwise.csv": pair,
              "timing_statistics.csv": stats}
    for name, value in values.items():
        if name in omit:
            continue
        write_json(root / name, value) if name.endswith(".json") else write_csv(root / name, value)
    return root


class CompilerPerformanceReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ptx = self.root / "ptx.json"
        self.sass = self.root / "sass.json"
        write_json(self.ptx, ptx_fixture())
        write_json(self.sass, sass_fixture())
        self.benchmark = create_benchmark(self.root / "benchmark")
        self.reference_map = "ptx=mi_naive_fma,sass=mi_naive_fma,benchmark=cuda-naive-fma"
        self.candidate_map = "ptx=mi_naive_no_fma,sass=mi_naive_no_fma,benchmark=cuda-naive-no-fma"

    def tearDown(self):
        self.temporary.cleanup()

    def combined_arguments(self, output):
        return ["--ptx-comparison", str(self.ptx), "--sass-comparison", str(self.sass),
                "--benchmark", str(self.benchmark), "--reference-map", self.reference_map,
                "--candidate-map", self.candidate_map, "--output", str(output)]

    def test_mapping_is_explicit_but_compilation_identity_is_unverified(self):
        result = report.run(self.combined_arguments(self.root / "output"))
        self.assertEqual(result["identity_mapping"]["status"], "explicit")
        self.assertEqual(result["compilation_provenance"]["status"], "unverified")
        self.assertTrue(all(item["status"] == "unverified"
                            for item in result["compilation_provenance"]["relationships"]))
        self.assertNotIn("binary_sha256", result["compilation_provenance"]["strong_identifiers"]["sass"])

    def test_scopes_virtual_and_physical_resources_are_preserved(self):
        result = report.run(self.combined_arguments(self.root / "scope"))
        self.assertEqual(result["evidence"]["ptx"]["text_comparison"]["raw"]["scope"], "full_module")
        self.assertEqual(result["evidence"]["ptx"]["selected_scope"], "selected_entry_only")
        self.assertEqual(result["evidence"]["sass"]["text_comparison"]["raw"]["scope"], "full_file")
        self.assertEqual(result["evidence"]["sass"]["selected_scope"], "selected_function_only")
        text = (self.root / "scope" / "report.md").read_text()
        self.assertIn("PTX declarations, not physical allocation", text)
        self.assertIn("Physical registers/thread", text)

    def test_benchmark_statuses_and_failed_candidate_remain_separate(self):
        result = report.run(self.combined_arguments(self.root / "status"))
        benchmark = result["evidence"]["benchmark"]
        self.assertEqual(benchmark["completion_status"], "tolerance_failed")
        self.assertEqual(benchmark["data_completeness"], "complete")
        self.assertEqual(benchmark["correctness_status"], "failed")
        self.assertEqual(benchmark["rows"][0]["measurement_count"], 500)
        text = (self.root / "status" / "report.md").read_text()
        self.assertIn("not presented as a successful optimization", text)
        self.assertIn("| 512×512×512 | 0.26 | 0.25 |", text)

    def test_missing_required_benchmark_file_is_incomplete_not_tolerance_failure(self):
        incomplete = create_benchmark(self.root / "incomplete", status="complete", omit=("summary.csv",))
        result = report.run(["--benchmark", str(incomplete), "--output", str(self.root / "incomplete-out")])
        benchmark = result["evidence"]["benchmark"]
        self.assertEqual(benchmark["data_completeness"], "incomplete")
        self.assertEqual(benchmark["correctness_status"], "unavailable")
        self.assertTrue(any(item["code"] == "benchmark_required_files_missing" for item in result["findings"]))

    def test_explicit_mapping_is_required_and_validated(self):
        with self.assertRaisesRegex(ValueError, "explicitly map"):
            report.run(["--ptx-comparison", str(self.ptx), "--sass-comparison", str(self.sass),
                        "--output", str(self.root / "missing-map")])
        with self.assertRaisesRegex(ValueError, "does not match"):
            report.run(["--ptx-comparison", str(self.ptx), "--sass-comparison", str(self.sass),
                        "--reference-map", "ptx=wrong,sass=mi_naive_fma",
                        "--candidate-map", "ptx=mi_naive_no_fma,sass=mi_naive_no_fma",
                        "--output", str(self.root / "wrong-map")])

    def test_conflicting_architecture_is_incompatible(self):
        write_json(self.sass, sass_fixture("sm_90"))
        result = report.run(self.combined_arguments(self.root / "conflict"))
        self.assertEqual(result["compilation_provenance"]["status"], "incompatible")
        self.assertTrue(any(item["severity"] == "incompatible" for item in result["findings"]))

    def test_partial_evidence_and_absence_are_explicit(self):
        result = report.run(["--sass-comparison", str(self.sass), "--output", str(self.root / "partial")])
        self.assertEqual(result["evidence"]["ptx"]["status"], "absent")
        self.assertEqual(result["evidence"]["benchmark"]["status"], "absent")
        self.assertEqual(result["compilation_provenance"]["status"], "absent")

    def test_output_is_deterministic_and_never_overwritten(self):
        first, second = self.root / "first", self.root / "second"
        report.run(self.combined_arguments(first))
        report.run(self.combined_arguments(second))
        self.assertEqual((first / "experiment.json").read_bytes(), (second / "experiment.json").read_bytes())
        with self.assertRaises(FileExistsError):
            report.run(self.combined_arguments(first))


if __name__ == "__main__":
    unittest.main()
