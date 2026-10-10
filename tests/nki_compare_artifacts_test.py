#!/usr/bin/env python3
"""CPU-only tests for NKI compiler-artifact comparison."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nki_compare_artifacts as compare


INSTRUCTIONS = """
┌─────────────────┬───────┐
│ Opcode          │ Count │
├─────────────────┼───────┤
│ MATMUL          │ 128   │
│ UNKNOWN(0xd4)   │ 48    │
│ COPY            │ 16    │
└─────────────────┴───────┘
"""

DMA_EMPTY = """Number of DMA descriptors for each op type:
┌────┬───────┐
│ Op │ Count │
└────┴───────┘

Total descriptors: 0 (0 GB)

Number of DMA engines used by each queue:
┌────────────────┬────────────────┐
│ Queue          │ DMA Engines    │
├────────────────┼────────────────┤
│ qPoolDynamic   │ 16             │
├────────────────┼────────────────┤
│ TOTAL          │ 16             │
└────────────────┴────────────────┘
"""


def mlir(symbol="baseline.kernel", constant=1, source="baseline.py"):
    return f'''module attributes {{nisa.target = #nisa.target<trn1>}} {{
  func.func @{symbol}(%arg0: memref<128xf32>) attributes {{kernel}} {{
    %c = arith.constant {constant} : i32 loc(callsite("sdk.py":2:3 at callsite("{source}":4:5 at "{source}":1:1)))
    return loc("{source}":9:1)
  }}
}}
'''


class NkiArtifactComparisonTests(unittest.TestCase):
    def make_capture(self, root, name, *, source, symbol, constant=1,
                     instructions=INSTRUCTIONS, dma=DMA_EMPTY, neff=b"neff"):
        capture = root / name
        (capture / "sg00").mkdir(parents=True)
        (capture / "module.mlir").write_text(mlir(symbol, constant, source))
        (capture / "kernel.neff").write_bytes(neff)
        (capture / "sg00" / "instruction_stats.txt").write_text(instructions)
        (capture / "sg00" / "dma_stats.txt").write_text(dma)
        (capture / "info.json").write_text(json.dumps({"tool_version": "2.27", "nki_version": "0.6"}))
        (capture / "kernel_info.json").write_text(json.dumps({
            "global": {"kernels": {symbol: {}}, "summary": {"total_subgraphs": 1}}
        }))
        return capture

    def test_location_scanner_and_canonicalization(self):
        text = mlir("first.kernel", source="a.py")
        stripped = compare.strip_mlir_locations(text)
        self.assertNotIn("loc(", stripped)
        self.assertIn("arith.constant 1 : i32", stripped)
        left = compare.normalize_mlir(text)
        right = compare.normalize_mlir(mlir("second.kernel", source="elsewhere.py"))
        self.assertEqual(left, right)
        self.assertIn("@__nki_entry", left)
        with self.assertRaises(ValueError):
            compare.strip_mlir_locations("%0 = op loc(callsite(\"x\")")

    def test_canonicalization_preserves_semantics(self):
        self.assertNotEqual(compare.normalize_mlir(mlir(constant=1)),
                            compare.normalize_mlir(mlir(constant=2)))

    def test_statistics_parsers(self):
        self.assertEqual(compare.parse_instruction_stats(INSTRUCTIONS), {
            "MATMUL": 128, "UNKNOWN(0xd4)": 48, "COPY": 16
        })
        dma = compare.parse_dma_stats(DMA_EMPTY)
        self.assertEqual(dma["total_descriptors"], 0)
        self.assertEqual(dma["sections"]["number_of_dma_descriptors_for_each_op_type"], [])
        self.assertEqual(dma["sections"]["number_of_dma_engines_used_by_each_queue"][0]["Queue"], "qPoolDynamic")

    def test_k_blocking_style_equivalence(self):
        """Raw IR and NEFF differ, while canonical IR and aggregate stats match."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self.make_capture(root, "baseline", source="baseline.py", symbol="base.kernel", neff=b"one")
            candidate = self.make_capture(root, "blocked", source="blocked.py", symbol="blocked.kernel", neff=b"two")
            output = root / "result"
            result = compare.run([str(baseline), str(candidate), "--output", str(output)])
            comparison = result["comparison"]
            self.assertFalse(comparison["mlir"]["raw"]["identical"])
            self.assertTrue(comparison["mlir"]["canonical"]["identical"])
            self.assertTrue(comparison["instructions"]["identical"])
            self.assertTrue(comparison["dma"]["identical"])
            self.assertFalse(comparison["neff"]["byte_identical"])
            self.assertTrue((output / "comparison.json").is_file())
            self.assertTrue((output / "report.md").is_file())
            self.assertTrue((output / "mlir" / "normalized.diff").is_file())
            report = (output / "report.md").read_text()
            self.assertIn("| Raw MLIR | different |", report)
            self.assertIn("| Canonical MLIR | identical |", report)
            self.assertIn("## MLIR comparison", report)
            self.assertIn("## DMA comparison", report)
            self.assertIn("Both DMA reports contained valid parsed descriptor tables with no descriptor data rows", report)
            self.assertIn("## NEFF files", report)
            self.assertIn("`7692c3ad3540bb803c020b3aee66cd8887123234ea0c6e7143c0add73ff431ed`", report)
            self.assertIn("| Baseline | 3 |", report)
            self.assertIn("## Conclusions", report)
            self.assertIn("critical paths", report)
            self.assertNotIn("criticalpaths", report)
            with self.assertRaises(FileExistsError):
                compare.run([str(baseline), str(candidate), "--output", str(output)])

    def test_a_reuse_style_detects_ir_difference_with_equal_counts(self):
        """A source rewrite can retain statistics while changing canonical IR and NEFF."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self.make_capture(root, "v1", source="v1.py", symbol="v1.kernel", neff=b"v1")
            candidate = self.make_capture(root, "v2", source="v2.py", symbol="v2.kernel",
                                          constant=2, neff=b"v2")
            result = compare.run([str(baseline), str(candidate), "--output", str(root / "result")])
            comparison = result["comparison"]
            self.assertFalse(comparison["mlir"]["canonical"]["identical"])
            self.assertTrue(comparison["instructions"]["identical"])
            self.assertTrue(comparison["dma"]["identical"])
            self.assertFalse(comparison["neff"]["byte_identical"])

    def test_missing_optional_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = root / "left", root / "right"
            left.mkdir()
            right.mkdir()
            result = compare.run([str(left), str(right), "--output", str(root / "result")])
            self.assertEqual(result["comparison"]["mlir"]["status"], "unavailable")
            self.assertEqual(result["comparison"]["instructions"]["status"], "unavailable")
            self.assertEqual(result["comparison"]["neff"]["status"], "unavailable")

    def test_subgraph_ambiguity_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = self.make_capture(root, "capture", source="a", symbol="a")
            (capture / "sg01").mkdir()
            (capture / "sg01" / "instruction_stats.txt").write_text(INSTRUCTIONS)
            with self.assertRaises(ValueError):
                compare.collect(capture, None)
            selected = compare.collect(capture, "sg00")
            self.assertEqual(selected["instruction_counts"]["MATMUL"], 128)

    def test_instruction_delta_and_provenance_mismatch(self):
        changed = INSTRUCTIONS.replace("│ MATMUL          │ 128   │", "│ MATMUL          │ 64    │")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = self.make_capture(root, "left", source="a", symbol="a")
            right = self.make_capture(root, "right", source="b", symbol="b", instructions=changed)
            info = json.loads((right / "info.json").read_text())
            info["nki_version"] = "0.7"
            (right / "info.json").write_text(json.dumps(info))
            result = compare.run([str(left), str(right), "--output", str(root / "result")])
            self.assertFalse(result["comparison"]["instructions"]["identical"])
            matmul = next(row for row in result["comparison"]["instructions"]["opcodes"] if row["name"] == "MATMUL")
            self.assertEqual(matmul["delta"], -64)
            self.assertFalse(result["comparison"]["provenance"]["compatible"])

    def test_benchmark_fields_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "benchmark" / "nested"
            run.mkdir(parents=True)
            (run / "info.json").write_text(json.dumps({
                "nc_latency": {"50": 40}, "latency": {"50": 92},
                "throughput": 12.5, "nc_node_count": 1,
            }))
            parsed = compare.read_benchmark(root / "benchmark")
            self.assertEqual(parsed["nc_latency_us"]["50"], 40)
            self.assertEqual(parsed["runtime_latency_us"]["50"], 92)


if __name__ == "__main__":
    unittest.main()
