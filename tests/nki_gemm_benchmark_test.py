#!/usr/bin/env python3
"""CPU-only tests for NKI GEMM harness plumbing; no NKI installation required."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nki_gemm_benchmark as benchmark


class FakeBackend:
    name = "synthetic"
    timing_mode = "synthetic_test"
    timing_note = "deterministic synthetic samples"

    def __init__(self):
        self.calls = []

    def execute(self, variant, a, b, warmups, iterations):
        self.calls.append((variant, a.copy(), b.copy(), warmups, iterations))
        offset = 1.0 if variant == "full-tiled" else 2.0
        logical_a = a.T if variant == "pretransposed" else a
        return benchmark.Execution(logical_a @ b, [offset + i for i in range(iterations)])

    def metadata(self):
        return {"backend": self.name, "timing_mode": self.timing_mode}


class NkiGemmBenchmarkTests(unittest.TestCase):
    def test_shape_and_variant_parsing(self):
        parsed = benchmark.parse_args([
            "--shapes", "128x128x512,256x256x1024",
            "--variants", "pretransposed", "--warmups", "0", "--iterations", "2",
        ])
        self.assertEqual(parsed.shapes, (
            benchmark.Shape(128, 128, 512), benchmark.Shape(256, 256, 1024)
        ))
        self.assertEqual(parsed.variants, ("pretransposed",))
        cartesian = benchmark.parse_args([
            "--cartesian", "--m", "128,256", "--k", "128", "--n", "512,1024"
        ])
        self.assertEqual(len(cartesian.shapes), 4)
        self.assertEqual(cartesian.shapes[0], benchmark.Shape(128, 128, 512))
        self.assertEqual(cartesian.shapes[-1], benchmark.Shape(256, 128, 1024))

    def test_invalid_configuration(self):
        for arguments in (
            ["--shapes", "129x128x512"],
            ["--shapes", "128x127x512"],
            ["--shapes", "128x128x513"],
            ["--variants", "unknown"],
            ["--iterations", "0"],
            ["--backend", "hardware"],
        ):
            with self.assertRaises((SystemExit, ValueError), msg=arguments):
                benchmark.parse_args(arguments)

    def test_statistics_validation(self):
        stats = benchmark.timing_statistics([3.0, 1.0, 2.0])
        self.assertEqual(stats["mean_ms"], 2.0)
        self.assertEqual(stats["median_ms"], 2.0)
        self.assertEqual(stats["min_ms"], 1.0)
        self.assertEqual(stats["max_ms"], 3.0)
        self.assertAlmostEqual(stats["stddev_ms"], np.std([3.0, 1.0, 2.0]))
        for samples in ([], [-1.0], [float("nan")]):
            with self.assertRaises(ValueError):
                benchmark.timing_statistics(samples)

    def test_correctness_failure_is_recorded(self):
        class IncorrectBackend(FakeBackend):
            def execute(self, variant, a, b, warmups, iterations):
                result = super().execute(variant, a, b, warmups, iterations)
                result.output = result.output + np.float32(1.0)
                return result

        config = benchmark.Config(
            shapes=(benchmark.Shape(128, 128, 512),),
            variants=("full-tiled",), seed=7, warmups=0, iterations=1,
            atol=0.0, rtol=0.0, backend="simulate", output=Path("unused"),
        )
        row = benchmark.run_benchmark(config, IncorrectBackend())[0]
        self.assertEqual(row["correct"], "false")
        self.assertAlmostEqual(row["max_abs_error"], 1.0, delta=1e-4)
        self.assertAlmostEqual(row["mean_abs_error"], 1.0, delta=1e-4)

    def test_shared_inputs_metrics_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            config = benchmark.Config(
                shapes=(benchmark.Shape(128, 128, 512),),
                variants=benchmark.VARIANTS,
                seed=42,
                warmups=1,
                iterations=3,
                atol=1e-4,
                rtol=1e-4,
                backend="simulate",
                output=output,
            )
            backend = FakeBackend()
            rows = benchmark.run_benchmark(config, backend)
            self.assertEqual(len(rows), 2)
            self.assertTrue(np.array_equal(backend.calls[0][1].T, backend.calls[1][1]))
            self.assertTrue(np.array_equal(backend.calls[0][2], backend.calls[1][2]))
            self.assertEqual({row["correct"] for row in rows}, {"true"})
            self.assertEqual({row["max_abs_error"] for row in rows}, {0.0})
            self.assertEqual({row["mean_abs_error"] for row in rows}, {0.0})
            self.assertEqual(rows[0]["median_ms"], 2.0)
            self.assertEqual(rows[1]["median_ms"], 3.0)

            benchmark.write_artifacts(config, backend, rows)
            with (output / "summary.csv").open(newline="") as stream:
                captured = list(csv.DictReader(stream))
            self.assertEqual(len(captured), 2)
            for field in (
                "variant", "M", "K", "N", "correct", "max_abs_error",
                "mean_abs_error", "mean_ms", "median_ms", "min_ms", "max_ms",
                "stddev_ms", "timing_mode", "timing_note",
            ):
                self.assertIn(field, captured[0])
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["experiment"], "nki-gemm")
            self.assertEqual(metadata["backend"]["backend"], "synthetic")
            with self.assertRaises(FileExistsError):
                benchmark.write_artifacts(config, backend, rows)


if __name__ == "__main__":
    unittest.main()
