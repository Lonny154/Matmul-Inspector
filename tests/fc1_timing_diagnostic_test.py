#!/usr/bin/env python3
"""CPU-only tests for the FC1 timing diagnostic's scheduling and summaries."""

from pathlib import Path
import importlib.util
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "fc1_timing_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("fc1_timing_diagnostic", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TimingDiagnosticTests(unittest.TestCase):
    def test_schedule_is_reproducible_balanced_and_alternating(self):
        first = MODULE.build_schedule(blocks=20, seed=17)
        self.assertEqual(first, MODULE.build_schedule(blocks=20, seed=17))
        self.assertEqual(len(first), 120)
        for mode in MODULE.MODES:
            for implementation in MODULE.IMPLEMENTATIONS:
                self.assertEqual(sum(r["mode"] == mode and r["implementation"] == implementation
                                     for r in first), 20)
        for block in range(1, 21):
            rows = [r for r in first if r["block"] == block]
            expected = "baseline" if block % 2 else "candidate"
            self.assertTrue(all(next(r for r in rows if r["mode"] == mode)["implementation"] == expected
                                for mode in MODULE.MODES))

    def test_summary_keeps_timing_domains_separate(self):
        rows = []
        for value in (1.0, 3.0):
            rows.append({"mode": "allocating", "implementation": "baseline",
                         "cuda_event_ms_per_operation": value,
                         "host_submission_ms_per_block": value * 10,
                         "host_wall_ms_per_block": value * 20})
        summary = MODULE.summarize_measurements(rows)["allocating"]["baseline"]
        self.assertEqual(summary["cuda_event_ms_per_operation"]["median"], 2.0)
        self.assertEqual(summary["host_submission_ms_per_block"]["median"], 20.0)
        self.assertEqual(summary["host_wall_ms_per_block"]["median"], 40.0)


if __name__ == "__main__":
    unittest.main()
