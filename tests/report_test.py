#!/usr/bin/env python3
"""CPU-only report parsing tests; render checks run when matplotlib is installed."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import report


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'run'
        self.run.mkdir()

    def write(self, rows, metadata=None):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (self.run / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        if metadata is not None:
            (self.run / 'metadata.json').write_text(json.dumps(metadata))

    @staticmethod
    def row(kernel='tiled', **extra):
        return dict(M=2, N=3, K=7, kernel=kernel, reference_kernel='naive',
                    mean_ms=2, gflops=0.000042, speedup=2, divergent_count=3, **extra)

    def test_legacy_and_rectangular_percentages(self):
        self.write([self.row()])
        metadata, rows, warnings = report.load_report(self.run)
        self.assertEqual(metadata, {})
        self.assertTrue(warnings)
        self.assertEqual(rows[0]['shape'], (2, 3, 7))
        self.assertEqual(report.percentage(rows[0], 'divergence'), 50)
        self.assertIsNone(report.percentage(rows[0], 'tolerance'))
        self.assertEqual(rows[0]['mean_ms'], 2)

    def test_kernel_grouping_and_self_comparison_exclusion(self):
        self.write([self.row('naive'), self.row(), self.row('cuda-naive-no-fma')])
        _, rows, _ = report.load_report(self.run)
        self.assertEqual(len(report.group_rows(rows)), 3)
        groups = report.group_rows(rows, numerical=True)
        self.assertEqual(set(groups), {'tiled vs naive', 'cuda-naive-no-fma vs naive'})

    def test_numerical_aggregates_are_full_output_not_sample_counts(self):
        self.write([self.row(divergent_percent=50, tolerance_failures=1)])
        (self.run / 'mismatches.csv').write_text('row,col\n0,0\n')
        _, rows, _ = report.load_report(self.run)
        self.assertEqual(report.percentage(rows[0], 'divergence'), 50)
        self.assertAlmostEqual(report.percentage(rows[0], 'tolerance'), 100 / 6)

    def test_invalid_optional_values_and_metadata(self):
        row = self.row()
        row.update(mean_ms='nan', gflops='unknown', divergent_count=999, speedup=-1)
        self.write([row])
        (self.run / 'metadata.json').write_text('[]')
        _, rows, warnings = report.load_report(self.run)
        self.assertGreaterEqual(len(warnings), 5)
        for field in ('mean_ms', 'gflops', 'divergent_count', 'speedup'):
            self.assertIsNone(rows[0][field])

    def test_missing_required_columns_and_duplicates(self):
        self.write([dict(kernel='naive')])
        with self.assertRaises(ValueError):
            report.load_report(self.run)
        self.write([self.row(), self.row()])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            report.load_report(self.run)

    def test_report_preserves_context(self):
        self.write([self.row()], dict(gpu_name='Synthetic GPU', git_dirty=True,
                                      config=dict(seed=42)))
        metadata, rows, warnings = report.load_report(self.run)
        text = report.markdown(self.run, metadata, rows, [], warnings)
        self.assertIn('Synthetic GPU', text)
        self.assertIn('dirty', text)
        self.assertIn('42', text)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'optional matplotlib not installed')
    def test_render_correct_values_and_immutable_sources(self):
        self.write([self.row('naive'), self.row(tolerance_failures=0)], dict(config={}))
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.run.iterdir()}
        # Capture plotted values before closing to verify the actual figure data.
        from matplotlib.axes import Axes
        original = Axes.plot
        plotted = []

        def record(ax, x, y, *args, **kwargs):
            plotted.append(list(y))
            return original(ax, x, y, *args, **kwargs)

        with patch.object(Axes, 'plot', record):
            self.assertEqual(report.main([str(self.run)]), 0)
        self.assertIn([2.0], plotted)
        self.assertIn([0.000042], plotted)
        self.assertIn([50.0], plotted)
        self.assertIn([0.0], plotted)
        for name, digest in before.items():
            self.assertEqual(hashlib.sha256((self.run / name).read_bytes()).hexdigest(), digest)
        output = self.run / 'report'
        self.assertEqual(len(list(output.glob('*.png'))), 4)
        for image in output.glob('*.png'):
            self.assertTrue(image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertIn('M×N', (output / 'report.md').read_text())
        self.assertEqual(report.main([str(self.run)]), 2)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'optional matplotlib not installed')
    def test_compare_only_and_missing_metrics(self):
        self.write([dict(M=4, N=4, K=4, kernel='tiled', reference_kernel='naive', divergent_count=0)])
        self.assertEqual(report.main([str(self.run)]), 0)
        self.assertEqual([p.name for p in (self.run / 'report').glob('*.png')], ['numerics.png'])
        self.assertIn('Skipped latency.png', (self.run / 'report' / 'report.md').read_text())


if __name__ == '__main__':
    unittest.main()
