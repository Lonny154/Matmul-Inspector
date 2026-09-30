"""Lightweight crossover analysis, serialization and optional render/replay tests."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import crossover
import report
import reproduce

FIXTURE = sys.argv[1] if len(sys.argv) > 1 else None
EXECUTABLE = sys.argv[2] if len(sys.argv) > 2 else None
sys.argv = sys.argv[:1]


def rows(cpu, gpu, mode='kernel_only'):
    result = []
    for n, a, b in zip((2,4,8), cpu, gpu):
        result += [dict(shape=(n,n,n), kernel='cpu', timing_mode='host_matmul', median_ms=a),
                   dict(shape=(n,n,n), kernel='tiled', timing_mode=mode, median_ms=b)]
    return result


class CrossoverTests(unittest.TestCase):
    def test_first_win_and_reversal(self):
        points = crossover.analyze(list(reversed(rows([1,1,1], [2,0.5,2]))))[('tiled','kernel_only')]
        self.assertEqual([p['speedup'] for p in points], [0.5,2,0.5])
        self.assertEqual([p['status'] for p in points], ['cpu_faster_or_equal','first_sampled_gpu_win','cpu_faster_or_equal'])

    def test_no_win_and_ties(self):
        points = crossover.analyze(rows([1,1,1], [2,1,3]))[('tiled','kernel_only')]
        self.assertTrue(all(p['status'] == 'cpu_faster_or_equal' for p in points))
        self.assertIn('none observed', crossover.section({}, rows([1,1,1], [2,1,3])))

    def test_smallest_wins_and_modes_separate(self):
        data = rows([1,1,1], [.5,.5,.5]) + [r for r in rows([1,1,1], [3,3,.5], 'end_to_end') if r['kernel'] != 'cpu']
        groups = crossover.analyze(data)
        self.assertEqual(groups[('tiled','kernel_only')][0]['status'], 'first_sampled_gpu_win')
        self.assertEqual(groups[('tiled','end_to_end')][2]['status'], 'first_sampled_gpu_win')

    def test_missing_and_zero_not_infinite(self):
        points = crossover.analyze(rows([1,0,None], [0,1,2]))[('tiled','kernel_only')]
        self.assertTrue(all(p['speedup'] is None and p['status'] == 'unavailable' for p in points))
        points = crossover.analyze(rows([1,1,1], [float('inf'),float('nan'),0]))[('tiled','kernel_only')]
        self.assertTrue(all(p['speedup'] is None for p in points))

    @unittest.skipUnless(FIXTURE, 'C++ fixture path required')
    def test_artifact_roundtrip_replay_and_render(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'run'
            subprocess.run([FIXTURE, str(run)], check=True)
            metadata, data, warnings = report.load_report(run)
            self.assertEqual(metadata['config']['mode'], 'crossover')
            self.assertEqual(metadata['config']['iterations'], 2)
            groups = crossover.analyze(data)
            with (run / 'crossover.csv').open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(len(saved), 6)
            for record in saved:
                point = next(p for p in groups[(record['kernel'],record['timing_mode'])] if p['size'] == int(record['M']))
                self.assertEqual(record['crossover_status'], point['status'])
                self.assertEqual(float(record['speedup']), point['speedup'])
            command = reproduce.command(metadata, 'binary', 'new-run')
            self.assertIn('crossover', command)
            self.assertIn('--warmups', command)
            self.assertIn('--iterations', command)
            text = report.markdown(run, metadata, data, [], warnings)
            self.assertIn('N=4', text)
            self.assertIn('none observed', text)
            if importlib.util.find_spec('matplotlib'):
                before = {p: p.read_bytes() for p in run.iterdir() if p.is_file()}
                from matplotlib.axes import Axes
                original = Axes.plot
                plotted = []

                def record(ax, x, y, *args, **kwargs):
                    plotted.append((list(x), list(y), kwargs.get('label')))
                    return original(ax, x, y, *args, **kwargs)

                with patch.object(Axes, 'plot', record):
                    self.assertEqual(report.main([str(run)]), 0)
                self.assertIn(([2,4,8], [2,0.5,2], 'tiled: kernel_only'), plotted)
                self.assertIn(([2,4,8], [0.5,2,0.5], 'tiled: kernel_only'), plotted)
                self.assertIn(([2,4,8], [1,1,1], 'CPU: host_matmul'), plotted)
                self.assertEqual(len(list((run / 'report').glob('*.png'))), 2)
                self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))

    @unittest.skipUnless(EXECUTABLE, 'CLI path required')
    def test_cuda_unavailable_is_recorded_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'skip'
            result = subprocess.run([EXECUTABLE, 'crossover', '--sizes', '2', '--output', str(run)],
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True)
            self.assertEqual(result.returncode, 77, result.stderr)
            metadata = json.loads((run / 'metadata.json').read_text())
            self.assertEqual(metadata['status'], 'skipped')
            self.assertIn('steady_clock', metadata['end_to_end_timing_methodology'])
            self.assertIn('one untimed', metadata['crossover_execution_order'])


if __name__ == '__main__':
    unittest.main()
