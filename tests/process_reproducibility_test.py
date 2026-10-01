"""Fresh subprocess capture and portable comparison; no second GPU needed."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import process_reproducibility as study
import compare_hardware as hardware

EXE = Path(sys.argv.pop(1)).resolve()
COMPARATOR = Path(sys.argv.pop(1)).resolve()
GPU = '--gpu' in sys.argv
if GPU:
    sys.argv.remove('--gpu')


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def capture(path, operation='reduction_sum', method='fp32_pairwise', extra=()):
    command = [sys.executable, str(ROOT/'scripts/process_reproducibility.py'), '--executable', str(EXE),
               '--comparator', str(COMPARATOR), '--operation', operation, '--method', method,
               '--process-repeats', '3', '--output', str(path), *extra]
    result = subprocess.run(command, capture_output=True, text=True)
    if GPU and result.returncode == 77:
        raise unittest.SkipTest('CUDA runtime/device unavailable')
    if result.returncode:
        raise AssertionError(result.stdout+result.stderr)
    return study.read_capture(path)


def alter_output(run, bits):
    """Synthetic bits, with consistent native payload/sidecar, for edge-case diagnostics."""
    row = run['rows'][0]
    path = hardware.binary(run, row)
    payload = struct.pack('<I', bits)
    path.write_bytes(path.read_bytes()[:24]+payload)
    row['output_sha256'] = hashlib.sha256(payload).hexdigest()
    sidecar = Path(str(path)+'.json')
    context = hardware.load_json(sidecar)
    context['output_sha256'] = row['output_sha256']
    study.write_json(sidecar, context)


class ProcessTests(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(study.classification([dict(divergent_count=0, tolerance_failures=0)]), 'bitwise_stable')
        self.assertEqual(study.classification([dict(divergent_count=1, tolerance_failures=0)]), 'numerically_stable_but_not_bitwise')
        self.assertEqual(study.classification([dict(divergent_count=1, tolerance_failures=1)]), 'varying')
        # Identical NaNs are bitwise stable but tolerance still fails.
        self.assertEqual(study.classification([dict(divergent_count=0, tolerance_failures=1)]), 'bitwise_stable')
        with self.assertRaises(ValueError):
            study.classification([])

    def test_capture_portability_and_compatibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for op, method in [('reduction_sum', 'neumaier_fp32'), ('dot', 'kahan_fp32'), ('matmul', 'cpu')]:
                a = capture(root/op, op, method, ['--size', '5'])
                meta = a['metadata']
                self.assertEqual(meta['classification'], 'bitwise_stable')
                self.assertEqual(len({l['run_id'] for l in meta['launches']}), 3)
                self.assertTrue(all(isinstance(l['pid'], int) and l['pid'] > 0 for l in meta['launches']))
                self.assertEqual(len(rows(a['path']/'process_pairs.csv')), 3)
                records = rows(a['path']/'process_reproducibility.csv')
                self.assertEqual(len({r['output_sha256'] for r in records}), 1)
                self.assertTrue(all(r['bitwise_equal'] == 'true' and r['numerical_agreement'] == 'true' for r in records))
                self.assertEqual(len(records[0]['output_bits']), 0 if op == 'matmul' else 32)
                # Relocation must not depend on original executable paths/commands.
                moved = root/(op+'-moved')
                shutil.copytree(a['path'], moved)
                b = study.read_capture(moved)
                study.require_compatible(b['runs'][0], a['runs'][0])
                aggregate = root/(op+'-aggregate')
                result = subprocess.run([sys.executable, str(ROOT/'scripts/compare_hardware.py'), '--process-captures',
                    str(a['path']), str(moved), '--baseline', str(moved), '--comparator', str(COMPARATOR),
                    '--output', str(aggregate)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(rows(aggregate/'cross_hardware_comparison.csv')), 6)
                self.assertTrue((aggregate/'report.md').is_file())
                self.assertEqual(hardware.load_json(aggregate/'cross_hardware_metadata.json')['baseline_repeat'], 0)
                for field, value in [('seed', 0), ('input', 'cancellation'), ('dtype', 'float64'),
                                     ('atol', 0), ('shapes', [{'M':1,'N':1,'K':19}])]:
                    incompatible = copy.deepcopy(b['runs'][0])
                    incompatible['config'][field] = value
                    with self.assertRaisesRegex(ValueError, 'Incompatible captures'):
                        study.require_compatible(incompatible, a['runs'][0])
                different = copy.deepcopy(b['runs'][0])
                different['metadata'].update(gpu_name='Synthetic test GPU', cuda_compiler='Synthetic toolchain')
                study.require_compatible(different, a['runs'][0])
                _, _, _, warnings = hardware.compatibility(different, a['runs'][0], different['rows'][0], a['runs'][0]['rows'][0])
                self.assertTrue(any('cross_toolchain' in w for w in warnings))
                # Tampering and path traversal fail before native comparison.
                manifest = hardware.load_json(moved/study.MANIFEST)
                manifest['launches'][0]['directory'] = '../outside'
                study.write_json(moved/study.MANIFEST, manifest)
                with self.assertRaisesRegex(ValueError, 'escapes'):
                    study.read_capture(moved)

    def test_differences_and_rejections(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = capture(root/'a', extra=['--size', '1'])
            ref, candidate = a['runs'][:2]
            for left, right, equal, passed, ulp in [(0, 0x80000000, False, True, 0),
                    (0x3f800000, 0x3f800001, False, True, 1),
                    (0x3f800000, 0x40000000, False, False, 8388608),
                    (0x7fc00001, 0x7fc00001, True, False, 0)]:
                alter_output(ref, left); alter_output(candidate, right)
                metric = study.compare(COMPARATOR, ref, candidate)
                self.assertEqual((metric['bitwise_equal'], metric['numerical_agreement'], metric['max_ulp']), (equal, passed, ulp))
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                study.read_capture(root/'a')
            b = capture(root/'b', extra=['--size', '3'])
            c = capture(root/'c', extra=['--size', '4'])
            dest = root/'incompatible'
            with self.assertRaisesRegex(ValueError, 'Incompatible captures'):
                study.compare_captures([b['path'], c['path']], b['path'], dest, COMPARATOR)
            self.assertFalse(dest.exists())
            dest.mkdir()
            with self.assertRaises(FileExistsError):
                study.compare_captures([b['path'], b['path']], b['path'], dest, COMPARATOR)
            self.assertEqual(study.main(['--executable', str(EXE), '--output', str(root/'b')]), 2)
            self.assertEqual(study.main(['--executable', str(EXE), '--output', str(root/'invalid'), '--process-repeats', '1']), 2)


class GpuTests(unittest.TestCase):
    def test_fresh_cuda_processes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for op in ('reduction_sum', 'dot', 'matmul'):
                cap = capture(root/op, op, 'tiled' if op == 'matmul' else 'cuda-tree', ['--size', '17'])
                self.assertEqual(cap['metadata']['classification'], 'bitwise_stable')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GpuTests if GPU else ProcessTests)
    result = unittest.TextTestRunner().run(suite)
    sys.exit(77 if GPU and result.skipped else 0 if result.wasSuccessful() else 1)
