"""GPU execution/replay integration; CTest treats exit 77 as a clean skip."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

exe, source = sys.argv[1:]
spec = importlib.util.spec_from_file_location('reproduce', Path(source) / 'scripts/reproduce.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    for candidate, input_mode, expected_status in (
        ('cuda-naive-no-fma', 'fma-sensitive', 0),
        ('cuda-naive-reordered', 'cancellation', 1)):
        original = root / candidate
        args = [exe, 'compare', '--sizes', '4,17', '--seed', '42', '--reference', 'cuda-naive-fma',
                '--candidate', candidate, '--input', input_mode, '--max-mismatches', '3', '--output', str(original)]
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode == 77:
            print(result.stdout)
            raise SystemExit(77)
        assert result.returncode == expected_status, (result.stdout, result.stderr)
        metadata = json.loads((original / 'metadata.json').read_text())
        assert metadata['fp_verified'] is True
        assert metadata['mismatches_truncated'] is True
        assert metadata['config']['candidate_contraction'] == ('separate_rn_mul_add' if input_mode == 'fma-sensitive' else 'explicit_fma_rn')
        rows = list(csv.DictReader((original / 'summary.csv').open()))
        assert all(float(row['divergent_percent']) == 100 for row in rows)
        assert all(row['mismatches_saved'] == '3' for row in rows)
        repeat = root / (candidate + '-replay')
        rerun = subprocess.run(replay.command(metadata, exe, repeat), capture_output=True, text=True)
        assert rerun.returncode == expected_status, (rerun.stdout, rerun.stderr)
        for filename in ('summary.csv', 'mismatches.csv'):
            assert (original / filename).read_bytes() == (repeat / filename).read_bytes()
    benchmark = subprocess.run([exe, 'benchmark', '--sizes', '4', '--reference', 'cuda-naive-fma',
        '--candidate', 'cuda-naive-reordered', '--iterations', '2', '--warmups', '0', '--output', str(root / 'timed')],
        capture_output=True, text=True)
    assert benchmark.returncode in (0,1), (benchmark.stdout, benchmark.stderr)
    rows = list(csv.DictReader((root / 'timed/summary.csv').open()))
    assert len(rows) == 2 and all(float(row['mean_ms']) > 0 for row in rows)
print('Controlled FMA/reordering artifact and replay checks passed.')
