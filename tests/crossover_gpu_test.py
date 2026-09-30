"""Small real-device crossover/replay smoke test; no performance assertions."""
import csv
import json
from pathlib import Path
import subprocess
import statistics
import sys
import tempfile

exe = sys.argv[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import reproduce


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


with tempfile.TemporaryDirectory() as directory:
    run = Path(directory) / 'run'
    result = subprocess.run([exe, 'crossover', '--sizes', '17,2', '--candidate', 'tiled',
                             '--iterations', '3', '--warmups', '1', '--trials', '2', '--bootstrap-samples', '30', '--save-output', '--output', str(run)],
                            capture_output=True, text=True)
    if result.returncode == 77:
        print('SKIP: CUDA execution unavailable')
        sys.exit(77)
    assert result.returncode == 0, (result.stdout, result.stderr)
    meta = json.loads((run / 'metadata.json').read_text())
    summary = read_rows(run / 'summary.csv')
    points = read_rows(run / 'crossover.csv')
    assert len(summary) == 6 and len(points) == 4
    assert meta['config']['warmups'] == 1 and meta['config']['iterations'] == 3
    assert 'cudaMalloc' in meta['end_to_end_timing_methodology']
    captured = read_rows(run / 'timing_samples.csv')
    assert len(captured) == 48 and {s['trial'] for s in captured} == {'0','1'}
    for row in summary:
        measured = [float(s['latency_ms']) for s in captured if s['row_id']==row['row_id'] and s['phase']=='measurement']
        assert float(row['median_ms']) == statistics.median(measured)
        assert row['median_ci_low_ms'] and row['median_ci_high_ms']
    assert len(read_rows(run / 'timing_statistics.csv')) == 18

    for n in ('2','17'):
        by_mode = {r['timing_mode']: r for r in summary if r['M'] == n}
        assert set(by_mode) == {'host_matmul','kernel_only','end_to_end'}
        assert by_mode['kernel_only']['output_sha256'] == by_mode['end_to_end']['output_sha256']
        for mode, row in by_mode.items():
            assert row['tolerance_pass'] == 'true' and float(row['median_ms']) > 0
            assert (run / row['output_file']).is_file()
        for point in (p for p in points if p['M'] == n):
            expected = float(by_mode['host_matmul']['median_ms']) / float(by_mode[point['timing_mode']]['median_ms'])
            assert abs(float(point['speedup']) - expected) < 1e-12 * expected
    replay = Path(directory) / 'replay'
    result = subprocess.run(reproduce.command(meta, exe, replay), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    repeated = read_rows(replay / 'summary.csv')
    assert [(r['kernel'], r['timing_mode'], r['output_sha256']) for r in summary] == [
        (r['kernel'], r['timing_mode'], r['output_sha256']) for r in repeated]
    bench = Path(directory) / 'benchmark'
    result = subprocess.run([exe,'benchmark','--sizes','2,17','--trials','2','--iterations','3','--warmups','1',
                             '--bootstrap-samples','30','--output',str(bench)],capture_output=True,text=True)
    assert result.returncode == 0, (result.stdout,result.stderr)
    assert len(read_rows(bench / 'timing_samples.csv')) == 32
    assert len(read_rows(bench / 'pairwise.csv')) == 2
print('Crossover/benchmark trials, timing samples, artifacts and replay passed')
