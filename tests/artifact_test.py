"""CPU-only CLI/serialization/provenance integration tests (stdlib only)."""
import csv
import json
import importlib.util
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile

exe, fixture, source = sys.argv[1:]


def run(*args, code=0):
    result = subprocess.run([exe, *args], cwd=root, capture_output=True, text=True)
    assert result.returncode == code, (args, result.returncode, result.stdout, result.stderr)
    return result


def load(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    output = root / 'run with spaces'
    run('compare', '--reference', 'cpu', '--candidate', 'cpu', '--sizes', '4,17',
        '--seed', '42', '--output', str(output))
    meta = load(output / 'metadata.json')
    assert meta['status'] == 'complete'
    assert meta['config']['seed'] == 42 and meta['config']['seed_b'] == 123
    assert meta['config']['warmups'] == meta['config']['iterations'] == 0
    assert meta['git_dirty'] in (True, False, 'unknown')
    if meta['git_dirty'] is True:
        assert 'WARNING: DIRTY' in (output / 'console.txt').read_text()
    assert 'cpu' in (output / 'console.txt').read_text()
    assert not (output / 'mismatches.csv').exists()
    rows = list(csv.DictReader((output / 'summary.csv').open()))
    assert len(rows) == 2 and rows[1]['M'] == '17'
    for row in rows:
        assert None not in row and row['bitwise_equal'] == row['tolerance_pass'] == 'true'
        assert row['mean_ms'] == row['gflops'] == row['speedup'] == ''
    before = (output / 'metadata.json').read_bytes()
    run('compare', '--reference', 'cpu', '--candidate', 'cpu', '--output', str(output), code=1)
    assert (output / 'metadata.json').read_bytes() == before

    fixture_dir = root / 'fixture'
    subprocess.run([fixture, str(fixture_dir)], check=True)
    fixture_meta = load(fixture_dir / 'metadata.json')
    assert fixture_meta['escape_test'] == 'quote" slash\\ newline\n tab\t control\x01'
    mismatch = list(csv.DictReader((fixture_dir / 'mismatches.csv').open()))
    assert len(mismatch) == 2 and mismatch[0]['kernel'] == 'quoted,"kernel'
    assert mismatch[0]['ulp_distance'] == '1' and mismatch[0]['tolerance_pass'] == 'true'
    assert mismatch[1]['kind'] == 'first_numeric' and mismatch[1]['tolerance_pass'] == 'false'
    assert len(mismatch[0]['reference_bits']) == 32

    for args in [('compare', '--seed', '-1'), ('benchmark', '--iterations', '0'),
                 ('compare', '--atol', 'NaN'), ('compare', '--sizes', '4,'), ('unknown',)]:
        run(*args, code=2)
    run('--help')
    run('compare', '--help')

    spec = importlib.util.spec_from_file_location('reproduce', Path(source) / 'scripts/reproduce.py')
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    replay_dir = root / 'replay'
    subprocess.run(replay.command(meta, exe, replay_dir), check=True, capture_output=True)
    assert (replay_dir / 'summary.csv').read_text() == (output / 'summary.csv').read_text()
    invalid_schema = dict(meta, schema_version=999)
    try:
        replay.command(invalid_schema, exe, replay_dir)
        raise AssertionError('Unsupported replay schema accepted')
    except ValueError:
        pass

    # New schema carries the experiment controls, including benchmark replay.
    controlled = json.loads(json.dumps(meta))
    controlled['config'].update(mode='benchmark', reference='cuda-naive-fma',
        candidate='cuda-naive-reordered', input='cancellation', generator='cancellation-v1',
        warmups=3, iterations=50, max_mismatches=10)
    invocation = replay.command(controlled, exe, root / 'controlled-replay')
    for value in ('cuda-naive-fma', 'cuda-naive-reordered', 'cancellation', '--max-mismatches'):
        assert value in invocation
    assert rows[0]['contraction_mode'] == 'compiler_default'
    assert rows[0]['max_ulp'] == rows[0]['tolerance_failures'] == '0'
    assert meta['fp_verified'] in (True, False)
    assert meta['config']['reference_accumulation'] == 'increasing_k'
    fixture_rows = list(csv.DictReader((fixture_dir / 'summary.csv').open()))
    assert fixture_rows[0]['divergent_count'] == '2'
    assert fixture_rows[0]['tolerance_failures'] == '1'
    assert fixture_rows[0]['ulp_1'] == '1'
    assert fixture_rows[0]['mismatches_saved'] == '2'

    # Instruction inspection is tested without CUDA tools or a GPU.
    spec = importlib.util.spec_from_file_location('verify_fp', Path(source) / 'scripts/verify_fp.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    ptx = ('.entry mi_naive_fma() { fma.rn.f32 %f1,%f2,%f3,%f4; }\n'
           '.entry mi_naive_no_fma() { mul.rn.f32 %f1,%f2,%f3; add.rn.f32 %f4,%f1,%f5; }\n'
           '.entry mi_naive_reordered() { fma.rn.f32 %f1,%f2,%f3,%f4; add.rn.f32 %f1,%f2,%f3; }')
    assert len(verifier.inspect_ptx(ptx + ptx)) == 6
    for invalid in ('', ptx.replace('mul.rn.f32', 'fma.rn.f32'),
                    ptx.replace('mul.rn.f32', 'mad.rn.f32'), ptx[:-1],
                    ptx.replace('mul.rn.f32', 'call fn; mul.rn.f32')):
        try:
            verifier.inspect_ptx(invalid)
            raise AssertionError('Invalid PTX accepted')
        except ValueError:
            pass
    verification_report = root / 'unverified.json'
    failed_check = subprocess.run([sys.executable, str(Path(source) / 'scripts/verify_fp.py'),
        '--binary', exe, '--cuobjdump', str(root / 'missing-tool'), '--report', str(verification_report)], capture_output=True)
    assert failed_check.returncode == 1 and load(verification_report)['verified'] is False

    # Simulate unavailable hardware even on a machine that does have a GPU.
    import os
    skipped_dir = root / 'skipped'
    skipped = subprocess.run([exe, 'benchmark', '--sizes', '4', '--output', str(skipped_dir)],
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True)
    assert skipped.returncode == 77, (skipped.stdout, skipped.stderr)
    assert load(skipped_dir / 'metadata.json')['status'] == 'skipped'

    # A recorded execution failure is distinct from a numerical mismatch or skip.
    failed_dir = root / 'failed'
    run('compare', '--reference', 'cpu', '--candidate', 'cpu', '--m', '18446744073709551615',
        '--n', '2', '--k', '2', '--output', str(failed_dir), code=1)
    assert load(failed_dir / 'metadata.json')['status'] == 'failed'

    # Build-time provenance refresh: clean, untracked, staged, and unavailable Git.
    if shutil.which('git'):
        repo = root / 'source'
        repo.mkdir()
        subprocess.run(['git', 'init', '-q', str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false',
                        'commit', '-q', '--allow-empty', '-m', 'fixture'], check=True)
        header = root / 'provenance.hpp'
    
        def provenance(path):
            subprocess.run(['cmake', f'-DSOURCE_DIR={path}', f'-DOUTPUT={header}',
                            '-P', str(Path(source) / 'cmake/provenance.cmake')], check=True)
            return header.read_text()
    
        assert 'git_dirty = "false"' in provenance(repo)
        (repo / 'new.txt').write_text('new source')
        assert 'git_dirty = "true"' in provenance(repo)
        subprocess.run(['git', '-C', str(repo), 'add', 'new.txt'], check=True)
        assert 'git_dirty = "true"' in provenance(repo)
        assert 'git_commit = "unknown"' in provenance(root)

print('CLI, JSON/CSV, mismatch, and Git provenance checks passed.')
