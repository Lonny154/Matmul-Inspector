"""Lightweight CPU-only hashing, capture, aggregation and cross-run diagnostics tests."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

exe, comparator, output_test, source = sys.argv[1:]
source = Path(source)


def call(args, code=0):
    p = subprocess.run([str(x) for x in args], capture_output=True, text=True)
    assert p.returncode == code, (args, p.returncode, p.stdout, p.stderr)
    return p


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_rows(path, records):
    with path.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(records[0]))
        w.writeheader(); w.writerows(records)


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    # Golden SHA-256 verification covers block and final-padding boundaries.
    binaries = root / 'binaries'
    generated = call([output_test, binaries])
    for line in generated.stdout.splitlines():
        filename, fingerprint = line.split(',')
        raw = (binaries / filename).read_bytes()
        assert raw[:8] == b'MIFP32LE'
        assert hashlib.sha256(raw[24:]).hexdigest() == fingerprint
        n = int(filename.split('.')[0])
        assert raw[8:24] == struct.pack('<QQ', 1, n)
        assert raw[24:] == b''.join(struct.pack('<I', 0x9e3779b9 * i & 0xffffffff) for i in range(n))
    assert (binaries / 'special.bin').read_bytes()[24:] == struct.pack('<5I', 0, 0x80000000, 0x7fa12345, 0x7fc12345, 0xff800000)
    for content in (b'bad', b'MIFP32LE' + struct.pack('<QQ', 2, 2) + b'\0'*4,
                    b'MIFP64LE' + struct.pack('<QQ', 1, 1) + b'\0'*4):
        bad = root / 'bad.bin'; bad.write_bytes(content)
        call([comparator, bad, binaries / '1.bin', '1e-6', '1e-5'], code=2)
    call([comparator, binaries / '1.bin', binaries / '2.bin', '1e-6', '1e-5'], code=2)

    captured = root / 'captured'
    result = call([exe, 'compare', '--reference', 'cpu', '--candidate', 'cpu', '--sizes', '4,17',
                   '--seed', '42', '--save-output', '--output', captured])
    assert 'reference SHA-256:' in result.stdout and 'candidate SHA-256:' in result.stdout
    metadata = read_json(captured / 'metadata.json')
    assert metadata['schema_version'] == 3 and metadata['config']['save_output'] is True
    for row in rows(captured / 'summary.csv'):
        for prefix in ('output', 'reference'):
            filename = row['output_file' if prefix == 'output' else 'reference_output_file']
            assert hashlib.sha256((captured / filename).read_bytes()[24:]).hexdigest() == row[prefix + '_sha256']
    call([exe, 'compare', '--save-output'], code=2)
    call([exe, 'compare', '--save-output', '--save-output', '--output', root / 'invalid'], code=2)
    replayed = root / 'replayed'
    call([sys.executable, source / 'scripts/reproduce.py', captured / 'metadata.json', '--executable', exe, '--output', replayed])
    assert (captured / 'summary.csv').read_bytes() == (replayed / 'summary.csv').read_bytes()
    for filename in captured.glob('*.bin*'):
        assert filename.read_bytes() == (replayed / filename.name).read_bytes()

    # Synthetic performance/environment data below exist only in this temporary
    # test fixture. They are not measurements or checked-in hardware claims.
    def synthetic(name, gpu='Synthetic GPU A'):
        path = root / name
        shutil.copytree(captured, path)
        meta = read_json(path / 'metadata.json')
        meta.update(gpu_name=gpu, gpu_compute_capability='8.9', cpu_model='Synthetic CPU',
                    git_commit='a'*40, runtime_git_commit='a'*40, git_dirty=False,
                    build_git_dirty=False, runtime_git_dirty=False, build_type='Release',
                    cuda_compiler='Synthetic NVCC 1', cuda_runtime_version='12000',
                    timing_methodology='synthetic event fixture')
        meta['config'].update(mode='benchmark', reference='naive', candidate='tiled', warmups=3, iterations=5)
        write_json(path / 'metadata.json', meta)
        records = []
        for original in rows(path / 'summary.csv'):
            for kernel, milliseconds, filekey in [('naive', 2, 'reference_output_file'), ('tiled', 1, 'output_file')]:
                row = dict(original, kernel=kernel, reference_kernel='naive', mean_ms=str(milliseconds),
                           median_ms=str(milliseconds), min_ms=str(milliseconds), stddev_ms='0',
                           gflops=str(2*int(original['M'])*int(original['N'])*int(original['K'])/(milliseconds*1e6)),
                           speedup=str(2/milliseconds), output_file=original[filekey],
                           tile_size='16' if kernel == 'tiled' else '0', contraction_mode='compiler_default',
                           accumulation_mode='increasing_k_zero_padded_tiles' if kernel == 'tiled' else 'increasing_k')
                side = path / (row['output_file'] + '.json')
                context = read_json(side)
                context.update(kernel=kernel, accumulation_mode=row['accumulation_mode'])
                write_json(side, context)
                records.append(row)
        write_rows(path / 'summary.csv', records)
        return path

    script = source / 'scripts/compare_hardware.py'
    a, b = synthetic('a'), synthetic('b', 'Synthetic GPU B')
    def aggregate(name, inputs=(a, b), extra=(), code=0):
        target = root / name
        result = call([sys.executable, script, *inputs, '--comparator', comparator, '--output', target, *extra], code)
        return target, result

    combined, printed = aggregate('combined')
    report = read_json(combined / 'cross_hardware_metadata.json')
    assert report['baseline'] == str(a) and report['baseline_selection'] == 'first_input'
    assert len(report['sources']) == 2 and report['sources'][1]['metadata']['gpu_name'] == 'Synthetic GPU B'
    performance = rows(combined / 'cross_hardware_summary.csv')
    assert all(r['baseline_performance_ratio'] == '1.0' for r in performance)
    assert [r['within_run_naive_speedup'] for r in performance[:2]] == ['1.0', '2.0']
    numerical = rows(combined / 'cross_hardware_numerics.csv')
    assert all(r['status'] == 'bitwise_identical' and r['tolerance_failures'] == '0' for r in numerical)
    assert 'Synthetic GPU B' in printed.stdout
    aggregate('combined', code=2)  # refuse overwrite
    explicit, _ = aggregate('explicit', extra=('--baseline', b))
    assert read_json(explicit / 'cross_hardware_metadata.json')['baseline'] == str(b)
    aggregate('bad-baseline', extra=('--baseline', root / 'missing'), code=2)

    # Genuine differing bytes; the C++ comparator defines numerical semantics.
    different = synthetic('different', 'Synthetic GPU C')
    records = rows(different / 'summary.csv')
    selected = records[0]
    filename = selected['output_file']
    raw = bytearray((different / filename).read_bytes())
    bits = struct.unpack_from('<I', raw, 24)[0]
    struct.pack_into('<I', raw, 24, bits + 1)
    (different / filename).write_bytes(raw)
    selected['output_sha256'] = hashlib.sha256(raw[24:]).hexdigest()
    side = different / (filename + '.json'); context = read_json(side)
    context['output_sha256'] = selected['output_sha256']; write_json(side, context)
    write_rows(different / 'summary.csv', records)
    differing, _ = aggregate('differing', (a, different))
    numerical = rows(differing / 'cross_hardware_numerics.csv')[4]
    assert numerical['status'] == 'numerically_compared'
    assert numerical['divergent_count'] == '1' and numerical['max_ulp'] == '1'
    assert numerical['mean_divergent_ulp'] == '1' and numerical['divergent_percent'] == '6.25'
    assert numerical['tolerance_failures'] == '0'
    assert json.loads(numerical['first_divergence'])['row'] == 0

    # NaNs and signed zeros: identical hashes do not imply tolerance passes.
    special_ref = root / 'special-ref.bin'; special_actual = root / 'special-actual.bin'
    special_ref.write_bytes(b'MIFP32LE' + struct.pack('<QQ4I', 1,4,0,0x7fc00001,0x7f800000,0xbf800000))
    special_actual.write_bytes(b'MIFP32LE' + struct.pack('<QQ4I', 1,4,0x80000000,0x7fc00001,0xff800000,0xbf800001))
    special = json.loads(call([comparator, special_ref, special_actual, '1e-6', '1e-5']).stdout)
    assert special['divergent_count'] == 3 and special['tolerance_failures'] == 2
    assert special['max_ulp'] == 1 and special['mean_divergent_ulp'] == 0.5
    assert special['ulp_bins'] == [1,1,0,0,0,0]
    assert special['nan_pairs'] == special['infinity_pairs'] == 1

    incompatible = synthetic('incompatible')
    meta = read_json(incompatible / 'metadata.json'); meta['config']['seed'] = 43
    write_json(incompatible / 'metadata.json', meta)
    for side in incompatible.glob('*.bin.json'):
        context = read_json(side); context['seed'] = 43; write_json(side, context)
    incompatible_out, _ = aggregate('incompatible-report', (a, incompatible))
    assert all(r['status'] == 'incompatible' for r in rows(incompatible_out / 'cross_hardware_numerics.csv')[4:])
    assert all(r['baseline_performance_ratio'] == '' for r in rows(incompatible_out / 'cross_hardware_summary.csv')[4:])

    # A missing shape is a separate configuration, not a bitwise mismatch.
    subset = synthetic('subset')
    write_rows(subset / 'summary.csv', rows(subset / 'summary.csv')[:2])
    subset_out, _ = aggregate('subset-report', (subset, a))
    assert any(r['status'] == 'no_baseline_configuration' for r in rows(subset_out / 'cross_hardware_numerics.csv'))

    toolchain = synthetic('toolchain')
    meta = read_json(toolchain / 'metadata.json'); meta['cuda_compiler'] = 'Synthetic NVCC 2'
    write_json(toolchain / 'metadata.json', meta)
    toolchain_out, _ = aggregate('toolchain-report', (a, toolchain))
    assert all(r['compatibility'] == 'cross_toolchain' for r in rows(toolchain_out / 'cross_hardware_summary.csv')[4:])
    dirty = synthetic('dirty')
    meta = read_json(dirty / 'metadata.json'); meta['git_dirty'] = True
    write_json(dirty / 'metadata.json', meta)
    dirty_out, _ = aggregate('dirty-report', (a, dirty))
    assert all(r['compatibility'] == 'uncontrolled_source' and r['baseline_performance_ratio'] == ''
               for r in rows(dirty_out / 'cross_hardware_summary.csv')[4:])

    # Changing timing controls suppresses ratios; changing thresholds uses baseline
    # tolerances explicitly for detailed diagnostics, while hashes remain comparable.
    settings = synthetic('settings')
    meta = read_json(settings / 'metadata.json')
    meta['config'].update(iterations=7, atol=0.0)
    write_json(settings / 'metadata.json', meta)
    settings_out, _ = aggregate('settings-report', (a, settings))
    assert all(r['baseline_performance_ratio'] == '' for r in rows(settings_out / 'cross_hardware_summary.csv')[4:])
    assert all('baseline_tolerances' in r['warnings'] for r in rows(settings_out / 'cross_hardware_numerics.csv')[4:])

    # Legacy schema directories remain readable with numerical hashes unavailable.
    for version in (1, 2):
        legacy = synthetic('legacy-' + str(version))
        meta = read_json(legacy / 'metadata.json'); meta['schema_version'] = version
        write_json(legacy / 'metadata.json', meta)
        records = rows(legacy / 'summary.csv')
        for r in records:
            for field in ('output_sha256', 'reference_sha256', 'output_file', 'reference_output_file'):
                del r[field]
        write_rows(legacy / 'summary.csv', records)
        legacy_out, _ = aggregate('legacy-report-' + str(version), (a, legacy))
        assert all(r['status'] == 'hash_unavailable_performance_only' for r in rows(legacy_out / 'cross_hardware_numerics.csv')[4:])

    # Missing hashes remain performance-only, and missing binaries do not invent metrics.
    old = synthetic('old')
    records = rows(old / 'summary.csv')
    for r in records: r['output_sha256'] = ''; r['output_file'] = ''
    write_rows(old / 'summary.csv', records)
    old_out, _ = aggregate('old-report', (a, old))
    assert all(r['status'] == 'hash_unavailable_performance_only' for r in rows(old_out / 'cross_hardware_numerics.csv')[4:])
    records = rows(different / 'summary.csv')
    for r in records: r['output_file'] = ''
    write_rows(different / 'summary.csv', records)
    missing_out, _ = aggregate('missing-report', (a, different))
    r = rows(missing_out / 'cross_hardware_numerics.csv')[4]
    assert r['status'] == 'hash_differs_details_unavailable' and r['max_ulp'] == ''

    # Corruption and incompatible binary metadata fail explicitly.
    corrupt = synthetic('corrupt')
    record = rows(corrupt / 'summary.csv')[0]
    side = corrupt / (record['output_file'] + '.json'); context = read_json(side)
    context['dtype'] = 'float64'; write_json(side, context)
    aggregate('corrupt-report', (a, corrupt), code=2)
    context['dtype'] = 'float32'; write_json(side, context)
    path = corrupt / record['output_file']; path.write_bytes(path.read_bytes()[:-1])
    aggregate('truncated-report', (a, corrupt), code=2)

print('Output hashing, binary roundtrip, cross-hardware aggregation and replay tests passed.')
