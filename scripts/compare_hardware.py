#!/usr/bin/env python3
"""Aggregate collected Matmul-Inspector runs; no remote execution or GPU required."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess

ENVIRONMENT = ('gpu_name', 'gpu_compute_capability', 'cpu_model', 'cuda_runtime_version',
               'cuda_driver_api_version', 'nvidia_driver_version', 'cxx_compiler', 'cuda_compiler',
               'cuda_architectures', 'cxx_flags', 'cuda_flags', 'fast_math', 'os')
TOOLCHAIN = ('cuda_runtime_version', 'cuda_driver_api_version', 'nvidia_driver_version',
             'cxx_compiler', 'cuda_compiler', 'cuda_architectures', 'cxx_flags', 'cuda_flags', 'fast_math')
INPUT_FIELDS = ('seed', 'seed_b', 'dtype', 'generator', 'input')
METRICS = ('divergent_count', 'divergent_percent', 'max_ulp', 'mean_divergent_ulp',
           'max_absolute_error', 'max_relative_error', 'tolerance_failures',
           'tolerance_failure_percent', 'nan_pairs', 'infinity_pairs', 'zero_reference_nonzero',
           'ulp_bins', 'first_divergence', 'first_tolerance_failure')


def load_json(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Invalid JSON number: ' + x)))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def finite_number(value):
    if value in ('', None):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError('Invalid nonnegative timing/config value: ' + str(value))
    return result


def key(row):
    return (row['kernel'], int(row['M']), int(row['N']), int(row['K']))


def semantics(row):
    kernel = row['kernel']
    contraction = 'separate_rn_mul_add' if kernel == 'cuda-naive-no-fma' else (
        'explicit_fma_rn' if kernel in ('cuda-naive-fma', 'cuda-naive-reordered') else 'compiler_default')
    accumulation = 'even_odd_partials' if kernel == 'cuda-naive-reordered' else (
        'increasing_k_zero_padded_tiles' if kernel == 'tiled' else 'increasing_k')
    return (int(row['tile_size']), row.get('contraction_mode', contraction), row.get('accumulation_mode', accumulation))


def input_config(config):
    return {field: config.get(field, 'random' if field == 'input' else 'unknown') for field in INPUT_FIELDS}


def clean_source(meta):
    commit = meta.get('git_commit', 'unknown')
    runtime = meta.get('runtime_git_commit', commit)
    return (meta.get('git_dirty') is False and commit not in ('unknown', '')
            and runtime == commit and meta.get('build_git_dirty', False) is False
            and meta.get('runtime_git_dirty', False) is False)


def hardware(meta):
    gpu = meta.get('gpu_name', 'unknown')
    cc = meta.get('gpu_compute_capability', 'unknown')
    cpu = meta.get('cpu_model', 'unknown')
    return f'{gpu} (CC {cc}); CPU {cpu}'


def read_run(path):
    path = path.resolve()
    metadata = load_json(path / 'metadata.json')
    if metadata.get('schema_version') not in (1, 2, 3):
        raise ValueError(f'{path}: unsupported result schema')
    if metadata.get('status') not in ('complete', 'tolerance_failed'):
        raise ValueError(f'{path}: incomplete/failed/skipped run cannot be aggregated')
    config = metadata['config']
    for name in ('seed', 'seed_b', 'dtype', 'generator', 'mode', 'atol', 'rtol', 'shapes', 'warmups', 'iterations', 'reference', 'candidate'):
        if name not in config:
            raise ValueError(f'{path}: missing config {name}')
    if any(finite_number(config[field]) is None for field in ('atol', 'rtol')):
        raise ValueError(f'{path}: missing numerical tolerances')
    if metadata['schema_version'] == 3 and (config.get('output_encoding') != 'fp32-le-row-major-v1' or config.get('output_hash') != 'sha256'):
        raise ValueError(f'{path}: unsupported output hash encoding')
    with (path / 'summary.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'{path}: empty summary')
    indexed = {}
    shapes = {(s['M'], s['N'], s['K']) for s in config['shapes']}
    for row in rows:
        if None in row or None in row.values():
            raise ValueError(f'{path}: malformed summary CSV')
        identity = key(row)
        if row['kernel'] not in (config['reference'], config['candidate']):
            raise ValueError(f'{path}: summary kernel not declared in metadata')
        if identity in indexed:
            raise ValueError(f'{path}: ambiguous duplicate kernel/shape {identity}')
        if identity[1:] not in shapes:
            raise ValueError(f'{path}: summary shape not declared in metadata')
        for field in ('mean_ms', 'median_ms', 'min_ms', 'stddev_ms', 'gflops'):
            finite_number(row[field])
        for field in ('output_sha256', 'reference_sha256'):
            if row.get(field) and not re.fullmatch('[0-9a-f]{64}', row[field]):
                raise ValueError(f'{path}: invalid SHA-256')
        indexed[identity] = row
    return dict(path=path, metadata=metadata, config=config, rows=rows, indexed=indexed,
                hardware=hardware(metadata), metadata_sha256=digest(path / 'metadata.json'),
                summary_sha256=digest(path / 'summary.csv'), binaries={})


def binary(run, row):
    """Validate header, exact length, sidecar context and payload digest before use."""
    filename = row.get('output_file', '')
    if not filename:
        return None
    path = (run['path'] / filename).resolve()
    if run['path'] not in path.parents:
        raise ValueError('Output path escapes source run directory')
    sidecar_path = Path(str(path) + '.json')
    if run['path'] not in sidecar_path.resolve().parents:
        raise ValueError('Sidecar path escapes source run directory')
    context = load_json(sidecar_path)
    shape = {field: int(row[field]) for field in ('M', 'N', 'K')}
    expected = dict(format='mifp32le-v1', dtype='float32', byte_order='little',
                    rows=shape['M'], cols=shape['N'], kernel=row['kernel'], **shape,
                    seed=run['config']['seed'], seed_b=run['config']['seed_b'],
                    generator=run['config']['generator'], contraction_mode=semantics(row)[1],
                    accumulation_mode=semantics(row)[2], output_sha256=row.get('output_sha256'))
    if run['config']['dtype'] != 'float32' or any(context.get(k) != v for k, v in expected.items()):
        raise ValueError(f'{path}: binary sidecar incompatible with summary/configuration')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        header = stream.read(24)
        if len(header) != 24 or header[:8] != b'MIFP32LE':
            raise ValueError(f'{path}: invalid binary header/dtype/byte order')
        rows, cols = struct.unpack('<QQ', header[8:])
        if (rows, cols) != (shape['M'], shape['N']) or path.stat().st_size != 24 + rows * cols * 4:
            raise ValueError(f'{path}: binary dimensions/length disagree')
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != expected['output_sha256']:
        raise ValueError(f'{path}: output SHA-256 mismatch (corrupt/stale artifact)')
    run['binaries'][filename] = path
    return path


def compatibility(run, baseline, row, ref):
    warnings = []
    numerical = True
    a, b = input_config(run['config']), input_config(baseline['config'])
    for field in INPUT_FIELDS:
        if a[field] != b[field] or a[field] in (None, '', 'unknown'):
            warnings.append('incompatible_' + field)
            numerical = False
    if semantics(row) != semantics(ref):
        warnings.append('incompatible_kernel_semantics')
        numerical = False
    ma, mb = run['metadata'], baseline['metadata']
    source_ok = clean_source(ma) and clean_source(mb) and ma.get('git_commit') == mb.get('git_commit')
    if not source_ok:
        warnings.append('uncontrolled_source: commit differs, dirty, stale or unknown')
    differences = [field for field in TOOLCHAIN if ma.get(field, 'unknown') != mb.get(field, 'unknown')]
    if differences:
        warnings.append('cross_toolchain: ' + ','.join(differences))
    performance = numerical and source_ok
    for field in ('build_type', 'timing_methodology'):
        if ma.get(field, 'unknown') != mb.get(field, 'unknown') or ma.get(field, 'unknown') in (None, '', 'unknown'):
            warnings.append('different_or_unknown_' + field)
            performance = False
    for field in ('mode', 'warmups', 'iterations', 'atol', 'rtol', 'reference', 'candidate'):
        if run['config'][field] != baseline['config'][field]:
            warnings.append('different_' + field)
            performance = False
    if run['config']['atol'] != baseline['config']['atol'] or run['config']['rtol'] != baseline['config']['rtol']:
        warnings.append('detailed_comparison_uses_baseline_tolerances')
    classification = ('incompatible_inputs_or_semantics' if not numerical else
                      'uncontrolled_source' if not source_ok else
                      'configuration_differs' if not performance else
                      'cross_toolchain' if differences else 'controlled')
    return numerical, performance, classification, warnings


def detailed(comparator, reference, candidate, config):
    result = subprocess.run([str(comparator), str(reference), str(candidate), str(config['atol']), str(config['rtol'])],
                            capture_output=True, text=True, check=True)
    metrics = json.loads(result.stdout)
    if metrics.get('diagnostics_version') != 1 or any(field not in metrics for field in METRICS):
        raise ValueError('Unsupported/incomplete C++ comparison response')
    return {field: metrics[field] for field in METRICS}


def aggregate(runs, baseline, comparator=None):
    summaries, numerics = [], []
    for run in runs:
        for row in run['rows']:
            # Validate any saved output even when no detailed comparison is requested.
            saved = binary(run, row)
            ref = baseline['indexed'].get(key(row))
            info = dict(source_directory=str(run['path']), baseline_directory=str(baseline['path']),
                        hardware_label=run['hardware'], **{k: run['metadata'].get(k, 'unknown') for k in ENVIRONMENT},
                        git_commit=run['metadata'].get('git_commit', 'unknown'), git_dirty=run['metadata'].get('git_dirty', 'unknown'),
                        kernel=row['kernel'], M=int(row['M']), N=int(row['N']), K=int(row['K']),
                        seed=run['config']['seed'], seed_b=run['config']['seed_b'], tile_size=row['tile_size'],
                        contraction_mode=semantics(row)[1], accumulation_mode=semantics(row)[2],
                        output_sha256=row.get('output_sha256', ''),
                        **{k: row[k] for k in ('mean_ms', 'median_ms', 'min_ms', 'stddev_ms', 'gflops')})
            mean = finite_number(row['mean_ms'])
            naive = run['indexed'].get(('naive', *key(row)[1:]))
            naive_mean = finite_number(naive['mean_ms']) if naive else None
            info['within_run_naive_speedup'] = naive_mean / mean if naive_mean and mean else None
            info['baseline_performance_ratio'] = None
            numerical = dict(source_directory=str(run['path']), baseline_directory=str(baseline['path']),
                             hardware_label=run['hardware'], kernel=row['kernel'], M=info['M'], N=info['N'], K=info['K'],
                             output_sha256=row.get('output_sha256', ''), baseline_sha256=ref.get('output_sha256', '') if ref else '',
                             atol=baseline['config']['atol'], rtol=baseline['config']['rtol'],
                             **{field: None for field in METRICS})
            if ref is None:
                classification, warnings = 'separate_configuration', ['no baseline row for this kernel/shape']
                numerical['status'] = 'no_baseline_configuration'
            else:
                eligible, performance, classification, warnings = compatibility(run, baseline, row, ref)
                baseline_mean = finite_number(ref['mean_ms'])
                if performance and run['config']['mode'] == 'benchmark' and baseline_mean and mean:
                    info['baseline_performance_ratio'] = baseline_mean / mean
                ref_saved = binary(baseline, ref)
                if not eligible:
                    numerical['status'] = 'incompatible'
                elif not row.get('output_sha256') or not ref.get('output_sha256'):
                    numerical['status'] = 'hash_unavailable_performance_only'
                else:
                    same = row['output_sha256'] == ref['output_sha256']
                    numerical['status'] = 'bitwise_identical' if same else 'hash_differs_details_unavailable'
                    if same:
                        numerical['divergent_count'] = 0
                        numerical['divergent_percent'] = 0
                    if comparator and saved and ref_saved:
                        numerical.update(detailed(comparator, ref_saved, saved, baseline['config']))
                        if (numerical['divergent_count'] == 0) != same:
                            raise ValueError('C++ comparison disagrees with output fingerprints')
                        numerical['status'] = 'bitwise_identical' if same else 'numerically_compared'
                    elif not same:
                        warnings.append('save both outputs and supply --comparator for element-wise diagnostics')
            if run['path'] == baseline['path']:
                warnings.append('selected_baseline')
            info.update(compatibility=classification, warnings=warnings)
            numerical.update(compatibility=classification, warnings=warnings)
            summaries.append(info); numerics.append(numerical)
    return summaries, numerics


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, allow_nan=False) if isinstance(v, (dict, list, bool)) else v
                             for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--baseline', type=Path, help='Must name one of the supplied runs; default is the first')
    parser.add_argument('--output', type=Path, default=Path('cross-hardware-results'), help='New aggregate directory; never overwritten')
    parser.add_argument('--comparator', type=Path, help='CPU-only matmul-compare-outputs executable; otherwise searched on PATH')
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError('Aggregate output directory already exists: ' + str(args.output))
        paths = [path.resolve() for path in args.runs]
        if len(paths) != len(set(paths)):
            raise ValueError('Duplicate source directories; supply each run once')
        baseline_path = args.baseline.resolve() if args.baseline else paths[0]
        if baseline_path not in paths:
            raise ValueError('--baseline must be one of the supplied run directories')
        runs = [read_run(path) for path in paths]
        baseline = next(run for run in runs if run['path'] == baseline_path)
        comparator = args.comparator.resolve() if args.comparator else shutil.which('matmul-compare-outputs')
        if comparator and not Path(comparator).is_file():
            raise ValueError('Comparison executable does not exist')
        summaries, numerics = aggregate(runs, baseline, comparator)
        print('Cross-Hardware Comparison\nBaseline: ' + baseline['hardware'])
        print(f"  directory: {baseline_path}\n  commit: {baseline['metadata'].get('git_commit', 'unknown')}\n  seed: {baseline['config']['seed']}")
        print('Ratios are observations for matched configurations, not hardware rankings.')
        def rate(value):
            return f'{value:.6g}' if value is not None else 'n/a'
        for info, numeric in zip(summaries, numerics):
            print(f"{info['hardware_label']} | {info['kernel']} {info['M']}x{info['N']} K={info['K']} | "
                  f"{info['output_sha256'][:12] or 'no hash'} | {numeric['status']} | "
                  f"mean={info['mean_ms'] or 'untimed'} ms GFLOP/s={info['gflops'] or 'n/a'} | "
                  f"naive speedup={rate(info['within_run_naive_speedup'])} baseline ratio={rate(info['baseline_performance_ratio'])}")
            if numeric['max_ulp'] is not None:
                print(f"  divergent={numeric['divergent_percent']}% max ULP={numeric['max_ulp']} "
                      f"mean ULP={numeric['mean_divergent_ulp']} max abs={numeric['max_absolute_error']} "
                      f"max rel={numeric['max_relative_error']} tolerance failures={numeric['tolerance_failures']} "
                      f"({numeric['tolerance_failure_percent']}%) first={numeric['first_divergence']}")
            if info['warnings']:
                print('  ' + '; '.join(info['warnings']))
        args.output.mkdir(parents=True, exist_ok=False)
        write_csv(args.output / 'cross_hardware_summary.csv', summaries)
        write_csv(args.output / 'cross_hardware_numerics.csv', numerics)
        metadata = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        baseline=str(baseline_path), baseline_selection='explicit' if args.baseline else 'first_input',
                        aggregator_sha256=digest(Path(__file__)), comparator=str(comparator) if comparator else None,
                        comparator_sha256=digest(Path(comparator)) if comparator else None,
                        numerical_tolerances='baseline config', ratio_definition='baseline mean / candidate mean; only eligible configurations',
                        sources=[dict(directory=str(run['path']), hardware_label=run['hardware'], metadata=run['metadata'],
                                      metadata_sha256=run['metadata_sha256'], summary_sha256=run['summary_sha256']) for run in runs])
        (args.output / 'cross_hardware_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(2, f'Aggregation failed: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
