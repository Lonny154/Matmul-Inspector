#!/usr/bin/env python3
"""Capture one configuration in fresh native processes; reuse native artifacts and diagnostics."""
import argparse
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import uuid

import compare_hardware as hardware
from operations import KINDS, MATMUL_KERNELS, SUMMATION_METHODS, VECTOR_FIXTURES
from reproduce import command as replay_command

MANIFEST = 'process_reproducibility_metadata.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def classification(metrics):
    if not metrics:
        raise ValueError('At least two successful processes are required')
    if all(m['divergent_count'] == 0 for m in metrics):
        return 'bitwise_stable'
    return ('numerically_stable_but_not_bitwise' if all(m['tolerance_failures'] == 0 for m in metrics)
            else 'varying')


def identity(run):
    if len(run['rows']) != 1 or run['config']['mode'] != 'compare':
        raise ValueError('Process captures require one compare configuration per launch')
    row, config = run['rows'][0], run['config']
    fields = ('operation', 'dtype', 'layout', 'seed', 'seed_b', 'generator', 'input',
              'output_encoding', 'output_hash', 'atol', 'rtol', 'shapes')
    result = {field: config[field] for field in fields}
    result.update(kernel=row['kernel'], semantics=hardware.semantics(row),
                  block_size=row.get('reduction_block_size', ''), stages=row.get('reduction_stages', ''))
    # JSON normalization makes the identity portable (tuples become lists).
    return json.loads(json.dumps(result))


def require_compatible(run, baseline):
    left, right = identity(run), identity(baseline)
    differences = [field for field in left if left[field] != right[field]]
    if differences:
        raise ValueError('Incompatible captures: '+', '.join(differences))


def output_record(run):
    row = run['rows'][0]
    path = hardware.binary(run, row)
    if path is None:
        raise ValueError('Process captures require saved output matrices')
    result = dict(output_sha256=row['output_sha256'], output_value='', output_bits='')
    if int(row['M']) == int(row['N']) == 1:
        payload = path.read_bytes()[24:]
        result.update(output_value=str(struct.unpack('<f', payload)[0]),
                      output_bits=f'{struct.unpack("<I", payload)[0]:032b}')
    return path, result


def compare(comparator, baseline, candidate):
    require_compatible(candidate, baseline)
    ref, a = output_record(baseline)
    actual, b = output_record(candidate)
    metrics = hardware.detailed(comparator, ref, actual, baseline['config'])
    same = a['output_sha256'] == b['output_sha256']
    if same != (metrics['divergent_count'] == 0):
        raise ValueError('Output hash disagrees with native comparison')
    return dict(bitwise_equal=same, numerical_agreement=metrics['tolerance_failures'] == 0, **metrics)


def contained(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError('Capture path escapes directory')
    return path


def read_capture(path):
    root = path.resolve()
    meta = hardware.load_json(root / MANIFEST)
    if (meta.get('schema_version') != 1 or meta.get('artifact_type') != 'independent-process-capture'
            or meta.get('status') != 'complete'):
        raise ValueError('Unsupported or incomplete process capture')
    launches = meta['launches']
    if len(launches) < 2 or len(launches) != meta['process_repeats']:
        raise ValueError('Invalid process repeat count')
    runs, ids, directories = [], set(), set()
    for index, launch in enumerate(launches):
        child = contained(root, launch['directory'])
        if launch['repeat_index'] != index or launch['run_id'] in ids or child in directories:
            raise ValueError('Duplicate/invalid process identity')
        ids.add(launch['run_id']); directories.add(child)
        if launch['executable_sha256'] != meta['executable_sha256']:
            raise ValueError('Executable changed within capture')
        for filename, expected in launch['artifact_sha256'].items():
            if hardware.digest(contained(child, filename)) != expected:
                raise ValueError('Capture artifact digest mismatch: '+filename)
        if not {'metadata.json', 'summary.csv'}.issubset(launch['artifact_sha256']):
            raise ValueError('Missing native artifact digests')
        run = hardware.read_run(child)
        if identity(run) != meta['configuration']:
            raise ValueError('Capture configuration disagrees with native artifacts')
        output_record(run)
        runs.append(run)
    for filename, expected in meta['artifact_sha256'].items():
        if hardware.digest(contained(root, filename)) != expected:
            raise ValueError('Capture report digest mismatch: '+filename)
    return dict(path=root, metadata=meta, runs=runs)


def capture(args):
    exe = args.executable.resolve()
    comparator = (args.comparator or exe.with_name('matmul-compare-outputs')).resolve()
    if not exe.is_file() or not comparator.is_file():
        raise ValueError('Build the native executable and comparator first')
    if args.process_repeats < 2:
        raise ValueError('At least two process repeats are required')
    if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
        raise ValueError('Tolerances must be finite and nonnegative')
    if not 0 <= args.seed <= 0xffffffff or (args.seed_b is not None and not 0 <= args.seed_b <= 0xffffffff):
        raise ValueError('Seeds must be uint32')
    method = args.method or ('cpu' if args.operation == 'matmul' else 'fp32_forward')
    fixture = args.fixture or ('random' if args.operation == 'matmul' else 'random_uniform')
    allowed = MATMUL_KERNELS if args.operation == 'matmul' else {'cpu', 'cpu-reverse', 'cuda-tree', *SUMMATION_METHODS}
    fixtures = ('random', 'cancellation', 'fma-sensitive') if args.operation == 'matmul' else VECTOR_FIXTURES
    if method not in allowed or fixture not in fixtures:
        raise ValueError('Method or fixture is incompatible with operation')
    dimensions = (args.m, args.n, args.k)
    rectangular = any(v is not None for v in dimensions)
    if rectangular and (args.operation != 'matmul' or args.size is not None or any(v is None or v <= 0 for v in dimensions)):
        raise ValueError('Supply positive --m --n --k together for matmul, without --size')
    size = args.size if args.size is not None else (4 if args.operation == 'matmul' else 257)
    if size <= 0:
        raise ValueError('Size must be positive')
    if args.output.exists():
        raise ValueError('Output directory already exists')
    base = [str(exe), 'compare', '--operation', args.operation, '--reference', 'cpu', '--candidate', method,
            '--input', fixture, '--seed', str(args.seed), '--seed-b', str(args.seed_b if args.seed_b is not None else (args.seed+81)&0xffffffff),
            '--atol', str(args.atol), '--rtol', str(args.rtol), '--save-output']
    if rectangular:
        for flag, value in zip(('--m', '--n', '--k'), dimensions):
            base += [flag, str(value)]
    else:
        base += ['--sizes', str(size)]
    if args.operation != 'matmul':
        base += ['--block-size', str(args.block_size), '--repeats', '1']
    args.output.mkdir(parents=True)
    executable_hash = hardware.digest(exe)
    meta = dict(schema_version=1, artifact_type='independent-process-capture', status='incomplete',
                capture_id=str(uuid.uuid4()), timestamp_utc=now(), process_repeats=args.process_repeats,
                executable_sha256=executable_hash, comparator_sha256=hardware.digest(comparator),
                script_sha256=hardware.digest(Path(__file__)),
                helper_sha256={name:hardware.digest(Path(__file__).with_name(name)) for name in
                               ('compare_hardware.py', 'reproduce.py', 'operations.py')}, launches=[])
    write_json(args.output / MANIFEST, meta)
    runs, rows, pairs = [], [], []
    for index in range(args.process_repeats):
        child = args.output / 'runs' / f'process-{index:04d}'
        invocation = replay_command(runs[0]['metadata'], exe, child) if runs else base+['--output', str(child)]
        started = now()
        with subprocess.Popen(invocation, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            stdout, stderr = process.communicate()
            pid, code = process.pid, process.returncode
        launch = dict(repeat_index=index, run_id=str(uuid.uuid4()), pid=pid, timestamp_utc=started,
                      finished_utc=now(), directory=str(child.relative_to(args.output)), command=invocation,
                      exit_code=code, executable_sha256=hardware.digest(exe))
        if launch['executable_sha256'] != executable_hash:
            raise ValueError('Executable changed during capture')
        if code == 77:
            meta.update(status='skipped', skipped_launch=launch, reason=(stderr or stdout)[-2000:])
            write_json(args.output / MANIFEST, meta)
            print('Capture skipped: CUDA hardware/runtime unavailable')
            return 77
        if code not in (0, 1):
            raise ValueError('Native process failed: '+(stderr or stdout)[-1500:])
        run = hardware.read_run(child)
        if runs:
            require_compatible(run, runs[0])
        _, value = output_record(run)
        launch['artifact_sha256'] = {p.name:hardware.digest(p) for p in child.iterdir() if p.is_file()}
        meta['launches'].append(launch)
        runs.append(run)
        metrics = compare(comparator, runs[0], run)
        rows.append(dict(repeat_index=index, run_id=launch['run_id'], pid=pid, timestamp_utc=started,
                         **identity(run), **value, **metrics))
    for a, b in itertools.combinations(range(len(runs)), 2):
        pairs.append(dict(reference_repeat=a, candidate_repeat=b, **compare(comparator, runs[a], runs[b])))
    state = classification(pairs)
    hardware.write_csv(args.output / 'process_reproducibility.csv', rows)
    hardware.write_csv(args.output / 'process_pairs.csv', pairs)
    dirty = any(not hardware.clean_source(r['metadata']) for r in runs)
    text = ['# Fresh-process reproducibility', '', f'Classification: **{state}**; {len(runs)} independent launches.',
            f"Operation: {args.operation}; method: {method}; fixture: {fixture}; seed: {args.seed}.",
            f"Hardware: {runs[0]['hardware']}.", '',
            f"All-pair numerical agreement: {all(p['numerical_agreement'] for p in pairs)}; largest ULP: {max(p['max_ulp'] for p in pairs)}.",
            f"Largest absolute difference: {max(float(p['max_absolute_error']) for p in pairs)}.", '',
            'Per-launch CSV compares to launch zero; process_pairs.csv contains every earlier/later launch pair.',
            'Bitwise stability is separate from numerical tolerance (identical NaNs can fail tolerance).',
            'This tests fresh processes on this machine, not cross-hardware agreement or mathematical accuracy.',
            'FP64 scalar diagnostics in child runs remain approximate analysis references.',
            'Each runs/ directory is a normal replayable native capture; copy the entire capture for portability.']
    if dirty:
        text += ['', '**Uncontrolled source:** dirty, stale or unknown source; a commit alone cannot recreate this build.']
        print('WARNING: dirty/stale/unknown source; provenance is recorded.', file=sys.stderr)
    (args.output / 'report.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    meta.update(status='complete', configuration=identity(runs[0]), classification=state,
                comparison_policy='All earlier/later process pairs; native directional tolerance; per-launch CSV vs repeat zero',
                provenance=[r['metadata'] for r in runs],
                artifact_sha256={name:hardware.digest(args.output/name) for name in
                                 ('process_reproducibility.csv', 'process_pairs.csv', 'report.md')})
    write_json(args.output / MANIFEST, meta)
    print(f'{state}: {len(runs)} fresh processes; report: {args.output / "report.md"}')
    return 0


def compare_captures(paths, baseline_path, output, comparator):
    """Strict captured-configuration matching; native diagnostics for every launch."""
    if output.exists():
        raise FileExistsError('Aggregate output directory already exists')
    if comparator is None:
        raise ValueError('Process-capture comparison requires --comparator')
    if len(paths) < 2:
        raise ValueError('Supply at least two process captures')
    captures = [read_capture(p) for p in paths]
    baseline = next(c for c in captures if c['path'] == baseline_path)
    reference = baseline['runs'][0]
    # Validate everything before writing the output directory.
    for capture_ in captures:
        for run in capture_['runs']:
            require_compatible(run, reference)
    rows = []
    for capture_ in captures:
        for index, run in enumerate(capture_['runs']):
            _, values = output_record(run)
            _, _, status, warnings = hardware.compatibility(run, reference, run['rows'][0], reference['rows'][0])
            if capture_['metadata']['executable_sha256'] != baseline['metadata']['executable_sha256']:
                warnings.append('different_executable_sha256: build identity differs')
            differences = [field for field in (*hardware.ENVIRONMENT, 'build_type', 'git_commit', 'git_dirty')
                           if run['metadata'].get(field) != reference['metadata'].get(field)]
            rows.append(dict(capture_id=capture_['metadata']['capture_id'], source_directory=str(capture_['path']),
                repeat_index=index, run_id=capture_['metadata']['launches'][index]['run_id'],
                hardware_label=run['hardware'], baseline_hardware=reference['hardware'],
                process_classification=capture_['metadata']['classification'], **identity(run), **values,
                baseline_sha256=reference['rows'][0]['output_sha256'],
                executable_sha256=capture_['metadata']['executable_sha256'],
                baseline_executable_sha256=baseline['metadata']['executable_sha256'],
                compatibility=status, environment_differences=differences, warnings=warnings,
                **compare(comparator, reference, run)))
    output.mkdir(parents=True, exist_ok=False)
    hardware.write_csv(output / 'cross_hardware_comparison.csv', rows)
    lines = ['# Captured-process comparison', '', f"Baseline: {reference['hardware']}; launch zero in `{baseline_path}`.", '',
             'Same configuration across captures; every launch is compared with baseline launch zero.',
             'Matching outputs do not prove mathematical accuracy; environment differences do not establish causality.', '',
             '| Capture | Hardware | Process classification | Bitwise matches | Tolerance passes | Max ULP | Max absolute difference |',
             '|---|---|---|---:|---:|---:|---:|']
    environment_notes = []
    for capture_ in captures:
        group = [r for r in rows if r['source_directory'] == str(capture_['path'])]
        lines.append(f"| {capture_['metadata']['capture_id']} | {group[0]['hardware_label']} | {group[0]['process_classification']} | "
                     f"{sum(r['bitwise_equal'] for r in group)}/{len(group)} | {sum(r['numerical_agreement'] for r in group)}/{len(group)} | "
                     f"{max(r['max_ulp'] for r in group)} | {max(float(r['max_absolute_error']) for r in group)} |")
        environment_notes += ['', f"Capture {capture_['metadata']['capture_id']}:", 'Environment differences: '+(', '.join(group[0]['environment_differences']) or 'none recorded')+'.',
                  'Provenance flags: '+('; '.join(group[0]['warnings']) or 'none')+'.', '']
    lines += environment_notes
    lines += ['Full hashes, relative differences, first divergence, and native ULP diagnostics are in cross_hardware_comparison.csv.',
              'Original hardware/toolchain metadata and build hashes are retained in cross_hardware_metadata.json.']
    (output / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(output / 'cross_hardware_metadata.json', dict(schema_version=1, timestamp_utc=now(),
        baseline_capture_id=baseline['metadata']['capture_id'], baseline_repeat=0,
        comparator_sha256=hardware.digest(Path(comparator)), script_sha256=hardware.digest(Path(__file__)),
        sources=[dict(directory=str(c['path']), manifest_sha256=hardware.digest(c['path']/MANIFEST),
                      metadata=c['metadata']) for c in captures]))
    print(f"Capture comparison: {sum(r['bitwise_equal'] for r in rows)}/{len(rows)} outputs match baseline; {output / 'report.md'}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', choices=sorted(KINDS), default='reduction_sum')
    parser.add_argument('--size', type=int, help='Vector length or square matmul size')
    for name in ('m', 'n', 'k'):
        parser.add_argument('--'+name, type=int)
    parser.add_argument('--fixture')
    parser.add_argument('--method', help='Existing kernel/method name; defaults to cpu for matmul, fp32_forward for vectors')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed-b', type=int)
    parser.add_argument('--block-size', type=int, choices=(64,128,256,512), default=256)
    parser.add_argument('--process-repeats', type=int, default=3)
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--executable', type=Path, default=Path('build/matmul-inspector'))
    parser.add_argument('--comparator', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return capture(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print('Process capture failed: '+str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
