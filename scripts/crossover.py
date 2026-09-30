#!/usr/bin/env python3
"""Run a CPU/GPU crossover sweep, optionally render it; analyze recorded medians."""
import argparse
import math
from pathlib import Path
import subprocess
import sys


MODES = ('kernel_only', 'end_to_end')


def analyze(rows):
    """Group square configurations without pooling modes or interpolating sizes.

    Invalid/zero/missing latencies remain unavailable, never infinite speedups.
    A later loss is preserved even after an earlier GPU win.
    """
    cpu = {r['shape']: r.get('median_ms') for r in rows
           if r['kernel'] == 'cpu' and r.get('timing_mode') == 'host_matmul'}
    groups = {}
    for row in rows:
        shape = row['shape']
        if (row['kernel'] == 'cpu' or row.get('timing_mode') not in MODES
                or len(set(shape)) != 1):
            continue
        cpu_ms, gpu_ms = cpu.get(shape), row.get('median_ms')
        valid = all(value is not None and math.isfinite(value) and value > 0 for value in (cpu_ms, gpu_ms))
        point = dict(size=shape[0], cpu_ms=cpu_ms, gpu_ms=gpu_ms,
                     speedup=cpu_ms / gpu_ms if valid else None,
                     status=('gpu_faster' if gpu_ms < cpu_ms else 'cpu_faster_or_equal') if valid else 'unavailable',
                     tolerance_pass=row.get('tolerance_pass'))
        groups.setdefault((row['kernel'], row['timing_mode']), []).append(point)
    for points in groups.values():
        points.sort(key=lambda p: p['size'])
        first = next((p for p in points if p['status'] == 'gpu_faster'), None)
        if first:
            first['status'] = 'first_sampled_gpu_win'
    return groups


def render(rows, destination, warnings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = analyze(rows)
    figures = []
    kernels = sorted({key[0] for key in groups})
    # Two curves per kernel; separate figures prevent a many-kernel legend pileup.
    for index, kernel in enumerate(kernels):
        for metric in ('latency', 'speedup'):
            fig, ax = plt.subplots(figsize=(8, 4.8))
            series = 0
            cpu_points = {}
            for mode in MODES:
                points = groups.get((kernel, mode), [])
                for point in points:
                    if point['cpu_ms'] is not None and point['cpu_ms'] > 0:
                        cpu_points[point['size']] = point['cpu_ms']
                points = [p for p in points if p['speedup'] is not None]
                if points:
                    ax.plot([p['size'] for p in points],
                            [p['gpu_ms'] if metric == 'latency' else p['speedup'] for p in points],
                            marker='o', label=f'{kernel}: {mode}')
                    series += 1
            if metric == 'latency' and cpu_points:
                sizes = sorted(cpu_points)
                ax.plot(sizes, [cpu_points[n] for n in sizes], marker='s', color='black',
                        linestyle='--', label='CPU: host_matmul')
                series += 1
            if not series:
                plt.close(fig)
                warnings.append(f'Skipped crossover {kernel} {metric}: no positive paired medians.')
                continue
            if metric == 'speedup':
                ax.axhline(1, linestyle='--', color='black', label='Parity (1×)')
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set(xlabel='Square matrix dimension N (M=N=K)',
                   ylabel='Median latency (ms)' if metric == 'latency' else 'CPU median / GPU median (×)',
                   title=f'CPU/GPU crossover: {kernel} — {metric}')
            ax.grid(which='both', alpha=0.2)
            ax.legend(fontsize='small')
            fig.tight_layout()
            filename = f'crossover_{index}_{metric}.png'
            fig.savefig(destination / filename, dpi=160)
            plt.close(fig)
            figures.append((filename, f'{kernel}: {metric}',
                            'Measured medians on logarithmic axes. Lines connect tested sizes only; '
                            'they are not interpolated crossover estimates.'))
    if not groups:
        warnings.append('No crossover measurements available (the run may have been skipped).')
    return figures


def section(metadata, rows):
    groups = analyze(rows)
    lines = ['## CPU/GPU crossover', '',
             'CPU baseline: ' + str(metadata.get('cpu_implementation', 'unknown')) + '.', '',
             '| CUDA kernel | Timing mode | First sampled GPU win | Largest paired size: speedup |',
             '|---|---|---|---|']
    for (kernel, mode), points in sorted(groups.items()):
        first = next((p['size'] for p in points if p['status'] == 'first_sampled_gpu_win'), None)
        valid = [p for p in points if p['speedup'] is not None]
        last = f"N={valid[-1]['size']}: {valid[-1]['speedup']:.3f}×" if valid else 'unavailable'
        result = f'N={first}' if first is not None else ('none observed' if valid else 'unavailable')
        lines.append(f'| {kernel} | {mode} | {result} | {last} |')
    lines += ['', 'Timing boundaries:', '']
    for name in ('cpu_timing_methodology', 'kernel_only_timing_methodology', 'end_to_end_timing_methodology',
                 'crossover_execution_order'):
        lines.append('- ' + str(metadata.get(name, name + ': unknown')))
    failures = sum(p['tolerance_pass'] is False for points in groups.values() for p in points)
    unknown = sum(p['tolerance_pass'] is None for points in groups.values() for p in points)
    unavailable = sum(p['speedup'] is None for points in groups.values() for p in points)
    lines += ['', f'GPU configuration/mode rows failing numerical tolerance: {failures}. '
              f'Rows without recorded tolerance status: {unknown}. '
              f'Rows without usable paired medians: {unavailable}.', '',
              'Speedup uses CPU median / GPU median, distinct from the mean-based speedup in summary.csv. '
              'The first strict GPU win is only a sampled observation, not an exact or necessarily sustained '
              'crossover. No win means none among usable tested sizes. Numerical failures remain visible '
              'and do not establish correctness.', '',
              'This tests the repository’s single-threaded CPU implementation, not an optimized BLAS. '
              'Kernel-only GPU timing assumes operands already on device and is not a host-to-host comparison. '
              'End-to-end timing includes allocation and pageable transfers without overlap or buffer reuse. '
              'Results depend on implementation, CPU/GPU, compilers, clocks, load, warmups and transfer assumptions. '
              'Tiny CPU samples may approach clock/measurement overhead; no overhead subtraction is applied.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Remaining options are passed to matmul-inspector crossover.')
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', action='store_true')
    args, options = parser.parse_known_args()
    if args.output.exists():
        parser.error('Output directory already exists: ' + str(args.output))
    if args.report:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            parser.error('Install scripts/requirements-report.txt before requesting --report')
    try:
        result = subprocess.run([str(args.executable.resolve()), 'crossover', *options, '--output', str(args.output)])
        if args.report and result.returncode in (0, 1):
            from compare_hardware import load_json
            metadata = load_json(args.output / 'metadata.json')
            if metadata.get('status') in ('complete', 'tolerance_failed'):
                import report
                report_code = report.main([str(args.output)])
                if report_code:
                    return report_code
        return result.returncode
    except (OSError, ValueError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
