#!/usr/bin/env python3
"""Render existing Matmul-Inspector summaries; never execute benchmarks."""
import argparse
import csv
import math
import os
from pathlib import Path
import sys

from compare_hardware import load_json, finite_number


METRICS = ('median_speedup', 'median_ms', 'mean_ms', 'gflops', 'speedup', 'divergent_count', 'divergent_percent',
           'tolerance_failures', 'tolerance_failure_percent')


def load_report(run):
    """Require row identities, but tolerate absent/invalid optional measurements."""
    run = Path(run).resolve()
    warnings = []
    metadata = {}
    try:
        metadata = load_json(run / 'metadata.json')
        if not isinstance(metadata, dict):
            raise ValueError('expected a JSON object')
    except (OSError, ValueError) as error:
        warnings.append(f'Metadata unavailable: {error}')
        metadata = {}
    rows = []
    with (run / 'summary.csv').open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if not {'M', 'N', 'K', 'kernel'}.issubset(reader.fieldnames or []):
            raise ValueError('summary.csv requires M, N, K and kernel columns')
        seen = set()
        for line, raw in enumerate(reader, 2):
            try:
                shape = tuple(int(raw[field]) for field in ('M', 'N', 'K'))
                if min(shape) <= 0 or not raw['kernel'] or None in raw:
                    raise ValueError('invalid dimensions/kernel or malformed CSV')
            except (ValueError, TypeError) as error:
                warnings.append(f'Skipping summary line {line}: {error}')
                continue
            # A same-kernel reference/candidate pair can have two timing rows.
            identity = (raw['kernel'], raw.get('reference_kernel') or 'unknown', shape, raw.get('timing_mode', ''))
            if identity in seen:
                raise ValueError(f'Ambiguous duplicate kernel/reference/shape at line {line}')
            seen.add(identity)
            row = dict(source_fields=dict(raw), row_id=raw.get('row_id', str(line-2)), kernel=identity[0], reference=identity[1], shape=shape, timing_mode=identity[3],
                       tolerance_pass={'true': True, 'false': False}.get(raw.get('tolerance_pass')))
            for field in METRICS:
                try:
                    value = finite_number(raw.get(field))
                    if value is not None and field.endswith('percent') and value > 100:
                        raise ValueError('percentage exceeds 100')
                    if value is not None and field in ('divergent_count', 'tolerance_failures'):
                        if not value.is_integer() or value > shape[0] * shape[1]:
                            raise ValueError('invalid output count')
                    row[field] = value
                except (ValueError, TypeError, OverflowError):
                    row[field] = None
                    warnings.append(f'Ignoring invalid {field} at summary line {line}')
            rows.append(row)
    if not rows:
        warnings.append('No usable summary rows.')
    if metadata.get('git_dirty') is True:
        warnings.append('Source tree was dirty; this capture is not fully reproducible.')
    if metadata.get('status', 'complete') not in ('complete', 'tolerance_failed'):
        warnings.append(f"Run status: {metadata['status']}; results may be incomplete.")
    return metadata, rows, warnings


def group_rows(rows, numerical=False):
    groups = {}
    for row in rows:
        if numerical and row['kernel'] == row['reference']:
            continue  # Reference self-comparisons do not characterize a candidate.
        label = row['kernel']
        if row.get('timing_mode'):
            label += ' [' + row['timing_mode'] + ']'
        if numerical:
            label += ' vs ' + row['reference']
        groups.setdefault(label, []).append(row)
    return groups


def percentage(row, kind):
    """Prefer a recorded aggregate; derive older-schema percentages from M*N."""
    count_field, percent_field = {
        'divergence': ('divergent_count', 'divergent_percent'),
        'tolerance': ('tolerance_failures', 'tolerance_failure_percent'),
    }[kind]
    if row[percent_field] is not None:
        return row[percent_field]
    count = row[count_field]
    return None if count is None else 100 * count / (row['shape'][0] * row['shape'][1])


def render_plots(rows, destination, warnings):
    # Lazy import keeps data parsing tests and CPU builds dependency-free.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    shapes = sorted({r['shape'] for r in rows}, key=lambda s: (math.prod(s), s))
    positions = {shape: i for i, shape in enumerate(shapes)}
    labels = ['×'.join(map(str, shape)) for shape in shapes]
    figures = []

    def finish(fig, ax, filename, title, ylabel, description, series):
        if not series:
            plt.close(fig)
            warnings.append(f'Skipped {filename}: no supported measurements.')
            return
        ax.set(title=title, xlabel='Matrix shape M×N×K (categorical; ordered by MNK)', ylabel=ylabel)
        ax.set_xticks(range(len(shapes)), labels, rotation=35, ha='right')
        ax.grid(axis='y', alpha=0.25)
        ax.legend(fontsize='small')
        fig.tight_layout()
        fig.savefig(destination / filename, dpi=160)
        plt.close(fig)
        figures.append((filename, title, description))

    width = min(20, max(8, len(shapes) * 0.9))
    primary = 'median_ms' if any(r['median_ms'] is not None for r in rows) else 'mean_ms'
    if primary == 'mean_ms':
        warnings.append('Median unavailable: latency plot falls back to recorded means.')
    for field, filename, title, ylabel in (
        (primary, 'latency.png', 'Kernel latency by shape', ('Median' if primary == 'median_ms' else 'Mean') + ' kernel latency (ms)'),
        ('gflops', 'throughput.png', 'Kernel throughput by shape', 'Throughput (GFLOP/s)'),
    ):
        fig, ax = plt.subplots(figsize=(width, 4.8))
        series = 0
        for label, group in group_rows(rows).items():
            values = {r['shape']: r[field] for r in group}
            if any(v is not None for v in values.values()):
                ax.plot(range(len(shapes)), [values.get(s) if values.get(s) is not None else math.nan
                                            for s in shapes], marker='o', label=label)
                series += 1
        ax.set_ylim(bottom=0)
        finish(fig, ax, filename, title, ylabel,
               'Recorded summary values. Gaps indicate unavailable measurements; connecting lines '
               'follow categorical shapes, not equal increments in problem size.', series)

    fig, ax = plt.subplots(figsize=(width, 4.8))
    groups = group_rows(rows, numerical=True)
    series = 0
    ratio_field = 'median_speedup' if any(r['median_speedup'] is not None for r in rows) else 'speedup'
    ratio_stat = 'median' if ratio_field == 'median_speedup' else 'mean'
    bar_width = 0.8 / max(1, len(groups))
    for index, (label, group) in enumerate(groups.items()):
        available = [r for r in group if r[ratio_field] is not None and r['mean_ms'] is not None]
        if available:
            ax.bar([positions[r['shape']] - 0.4 + bar_width * (index + 0.5) for r in available],
                   [r[ratio_field] for r in available], width=bar_width, label=label)
            series += 1
    ax.axhline(1, color='black', linewidth=0.8, linestyle='--')
    finish(fig, ax, 'kernel_comparison.png', 'Kernel comparison against recorded reference',
           f'Reference {ratio_stat} / candidate {ratio_stat} (×)',
           'Recorded within-run speedup. The dashed line is parity; values above one mean lower '
           'candidate latency for that configuration. References are named in the legend.', series)

    fig, ax = plt.subplots(figsize=(width, 4.8))
    series = 0
    for index, (label, group) in enumerate(groups.items()):
        for kind, style in (('divergence', '-'), ('tolerance', '--')):
            values = {r['shape']: percentage(r, kind) for r in group}
            if any(v is not None for v in values.values()):
                ax.plot(range(len(shapes)), [values.get(s) if values.get(s) is not None else math.nan
                                            for s in shapes], linestyle=style, marker='o',
                        color=f'C{index % 10}', label=f'{label}: {kind}')
                series += 1
    ax.set_ylim(-1, 101)
    finish(fig, ax, 'numerics.png', 'Numerical behavior against recorded reference',
           'Output elements (%)',
           'Solid lines show bitwise divergence; dashed lines show tolerance failures when recorded. '
           'Older divergence counts are divided by M×N. Self-comparisons are excluded. '
           'No distribution is inferred from the bounded mismatches.csv samples.', series)
    return figures


def markdown(run, metadata, rows, figures, warnings):
    def safe(value):
        return str(value).replace('\n', ' ').replace('|', '\\|').replace('`', "'")

    config = metadata.get('config', {})
    if not isinstance(config, dict):
        config = {}
    lines = ['# Matmul-Inspector experiment report', '', f'Run: `{safe(run)}`', '',
             '| Metadata | Value |', '|---|---|']
    for name in ('timestamp_utc', 'status', 'gpu_name', 'gpu_compute_capability', 'git_commit',
                 'git_dirty', 'build_type', 'cpu_model', 'cxx_compiler', 'cuda_compiler', 'cuda_runtime_version'):
        lines.append(f'| {name} | {safe(metadata.get(name, "unknown"))} |')
    for name in ('mode', 'seed', 'seed_b', 'input', 'warmups', 'iterations', 'atol', 'rtol'):
        lines.append(f'| {name} | {safe(config.get(name, "unknown"))} |')
    lines += ['', f'Rows: {len(rows)}. Kernels: {safe(", ".join(sorted(group_rows(rows))))}.', '',
              'Timing methodology: ' + safe(metadata.get('timing_methodology', 'unknown')) + '.', '',
              'These figures describe this capture only. Hardware/software, clocks, load and '
              'measurement variability limit broader conclusions. Bitwise divergence is distinct '
              'from a numerical tolerance failure.', '']
    if config.get('mode') == 'crossover':
        from crossover import section
        lines += [section(metadata, rows)]
    for filename, title, description in figures:
        lines += [f'## {title}', '', f'![{title}]({filename})', '', description, '']
    if warnings:
        lines += ['## Notes', ''] + ['- ' + safe(w) for w in warnings] + ['']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='run directory containing summary.csv')
    parser.add_argument('--output', type=Path, help='new report directory (default: RUN/report)')
    args = parser.parse_args(argv)
    try:
        metadata, rows, warnings = load_report(args.run)
        destination = args.output or args.run / 'report'
        # No clobbering source artifacts or older reports, even with --output.
        if destination.exists():
            raise ValueError(f'Report directory already exists: {destination}')
        try:
            import matplotlib  # noqa: F401
        except ImportError as error:
            raise ValueError('Install plotting support: python3 -m pip install -r scripts/requirements-report.txt') from error
        destination.mkdir(parents=True)
        if isinstance(metadata.get('config'), dict) and metadata['config'].get('mode') == 'crossover':
            from crossover import render
            figures = render(rows, destination, warnings)
        else:
            figures = render_plots(rows, destination, warnings)
        import reliability
        reliability_text = ''
        try:
            data = reliability.load(args.run, rows)
            if data:
                figures += reliability.render(data, destination)
                reliability_text = reliability.section(data, metadata.get('config', {}),
                                                       os.path.relpath(args.run.resolve(), destination.resolve()))
            elif any(row['mean_ms'] is not None for row in rows):
                warnings.append('Raw timing samples unavailable; reliability plots/CIs cannot be reconstructed.')
        except (OSError, ValueError, KeyError, TypeError) as error:
            warnings.append('Reliability artifacts unavailable/invalid: ' + str(error))
        (destination / 'report.md').write_text(
            markdown(args.run.resolve(), metadata, rows, figures, warnings) + '\n' + reliability_text, encoding='utf-8')
        for warning in warnings:
            print('warning: ' + warning, file=sys.stderr)
        print(f'Report: {destination / "report.md"} ({len(figures)} plots)')
        return 0
    except (OSError, ValueError, csv.Error) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
