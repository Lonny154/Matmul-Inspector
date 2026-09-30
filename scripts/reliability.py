"""Read native timing artifacts and visualize them without recomputing CIs."""
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f'Malformed {Path(path).name}')
    return rows


def number(text):
    if text in ('', None):
        return None
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Invalid timing statistic')
    return value


def load(run, summary):
    """Missing legacy raw samples are fine; malformed modern samples are not plotted."""
    path = Path(run) / 'timing_samples.csv'
    if not path.exists():
        return None
    identities = {str(row['row_id']): row for row in summary}
    groups = defaultdict(list)
    expected = defaultdict(int)
    measured = set()
    for row in read_csv(path):
        identity = identities[row['row_id']]
        if (row['kernel'] != identity['kernel'] or row['timing_mode'] != identity['timing_mode']
                or tuple(int(row[key]) for key in ('M','N','K')) != identity['shape']):
            raise ValueError('Timing sample identity disagrees with summary')
        trial, iteration = int(row['trial']), int(row['iteration'])
        phase = row['phase']
        key = (row['row_id'], trial, phase)
        trial_key = key[:2]
        value = number(row['latency_ms'])
        if (trial < 0 or iteration != expected[key] or phase not in ('warmup','measurement')
                or value is None or (phase == 'warmup' and trial_key in measured)):
            raise ValueError('Invalid timing sample phase/iteration')
        expected[key] += 1
        if phase == 'measurement':
            measured.add(trial_key)
        groups[row['row_id']].append(dict(trial=trial, iteration=iteration, phase=phase, latency_ms=value))
    if not groups or any(not any(p['phase'] == 'measurement' for p in points) for points in groups.values()):
        raise ValueError('No measured timing samples')
    stats = read_csv(Path(run) / 'timing_statistics.csv')
    overall = {}
    trials = defaultdict(list)
    for row in stats:
        if row['row_id'] not in groups or row['scope'] not in ('overall','trial'):
            raise ValueError('Invalid timing statistics identity')
        for key in ('min_ms','max_ms','mean_ms','median_ms','stddev_ms','iqr_ms','mad_ms','cv',
                    'median_ci_low_ms','median_ci_high_ms'):
            row[key] = number(row.get(key))
        low, high = row['median_ci_low_ms'], row['median_ci_high_ms']
        if (low is None) != (high is None) or (low is not None and low > high):
            raise ValueError('Invalid interval bounds')
        if row['scope'] == 'overall':
            if row['row_id'] in overall:
                raise ValueError('Duplicate overall timing statistics')
            overall[row['row_id']] = row
        else:
            trials[row['row_id']].append(row)
    if set(overall) != set(groups):
        raise ValueError('Missing overall timing statistics')
    return dict(groups=dict(groups), overall=overall, trials=dict(trials), identities=identities,
                pairs=read_csv(Path(run) / 'pairwise.csv') if (Path(run) / 'pairwise.csv').exists() else [])


def label(data, identity):
    row = data['identities'][identity]
    return f"{row['kernel']} / {row['timing_mode']}\n" + '×'.join(map(str,row['shape']))


def selected(data, limit=6):
    ordered = sorted(data['groups'], key=lambda key: (math.prod(data['identities'][key]['shape']),
                                                     data['identities'][key]['kernel'],
                                                     data['identities'][key]['timing_mode']))
    if len(ordered) <= limit:
        return ordered
    return [ordered[round(i*(len(ordered)-1)/(limit-1))] for i in range(limit)]


def render(data, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids = selected(data)
    labels = [label(data,key) for key in ids]
    figures = []

    def save(fig, name, title, caption):
        fig.tight_layout()
        fig.savefig(Path(destination) / name, dpi=160)
        plt.close(fig)
        figures.append((name,title,caption))

    def latency_scale(ax, values):
        # A narrow sub-decade range on symlog may have no labeled major ticks.
        positive = [v for v in values if v is not None and v > 0]
        if positive and max(positive) / min(positive) > 20:
            ax.set_yscale('symlog', linthresh=min(positive))
        else:
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)

    caption = ('At most six configurations, evenly selected across configurations ordered by work size; '
               'all measurements remain in timing_samples.csv. Warmups are excluded from distributions and CIs.')
    fig, ax = plt.subplots(figsize=(11,5))
    ax.boxplot([[p['latency_ms'] for p in data['groups'][key] if p['phase']=='measurement'] for key in ids])
    ax.set_xticks(range(1,len(ids)+1),labels,rotation=20,ha='right',fontsize=8)
    ax.set(ylabel='Measured latency (ms)',title='Latency distributions: measurements only')
    latency_scale(ax,[p['latency_ms'] for key in ids for p in data['groups'][key] if p['phase']=='measurement'])
    ax.grid(axis='y',alpha=.2)
    save(fig,'reliability_distribution.png','Latency distributions',caption + ' Boxes: quartiles and median; whiskers: 1.5 IQR; points: outside whiskers. Symlog is used for wide ranges and preserves zero samples.')

    fig, axes = plt.subplots(len(ids),1,figsize=(10,max(3,len(ids)*2.1)),squeeze=False)
    for ax, key in zip(axes[:,0],ids):
        by_trial = defaultdict(list)
        for sample in data['groups'][key]:
            by_trial[sample['trial']].append(sample)
        for trial, points in sorted(by_trial.items()):
            for phase, style in (('warmup',':'),('measurement','-')):
                phase_points = [p for p in points if p['phase']==phase]
                x = [p['iteration'] - len(phase_points) if phase=='warmup' else p['iteration'] for p in phase_points]
                if x:
                    ax.plot(x,[p['latency_ms'] for p in phase_points],style,marker='.',markersize=3,
                            color=f'C{trial%10}',label=f'Trial {trial} {phase}')
        ax.axvline(-.5,color='black',linewidth=.6)
        ax.set(title=label(data,key).replace('\n',' — '),ylabel='Latency (ms)')
        ax.grid(alpha=.2)
        # Avoid giant legends on large trial counts; the full identity is in CSV.
        if len(by_trial) <= 5:
            ax.legend(fontsize=6,ncol=2)
    axes[-1,0].set_xlabel('Iteration within trial (negative = warmup; measurements start at 0)')
    save(fig,'reliability_sequence.png','Warmup and measurement sequences',caption + ' Trial identity is preserved; dotted traces are warmups. No adaptive warmup is applied.')

    fig, ax = plt.subplots(figsize=(11,5))
    for i, key in enumerate(ids):
        stats = data['overall'][key]
        low,high=stats['median_ci_low_ms'],stats['median_ci_high_ms']
        if low is not None:
            ax.vlines(i,low,high,color='C0',linewidth=2)
            ax.scatter([i,i],[low,high],marker='_',color='C0')
        ax.scatter(i,stats['median_ms'],color='C0',marker='o',label='Overall median / CI' if i==0 else None)
        medians = [s['median_ms'] for s in data['trials'].get(key,[])]
        ax.scatter([i+.12]*len(medians),medians,color='C1',marker='.',label='Trial medians' if i==0 else None)
    ax.set_xticks(range(len(ids)),labels,rotation=20,ha='right',fontsize=8)
    ax.set(ylabel='Median latency (ms)',title='Median confidence intervals and trial variation')
    latency_scale(ax,[data['overall'][key][field] for key in ids for field in
                      ('median_ms','median_ci_low_ms','median_ci_high_ms')] +
                     [s['median_ms'] for key in ids for s in data['trials'].get(key,[])])
    ax.grid(axis='y',alpha=.2)
    ax.legend(fontsize=8)
    save(fig,'reliability_intervals.png','Median bootstrap intervals',caption + ' Interval level and method are recorded below; absent intervals are not invented. CI overlap is not a significance test.')
    return figures


def section(data, config, artifact_prefix=".."):
    from urllib.parse import quote
    prefix = quote(artifact_prefix.rstrip("/"), safe="/.")
    def fmt(value):
        return 'unavailable' if value is None else f'{value:.6g}'
    lines=['## Benchmark methodology and reliability','',
           f"Warmups per trial: {config.get('warmups','unknown')}; measured iterations per trial: {config.get('iterations','unknown')}; "
           f"trials: {config.get('trials',1)}. Primary latency statistic: median.", '',
           f"Bootstrap replicates: {config.get('bootstrap_samples','unknown')}; confidence level: {config.get('confidence_level','unknown')}; "
           f"bootstrap seed: {config.get('bootstrap_seed','unknown')}.", '',
           'Multiple trials: resample whole trial blocks with replacement and take the pooled median. '
           'Single trial: resample individual measured latencies with replacement (IID assumption). '
           'Intervals are percentile bootstrap bounds, using linear quantiles. Warmups are timed separately and excluded. '
           'Blocks are sequential runs in one process, not independent process restarts; cross-trial drift can violate bootstrap assumptions. '
           'Small numbers of trials give weak uncertainty estimates. Zero/disabled/insufficient-sample intervals are left unavailable.', '',
           '| Configuration | Median ms | IQR ms | MAD ms | CV | Median CI ms | Heuristic flags |',
           '|---|---:|---:|---:|---:|---|---|']
    ids=list(data['groups'])
    for key in ids[:12]:
        s=data['overall'][key]
        lines.append(f"| {label(data,key).replace(chr(10),' ')} | {fmt(s['median_ms'])} | {fmt(s['iqr_ms'])} | "
                     f"{fmt(s['mad_ms'])} | {fmt(s['cv'])} | {fmt(s['median_ci_low_ms'])} – {fmt(s['median_ci_high_ms'])} | "
                     f"{s.get('stability_warnings') or 'none'} |")
    lines += ['', 'Descriptive spread is not a confidence claim. Flags are conservative heuristics, not correctness failures: '
              'CV >20%, max/min >3, IQR/median >25%, first measurement differs >50% from the remaining median (n≥5), '
              'or last-quarter median differs >20% from first-quarter median (n≥8). Fewer than 10 measurements or 5 trials '
              'indicates limited evidence. Zero samples flag timer resolution.', '',
              'Pairwise comparisons use candidate/reference median latency ratio, (ratio−1)×100 percent difference, '
              'and reference/candidate median speedup. CI overlap is descriptive only; neither overlap nor non-overlap '
              'alone establishes statistical significance. No formal hypothesis test is performed.', '',
              '| Candidate row | Reference row | Latency ratio | Difference % | Speedup | CI overlap |',
              '|---|---|---:|---:|---:|---|']
    for pair in data['pairs'][:12]:
        lines.append('| ' + ' | '.join(pair.get(key,'') or 'unavailable' for key in
                     ('candidate_row_id','reference_row_id','latency_ratio','percent_difference','speedup','median_ci_overlap')) + ' |')
    lines += ['', f'Tables show at most 12 rows. Complete data: [raw samples]({prefix}/timing_samples.csv), '
              f'[overall/per-trial statistics]({prefix}/timing_statistics.csv), [pairwise comparisons]({prefix}/pairwise.csv). '
              'These results apply only to the recorded hardware/software environment. Initialization, frequency/power changes, '
              'thermal throttling, background load, CPU frequency scaling and allocation/transfer choices can affect timing. '
              'Fixed reference-before-candidate ordering is not a randomized experiment.', '']
    return '\n'.join(lines)
