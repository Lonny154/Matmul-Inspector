#!/usr/bin/env python3
"""Post-process CUDA matmul timings with explicit, auditable simple-roofline ceilings."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
from urllib.parse import quote

from compare_hardware import digest
from report import load_report

MODEL = 'fp32-compulsory-read-A-read-B-write-C-v1'
SOURCES = ('user-supplied', 'documented-vendor-spec', 'measured-separately')


def positive(value):
    if isinstance(value, bool):
        raise ValueError('Expected a finite positive number')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Expected a finite positive number') from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError('Expected a finite positive number')
    return number


def ceilings(peak_fp32_tflops, memory_bandwidth_gbps, compute_source='user-supplied',
             bandwidth_source='user-supplied', source_note=''):
    compute = positive(peak_fp32_tflops)
    bandwidth = positive(memory_bandwidth_gbps)
    if compute_source not in SOURCES or bandwidth_source not in SOURCES:
        raise ValueError('Unknown ceiling source label')
    # Validate derived ratios too: avoid zero/overflow roofs from extreme inputs.
    peak_gflops = positive(compute * 1000)
    ridge = positive(peak_gflops / bandwidth)
    return dict(peak_fp32_tflops=compute, peak_compute_gflops=peak_gflops,
                peak_memory_bandwidth_gbps=bandwidth, ridge_point_flop_per_byte=ridge,
                compute_source=compute_source, bandwidth_source=bandwidth_source,
                source_note=source_note, units='decimal: TFLOP/s=1e12 FLOP/s; GB/s=1e9 bytes/s (not gigabits/s)')


def metrics(m, n, k, latency_ms, hardware):
    """Nominal math operations and compulsory bytes; never measured DRAM traffic."""
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (m,n,k)):
        raise ValueError('Matrix dimensions must be positive integers')
    latency_ms = positive(latency_ms)
    flop_count = 2*m*n*k
    model_bytes = 4*(m*k+k*n+m*n)
    intensity = positive(flop_count / model_bytes)
    achieved = positive(flop_count / (latency_ms*1e6))
    compute = positive(hardware['peak_compute_gflops'])
    bandwidth = positive(hardware['peak_memory_bandwidth_gbps'])
    ridge = positive(compute / bandwidth)
    memory_roof = positive(intensity*bandwidth)
    attainable = min(compute,memory_roof)
    model_bandwidth = positive(model_bytes/(latency_ms*1e6))
    result = dict(flop_count=flop_count, model_bytes=model_bytes,
                  arithmetic_intensity_flop_per_byte=intensity, latency_ms=latency_ms,
                  achieved_flops=positive(achieved*1e9), achieved_gflops=achieved,
                  peak_compute_gflops=compute, peak_memory_bandwidth_gbps=bandwidth,
                  ridge_point_flop_per_byte=ridge, memory_roof_gflops=memory_roof,
                  compute_roof_gflops=compute, attainable_gflops=attainable,
                  roofline_bound='memory-bound' if intensity < ridge else 'compute-bound',
                  efficiency_vs_attainable=positive(achieved/attainable),
                  efficiency_vs_compute_peak=positive(achieved/compute),
                  compute_peak_percent=positive(100*achieved/compute),
                  model_bandwidth_gbps=model_bandwidth,
                  model_bandwidth_peak_percent=positive(100*model_bandwidth/bandwidth))
    flags=[]
    if achieved > compute:
        flags.append('exceeds_configured_compute_peak')
    if achieved > memory_roof:
        flags.append('exceeds_model_memory_roof')
    if achieved > attainable:
        flags.append('exceeds_attainable_roof')
    result['roofline_warnings']=';'.join(flags)
    return result


def prepare(run, hardware, statistic='median'):
    if statistic not in ('median','mean'):
        raise ValueError('Latency statistic must be median or mean')
    meta, source_rows, warnings = load_report(run)
    if meta.get('schema_version') not in (1,2,3):
        raise ValueError('Expected a supported benchmark metadata schema (1, 2, or 3)')
    config=meta.get('config',{})
    if not isinstance(config,dict):
        raise ValueError('Expected metadata config object')
    if config.get('dtype') != 'float32':
        raise ValueError('The compulsory traffic model requires recorded float32 dtype')
    if config.get('mode') not in ('benchmark','crossover'):
        raise ValueError('Roofline requires timed benchmark or crossover artifacts')
    if meta.get('status') not in ('complete','tolerance_failed'):
        raise ValueError('Source run is incomplete, failed or skipped')
    points, excluded = [], []
    for row in source_rows:
        raw=row['source_fields']
        mode=row['timing_mode']
        inferred=False
        # Before per-row modes existed, benchmark schema 1–3 was CUDA kernel-only.
        if not mode and config['mode']=='benchmark' and row['kernel']!='cpu':
            mode='kernel_only'
            inferred=True
        reason=None
        if row['kernel']=='cpu' or raw.get('backend')=='cpu':
            reason='CPU row: GPU ceilings do not apply'
        elif raw.get('backend') not in (None,'','cuda'):
            reason='Unknown backend'
        elif mode != 'kernel_only':
            reason='Not established CUDA kernel-only timing: ' + (mode or 'unknown')
        latency=row[statistic+'_ms']
        if reason is None and (latency is None or latency<=0):
            reason=f'No positive {statistic}_ms; choose --statistic mean explicitly for mean-only legacy runs'
        if reason:
            excluded.append(dict(row_id=row['row_id'],kernel=row['kernel'],shape=row['shape'],reason=reason))
            continue
        point=dict(raw)
        point.update(row_id=row['row_id'], backend='cuda', timing_mode=mode,
                     timing_mode_inferred=inferred, latency_statistic=statistic,
                     model_name=MODEL)
        try:
            point.update(metrics(*row['shape'],latency,hardware))
        except (ValueError, OverflowError) as error:
            excluded.append(dict(row_id=row['row_id'],kernel=row['kernel'],shape=row['shape'],reason=str(error)))
            continue
        points.append(point)
        if inferred:
            warnings.append(f"Row {row['row_id']}: kernel_only inferred from legacy benchmark schema contract")
        if row['tolerance_pass'] is False:
            warnings.append(f"Row {row['row_id']}: numerical tolerance failed; performance does not establish correctness")
        elif row['tolerance_pass'] is None:
            warnings.append(f"Row {row['row_id']}: numerical tolerance status unavailable")
        if point['roofline_warnings']:
            warnings.append(f"Row {row['row_id']}: {point['roofline_warnings']} (values retained, not clamped)")
    flagged=sum(bool(p.get('stability_warnings')) for p in points)
    if flagged:
        warnings.append(f'{flagged} analyzed row(s) carry source timing-stability flags; retained in roofline.csv')
    if excluded:
        warnings.append(f'{len(excluded)} row(s) excluded; see roofline_metadata.json for reasons')
    if not points:
        raise ValueError('No usable CUDA kernel-only rows: ' + '; '.join(e['reason'] for e in excluded[:3]))
    return meta, points, excluded, warnings


def plot(points, hardware, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ridge=hardware['ridge_point_flop_per_byte']
    intensities=[p['arithmetic_intensity_flop_per_byte'] for p in points]
    # Keep both the measured range and ridge visible on the logarithmic axis.
    lo=min(min(intensities),ridge)/3
    hi=max(max(intensities),ridge)*3
    x=[10**(math.log10(lo)+(math.log10(hi)-math.log10(lo))*i/200) for i in range(201)]
    compute=hardware['peak_compute_gflops']
    memory=[v*hardware['peak_memory_bandwidth_gbps'] for v in x]
    fig,ax=plt.subplots(figsize=(10,6))
    ax.loglog(x,memory,'--',color='C1',label='Model memory-bandwidth roof')
    ax.axhline(compute,linestyle='--',color='C2',label='Configured FP32 compute roof')
    ax.loglog(x,[min(compute,v) for v in memory],color='black',linewidth=2,label='Simple attainable roof')
    ax.axvline(ridge,linestyle=':',color='gray',label=f'Ridge: {ridge:.4g} FLOP/byte')
    groups={}
    for point in points:
        # Preserve variant controls rather than assuming only kernel names matter.
        key=(point['kernel'],point.get('tile_size',''),point.get('contraction_mode',''),point.get('accumulation_mode',''))
        groups.setdefault(key,[]).append(point)
    markers=('o','s','^','D','v','P','X')
    for i,(key,group) in enumerate(groups.items()):
        label=key[0]
        if key[1] not in ('','0',0):
            label+=f' (tile {key[1]})'
        ax.scatter([p['arithmetic_intensity_flop_per_byte'] for p in group],
                   [p['achieved_gflops'] for p in group],s=48,marker=markers[i%len(markers)],label=label,zorder=4)
    # Label each distinct shape once so many variants don't repeat overlapping text.
    shapes={}
    for p in points:
        shapes.setdefault((p['M'],p['N'],p['K']),p)
    ordered=sorted(shapes.values(),key=lambda p:p['flop_count'])
    selected=ordered if len(ordered)<=10 else [ordered[round(i*(len(ordered)-1)/9)] for i in range(10)]
    for p in selected:
        ax.annotate('×'.join(str(p[k]) for k in ('M','N','K')),
                    (p['arithmetic_intensity_flop_per_byte'],p['achieved_gflops']),
                    xytext=(5,6),textcoords='offset points',fontsize=7)
    ax.set(xlabel='Compulsory-model arithmetic intensity (FLOP/byte)',ylabel='Performance (GFLOP/s)',
           title='Simple FP32 matmul roofline — CUDA kernel-only timing',xlim=(lo,hi))
    ax.grid(which='both',alpha=.2)
    ax.legend(fontsize=8,loc='best')
    fig.tight_layout()
    fig.savefig(Path(destination)/'roofline.png',dpi=160)
    plt.close(fig)


def markdown(run, destination, meta, points, hardware, statistic, warnings):
    def safe(value):
        return str(value).replace('|','\\|').replace('\n',' ')
    def link(filename):
        path=os.path.relpath(Path(run)/filename,destination)
        return quote(Path(path).as_posix(),safe='/.' )
    lines=['# Matmul-Inspector performance report','', '## Roofline Analysis','',
           f"Source run: `{safe(Path(run).resolve())}`",'',
           f"GPU: {safe(meta.get('gpu_name','unknown'))}; compute capability: {safe(meta.get('gpu_compute_capability','unknown'))}. "
           f"Source commit: `{safe(meta.get('git_commit','unknown'))}`; dirty: {safe(meta.get('git_dirty','unknown'))}.", '',
           '| Explicit ceiling | Value | Source label |','|---|---:|---|',
           f"| FP32 compute | {hardware['peak_fp32_tflops']:.6g} TFLOP/s | {hardware['compute_source']} |",
           f"| Memory bandwidth | {hardware['peak_memory_bandwidth_gbps']:.6g} GB/s | {hardware['bandwidth_source']} |", '',
           'Ceiling source note: '+safe(hardware['source_note'] or 'none supplied')+'.', '',
           '**Model:** nominal FLOPs = `2*M*N*K`, counting multiply and add as two operations. '
           'Model bytes = `4*(M*K + K*N + M*N)`: read A once, read B once, write C once. '
           'This is logical FP32 compulsory traffic, excluding row padding, rereads, output initialization and transfers; C is not read.', '',
           f"Arithmetic intensity = FLOPs/model bytes. Ridge point = compute/bandwidth = {hardware['ridge_point_flop_per_byte']:.6g} FLOP/byte. "
           'Attainable GFLOP/s = min(compute GFLOP/s, intensity × bandwidth GB/s). '
           'Below the ridge is model memory-bound; at or above it is model compute-bound.', '',
           f"Achieved throughput is recomputed from **{statistic} latency**, not copied from the legacy mean-based GFLOP/s field. "
           'Decimal units: TFLOP/s = 10¹² FLOP/s; GB/s = 10⁹ bytes/s, not gigabits/s.', '',
           'Timing methodology: '+safe(meta.get('timing_methodology','unknown'))+'.', '',
           '![Simple roofline](roofline.png)', '',
           'Markers preserve kernel variants; annotations show up to ten distinct M×N×K shapes. '
           'All analyzed configurations are in [roofline.csv](roofline.csv).', '',
           '| Kernel / M×N×K | AI FLOP/byte | Achieved GFLOP/s | Model bound | Attainable efficiency | Compute peak % | Model bandwidth % |',
           '|---|---:|---:|---|---:|---:|---:|']
    for p in points[:20]:
        shape='×'.join(str(p[k]) for k in ('M','N','K'))
        lines.append(f"| {safe(p['kernel'])} / {shape} | {p['arithmetic_intensity_flop_per_byte']:.4g} | "
                     f"{p['achieved_gflops']:.5g} | {p['roofline_bound']} | {100*p['efficiency_vs_attainable']:.3g}% | "
                     f"{p['compute_peak_percent']:.3g}% | {p['model_bandwidth_peak_percent']:.3g}% |")
    lines+=['','Table limited to 20 rows; CSV contains all points.', '',
            '**Interpretation limits:** the traffic is a model, not measured DRAM traffic, cache reuse, '
            'shared-memory traffic, or hardware-counter transactions. Model bandwidth = model bytes / latency; '
            'its percentage is not measured DRAM utilization. The predicted bound describes this model, '
            'not a diagnosis of the kernel’s actual bottleneck. Different kernels at the same shape have '
            'the same modeled intensity despite different real memory behavior. Vendor peaks may not be '
            'sustainably achievable; ceilings apply to ordinary FP32 arithmetic, not Tensor Core rates. '
            'Values above roofs are retained and flagged, not clamped; check ceilings, units, timing noise '
            'and model mismatch. Kernel-only timing excludes transfers; CPU and end-to-end rows are excluded.', '',
            '**Correctness remains separate:** tolerance failures, bitwise diagnostics, output hashes and '
            'stability warnings are retained in CSV source columns. Roofline efficiency is not an accuracy result.', '',
            f"Source [metadata]({link('metadata.json')}) and [summary / numerical diagnostics]({link('summary.csv')}). "
            'Analysis assumptions and source SHA-256 digests: [roofline_metadata.json](roofline_metadata.json).']
    for name in ('mismatches.csv','timing_samples.csv','timing_statistics.csv','pairwise.csv'):
        if (Path(run)/name).is_file():
            lines+=['',f'Source [{name}]({link(name)}).']
    if warnings:
        lines+=['','### Warnings','']+['- '+safe(w) for w in warnings]
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='existing benchmark/crossover run')
    parser.add_argument('--output',type=Path,help='new analysis directory (default INPUT/roofline)')
    parser.add_argument('--peak-fp32-tflops',required=True,help='explicit ordinary FP32 TFLOP/s ceiling')
    parser.add_argument('--memory-bandwidth-gbps',required=True,help='explicit decimal GB/s (bytes, not bits)')
    parser.add_argument('--compute-source',choices=SOURCES,default='user-supplied')
    parser.add_argument('--bandwidth-source',choices=SOURCES,default='user-supplied')
    parser.add_argument('--source-note',default='',help='citation, measurement provenance, or assumptions')
    parser.add_argument('--statistic',choices=('median','mean'),default='median')
    parser.add_argument('--report',action='store_true',help='also generate roofline.png and report.md (matplotlib)')
    args=parser.parse_args(argv)
    try:
        hardware=ceilings(args.peak_fp32_tflops,args.memory_bandwidth_gbps,args.compute_source,args.bandwidth_source,args.source_note)
        run=args.input.resolve()
        destination=args.output or run/'roofline'
        if destination.exists():
            raise ValueError(f'Output directory already exists: {destination}')
        names=('metadata.json','summary.csv','mismatches.csv','timing_samples.csv','timing_statistics.csv','pairwise.csv')
        source_digests={name:digest(run/name) for name in names if (run/name).is_file()}
        meta,points,excluded,warnings=prepare(run,hardware,args.statistic)
        if any(digest(run/name)!=sha for name,sha in source_digests.items()):
            raise ValueError('Source artifacts changed during analysis')
        if args.report:
            try:
                import matplotlib
            except ImportError as error:
                raise ValueError('Install scripts/requirements-report.txt to render plots') from error
        provenance=dict(schema_version=1,artifact_type='matmul-simple-roofline',model=MODEL,
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        source_run=str(run),source_artifact_sha256=source_digests,source_metadata=meta,
                        ceilings=hardware,latency_statistic=args.statistic,
                        model_bytes_formula='4*(M*K+K*N+M*N)',flop_count_formula='2*M*N*K',
                        measured_memory_traffic=None,excluded_rows=excluded,warnings=warnings,
                        analysis_python=platform.python_version(),
                        analysis_tool_sha256={p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('report.py'),Path(__file__).with_name('compare_hardware.py'))})
        if args.report:
            provenance['matplotlib_version']=matplotlib.__version__
        destination.mkdir(parents=True)
        fields=list(dict.fromkeys(field for p in points for field in p))
        with (destination/'roofline.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader();writer.writerows(points)
        if args.report:
            plot(points,hardware,destination)
            (destination/'report.md').write_text(markdown(run,destination,meta,points,hardware,args.statistic,warnings),encoding='utf-8')
        # Completion metadata written last. No source file is written or replaced.
        (destination/'roofline_metadata.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        for warning in warnings[:8]:
            print('warning: '+warning,file=sys.stderr)
        if len(warnings)>8:
            print(f'warning: {len(warnings)-8} additional warnings in roofline_metadata.json',file=sys.stderr)
        print(f'Roofline: {destination} ({len(points)} kernel-only rows; {len(excluded)} excluded)')
        return 0
    except (OSError,ValueError,TypeError,OverflowError,csv.Error) as error:
        print(f'error: {error}',file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
