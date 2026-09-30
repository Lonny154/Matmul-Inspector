#!/usr/bin/env python3
"""Collect controlled reduction-order experiments; numerical analysis, not performance ranking."""
import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

from compare_hardware import read_run, binary, detailed, digest, load_json, write_csv
from operations import VECTOR_FIXTURES


def choices(text, allowed=None, integer=False):
    parts=text.split(',')
    if integer:
        parts=[int(v) for v in parts]
        if any(v<=0 for v in parts):
            raise ValueError('Sizes, repeats and blocks must be positive')
    if not parts or len(set(parts)) != len(parts) or any(v=='' for v in parts):
        raise ValueError('Lists must be nonempty and unique')
    if allowed is not None and any(v not in allowed for v in parts):
        raise ValueError('Unsupported selection: ' + text)
    return parts


def observations(path, repeats):
    with path.open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=repeats or any(None in r or None in r.values() for r in rows):
        raise ValueError('Malformed scalar repeat capture')
    for index,row in enumerate(rows):
        bits=row['fp32_bits']
        if int(row['repeat_index'])!=index or len(bits)!=32 or set(bits)-{'0','1'}:
            raise ValueError('Invalid repeat identity or FP32 bit representation')
        if int(row['fp32_hex'],16)!=int(bits,2):
            raise ValueError('Hex and binary representations disagree')
    states={r['determinism'] for r in rows}
    if len(states)!=1 or not states.issubset({'insufficient_repeats','bitwise_deterministic',
                                            'numerically_equal_bitwise_varying','numerically_varying'}):
        raise ValueError('Invalid determinism classification')
    for row in rows:
        row['bitwise_vs_baseline'] = row['fp32_bits'] == row['baseline_bits']
    return rows


def finite(value):
    if value in ('',None):
        return None
    result=float(value)
    return result if math.isfinite(result) else None


def summarize(rows, blocks):
    groups={}
    for row in rows:
        key=(row['fixture'],row['length'],row['implementation'],row['block_size'])
        groups.setdefault(key,[]).append(row)
    input_bits={}
    for group in groups.values():
        first=group[0]
        input_bits.setdefault((first['fixture'],first['length']),set()).update(r['fp32_bits'] for r in group)
    return dict(configurations=len(groups),
                bitwise_deterministic=sum(g[0]['determinism']=='bitwise_deterministic' for g in groups.values()),
                insufficient_repeats=sum(g[0]['determinism']=='insufficient_repeats' for g in groups.values()),
                varying=sum(g[0]['determinism'] in ('numerically_equal_bitwise_varying','numerically_varying') for g in groups.values()),
                max_ulp_vs_cpu=max((int(r['ulp_vs_baseline']) for r in rows if r['ulp_vs_baseline']),default=None),
                max_absolute_error_fp64=max((finite(r['absolute_error_fp64']) for r in rows if finite(r['absolute_error_fp64']) is not None),default=None),
                max_block_pair_ulp=max((r['max_ulp'] for r in blocks),default=None),
                inputs_all_bits_identical=sum(len(v)==1 for v in input_bits.values()),
                input_agreement=[dict(fixture=k[0],length=k[1],all_bits_identical=len(v)==1) for k,v in input_bits.items()],
                input_count=len(input_bits), tolerance_failures=sum(r['tolerance_pass']=='false' for r in rows))


def render(rows, blocks, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Use repeat zero per configuration. Every repeat remains in the CSV.
    first=[r for r in rows if r['repeat_index']=='0']
    fixtures=list(dict.fromkeys(r['fixture'] for r in first))
    figures=[]
    for field,name,ylabel in (('ulp_vs_baseline','ulp_drift.png','ULP distance vs FP32 serial'),
                               ('absolute_error_fp64','absolute_error.png','Absolute error vs FP64 accumulation')):
        cols=min(2,len(fixtures)); panels=(len(fixtures)+cols-1)//cols
        fig,axes=plt.subplots(panels,cols,figsize=(12,4*panels),squeeze=False)
        for ax,fixture in zip(axes.flat,fixtures):
            groups={}
            for row in first:
                if row['fixture']!=fixture:
                    continue
                label=row['implementation']+(' / block '+row['block_size'] if row['block_size'] else '')
                groups.setdefault(label,[]).append(row)
            for label,group in groups.items():
                points=sorted((int(r['length']),finite(r[field])) for r in group if finite(r[field]) is not None)
                if points:
                    ax.plot([p[0] for p in points],[p[1] for p in points],marker='o',label=label)
            ax.set(xlabel='Vector length (elements)',ylabel=ylabel,title=fixture)
            ax.set_xscale('log'); ax.set_yscale('symlog',linthresh=1 if field=='ulp_vs_baseline' else 1e-7)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=7); ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(fixtures):]:
            ax.set_visible(False)
        fig.tight_layout();fig.savefig(output/name,dpi=150);plt.close(fig);figures.append(name)
    if blocks:
        fig,ax=plt.subplots(figsize=(11,5))
        groups={}
        for row in blocks:
            if row['reference_block_size']!=row['selected_reference_block']:
                continue
            label=f"{row['fixture']} / length {row['length']}"
            groups.setdefault(label,[]).append(row)
        for label,group in groups.items():
            group=sorted(group,key=lambda r:r['candidate_block_size'])
            ax.plot([r['candidate_block_size'] for r in group],[r['max_ulp'] for r in group],marker='o',label=label)
        ax.set(xlabel='CUDA block size (threads)',ylabel='ULP vs selected reference block',title='Block-size sensitivity (first execution)')
        ax.set_yscale('symlog',linthresh=1);ax.set_ylim(bottom=0);ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1,1));ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(output/'block_sensitivity.png',dpi=150);plt.close(fig);figures.append('block_sensitivity.png')
    return figures


def report(rows, blocks, summary, config, sources, figures):
    lines=['# Reduction reproducibility experiment','',
        f"Operation: `{config['operation']}`. Seed: {config['seed']}; B seed: {config['seed_b']}. Repeats: {config['repeats']}.",
        f"Fixtures: {', '.join(config['fixtures'])}. Requested blocks: {config['block_sizes']}.",'',
        'FP32 addition is not associative: a serial loop and a tree group terms differently. '
        'The CPU baseline is forward serial; cpu-reverse visits decreasing indices. CUDA uses a fixed block tree followed by repeated partial reductions. '
        'CPU dot contraction follows compiler flags; CUDA dot uses separate rounded products. This is not a contraction-isolation experiment.','',
        'FP64 accumulation is a higher-precision analysis reference, not a mathematically exact answer. '
        'Relative FP64 error is omitted when |reference| ≤ 1e-12 or a value is nonfinite. '
        'ULPs are against the named FP32 baseline, not FP64. A bitwise mismatch need not fail tolerance.','',
        f"Repeat results: {summary['bitwise_deterministic']}/{summary['configurations']} configurations bitwise deterministic; "
        f"{summary['varying']} varying; {summary['insufficient_repeats']} with insufficient repeat evidence.",
        f"Largest ULP vs CPU serial: {summary['max_ulp_vs_cpu']}. Largest absolute error vs FP64: {summary['max_absolute_error_fp64']}.",
        f"Largest pairwise block ULP: {summary['max_block_pair_ulp']}. Tolerance failures across observations: {summary['tolerance_failures']}.",
        f"All selected implementations/repeats agreed bitwise for {summary['inputs_all_bits_identical']}/{summary['input_count']} inputs.",'',
        '**Deterministic repeated execution does not mean identical results across different reduction orders.** '
        'These are numerical experiments, not performance rankings. Lines join tested lengths; no monotonic error-growth claim is made.','',
        '| Fixture | Length | Implementation / block | Bits vs CPU | ULP vs CPU | Abs. error vs FP64 | Repeat classification |',
        '|---|---:|---|---|---:|---:|---|']
    for row in [r for r in rows if r['repeat_index']=='0'][:80]:
        lines.append('| '+' | '.join(str(v) for v in (row['fixture'],row['length'],row['implementation']+' / '+row['block_size'],
            'same' if row['bitwise_vs_baseline'] else 'different',row['ulp_vs_baseline'] or 'unavailable',row['absolute_error_fp64'] or 'unavailable',row['determinism']))+' |')
    lines+=['', 'Input agreement across selected implementations/repeats:']
    for item in summary['input_agreement']:
        lines.append(f"- {item['fixture']}, length {item['length']}: " + ('all bits identical' if item['all_bits_identical'] else 'bitwise disagreement'))
    lines+=['','Table limited to 80 configurations. Full observations: [CSV](reproducibility_results.csv). '
            'Every block pair: [CSV](block_sensitivity.csv). Provenance and selected reference block: '
            '[metadata](reproducibility_metadata.json). Native child runs retain replayable metadata, scalar binaries and diagnostics.','']
    for name in figures:
        lines+=['!['+name.removesuffix('.png').replace('_',' ')+']('+name+')','']
    if sources:
        meta=next((s['metadata'] for s in sources if s['metadata'].get('gpu_name') not in (None,'unknown','')),sources[0]['metadata'])
        lines += [f"Captured GPU: {meta.get('gpu_name','unknown')}; compiler: {meta.get('cxx_compiler','unknown')}; CUDA: {meta.get('cuda_compiler','unknown')}.",
                  f"Commit: {meta.get('git_commit','unknown')}; dirty: {meta.get('git_dirty','unknown')}. Results are hardware/build specific."]
    if any(s['metadata'].get('git_dirty') is not False for s in sources):
        lines += ['', '**Dirty or unknown source provenance:** commit alone cannot reconstruct this experiment.']
    skipped=[s['directory'] for s in sources if s['metadata'].get('status')=='skipped']
    if skipped:
        lines+=['',f'{len(skipped)} CUDA configurations skipped: runtime/device unavailable; see source metadata.']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation',choices=('reduction_sum','dot'),default='reduction_sum')
    parser.add_argument('--sizes',default='1024,1003')
    parser.add_argument('--fixtures',default='random_uniform,cancellation')
    parser.add_argument('--implementations',default='cpu,cpu-reverse,cuda-tree')
    parser.add_argument('--block-sizes',default='64,128,256,512')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--seed-b',type=int)
    parser.add_argument('--atol',type=float,default=1e-6)
    parser.add_argument('--rtol',type=float,default=1e-5)
    parser.add_argument('--executable',type=Path,default=Path('build/matmul-inspector'))
    parser.add_argument('--comparator',type=Path,help='default: matmul-compare-outputs beside executable')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-plots',action='store_true',help='CSV/JSON/Markdown without matplotlib')
    args=parser.parse_args(argv)
    try:
        sizes=choices(args.sizes,integer=True)
        fixtures=choices(args.fixtures,VECTOR_FIXTURES)
        implementations=choices(args.implementations,('cpu','cpu-reverse','cuda-tree'))
        blocks=choices(args.block_sizes,(64,128,256,512),integer=True)
        if not 1<=args.repeats<=10000 or not 0<=args.seed<=0xffffffff:
            raise ValueError('Invalid repeat count or seed')
        seed_b=args.seed_b if args.seed_b is not None else (args.seed+81)&0xffffffff
        if not 0<=seed_b<=0xffffffff or any(not math.isfinite(v) or v<0 for v in (args.atol,args.rtol)):
            raise ValueError('Invalid seed or tolerances')
        executable=args.executable.resolve()
        comparator=(args.comparator or executable.with_name('matmul-compare-outputs')).resolve()
        if not executable.is_file() or not comparator.is_file():
            raise ValueError('Build matmul-inspector and matmul-compare-outputs first')
        if args.output.exists():
            raise ValueError('Output directory already exists')
        if not args.no_plots:
            try:
                import matplotlib  # noqa: F401
            except ImportError as error:
                raise ValueError('Install scripts/requirements-report.txt or use --no-plots') from error
        config=dict(operation=args.operation,sizes=sizes,fixtures=fixtures,implementations=implementations,
                    block_sizes=blocks,repeats=args.repeats,seed=args.seed,seed_b=seed_b,atol=args.atol,rtol=args.rtol)
        args.output.mkdir(parents=True)
        rows,sources,block_rows=[],[],[]
        configurations=[(impl,block) for impl in implementations for block in (blocks if impl=='cuda-tree' else [256])]
        for fixture,length in itertools.product(fixtures,sizes):
            gpu=[]
            for impl,block in configurations:
                child=args.output/'runs'/f'{fixture}-{length}-{impl}-{block}'
                command=[str(executable),'compare','--operation',args.operation,'--size',str(length),'--input',fixture,
                         '--reference','cpu','--candidate',impl,'--block-size',str(block),'--repeats',str(args.repeats),
                         '--seed',str(args.seed),'--seed-b',str(seed_b),'--atol',str(args.atol),'--rtol',str(args.rtol),
                         '--save-output','--output',str(child)]
                result=subprocess.run(command,capture_output=True,text=True)
                meta=load_json(child/'metadata.json') if (child/'metadata.json').is_file() else {}
                if result.returncode not in (0,1,77) or meta.get('status') not in ('complete','tolerance_failed','skipped'):
                    raise ValueError('Native experiment failed: '+result.stderr[-1200:])
                sources.append(dict(directory=str(child.relative_to(args.output)),command=command,metadata=meta,
                                    metadata_sha256=digest(child/'metadata.json')))
                if meta['status']=='skipped':
                    continue
                sources[-1]['artifact_sha256']={name:digest(child/name) for name in ('scalar_results.csv','summary.csv','console.txt')}
                points=observations(child/'scalar_results.csv',args.repeats)
                for point in points:
                    point['source_directory']=str(child.relative_to(args.output))
                rows+=points
                run=read_run(child)
                saved=binary(run,run['rows'][0])
                if impl=='cuda-tree':
                    gpu.append((block,saved,points[0]['fp32_bits'],run['config']))
            all_same=len({g[2] for g in gpu})<=1
            for (ba,pa,_,ca),(bb,pb,_,_) in itertools.combinations(gpu,2):
                metrics=detailed(comparator,pa,pb,ca)
                block_rows.append(dict(operation=args.operation,fixture=fixture,length=length,seed=args.seed,
                    reference_block_size=ba,candidate_block_size=bb,selected_reference_block=blocks[0],
                    all_blocks_bitwise_equal=all_same,bitwise_equal=metrics['divergent_count']==0,
                    max_ulp=metrics['max_ulp'],absolute_error=metrics['max_absolute_error'],
                    relative_error=metrics['max_relative_error'],tolerance_failures=metrics['tolerance_failures']))
        if not rows:
            raise ValueError('No executed configurations; inspect skipped native runs')
        write_csv(args.output/'reproducibility_results.csv',rows)
        if block_rows:
            write_csv(args.output/'block_sensitivity.csv',block_rows)
        else:
            (args.output/'block_sensitivity.csv').write_text('operation,fixture,length,reference_block_size,candidate_block_size,max_ulp\n')
        summary=summarize(rows,block_rows)
        figures=[] if args.no_plots else render(rows,block_rows,args.output)
        (args.output/'report.md').write_text(report(rows,block_rows,summary,config,sources,figures),encoding='utf-8')
        metadata=dict(schema_version=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),artifact_type='reduction-reproducibility',status='complete',config=config,
                      summary=summary,sources=sources,selected_reference_block=blocks[0],
                      artifact_sha256={name:digest(args.output/name) for name in ("reproducibility_results.csv","block_sensitivity.csv","report.md",*figures)},
                      executable_sha256=digest(executable),comparator_sha256=digest(comparator),
                      script_sha256=digest(Path(__file__)),
                      helper_sha256={name:digest(Path(__file__).with_name(name)) for name in ("operations.py","compare_hardware.py")},
                      interpretation='FP64 is not exact; native FP32 diagnostics retained; repeated execution is distinct from agreement across orders')
        (args.output/'reproducibility_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        if any(s['metadata'].get('git_dirty') is not False for s in sources):
            print('WARNING: source/build dirty or unknown; commit alone cannot reproduce this suite.',file=sys.stderr)
        print(f"Reproducibility report: {args.output/'report.md'}; {summary['configurations']} configurations, "
              f"{summary['bitwise_deterministic']} bitwise deterministic, max ULP vs CPU={summary['max_ulp_vs_cpu']}")
        return 0
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
        print('error: '+str(error),file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
