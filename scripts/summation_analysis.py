#!/usr/bin/env python3
"""Analyze stable summation against forward FP32 and a rounded FP64 analysis target."""
import argparse
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

from compare_hardware import read_run, binary, detailed, digest, load_json, write_csv
from operations import SUMMATION_METHODS, VECTOR_FIXTURES
from reproducibility import choices, observations, finite


def label(row):
    return row['method'] + (' / block ' + row['block_size'] if row['block_size'] else '')


def summarize(rows):
    groups={}
    for row in rows:
        for fixture in ('all',row['fixture']):
            groups.setdefault((fixture,row['method'],row['block_size']),[]).append(row)
    result=[]
    for (fixture,method,block),group in groups.items():
        eligible=[r for r in group if all(finite(r[k]) is not None for k in
                  ('fp32_result','fp64_reference','fp64_rounded_to_fp32'))]
        matches=sum(r['rounded_target_bitwise_match']=='true' for r in eligible)
        def maximum(field):
            values=[finite(r[field]) for r in group if finite(r[field]) is not None]
            return max(values) if values else None
        result.append(dict(scope='overall' if fixture=='all' else 'fixture',fixture=fixture,
            method=method,block_size=block,cases=len(group),eligible_cases=len(eligible),exact_matches=matches,
            exact_match_percent=100*matches/len(eligible) if eligible else None,
            max_absolute_error_fp64=maximum('absolute_error_fp64'),max_ulp_to_target=maximum('ulp_to_rounded_target'),
            max_absolute_error_reduction=maximum('absolute_error_reduction'),
            max_finite_reduction_factor=maximum('error_reduction_factor'),
            eliminated_error_count=sum(r['factor_status']=='eliminated_error' for r in group),
            zero_baseline_count=sum(r['factor_status'] in ('both_zero','zero_baseline') for r in group),
            improved_count=sum(r['improvement_classification']=='improved' for r in group),
            worsened_count=sum(r['improvement_classification']=='worsened' for r in group),
            no_improvement_count=sum(r['improvement_classification']=='no_improvement' for r in group),
            nonpositive_error_reduction_count=sum(finite(r['absolute_error_reduction']) is not None and
                                                float(r['absolute_error_reduction'])<=0 for r in group),
            unavailable_count=sum(r['improvement_classification']=='unavailable' for r in group)))
    return result


def render(rows, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fixtures=list(dict.fromkeys(r['fixture'] for r in rows))
    figures=[]
    for field,name,title in (('absolute_error_fp64','absolute_error.png','Absolute error vs FP64 accumulation'),
                             ('error_reduction_factor','error_reduction.png','Error-reduction factor vs forward FP32')):
        cols=min(2,len(fixtures)); panels=(len(fixtures)+cols-1)//cols
        fig,axes=plt.subplots(panels,cols,figsize=(12,4*panels),squeeze=False)
        for ax,fixture in zip(axes.flat,fixtures):
            groups={}
            for row in rows:
                if row['fixture']==fixture:
                    groups.setdefault(label(row),[]).append(row)
            for name_,group in groups.items():
                group=sorted(group,key=lambda r:int(r['length']))
                suffix=''
                if field=='error_reduction_factor':
                    count=sum(r['factor_status']=='eliminated_error' for r in group)
                    if count:
                        suffix=f' (∞: {count}, omitted)'
                ax.plot([int(r['length']) for r in group],
                        [finite(r[field]) if finite(r[field]) is not None else math.nan for r in group],
                        marker='o',label=name_+suffix)
            ax.set(xlabel='Vector length (elements)',ylabel=title,title=fixture)
            ax.set_xscale('log'); ax.set_yscale('symlog',linthresh=1 if field=='error_reduction_factor' else 1e-7)
            ax.set_ylim(bottom=0)
            if field=='error_reduction_factor':
                ax.axhline(1,color='black',linestyle=':',linewidth=.8)
            ax.legend(fontsize=6); ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(fixtures):]:
            ax.set_visible(False)
        fig.tight_layout();fig.savefig(output/name,dpi=150);plt.close(fig);figures.append(name)
    overall=[r for r in summary if r['scope']=='overall']
    fig,ax=plt.subplots(figsize=(11,max(4,len(overall)*.5)))
    labels=[label(r)+(' (target check)' if r['method']=='fp64_accumulation' else '') for r in overall]
    values=[r['exact_match_percent'] if r['exact_match_percent'] is not None else math.nan for r in overall]
    ax.barh(labels,values)
    for i,row in enumerate(overall):
        ax.text((values[i] if math.isfinite(values[i]) else 0)+1,i,
                f"{row['exact_matches']}/{row['eligible_cases']}" if row['eligible_cases'] else "unavailable",va='center',fontsize=8)
    ax.set(xlabel='Bitwise matches to fp64_rounded_to_fp32 (%)',xlim=(0,115),title='Rounded-target match rate over tested finite cases')
    fig.tight_layout();fig.savefig(output/'exact_match_rate.png',dpi=150);plt.close(fig)
    figures.append('exact_match_rate.png')
    return figures


def report(rows, summary, pairs, sources, config, figures):
    def fmt(value):
        return 'unavailable' if value is None else f'{value:.7g}'
    overall=[r for r in summary if r['scope']=='overall']
    lines=['# Stable summation and error attribution','',
        f"Operation: `{config['operation']}`; seed {config['seed']}, B seed {config['seed_b']}. "
        f"Lengths: {config['sizes']}. Fixtures: {', '.join(config['fixtures'])}.",'',
        'Improvement baseline: **fp32_forward**. FP64 accumulation is an analysis reference, not an exact mathematical sum. '
        'All method outputs use FP32 storage; fp64_accumulation casts once at the output boundary. '
        '**exact_match** means identical bits to `fp64_rounded_to_fp32`, not zero error versus FP64. '
        'The FP64 method’s 100% target match is a construction check, not evidence that a summation algorithm is universally best.','',
        'Signed error is result minus FP64 reference. Absolute-error reduction is baseline error minus method error. '
        'The factor is baseline error / method error: both zero → 1; zero baseline with nonzero method error → 0; '
        'nonzero baseline with zero method error → infinity. Relative error is omitted for |FP64 reference| ≤ 1e-12 or nonfinite values. '
        'Nonfinite target comparisons are excluded from match-rate denominators.','',
        '| Method / block | Target matches | Largest abs. error | Largest error reduction | Largest finite factor | Error eliminated | No positive reduction |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in overall:
        lines.append(f"| {label(r)} | {r['exact_matches']}/{r['eligible_cases']} ({fmt(r['exact_match_percent'])}%) | "
            f"{fmt(r['max_absolute_error_fp64'])} | {fmt(r['max_absolute_error_reduction'])} | "
            f"{fmt(r['max_finite_reduction_factor'])} | {r['eliminated_error_count']} | {r['nonpositive_error_reduction_count']} |")
    lines+=['','## Observations','']
    for method in ('fp32_forward','cuda-tree','kahan_fp32','neumaier_fp32'):
        group=[r for r in rows if r['method']==method and finite(r['absolute_error_fp64']) is not None]
        if not group:
            continue
        worst=max(group,key=lambda r:float(r['absolute_error_fp64']))
        lines.append(f"- {method}: largest absolute error {worst['absolute_error_fp64']} on {worst['fixture']}, length {worst['length']}, block {worst['block_size'] or 'n/a'}.")
        if method in ('kahan_fp32','neumaier_fp32'):
            improvements=[r for r in group if finite(r['absolute_error_reduction']) is not None]
            if not improvements:
                lines.append(f'- {method}: improvement unavailable against a nonfinite baseline.')
                continue
            best=max(improvements,key=lambda r:float(r['absolute_error_reduction']))
            lines.append(f"- {method}: strongest absolute-error reduction {best['absolute_error_reduction']} on {best['fixture']}, length {best['length']}; factor {best['error_reduction_factor']} ({best['factor_status']}).")
            nohelp=[r for r in improvements if float(r['absolute_error_reduction'])<=0]
            descriptions=[f"{r['fixture']} L={r['length']} ({r['improvement_classification']})" for r in nohelp[:5]]
            lines.append(f"- {method}: {len(nohelp)} cases without positive error reduction. Examples: "+('; '.join(descriptions) or 'none observed')+'.')
    if pairs:
        lines+=['','## CUDA comparisons','',
            'The CUDA tree is unchanged. These first-output comparisons use the existing native FP32 comparator; ULPs here are to the named CPU method.', '',
            '| CPU reference | Matching CUDA pairs / tested | Largest pair ULP |', '|---|---:|---:|']
        for method in dict.fromkeys(r['reference_method'] for r in pairs):
            group=[r for r in pairs if r['reference_method']==method]
            lines.append(f"| {method} | {sum(r['divergent_count']==0 for r in group)}/{len(group)} | {max(r['max_ulp'] for r in group)} |")
    lines+=['','## Fixture dependence','', '| Fixture | Method / block | Rounded-target matches |', '|---|---|---:|']
    for r in [r for r in summary if r['scope']=='fixture']:
        lines.append(f"| {r['fixture']} | {label(r)} | {r['exact_matches']}/{r['eligible_cases']} |")
    lines+=['', 'No method is claimed universally best. Compensated FP32 can still lose information, and FP32 output rounding can leave a nonzero error even on an exact target match. '
        'For dot, pairwise/Kahan/Neumaier sum separately rounded FP32 products; forward/reverse retain compiler-dependent contraction. '
        'The FP64 reference multiplies promoted inputs. Dot errors therefore include product rounding and cannot all be attributed to summation order.', '',
        'Plots use measured numerical errors, not timings. Infinite factors are omitted from numeric plot coordinates and counted in legends; they remain `inf` in CSV. '
        'Connected sizes do not imply monotonic error growth.','']
    for name in figures:
        lines+=['!['+name.removesuffix('.png').replace('_',' ')+']('+name+')','']
    meta=next((s['metadata'] for s in sources if s['metadata'].get('gpu_name') not in (None,'unknown','')),sources[0]['metadata'])
    lines += [f"Hardware: {meta.get('gpu_name','unknown')} / {meta.get('cpu_model','unknown')}. "
              f"Compiler: {meta.get('cxx_compiler','unknown')}; CUDA: {meta.get('cuda_compiler','unknown')}.",
              'Stable-method compilation policy: '+str(meta.get('summation_fp_policy','unknown'))+'.',
              f"Commit: {meta.get('git_commit','unknown')}; dirty: {meta.get('git_dirty','unknown')}.", '',
              'Full data: [results](summation_results.csv), [summary](summation_summary.csv), '
              '[CUDA comparisons](summation_cuda_comparisons.csv), [provenance](summation_metadata.json). '
              'Child runs retain replayable native artifacts.']
    skipped=sum(s['metadata'].get('status')=='skipped' for s in sources)
    if skipped:
        lines+=['',f'{skipped} requested CUDA configurations were skipped; consult source metadata for runtime/device diagnostics.']
    if any(s['metadata'].get('git_dirty') is not False for s in sources):
        lines+=['','**Dirty/unknown source provenance:** the commit alone cannot reproduce this capture.']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation',choices=('reduction_sum','dot'),default='reduction_sum')
    parser.add_argument('--sizes',default='257,1024,100003')
    parser.add_argument('--fixtures',default='random_uniform,cancellation,large_dynamic_range')
    parser.add_argument('--methods',default=','.join(SUMMATION_METHODS))
    parser.add_argument('--cuda',action='store_true',help='include unchanged CUDA trees and the required CPU comparison targets')
    parser.add_argument('--block-sizes',default='64,128,256,512')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--seed-b',type=int)
    parser.add_argument('--executable',type=Path,default=Path('build/matmul-inspector'))
    parser.add_argument('--comparator',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args(argv)
    try:
        sizes=choices(args.sizes,integer=True)
        fixtures=choices(args.fixtures,VECTOR_FIXTURES)
        requested=choices(args.methods,SUMMATION_METHODS)
        blocks=choices(args.block_sizes,(64,128,256,512),integer=True)
        if not 0<=args.seed<=0xffffffff or (args.seed_b is not None and not 0<=args.seed_b<=0xffffffff):
            raise ValueError('Seeds must be uint32')
        seed_b=args.seed_b if args.seed_b is not None else (args.seed+81)&0xffffffff
        targets=['fp32_forward','fp32_pairwise','kahan_fp32','neumaier_fp32','fp64_accumulation']
        methods=list(dict.fromkeys(['fp32_forward',*requested,*(targets if args.cuda else [])]))
        executable=args.executable.resolve()
        comparator=(args.comparator or executable.with_name('matmul-compare-outputs')).resolve()
        if not executable.is_file() or (args.cuda and not comparator.is_file()):
            raise ValueError('Build the native executable/comparator first')
        if args.output.exists():
            raise ValueError('Output directory already exists')
        if not args.no_plots:
            try:
                import matplotlib  # noqa: F401
            except ImportError as error:
                raise ValueError('Install scripts/requirements-report.txt or use --no-plots') from error
        config=dict(operation=args.operation,sizes=sizes,fixtures=fixtures,requested_methods=requested,methods=methods,
                    cuda=args.cuda,block_sizes=blocks,seed=args.seed,seed_b=seed_b,baseline='fp32_forward')
        args.output.mkdir(parents=True)
        rows,sources,pairs=[],[],[]
        for fixture,length in itertools.product(fixtures,sizes):
            outputs={}
            configurations=[(m,256) for m in methods]+([('cuda-tree',b) for b in blocks] if args.cuda else [])
            for method,block in configurations:
                child=args.output/'runs'/f'{fixture}-{length}-{method}-{block}'
                command=[str(executable),'compare','--operation',args.operation,'--size',str(length),'--input',fixture,
                    '--reference','fp32_forward','--candidate',method,'--block-size',str(block),'--seed',str(args.seed),
                    '--seed-b',str(seed_b),'--save-output','--output',str(child)]
                result=subprocess.run(command,capture_output=True,text=True)
                meta=load_json(child/'metadata.json') if (child/'metadata.json').is_file() else {}
                if result.returncode not in (0,1,77) or meta.get('status') not in ('complete','tolerance_failed','skipped'):
                    raise ValueError('Native summation experiment failed: '+result.stderr[-1200:])
                sources.append(dict(directory=str(child.relative_to(args.output)),command=command,metadata=meta,
                                    metadata_sha256=digest(child/'metadata.json')))
                if meta['status']=='skipped':
                    continue
                row=observations(child/'scalar_results.csv',1)[0]
                if 'improvement_classification' not in row:
                    raise ValueError('Native executable lacks summation analysis fields; rebuild it')
                row.update(method=method,size=length,source_directory=str(child.relative_to(args.output)))
                for field in ('git_commit','git_dirty','build_type','cxx_compiler','cuda_compiler','gpu_name','cpu_model'):
                    row[field]=meta.get(field,'unknown')
                rows.append(row)
                sources[-1]['artifact_sha256']={name:digest(child/name) for name in ('scalar_results.csv','summary.csv','console.txt')}
                run=read_run(child)
                outputs[(method,block)]=(binary(run,run['rows'][0]),run['config'])
            for block in blocks if args.cuda else []:
                candidate=outputs.get(('cuda-tree',block))
                if not candidate:
                    continue
                for method in targets:
                    ref=outputs[(method,256)]
                    metrics=detailed(comparator,ref[0],candidate[0],ref[1])
                    pairs.append(dict(operation=args.operation,fixture=fixture,length=length,seed=args.seed,
                        reference_method='fp64_rounded_to_fp32' if method=='fp64_accumulation' else method,
                        candidate_method='cuda-tree',block_size=block,**metrics))
        summary=summarize(rows)
        write_csv(args.output/'summation_results.csv',rows)
        write_csv(args.output/'summation_summary.csv',summary)
        if pairs:
            write_csv(args.output/'summation_cuda_comparisons.csv',pairs)
        else:
            (args.output/'summation_cuda_comparisons.csv').write_text('operation,fixture,length,reference_method,candidate_method,block_size\n')
        figures=[] if args.no_plots else render(rows,summary,args.output)
        (args.output/'report.md').write_text(report(rows,summary,pairs,sources,config,figures),encoding='utf-8')
        metadata=dict(schema_version=1,artifact_type='stable-summation-analysis',timestamp_utc=datetime.now(timezone.utc).isoformat(),
            status='complete',config=config,sources=sources,summary=summary,
            executable_sha256=digest(executable),comparator_sha256=digest(comparator) if args.cuda else None,
            script_sha256=digest(Path(__file__)),
            helper_sha256={n:digest(Path(__file__).with_name(n)) for n in ('reproducibility.py','operations.py','compare_hardware.py')},
            artifact_sha256={n:digest(args.output/n) for n in ('summation_results.csv','summation_summary.csv','summation_cuda_comparisons.csv','report.md',*figures)},
            interpretation='FP64 is an analysis reference, not exact. Exact match means rounded FP32 target bits. Infinity factors are CSV inf, with eliminated_error status; nonfinite metrics are unavailable.')
        (args.output/'summation_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        if any(s['metadata'].get('git_dirty') is not False for s in sources):
            print('WARNING: dirty/unknown source; commit alone cannot reproduce this capture.',file=sys.stderr)
        print(f"Summation report: {args.output/'report.md'} ({len(rows)} results, {len(pairs)} CUDA comparisons)")
        return 0
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
        print('error: '+str(error),file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
