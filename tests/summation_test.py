"""Native summation capture, attribution, replay, plotting and CUDA pairing."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import summation_analysis as study
import reproduce
from compare_hardware import read_run, binary, detailed, digest

EXE=Path(sys.argv.pop(1))
COMPARATOR=Path(sys.argv.pop(1))
GPU='--gpu' in sys.argv
if GPU:
    sys.argv.remove('--gpu')


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


class SummationTests(unittest.TestCase):
    def test_native_metrics_replay_and_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/'study'
            args=['--executable',str(EXE),'--comparator',str(COMPARATOR),'--output',str(dest),
                  '--sizes','1,257','--fixtures','random_uniform,cancellation,large_dynamic_range','--no-plots']
            self.assertEqual(study.main(args),0)
            rows=csv_rows(dest/'summation_results.csv')
            self.assertEqual(len(rows),36)
            meta=json.loads((dest/'summation_metadata.json').read_text())
            self.assertEqual(meta['artifact_sha256']['summation_results.csv'],digest(dest/'summation_results.csv'))
            summaries=csv_rows(dest/'summation_summary.csv')
            target=next(r for r in summaries if r['scope']=='overall' and r['method']=='fp64_accumulation')
            self.assertEqual(float(target['exact_match_percent']),100)
            exact=next(r for r in rows if r['method']=='neumaier_fp32' and r['fixture']=='cancellation' and r['length']=='257')
            self.assertEqual(exact['improvement_classification'],'exact_match')
            self.assertEqual(exact['factor_status'],'eliminated_error')
            self.assertEqual(exact['error_reduction_factor'],'inf')
            zeros=[r for r in rows if r['length']=='1' and r['fixture']=='cancellation']
            self.assertTrue(all(r['factor_status']=='both_zero' for r in zeros))
            self.assertTrue(all(r['error_reduction_factor']=='1' for r in zeros))
            child=dest/exact['source_directory']
            native=json.loads((child/'metadata.json').read_text())
            replay=Path(temp)/'replay'
            result=subprocess.run(reproduce.command(native,EXE,replay),capture_output=True,text=True)
            self.assertIn(result.returncode,(0,1),result.stdout+result.stderr)
            self.assertEqual((child/'scalar_results.csv').read_bytes(),(replay/'scalar_results.csv').read_bytes())
            a,b=read_run(child),read_run(replay)
            self.assertEqual(detailed(COMPARATOR,binary(a,a['rows'][0]),binary(b,b['rows'][0]),a['config'])['divergent_count'],0)
            if importlib.util.find_spec('matplotlib'):
                figures=study.render(rows,meta['summary'],dest)
                self.assertEqual(len(figures),3)
                self.assertTrue(all((dest/f).read_bytes().startswith(b'\x89PNG') for f in figures))
            before=(dest/'summation_results.csv').read_bytes()
            self.assertEqual(study.main(args),2)
            self.assertEqual(before,(dest/'summation_results.csv').read_bytes())
            subset=Path(temp)/'subset'
            self.assertEqual(study.main(['--executable',str(EXE),'--output',str(subset),
                '--sizes','5','--fixtures','alternating_sign','--methods','kahan_fp32','--no-plots']),0)
            self.assertEqual({r['method'] for r in csv_rows(subset/'summation_results.csv')},{'fp32_forward','kahan_fp32'})

    def test_attribution_independent_of_selected_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/'run'
            result=subprocess.run([str(EXE),'compare','--operation','reduction_sum','--size','257',
                '--input','repeated_small_plus_large','--reference','fp32_reverse','--candidate','neumaier_fp32',
                '--output',str(dest)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            row=csv_rows(dest/'scalar_results.csv')[0]
            self.assertNotEqual(row['fp32_forward'],row['fp32_baseline'])
            self.assertEqual(float(row['forward_absolute_error_fp64']),256)
            self.assertEqual(row['factor_status'],'eliminated_error')
            # Nominal CPU benchmark support and timers still serialize correctly.
            bench=Path(temp)/'bench'
            result=subprocess.run([str(EXE),'benchmark','--operation','dot','--size','17',
                '--reference','fp32_forward','--candidate','kahan_fp32','--iterations','2',
                '--warmups','0','--bootstrap-samples','0','--output',str(bench)],capture_output=True,text=True)
            self.assertIn(result.returncode,(0,1),result.stderr)
            rows=csv_rows(bench/'summary.csv')
            self.assertEqual([r['backend'] for r in rows],['cpu','cpu'])
            self.assertEqual([r['timing_mode'] for r in rows],['host_serial','host_serial'])

    @unittest.skipUnless(GPU,'CUDA integration is separately labeled')
    def test_cuda_pairing_and_dot(self):
        with tempfile.TemporaryDirectory() as temp:
            for operation in ('reduction_sum','dot'):
                dest=Path(temp)/operation
                self.assertEqual(study.main(['--executable',str(EXE),'--comparator',str(COMPARATOR),
                    '--operation',operation,'--sizes','257','--fixtures','cancellation','--cuda',
                    '--methods','fp32_forward','--block-sizes','64,256','--no-plots','--output',str(dest)]),0)
                meta=json.loads((dest/'summation_metadata.json').read_text())
                if any(s['metadata'].get('status')=='skipped' for s in meta['sources']):
                    raise SystemExit(77)
                pairs=csv_rows(dest/'summation_cuda_comparisons.csv')
                self.assertEqual(len(pairs),10)
                self.assertEqual({r['reference_method'] for r in pairs},
                    {'fp32_forward','fp32_pairwise','kahan_fp32','neumaier_fp32','fp64_rounded_to_fp32'})
                self.assertEqual({r['block_size'] for r in pairs},{'64','256'})


if __name__=='__main__':
    unittest.main()
