"""Numerical experiment capture/replay tests, executable on CPU-only CI."""
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
import reproducibility as suite
import reproduce
from compare_hardware import read_run, binary, detailed

EXE=Path(sys.argv.pop(1))
COMPARATOR=Path(sys.argv.pop(1))
GPU="--gpu" in sys.argv
if GPU:
    sys.argv.remove("--gpu")


class ReproducibilityTest(unittest.TestCase):
    @unittest.skipUnless(GPU, 'GPU integration is separately labeled')
    def test_block_comparison_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'blocks'
            command=[str(EXE),'compare','--operation','reduction_sum','--size','257','--input','cancellation',
                '--reference','cuda-tree','--reference-block-size','64','--candidate','cuda-tree',
                '--block-size','512','--repeats','3','--save-output','--output',str(output)]
            result=subprocess.run(command,capture_output=True,text=True)
            if result.returncode==77:
                raise SystemExit(77)
            self.assertIn(result.returncode,(0,1),result.stdout+result.stderr)
            run=read_run(output)
            row=run['rows'][0]
            self.assertEqual(row['accumulation_mode'],'block_tree_512_multistage')
            self.assertEqual(row['reference_accumulation'],'block_tree_64_multistage')
            for field,block in (('output_file',512),('reference_output_file',64)):
                sidecar=json.loads((output/(row[field]+'.json')).read_text())
                self.assertEqual(sidecar['reduction_block_size'],block)
            replay=Path(temp)/'replay'
            result=subprocess.run(reproduce.command(run['metadata'],EXE,replay),capture_output=True,text=True)
            self.assertIn(result.returncode,(0,1),result.stderr)
            self.assertEqual((output/'scalar_results.csv').read_bytes(),(replay/'scalar_results.csv').read_bytes())
            self.assertTrue(all(r['determinism']=='bitwise_deterministic' for r in suite.observations(output/'scalar_results.csv',3)))

    def test_validation(self):
        for value in ('0','-1','64,64','96'):
            with self.assertRaises(ValueError):
                suite.choices(value,(64,128,256,512),integer=True)
        self.assertIsNone(suite.finite('nan'))
        self.assertIsNone(suite.finite(''))

    def test_capture_replay_and_report(self):
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/'suite'
            args=['--executable',str(EXE),'--comparator',str(COMPARATOR),'--output',str(dest),
                  '--implementations','cpu,cpu-reverse','--sizes','1,257','--repeats','3',
                  '--fixtures','cancellation,repeated_small_plus_large','--no-plots']
            self.assertEqual(suite.main(args),0)
            meta=json.loads((dest/'reproducibility_metadata.json').read_text())
            self.assertEqual(meta['summary']['configurations'],8)
            self.assertEqual(meta['summary']['bitwise_deterministic'],8)
            self.assertEqual(meta['summary']['varying'],0)
            with (dest/'reproducibility_results.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),24)
            self.assertTrue(any(float(r['absolute_error_fp64'])>0 for r in rows))
            self.assertTrue(all(r['deterministic_across_repeats']=='true' for r in rows))
            text=(dest/'report.md').read_text()
            self.assertIn('not a mathematically exact',text)
            self.assertIn('does not mean identical',text)
            child=dest/'runs'/'repeated_small_plus_large-257-cpu-reverse-256'
            native=json.loads((child/'metadata.json').read_text())
            self.assertEqual(native['config']['generator'],'vector-repeated_small_plus_large-v1')
            replay=Path(temp)/'replay'
            command=reproduce.command(native,EXE,replay)
            self.assertIn('--repeats',command)
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertIn(result.returncode,(0,1),result.stdout+result.stderr)
            self.assertEqual((child/'scalar_results.csv').read_bytes(),(replay/'scalar_results.csv').read_bytes())
            before=(dest/'reproducibility_results.csv').read_bytes()
            self.assertEqual(suite.main(args),2)
            self.assertEqual(before,(dest/'reproducibility_results.csv').read_bytes())
            a,b=read_run(child),read_run(replay)
            detail=detailed(COMPARATOR,binary(a,a['rows'][0]),binary(b,b['rows'][0]),a['config'])
            self.assertEqual(detail['divergent_count'],0)
            if importlib.util.find_spec('matplotlib'):
                self.assertEqual(len(suite.render(rows,[],dest)),2)
                self.assertTrue((dest/'ulp_drift.png').read_bytes().startswith(b'\x89PNG'))
            scalar=child/'scalar_results.csv'
            scalar.write_text('broken\n')
            with self.assertRaises(ValueError):
                suite.observations(scalar,3)

    def test_dot_fixture_and_repeat_bounds(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'dot'
            result=subprocess.run([str(EXE),'compare','--operation','dot','--size','257',
                '--input','cancellation','--candidate','cpu-reverse','--block-size','64',
                '--reference-block-size','128','--repeats','2','--output',str(path)],capture_output=True,text=True)
            self.assertIn(result.returncode,(0,1),result.stderr)
            rows=suite.observations(path/'scalar_results.csv',2)
            self.assertEqual(float(rows[0]['fp64_reference']),255)
            meta=json.loads((path/'metadata.json').read_text())
            command=reproduce.command(meta,EXE,Path(temp)/'again')
            self.assertEqual(command[command.index('--block-size')+1],'64')
            self.assertEqual(command[command.index('--reference-block-size')+1],'128')
            for flags in (['--repeats','0'],['--block-size','96']):
                bad=subprocess.run([str(EXE),'compare','--operation','dot',*flags],capture_output=True)
                self.assertEqual(bad.returncode,2)


if __name__=='__main__':
    unittest.main()
