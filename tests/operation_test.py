"""CPU-only vector artifact/replay/analysis integration; no GPU required."""
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
import compare_hardware
import operations
import report
import reproduce
import roofline
import reliability

EXE=Path(sys.argv.pop(1))
COMPARATOR=Path(sys.argv.pop(1))
GPU='--gpu' in sys.argv
if GPU:
    sys.argv.remove('--gpu')


class OperationsTest(unittest.TestCase):
    def test_models_and_identity(self):
        self.assertEqual(operations.model('dot',1,1,7),(14,60))
        self.assertEqual(operations.model('reduction_sum',1,1,7),(6,32))
        self.assertEqual(operations.identity({'reference':'naive','candidate':'tiled'}),'matmul')
        with self.assertRaises(ValueError):
            operations.identity({'candidate':'cuda-tree'})
        with self.assertRaises(ValueError):
            operations.identity({'operation':'dot'},dict(M=2,N=2,K=3,kernel='cuda-tree',length=3))
        h=roofline.ceilings(10,100)
        point=roofline.metrics(1,1,7,1,h,'dot')
        self.assertEqual(point['model_bytes'],60)
        self.assertAlmostEqual(point['achieved_gflops'],14/1e6)
        self.assertEqual(roofline.metrics(1,1,7,1,h,'reduction_sum')['flop_count'],6)
        with self.assertRaisesRegex(ValueError,'Zero nominal'):
            roofline.metrics(1,1,1,1,h,'reduction_sum')

    @unittest.skipUnless(GPU, 'GPU integration is separately labeled')
    def test_gpu_artifacts_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            for op in ('dot','reduction_sum'):
                run=Path(temp)/op
                args=[str(EXE),'benchmark','--operation',op,'--sizes','3,257',
                      '--iterations','2','--warmups','1','--trials','2','--bootstrap-samples','20',
                      '--atol','0.01','--rtol','0.001','--save-output','--output',str(run)]
                result=subprocess.run(args,capture_output=True,text=True)
                if result.returncode==77:
                    raise SystemExit(77)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                meta,rows,_=report.load_report(run)
                self.assertEqual(len(rows),4)
                data=reliability.load(run,rows)
                self.assertEqual(len(data['pairs']),2)
                gpu=[r for r in rows if r['kernel']=='cuda-tree']
                self.assertEqual([r['source_fields']['reduction_stages'] for r in gpu],['1','2'])
                self.assertEqual([r['source_fields']['flop_count'] for r in gpu],
                                 [str(operations.model(op,1,1,k)[0]) for k in (3,257)])
                replay=Path(temp)/(op+'-replay')
                result=subprocess.run(reproduce.command(meta,EXE,replay),capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                a,b=compare_hardware.read_run(run),compare_hardware.read_run(replay)
                _,numerics=compare_hardware.aggregate([a,b],a,COMPARATOR)
                self.assertTrue(all(r['status']=='bitwise_identical' for r in numerics))

    def test_cpu_artifacts_replay_and_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for op in ('dot','reduction_sum'):
                run=root/op
                result=subprocess.run([str(EXE),'benchmark','--operation',op,'--sizes','3,257',
                    '--reference','cpu','--candidate','cpu','--warmups','1','--iterations','3',
                    '--trials','2','--bootstrap-samples','20','--seed','42','--save-output','--output',str(run)],
                    capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                meta,rows,warnings=report.load_report(run)
                self.assertEqual(meta['config']['operation'],op)
                self.assertEqual(len(rows),2)
                self.assertTrue(all(r['operation']==op and r['shape'][:2]==(1,1) for r in rows))
                self.assertEqual(rows[0]['source_fields']['accumulation_mode'],'serial_increasing_index')
                self.assertEqual(rows[0]['source_fields']['model_bytes'],str(operations.model(op,1,1,3)[1]))
                data=reliability.load(run,rows)
                self.assertEqual(len(data['groups']),2)
                replay=root/(op+'-replay')
                command=reproduce.command(meta,EXE,replay)
                self.assertIn('--operation',command)
                result=subprocess.run(command,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                a,b=compare_hardware.read_run(run),compare_hardware.read_run(replay)
                _,num=compare_hardware.aggregate([a,b],a,COMPARATOR)
                self.assertTrue(all(r['status']=='bitwise_identical' for r in num))
                # Native comparator works on the scalar payload without Python FP emulation.
                first=a['rows'][0]
                path=compare_hardware.binary(a,first)
                self.assertEqual(path.stat().st_size,28)
                self.assertEqual(compare_hardware.detailed(COMPARATOR,path,path,a['config'])['max_ulp'],0)
                # Different operations must not be numerically eligible, even same scalar shape/kernel.
                b['config']['operation']='dot' if op=='reduction_sum' else 'reduction_sum'
                self.assertFalse(compare_hardware.compatibility(b,a,b['rows'][0],first)[0])
                if importlib.util.find_spec('matplotlib'):
                    self.assertEqual(report.main([str(run)]),0)
                    text=(run/'report'/'report.md').read_text()
                    self.assertIn('Vector implementation',text)
                    self.assertTrue((run/'report'/'latency.png').is_file())
                # Convert only this synthetic fixture to a GPU row for model serialization.
                fields=list(first)
                raw=dict(first,kernel='cuda-tree',backend='cuda',timing_mode='kernel_only',
                         reduction_block_size='256',reduction_stages='1')
                with (run/'summary.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=fields)
                    writer.writeheader();writer.writerow(raw)
                _,points,excluded,_=roofline.prepare(run,roofline.ceilings(10,100))
                self.assertEqual(points[0]['operation'],op)
                self.assertEqual(points[0]['model_bytes'],operations.model(op,1,1,3)[1])
                self.assertEqual(excluded,[])


if __name__=='__main__':
    unittest.main()
