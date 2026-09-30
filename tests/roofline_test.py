"""Roofline arithmetic, eligibility, provenance, and optional rendering; no GPU."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import roofline


class RooflineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name)/'run'
        self.run.mkdir()
        self.hardware=roofline.ceilings(10,100)

    def write(self, rows=None, mode='benchmark', schema=3):
        metadata=dict(schema_version=schema,status='complete',git_commit='synthetic-test',git_dirty=False,
                      gpu_name='Synthetic GPU',timing_methodology='CUDA events; transfers excluded',
                      config=dict(mode=mode,dtype='float32',seed=42))
        (self.run/'metadata.json').write_text(json.dumps(metadata))
        rows=rows if rows is not None else [dict(M=4,N=4,K=4,kernel='tiled',reference_kernel='naive',
                    backend='cuda',timing_mode='kernel_only',median_ms=1,mean_ms=2,tile_size=16,
                    tolerance_pass='false',divergent_count=1,output_sha256='a'*64,stability_warnings='test_noise')]
        with (self.run/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader();writer.writerows(rows)

    def test_rectangular_flops_bytes_units_and_efficiency(self):
        p=roofline.metrics(2,3,4,2,self.hardware)
        self.assertEqual(p['flop_count'],48)
        self.assertEqual(p['model_bytes'],104)  # 8 A + 12 B + 6 C floats
        self.assertAlmostEqual(p['arithmetic_intensity_flop_per_byte'],48/104)
        self.assertEqual(p['achieved_flops'],24000)
        self.assertAlmostEqual(p['achieved_gflops'],.000024)
        self.assertEqual(p['peak_compute_gflops'],10000)
        self.assertEqual(p['ridge_point_flop_per_byte'],100)
        self.assertAlmostEqual(p['memory_roof_gflops'],100*48/104)
        self.assertEqual(p['roofline_bound'],'memory-bound')
        self.assertAlmostEqual(p['efficiency_vs_attainable'],.000024/(100*48/104))
        self.assertAlmostEqual(p['efficiency_vs_compute_peak'],.000024/10000)
        self.assertAlmostEqual(p['model_bandwidth_gbps'],104/2e6)
        self.assertAlmostEqual(p['model_bandwidth_peak_percent'],(104/2e6)/100*100)

    def test_ridge_and_compute_bound(self):
        # Square AI = N/6; ridge is 100 FLOP/byte for these explicit limits.
        for n in (600,1200):
            p=roofline.metrics(n,n,n,100,self.hardware)
            self.assertEqual(p['roofline_bound'],'compute-bound')
            self.assertEqual(p['attainable_gflops'],10000)
        self.assertEqual(roofline.metrics(599,599,599,100,self.hardware)['roofline_bound'],'memory-bound')

    def test_exceeding_roofs_retains_raw_values(self):
        hardware=roofline.ceilings(.000001,.001)
        p=roofline.metrics(4,4,4,.001,hardware)
        self.assertGreater(p['efficiency_vs_compute_peak'],1)
        self.assertGreater(p['efficiency_vs_attainable'],1)
        self.assertGreater(p['model_bandwidth_peak_percent'],100)
        self.assertAlmostEqual(p['achieved_gflops'],.128)
        self.assertIn('exceeds_attainable_roof',p['roofline_warnings'])
        self.assertIn('exceeds_configured_compute_peak',p['roofline_warnings'])

    def test_invalid_ceiling_and_shape_inputs(self):
        for value in (0,-1,'unknown','nan','inf',None,True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    roofline.ceilings(value,100)
                with self.assertRaises(ValueError):
                    roofline.ceilings(10,value)
        with self.assertRaises(ValueError):
            roofline.ceilings(10,100,compute_source='guessed')
        for shape in ((0,3,4),(1.5,3,4),(True,3,4)):
            with self.assertRaises(ValueError):
                roofline.metrics(*shape,1,self.hardware)
        for latency in (0,-1,float('nan')):
            with self.assertRaises(ValueError):
                roofline.metrics(2,3,4,latency,self.hardware)
        with self.assertRaises(SystemExit):
            roofline.main(['--input',str(self.run)])

    def test_legacy_inference_and_explicit_mean(self):
        self.write([dict(M=4,N=4,K=4,kernel='cuda-naive-no-fma',median_ms=1,mean_ms=2)],schema=1)
        _,points,_,warnings=roofline.prepare(self.run,self.hardware)
        self.assertTrue(points[0]['timing_mode_inferred'])
        self.assertTrue(any('inferred' in w for w in warnings))
        _,mean,_,_=roofline.prepare(self.run,self.hardware,'mean')
        self.assertEqual(points[0]['achieved_gflops'],2*mean[0]['achieved_gflops'])
        self.write([dict(M=4,N=4,K=4,kernel='naive',mean_ms=2)])
        with self.assertRaisesRegex(ValueError,'No usable'):
            roofline.prepare(self.run,self.hardware)
        self.assertEqual(len(roofline.prepare(self.run,self.hardware,'mean')[1]),1)

    def test_crossover_timing_modes_never_mixed(self):
        rows=[dict(M=4,N=4,K=4,kernel=kernel,timing_mode=mode,median_ms=1)
              for kernel,mode in [('cpu','host_matmul'),('tiled','kernel_only'),('tiled','end_to_end')]]
        self.write(rows,mode='crossover')
        _,points,excluded,_=roofline.prepare(self.run,self.hardware)
        self.assertEqual(len(points),1)
        self.assertEqual(points[0]['timing_mode'],'kernel_only')
        self.assertEqual(len(excluded),2)
        self.write([rows[-1]],mode='crossover')
        with self.assertRaisesRegex(ValueError,'No usable'):
            roofline.prepare(self.run,self.hardware)

    def test_artifacts_provenance_correctness_and_no_overwrite(self):
        self.write()
        (self.run/'mismatches.csv').write_text('row,col\n0,0\n')
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.run.iterdir()}
        output=Path(self.temp.name)/'analysis'
        args=['--input',str(self.run),'--peak-fp32-tflops','10','--memory-bandwidth-gbps','100',
              '--source-note','Synthetic limits for testing','--output',str(output)]
        if importlib.util.find_spec('matplotlib'):
            args.append('--report')
            from matplotlib.axes import Axes
            original=Axes.scatter
            plotted=[]
            def record(ax,x,y,*args,**kwargs):
                plotted.append((list(x),list(y)))
                return original(ax,x,y,*args,**kwargs)
            with patch.object(Axes,'scatter',record):
                self.assertEqual(roofline.main(args),0)
            self.assertIn(([2/3],[128/1e6]),plotted)
            self.assertTrue((output/'roofline.png').read_bytes().startswith(b'\x89PNG'))
            text=(output/'report.md').read_text()
            self.assertIn('numerical tolerance failed',text)
            self.assertIn('not measured DRAM',text)
        else:
            self.assertEqual(roofline.main(args),0)
        metadata=json.loads((output/'roofline_metadata.json').read_text())
        self.assertEqual(metadata['source_artifact_sha256'],before)
        self.assertIsNone(metadata['measured_memory_traffic'])
        self.assertEqual(metadata['ceilings']['peak_fp32_tflops'],10)
        with (output/'roofline.csv').open() as stream:
            row=list(csv.DictReader(stream))[0]
        self.assertEqual(row['output_sha256'],'a'*64)
        self.assertEqual(row['tolerance_pass'],'false')
        self.assertEqual(row['stability_warnings'],'test_noise')
        self.assertEqual(row['tile_size'],'16')
        self.assertEqual(roofline.main(args),2)
        self.assertTrue(all(hashlib.sha256((self.run/name).read_bytes()).hexdigest()==sha for name,sha in before.items()))


if __name__=='__main__':
    unittest.main()
