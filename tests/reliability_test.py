"""Statistics artifact and report integration, CPU-only with synthetic timings."""
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import reliability
import report
import reproduce

FIXTURE=sys.argv[1]
sys.argv=sys.argv[:1]


class ReliabilityTests(unittest.TestCase):
    def test_roundtrip_statistics_and_plotting(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)/'run'
            subprocess.run([FIXTURE,str(run)],check=True)
            meta,summary,warnings=report.load_report(run)
            data=reliability.load(run,summary)
            self.assertEqual(len(data['groups']),2)
            for key,points in data['groups'].items():
                self.assertEqual(len(points),12)
                self.assertEqual({p['trial'] for p in points},{0,1})
                values=[p['latency_ms'] for p in points if p['phase']=='measurement']
                s=data['overall'][key]
                self.assertEqual(s['median_ms'],statistics.median(values))
                self.assertEqual(s['mean_ms'],statistics.mean(values))
                self.assertAlmostEqual(s['stddev_ms'],statistics.pstdev(values))
                self.assertEqual(len(data['trials'][key]),2)
                self.assertEqual(s['bootstrap_method'],'whole_trial')
                self.assertIn('p95_ms',s)
            pair=data['pairs'][0]
            self.assertEqual(float(pair['speedup']),2)
            self.assertEqual(float(pair['latency_ratio']),.5)
            self.assertEqual(float(pair['percent_difference']),-50)
            self.assertEqual(pair['median_ci_overlap'],'false')
            command=reproduce.command(meta,'binary','other')
            for flag in ('--trials','--bootstrap-samples','--confidence-level','--bootstrap-seed','--percentiles'):
                self.assertIn(flag,command)
            if importlib.util.find_spec('matplotlib'):
                before={p:p.read_bytes() for p in run.iterdir() if p.is_file()}
                self.assertEqual(report.main([str(run)]),0)
                self.assertEqual(len(list((run/'report').glob('reliability*.png'))),3)
                text=(run/'report'/'report.md').read_text()
                self.assertIn('not a confidence claim',text)
                self.assertIn('not correctness failures',text)
                self.assertTrue(all(p.read_bytes()==v for p,v in before.items()))
            # A bad sequence must not be graphed as a valid chronological trace.
            raw=run/'timing_samples.csv'
            with raw.open() as stream:
                captured=list(csv.DictReader(stream)); fields=list(captured[0])
            captured[0]['iteration']='10'
            with raw.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(captured)
            with self.assertRaises(ValueError):
                reliability.load(run,summary)

    def test_legacy_and_empty_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(reliability.load(directory,[]))
            (Path(directory)/'timing_samples.csv').write_text('row_id,phase,latency_ms\n')
            with self.assertRaises(ValueError):
                reliability.load(directory,[])


if __name__=='__main__':
    unittest.main()
