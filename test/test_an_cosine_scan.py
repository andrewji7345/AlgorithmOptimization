"""Regression checks for short inputs, the reference anchor, and durable failures."""
from array import array
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import run_an_cosine_scan as driver
from plot_an_cosine_scan import read
from run_scripts.sensitivity.worker import validate_event_coverage

try:
    import ROOT
except ImportError:
    ROOT = None


def fixture(path, recorded=3, bad_anchor=False):
    """One SR event, one reconstruction failure at c=0, and one baseline failure."""
    root = ROOT.TFile(str(path), 'RECREATE')
    root.mkdir('compactScan').cd()
    metadata = ROOT.TTree('Metadata', 'Metadata')
    processed = array('Q', [recorded])
    sample = ROOT.std.string('WbWb_4000_1000')
    reconstruction = ROOT.std.string('AN23-067-PATAK8-CA8-Thrust-source-port-v1')
    cuts = ROOT.std.vector('double')()
    for cut in (0, .85):
        cuts.push_back(cut)
    metadata.Branch('processedEvents', processed, 'processedEvents/l')
    metadata.Branch('sampleName', sample)
    metadata.Branch('referenceReconstruction', reconstruction)
    metadata.Branch('referenceCosThrustCuts', cuts)
    metadata.Fill()
    metadata.Write()
    events = ROOT.TTree('Events', 'Events')
    scalars = {}
    for name, code, leaf, value in [('run','I','i',1), ('lumi','I','i',1), ('event','Q','l',1),
            ('genWeight','f','F',1), ('referenceWeight','f','F',1),
            ('passesBaseline','b','O',1), ('analysisNBTags','H','s',1)]:
        scalars[name] = array(code, [value])
        events.Branch(name, scalars[name], name + '/' + leaf)
    vectors = {}
    for name in ('RecoStatus', 'Region', 'NAmbiguous', 'SJ1NCA4E300', 'SJ2NCA4E300',
                 'SJ1Mass', 'SJ2Mass', 'SuuMass'):
        is_mass = 'Mass' in name
        dtype = 'float' if is_mass else 'unsigned char' if name in ('RecoStatus','Region') else 'unsigned short'
        vectors[name] = ROOT.std.vector(dtype)()
        events.Branch('referenceScan' + name, vectors[name])
        if name != 'NAmbiguous':
            code, leaf = ('f','F') if is_mass else ('B','b') if name in ('RecoStatus','Region') else ('H','s')
            scalars['reference' + name] = array(code, [0])
            events.Branch('reference' + name, scalars['reference' + name], 'reference' + name + '/' + leaf)
    for index in range(3):
        scalars['event'][0] = index + 1
        scalars['passesBaseline'][0] = int(index != 2)
        values = dict(RecoStatus=[6 if index == 1 else 0, 0],
                      Region=[0 if index in (1,2) else 1, 0 if index == 2 else 1],
                      NAmbiguous=[15 if index == 1 else 0, 0],
                      SJ1NCA4E300=[0 if index == 1 else 2, 2],
                      SJ2NCA4E300=[0 if index == 1 else 2, 2],
                      SJ1Mass=[float('nan') if index == 1 else 1000, 1000],
                      SJ2Mass=[float('nan') if index == 1 else 1000, 1000],
                      SuuMass=[float('nan') if index == 1 else 4000, 4000])
        for name, vector in vectors.items():
            vector.clear()
            for value in values[name]:
                vector.push_back(value)
            if name != 'NAmbiguous':
                scalars['reference' + name][0] = values[name][1]
        if bad_anchor:
            scalars['referenceSJ1Mass'][0] = 999
        events.Fill()
    events.Write()
    root.Close()


class ScanPlanningTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((REPO / 'config/an_cosine_scan_2017.json').read_text())

    def test_committed_inputs_cover_every_requested_sample(self):
        driver.validate_config(self.config)
        self.assertEqual(len(self.config['samples']), 3)
        self.assertEqual(len(self.config['reference_cos_thrust_cuts']), 21)

    def test_short_inputs_are_rejected_before_preparation(self):
        self.config['samples'][1]['input_files'] = self.config['samples'][1]['input_files'][:1]
        with self.assertRaisesRegex(driver.JobError, 'Insufficient input coverage'):
            driver.validate_config(self.config)

    def test_duplicate_inputs_and_invalid_cutoffs_are_rejected(self):
        for cuts in ([.85, .5], [0, 0], [float('nan')], [True], [1.01]):
            config = copy.deepcopy(self.config)
            config['reference_cos_thrust_cuts'] = cuts
            with self.subTest(cuts=cuts), self.assertRaises(driver.JobError):
                driver.validate_config(config)
        self.config['samples'][1]['input_files'].append(self.config['samples'][0]['input_files'][0])
        with self.assertRaisesRegex(driver.JobError, 'Duplicate source'):
            driver.validate_config(self.config)

    def test_lost_submission_response_cannot_cause_resubmission(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            state = dict(phase='prepared', token='eo_abcd_0', job_ids=[])
            (run / 'status.json').write_text(json.dumps(state))
            backend = mock.Mock()
            backend.submit.side_effect = RuntimeError('lost submit response')
            with mock.patch.object(driver, 'load', return_value=(backend, {})):
                with self.assertRaisesRegex(RuntimeError, 'lost submit response'):
                    driver.submit(run)
                self.assertEqual(json.loads((run / 'status.json').read_text())['phase'], 'submitting')
                with self.assertRaisesRegex(driver.JobError, 'Refusing resubmission'):
                    driver.submit(run)
            self.assertEqual(backend.submit.call_count, 1)

    def test_plot_failure_is_saved_as_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'status.json').write_text(json.dumps(dict(phase='running', job_ids=['1.0'])))
            backend = mock.Mock()
            backend.lookup.return_value = dict(state='complete', job_ids=['1.0'])
            with mock.patch.object(driver, 'load', return_value=(backend, {})), \
                    mock.patch.object(driver.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'plot')):
                with self.assertRaises(subprocess.CalledProcessError):
                    driver.watch(run, once=True)
            state = json.loads((run / 'status.json').read_text())
            self.assertEqual(state['phase'], 'failed')
            self.assertIn('Plotting failed', state['reason'])


@unittest.skipUnless(ROOT is not None, 'Requires CMSSW PyROOT')
class ScanContentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'output.root'
        self.task = dict(output=str(self.path), sample='WbWb_4000_1000', index=0,
                         trial_token='eo_abcd_0', max_events=3, reference_cos_thrust_cuts=[0, .85])
        (self.path.parent / 'worker_result.json').write_text(json.dumps(dict(
            success=True, sample=self.task['sample'], index=0, trial_token=self.task['trial_token'])))

    def tearDown(self):
        self.tmp.cleanup()

    def test_reconstruction_failure_stays_in_fixed_baseline_denominator(self):
        fixture(self.path)
        rows, masses, audit = read(self.task)
        self.assertEqual(audit['processed_events'], 3)
        self.assertEqual([row['preselected_events'] for row in rows], [2,2])
        self.assertEqual([row['sr_events'] for row in rows], [1,2])
        self.assertEqual([row['efficiency'] for row in rows], [.5,1])
        self.assertEqual(rows[0]['complexity_guard_fraction'], .5)
        self.assertEqual(len(masses[.85]['sj']), 4)

    def test_worker_and_plotter_reject_underfilled_input(self):
        fixture(self.path)
        self.task['max_events'] = 4
        with self.assertRaisesRegex(RuntimeError, 'Incomplete requested input coverage'):
            validate_event_coverage(self.path, 4)
        with self.assertRaisesRegex(ValueError, 'Incomplete requested input coverage'):
            read(self.task)

    def test_worker_rejects_inconsistent_metadata(self):
        fixture(self.path, recorded=2)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete requested input coverage'):
            validate_event_coverage(self.path, 3)

    def test_fixed_reference_anchor_is_checked(self):
        from test.validate_reference_cosine_scan import validate
        fixture(self.path)
        validate(str(self.path))
        fixture(self.path, bad_anchor=True)
        with self.assertRaisesRegex(ValueError, '0.85 anchor mismatch'):
            validate(str(self.path))


if __name__ == '__main__':
    unittest.main()
