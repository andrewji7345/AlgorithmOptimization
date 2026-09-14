import contextlib
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_reference as evaluator
from compare_reference_likelihoods import validate_binning_study, evaluate_inputs, summarize, digest
from test.test_evaluate_reference import fixture


CONFIG = Path(__file__).resolve().parents[1] / 'config/reference_coarse_2017.json'


class ReferenceBinningTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(CONFIG.read_text())

    def test_coarsening_preserves_range_and_partitions_new_bins(self):
        self.assertEqual(validate_binning_study(self.spec, [0,3500,5500,7500,12000])['qcd_groups'], [[0],[1]])
        cases = [dict(self.spec, mass_bin_edges_gev=[100,3500,12000]),
                 dict(self.spec, mass_bin_edges_gev=[0,4000,12000]),
                 dict(self.spec, mass_bin_edges_gev=[False,3500,12000]),
                 dict(self.spec, qcd_groups=[[0,1],[1]]),
                 dict(self.spec, qcd_groups=[[0],[2]]),
                 dict(self.spec, qcd_groups=[[0],[True]]),
                 dict(self.spec, validation_scope='unverified'),
                 dict(self.spec, undeclared_setting=True)]
        for spec in cases:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                validate_binning_study(spec,[0,3500,5500,7500,12000])

    def test_event_level_coarsening_conserves_signed_yields_variances_and_shapes(self):
        campaign, metadata, payloads, extras = fixture()
        for name, payload in payloads.items():
            masses = np.resize(np.array([1000.,4000.,6000.,9000.]),payload.n_events)
            payload.suu_mass[:,0] = masses
            extras[name][1]['referenceSuuMass'][:] = masses
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(evaluator,'load_metadata',side_effect=lambda p: metadata[str(p)]))
            stack.enter_context(patch.object(evaluator,'read_event_payload',side_effect=lambda m: payloads[m.path]))
            stack.enter_context(patch.object(evaluator,'_read_additive',side_effect=lambda m: extras[m.path]))
            fine=evaluator.evaluate_reference_campaign(campaign)
            coarse=evaluator.evaluate_reference_campaign(campaign,bin_edges=[0,3500,12000],qcd_groups=[[0],[1]])
        self.assertEqual(fine['identity_audit'],coarse['identity_audit'])
        for model, old in fine['models'].items():
            new=coarse['models'][model]
            for sample, record in old['samples'].items():
                other=new['samples'][sample]
                for key in ('sum_gen_weights','generated_events','selected_events','cutflow','region_counts'):
                    self.assertEqual(record[key],other[key])
                if record['kind']=='signal':self.assertEqual(record['physicality'],other['physicality'])
            for signal, info in old['per_signal'].items():
                old_processes=info['input']['channels']['SR']['processes']
                new_processes=new['per_signal'][signal]['input']['channels']['SR']['processes']
                for process, hist in old_processes.items():
                    other=new_processes[process]
                    for key in ('sumw','sumw2'):
                        np.testing.assert_allclose(other[key],[hist[key][0],sum(hist[key][1:])],rtol=1e-14)
                    for nuisance, variation in hist.get('variations',{}).items():
                        for direction, values in variation.items():
                            np.testing.assert_allclose(other['variations'][nuisance][direction],[values[0],sum(values[1:])],rtol=1e-14)

    def test_cache_rejects_different_declared_binning_even_with_valid_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);production=root/'production';output=root/'output'
            production.mkdir();output.mkdir();nominal=production/'ak4_nominal';nominal.mkdir()
            (production/'definition.json').write_text(json.dumps({'frozen':{'radii':[.4],'campaign_contents':{'mass_bin_edges_gev':[0,3500,5500,7500,12000]}}}))
            (nominal/'state.json').write_text(json.dumps({'phase':'complete'}))
            cache=output/'ak4_development_evaluation.json'
            cache.write_text(json.dumps({'models':{},'mass_bin_edges_gev':[0,3500,5500,7500,12000],'qcd_groups':[[0,1],[2,3]]}))
            cache.with_suffix('.sha256.json').write_text(json.dumps({'sha256':digest(cache)}))
            with patch.object(evaluator,'validate_disjoint_folds',return_value={'disjoint':True}), patch.object(evaluator,'evaluate_reference_campaign') as evaluate:
                with self.assertRaisesRegex(ValueError,'different binning study'):
                    evaluate_inputs(production,output,True,'SR',self.spec)
                evaluate.assert_not_called()

    def test_completed_reused_fold_selection_remains_explicitly_exploratory(self):
        rows=[{'fold':fold,'signal':'s','model':'a','stage':'reference','status':'complete',
               'authoritative_fast_feasible':True,'mc_supported':True,'physicality_passed':True,
               'fast_significance':1.,'expected_limit':1.,'combine_significance':1.}
              for fold in ('development','validation')]
        result=summarize(rows,['s'],2,self.spec)
        self.assertFalse(result['provisional'])
        self.assertEqual(result['binning_study'],self.spec)
        self.assertEqual(result['global_development_selections']['reference']['status'],'reused_fold_gates_passed_exploratory_model')
        self.assertIn('independent confirmation',result['interpretation'])


if __name__=='__main__':unittest.main()
