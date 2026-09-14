import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from compare_reference_likelihoods import ranks,ranking_agreement,summarize,fit_one,global_selection,runtime_identity,verify_saved_inference


class ReferenceComparisonTest(unittest.TestCase):
    def test_ties_and_limit_direction(self):
        self.assertEqual(ranks([4,2,2,1]).tolist(),[4,2.5,2.5,1])
        result=ranking_agreement({'a':1,'b':2,'c':3},{'a':-4,'b':-2,'c':-1})
        self.assertAlmostEqual(result['spearman'],1);self.assertEqual(result['pairwise_agreement'],1)
        self.assertEqual(ranking_agreement({'a':1,'b':2},{'a':2,'b':1})['rank_reversals'],1)
        self.assertIsNone(ranking_agreement({'a':1},{'a':2})['spearman'])

    def test_selection_is_on_development_and_requires_both_gates(self):
        rows=[]
        for fold in ('development','validation'):
            for model,fast,limit,physical,mc in [('reference_AN2017',1,2,True,True),('a',3 if fold=='development' else 1,1.5,True,True),('b',2 if fold=='development' else 9,1,True,True),('unphysical',100,.01,False,True),('sparse',100,.01,True,False)]:
                rows.append({'stage':'reference','signal':'s','fold':fold,'model':model,'fast_significance':fast,'expected_limit':limit,'combine_significance':1/limit,'physicality_passed':physical,'mc_supported':mc,'authoritative_fast_feasible':mc,'status':'complete'})
        report=summarize(rows, sorted({r['signal'] for r in rows}), len(rows));selection=report['development_selections']['reference/s']
        self.assertEqual(selection['model_chosen_on_development'],'a')
        self.assertFalse(selection['validation_used_for_selection'])
        self.assertEqual(selection['validation_limit_ratio_to_reference'],.75)
        comparison=report['comparisons']['reference/s/development/physicality_and_mc_eligible/expected_limit']
        self.assertEqual(set(comparison['models']),{'reference_AN2017','a','b'})

    def test_invalid_template_has_durable_failure_and_never_runs_combine(self):
        task=('development','a','s','reference',{'fast_score':{'significance':2},'physicality':{'feasible':True},'input':{'channels':{'SR':{}}}})
        with tempfile.TemporaryDirectory() as directory,patch('compare_reference_likelihoods.export_model',side_effect=ValueError('negative bin')),patch('compare_reference_likelihoods.run_expected') as run:
            row=fit_one(task,Path(directory),{},'/runtime')
            self.assertEqual(row['status'],'unsupported_or_invalid');run.assert_not_called()
            self.assertEqual(fit_one(task,Path(directory),{},'/runtime'),row)
            self.assertIn('negative bin',row['error'])

    def test_global_selection_requires_every_benchmark_and_preserves_holdout(self):
        rows=[]
        for fold in ('development','validation'):
            for signal in ('s1','s2'):
                for model,fast,limit in [('reference_AN2017',1,2),('a',4 if fold=='development' else 1,1.5),('b',2 if fold=='development' else 20,1)]:
                    rows.append({'fold':fold,'signal':signal,'stage':'reference','model':model,'fast_significance':fast,'expected_limit':limit,'status':'complete','physicality_passed':True,'mc_supported':True,'authoritative_fast_feasible':True})
        rows.append(dict(rows[0],model='incomplete',fast_significance=1e6))
        report=global_selection(rows,'reference',['s1','s2'])
        self.assertEqual(report['selected_model'],'a');self.assertEqual(report['combine_model_chosen_on_development'],'b')
        self.assertEqual(report['validation_mean_limit_ratio_fast_choice_to_combine_choice'],1.5)
        self.assertEqual(report['eligible_development_models'],3)
        self.assertEqual(report['status'],'heldout_gates_passed_conditional_model')

    def test_authoritative_campaign_threshold_blocks_diagnostic_selection(self):
        from compare_reference_likelihoods import fixed_reference, _eligible
        row={'fold':'development','stage':'reference','signal':'s','model':'reference_AN2017',
             'status':'complete','physicality_passed':True,'mc_supported':True,
             'authoritative_fast_feasible':False,'fast_significance':10.,
             'combine_significance':5.,'expected_limit':1.}
        self.assertFalse(_eligible(row))
        self.assertIsNone(fixed_reference([row],'reference','development',['s']))
        self.assertNotIn('selected_model',global_selection([row],'reference',['s']))
        del row['authoritative_fast_feasible']
        self.assertFalse(_eligible(row))

    def test_fit_row_preserves_stricter_campaign_gate_with_diagnostic_score(self):
        from compare_reference_likelihoods import _eligible
        info={'fast_score':{'feasible':False,'significance':None,'failure_reasons':['insufficient_background_effective_events']},
              'diagnostic_fast_score':{'feasible':True,'significance':3.},
              'physicality':{'physicality_pass':True},'input':{'channels':{'SR':{}}}}
        manifest={'unsupported_background_bins':[],'missing_reference_nuisances':[], 'approximations':[]}
        result={'expected_limit':{'median':1.},'expected_significance':2.,'wall_seconds':1.,'fit_diagnostics':{}}
        with tempfile.TemporaryDirectory() as directory,patch('compare_reference_likelihoods.export_model',return_value=manifest),patch('compare_reference_likelihoods.run_expected',return_value=result):
            row=fit_one(('development','a','s','reference',info),Path(directory),{},'/runtime',True)
            self.assertEqual(row['status'],'complete')
            self.assertTrue(row['mc_supported'])
            self.assertTrue(row['diagnostic_fast_score_used'])
            self.assertEqual(row['fast_significance'],3.)
            self.assertFalse(_eligible(row))

    def test_partial_summary_keeps_required_benchmarks_and_withholds_selection(self):
        row={'fold':'development','stage':'reference','signal':'s1','model':'a',
             'status':'complete','physicality_passed':True,'mc_supported':True,
             'authoritative_fast_feasible':True,'fast_significance':2.,
             'combine_significance':1.,'expected_limit':1.}
        result=summarize([row],['s1','s2','s3'],12)
        self.assertTrue(result['provisional'])
        self.assertEqual(result['required_signals'],['s1','s2','s3'])
        self.assertEqual(result['global_development_selections'],{})
        self.assertEqual(result['development_selections'],{})
        self.assertEqual(result['aggregate_rankings']['reference/development']['models'],{})
        self.assertTrue(summarize([row])['provisional'])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            summarize([row,row],['s1'],2)

    def test_runtime_binary_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'CMSSW/bin').mkdir(parents=True)
            (root/'CMSSW/bin/combine').write_text('modified')
            wrapper=root/'wrapper';wrapper.write_text('wrapper')
            manifest=root/'runtime_manifest.json';manifest.write_text(json.dumps({'cmssw_version':'CMSSW','artifacts':{'bin/combine':{'sha256':'wrong'}}}))
            with self.assertRaisesRegex(ValueError,'runtime artifact'):runtime_identity(wrapper,manifest)

    def test_stage_comparison_requires_the_same_finite_mc_basis(self):
        row={'fold':'development','model':'a','signal':'s','stage':'reference',
             'status':'complete','stage_contract_version':2,'finite_mc_basis_sha256':'grouped'}
        other=dict(row,stage='statistics_only',finite_mc_basis_sha256='independent')
        with self.assertRaisesRegex(ValueError,'different finite-MC bases'):
            summarize([row,other])

    def test_fixed_failing_reference_can_normalize_but_cannot_be_selected(self):
        rows=[]
        for fold in ('development','validation'):
            for signal in ('s1','s2'):
                for model in ('reference_AN2017','a'):
                    rows.append({'fold':fold,'signal':signal,'stage':'reference','model':model,
                                 'fast_significance':100 if model=='reference_AN2017' else 2,
                                 'expected_limit':2 if model=='reference_AN2017' else 1,
                                 'combine_significance':1,'status':'complete','mc_supported':True,'authoritative_fast_feasible':True,
                                 'physicality_passed':model!='reference_AN2017'})
        report=summarize(rows, sorted({r['signal'] for r in rows}), len(rows))
        selected=report['global_development_selections']['reference']
        self.assertEqual(selected['selected_model'],'a')
        self.assertEqual(selected['validation_limit_ratio_to_reference'],{'s1':.5,'s2':.5})
        self.assertFalse(selected['reference_normalization']['development']['selection_eligible'])
        aggregate=report['aggregate_rankings']['reference/development']
        self.assertEqual(set(aggregate['models']),{'a'})
        self.assertEqual(aggregate['models']['a']['mean_sensitivity_regret'],0.)
        self.assertEqual(aggregate['models']['a']['mean_expected_limit_ratio_to_reference'],.5)
        # A sparse or missing reference is still insufficient for normalization.
        for row in rows:
            if row['model']=='reference_AN2017' and row['signal']=='s2':row['mc_supported']=False
        self.assertNotIn('validation_limit_ratio_to_reference',global_selection(rows,'reference',['s1','s2']))

    def test_resume_cannot_use_json_numbers_that_differ_from_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'higgsCombine.expected.Significance.mH125.root').touch()
            result={'mass_label':125,'injected_signal_strength':1,'expected_limit':{'median':99},'expected_significance':1,'fit_diagnostics':{'covariance_quality':3}}
            with patch('compare_reference_likelihoods.parse_expected_limits',return_value={'median':2}),patch('compare_reference_likelihoods.parse_expected_significance',return_value=1),patch('compare_reference_likelihoods.parse_fit_diagnostics',return_value={}):
                with self.assertRaisesRegex(ValueError,'JSON differs'):verify_saved_inference(result,root)

    def test_resume_requires_current_policies_commands_and_actual_minos_interval(self):
        import numpy as np
        import uproot
        from combine_runner import (CombineRunError, diagnostic_range_policy,
                                    parse_fit_diagnostics, significance_fit_policy)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'higgsCombine.expected.Significance.mH125.root').touch()
            diagnostic=root/'fitDiagnostics.expected.root'
            def write_fit(upper_error):
                with uproot.recreate(diagnostic) as output:
                    output['tree_fit_sb']={key:np.asarray([value]) for key,value in {
                        'fit_status':0,'r':1.,'rErr':13.,'rHiErr':upper_error,'numbadnll':0}.items()}
            write_fit(2.)
            limits={'median':2.,'quantiles':{'0.975':4.}}
            fit=parse_fit_diagnostics(diagnostic,r_max=20.)
            result={'mass_label':125,'injected_signal_strength':1.,'expected_limit':limits,
                    'expected_significance':1.,'fit_diagnostics':dict(fit,covariance_quality=3),
                    'significance_r_max':20.,'significance_fit_policy':significance_fit_policy(),
                    'fit_diagnostics_r_max':20.,'fit_diagnostics_range_policy':diagnostic_range_policy(4.),
                    'commands':[{'argv':['/runtime','combine','-M','Significance','--rMin','0','--rMax','20',
                                         '--cminDefaultMinimizerStrategy','2','--cminDefaultMinimizerTolerance','1e-5']},
                                {'argv':['/runtime','combine','-M','FitDiagnostics','--rMin','0','--rMax','20']}]}
            with patch('compare_reference_likelihoods.parse_expected_limits',return_value=limits), \
                 patch('compare_reference_likelihoods.parse_expected_significance',return_value=1.):
                verify_saved_inference(result,root)
                for missing in ('significance_fit_policy','fit_diagnostics_range_policy'):
                    stale=json.loads(json.dumps(result));del stale[missing]
                    with self.subTest(missing=missing),self.assertRaisesRegex(ValueError,'policy'):
                        verify_saved_inference(stale,root)
                for command,option,value in ((0,'--cminDefaultMinimizerTolerance','0.1'),
                                             (0,'--rMax','1000'),(1,'--rMax','1000')):
                    stale=json.loads(json.dumps(result));argv=stale['commands'][command]['argv']
                    argv[argv.index(option)+1]=value
                    with self.subTest(option=option,command=command),self.assertRaisesRegex(ValueError,'command'):
                        verify_saved_inference(stale,root)
                # Even fresh policy metadata and covariance3 cannot hide a
                # clipped upper interval in the saved ROOT artifact itself.
                write_fit(19.)
                with self.assertRaisesRegex(CombineRunError,'MINOS upper interval'):
                    verify_saved_inference(result,root)

    def test_changed_model_artifact_is_rejected_before_resume(self):
        task=('development','a','s','reference',{'fast_score':{},'physicality':{},'input':{'channels':{'SR':{}}}})
        with tempfile.TemporaryDirectory() as directory,patch('compare_reference_likelihoods.run_expected') as run:
            model=Path(directory)/'fits/development/a/s/reference/model';model.mkdir(parents=True)
            (model/'datacard.txt').write_text('changed')
            (model/'model.json').write_text(json.dumps({'artifacts_sha256':{'datacard.txt':'invalid'}}))
            row=fit_one(task,Path(directory),{},'/runtime')
            self.assertEqual(row['status'],'unsupported_or_invalid');run.assert_not_called()
            self.assertIn('Changed likelihood artifact',row['error'])


if __name__=='__main__':unittest.main()
