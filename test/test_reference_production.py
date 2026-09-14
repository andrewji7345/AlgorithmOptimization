import copy,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from produce_reference_ntuples import validated_settings,prepare_groups,tick,verify_group,main
from sensitivity_jobs import JobError,atomic_json
from run_scripts.sensitivity.worker import scan_grid_arguments
from test.test_sensitivity_evaluator import campaign_fixture

class ReferenceProductionTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        campaign=campaign_fixture();campaign['purpose']='pilot'
        self.campaign_path=self.root/'campaign.json';self.campaign_path.write_text(json.dumps(campaign))
        inputs=self.root/'inputs';inputs.mkdir()
        for sample in campaign['samples']:
            sample['normalization_scope']='representative_subset'
            (inputs/(sample['name']+'.txt')).write_text('\n'.join(f'root://example/{sample["name"]}/{i}.root' for i in range(4))+'\n')
        self.campaign_path.write_text(json.dumps(campaign))
        self.bundle=self.root/'bundle.tar.gz';self.bundle.write_bytes(b'test archive')
        self.grid={'collection_pt_cuts':[120,320],'ca_radii':[.6,1.2], 'cos_thrust_cuts':[.5,.85],'gate_counts':[0,3],'gate_pt_cuts':[200,400]}
        self.config={'run_dir':str(self.root/'run'),'campaign':str(self.campaign_path),'input_dir':str(inputs),
                     'backend':{'cmssw_bundle':str(self.bundle)},'radii':[.4,.8], 'systematics':['nominal','JECUp'],
                     'max_events_per_job':20000,'max_parallel_groups':1,'poll_seconds':1,'group_timeout_seconds':60,
                     'files_per_fold':1,'scan_grid':self.grid}
        self.path=self.root/'config.json';self.path.write_text(json.dumps(self.config))
    def tearDown(self):self.tmp.cleanup()
    def settings(self):return validated_settings(self.path)
    def test_folds_are_disjoint_stable_and_parameters_frozen(self):
        c,cam,inputs,frozen,fp=self.settings()
        for folds in inputs.values():self.assertFalse(set(folds['development'])&set(folds['validation']))
        self.assertEqual(inputs,self.settings()[2]);self.assertEqual(fp,self.settings()[4])
        self.config['max_parallel_groups']=3;self.path.write_text(json.dumps(self.config));self.assertEqual(fp,self.settings()[4])
        self.config['max_events_per_job']=5000;self.path.write_text(json.dumps(self.config));self.assertNotEqual(fp,self.settings()[4])
    def test_grid_rejects_invalid_or_unbounded_payload_before_jobs(self):
        for key,values in [('collection_pt_cuts',[0]),('gate_counts',[2.5]),('cos_thrust_cuts',[1.1]),('ca_radii',[.8,.8])]:
            grid=copy.deepcopy(self.grid);grid[key]=values
            with self.subTest(key=key),self.assertRaises(RuntimeError):scan_grid_arguments(grid)
        self.config['scan_grid']['ca_radii']=[0]
        self.path.write_text(json.dumps(self.config))
        with self.assertRaises(RuntimeError):self.settings()
    def test_previous_pilot_inputs_are_excluded_from_validation(self):
        name=json.loads(self.campaign_path.read_text())['samples'][0]['name']
        anchor=f'root://example/{name}/2.root'
        self.config['development_anchors']={name:[anchor]}
        self.path.write_text(json.dumps(self.config))
        inputs=self.settings()[2][name]
        self.assertEqual(inputs['development'],[anchor]);self.assertNotIn(anchor,inputs['validation'])
        self.config['development_anchors'][name]=['missing.root']
        self.path.write_text(json.dumps(self.config))
        with self.assertRaises(JobError):self.settings()
    def test_indented_source_comments_are_not_treated_as_input_files(self):
        name=json.loads(self.campaign_path.read_text())['samples'][0]['name']
        source=self.root/'inputs'/(name+'.txt')
        source.write_text('  # indented comment\n\n'+source.read_text()+'\t# another comment\n')
        selected=self.settings()[2][name]
        self.assertTrue(all(url.startswith('root://') for values in selected.values() for url in values))
    def test_event_limits_accept_all_events_and_reject_invalid_global_values(self):
        self.config['max_events_per_job']=-1
        self.path.write_text(json.dumps(self.config))
        self.assertEqual(self.settings()[0]['max_events_per_job'],-1)
        for value in (0,-2,True,False,1.5,'20000',None):
            with self.subTest(value=value):
                self.config['max_events_per_job']=value;self.path.write_text(json.dumps(self.config))
                with self.assertRaises(JobError):self.settings()
    def test_unknown_configuration_and_sample_override_keys_are_rejected(self):
        name=json.loads(self.campaign_path.read_text())['samples'][0]['name']
        cases=[{'sample_override':{}}, {'sample_overrides':{'unknown_sample':{}}},
               {'sample_overrides':{name:{'max_event_per_job':-1}}},
               {'sample_overrides':{name:[]}}, {'sample_overrides':[]},
               {'development_anchors':{'unknown_sample':[]}},
               {'validation_anchors':{'unknown_sample':[]}}, {'validation_anchors':[]}]
        for values in cases:
            with self.subTest(values=values):
                candidate=dict(self.config,**values);self.path.write_text(json.dumps(candidate))
                with self.assertRaises(JobError):self.settings()
    def test_invalid_sample_allocations_are_rejected_without_silent_file_capping(self):
        name=json.loads(self.campaign_path.read_text())['samples'][0]['name']
        for key,values in [('files_per_fold',[0,-1,True,1.5,3]),
                           ('max_events_per_job',[0,-2,True,1.5])]:
            for value in values:
                with self.subTest(key=key,value=value):
                    candidate=dict(self.config,sample_overrides={name:{key:value}})
                    self.path.write_text(json.dumps(candidate))
                    with self.assertRaises(JobError):self.settings()
        self.config['files_per_fold']=3;self.path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(JobError,'Insufficient independent'):self.settings()
    def test_both_fold_anchors_survive_expansion_and_never_cross_folds(self):
        original=self.settings()[2]
        self.config['development_anchors']={name:folds['development'] for name,folds in original.items()}
        self.config['validation_anchors']={name:folds['validation'] for name,folds in original.items()}
        self.config['files_per_fold']=2;self.path.write_text(json.dumps(self.config))
        _,_,expanded,frozen,fingerprint=self.settings()
        self.assertEqual(frozen['validation_anchors'],self.config['validation_anchors'])
        self.assertEqual(frozen['development_anchors'],self.config['development_anchors'])
        for name,folds in expanded.items():
            self.assertEqual(len(folds['development']),2);self.assertEqual(len(folds['validation']),2)
            self.assertFalse(set(folds['development'])&set(folds['validation']))
            for fold in ('development','validation'):
                self.assertEqual(folds[fold][:len(original[name][fold])],original[name][fold])
        # Adding an already-selected source to an anchor list changes the frozen
        # contract even if the selected inputs themselves do not change.
        name=next(iter(expanded));self.config['validation_anchors'][name]=expanded[name]['validation']
        self.path.write_text(json.dumps(self.config))
        self.assertEqual(expanded,self.settings()[2]);self.assertNotEqual(fingerprint,self.settings()[4])
        self.config['files_per_fold']=1;self.path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(JobError,'cannot omit existing validation anchors'):self.settings()
    def test_anchors_require_disjoint_unique_catalog_url_lists(self):
        name=json.loads(self.campaign_path.read_text())['samples'][0]['name']
        url=f'root://example/{name}/0.root'
        for fold in ('development','validation'):
            for urls in (url,None,[None],[url,url],['missing.root']):
                with self.subTest(fold=fold,urls=urls):
                    candidate=dict(self.config,**{fold+'_anchors':{name:urls}})
                    self.path.write_text(json.dumps(candidate))
                    with self.assertRaises(JobError):self.settings()
        candidate=dict(self.config,development_anchors={name:[url]},validation_anchors={name:[url]})
        self.path.write_text(json.dumps(candidate))
        with self.assertRaisesRegex(JobError,'anchors overlap'):self.settings()
    def test_catalogs_reject_duplicate_sources_including_redirector_aliases(self):
        samples=json.loads(self.campaign_path.read_text())['samples']
        first=self.root/'inputs'/(samples[0]['name']+'.txt')
        second=self.root/'inputs'/(samples[1]['name']+'.txt')
        first.write_text(first.read_text()+'root://first.example//store/mc/shared.root\n')
        second.write_text(second.read_text()+'root://second.example//store/mc/shared.root\n')
        with self.assertRaisesRegex(JobError,'multiple sample catalogs'):self.settings()
        second.write_text(second.read_text().replace('root://second.example//store/mc/shared.root\n',''))
        first.write_text(first.read_text()+'/store/mc/shared.root\n')
        with self.assertRaisesRegex(JobError,'Duplicate source'):self.settings()
    def test_sample_overrides_drive_tasks_and_frozen_campaign_provenance(self):
        baseline=self.settings();name=baseline[1]['samples'][0]['name']
        other=baseline[1]['samples'][1]['name']
        self.config['sample_overrides']={name:{'files_per_fold':2,'max_events_per_job':-1},
                                         other:{'max_events_per_job':50000}}
        self.path.write_text(json.dumps(self.config))
        c,cam,inputs,frozen,fp=self.settings()
        self.assertNotEqual(fp,baseline[4])
        self.assertEqual(frozen['sample_settings'][name],{'files_per_fold':2,'max_events_per_job':-1})
        self.assertEqual(frozen['sample_settings'][other],{'files_per_fold':1,'max_events_per_job':50000})
        _,groups=prepare_groups(c,cam,inputs,frozen,fp)
        for directory,descriptor in groups:
            for sample in cam['samples']:
                allocation=frozen['sample_settings'][sample['name']]
                tasks=[task for task in descriptor['tasks'] if task['sample']==sample['name']]
                self.assertEqual(len(tasks),2*allocation['files_per_fold'])
                self.assertTrue(all(task['max_events']==allocation['max_events_per_job'] for task in tasks))
                for fold in ('development','validation'):
                    self.assertEqual([task['input_files'][0] for task in tasks if task['fold']==fold],inputs[sample['name']][fold])
            for fold in ('development','validation'):
                produced=json.loads((directory/(fold+'_campaign.json')).read_text())
                provenance=produced['reference_production']
                self.assertEqual(provenance['fingerprint'],fp);self.assertEqual(provenance['fold'],fold)
                self.assertEqual(provenance['sample_settings'],frozen['sample_settings'])
                self.assertEqual(provenance['development_anchors'],{})
                self.assertEqual(provenance['validation_anchors'],{})
        # A new allocation requires a new run directory; immutable inputs may
        # not be overwritten by an apparently harmless resume.
        self.config['sample_overrides'][name]['max_events_per_job']=100000
        self.path.write_text(json.dumps(self.config))
        changed=self.settings()
        with self.assertRaises(JobError):prepare_groups(*changed)
    def test_prepare_resume_and_systematic_subset_do_not_duplicate_tasks(self):
        c,cam,inputs,frozen,fp=self.settings();backend,groups=prepare_groups(c,cam,inputs,frozen,fp)
        backend2,groups2=prepare_groups(c,cam,inputs,frozen,fp)
        self.assertEqual([d[1] for d in groups],[d[1] for d in groups2])
        with patch.object(backend,'submit',return_value=['42.0']) as submit:
            counts=tick(c,backend,groups,True,['nominal']);self.assertEqual(submit.call_count,1);self.assertEqual(counts['running'],1)
        with patch.object(backend,'lookup',return_value={'state':'running','job_ids':['42.0']}),patch.object(backend,'submit') as submit:
            tick(c,backend,groups,True,['nominal']);submit.assert_not_called()
        for directory,descriptor in groups:
            if descriptor['analysis_systematic']=='JECUp':self.assertEqual(json.loads((directory/'state.json').read_text())['phase'],'prepared')
    def test_lost_submit_response_is_reconciled_never_resubmitted(self):
        c,cam,inputs,frozen,fp=self.settings();backend,groups=prepare_groups(c,cam,inputs,frozen,fp)
        with patch.object(backend,'submit',side_effect=JobError('response lost')):tick(c,backend,groups,True,['nominal'])
        self.assertEqual(json.loads((groups[0][0]/'state.json').read_text())['phase'],'submitting')
        with patch.object(backend,'lookup',return_value={'state':'running','job_ids':['43.0']}),patch.object(backend,'submit') as submit:
            tick(c,backend,groups,True,['nominal']);submit.assert_not_called()
    def test_filter_switch_reconciles_unselected_live_groups_and_releases_capacity(self):
        c,cam,inputs,frozen,fp=self.settings();backend,groups=prepare_groups(c,cam,inputs,frozen,fp)
        with patch.object(backend,'submit',return_value=['42.0']):tick(c,backend,groups,True,['nominal'])
        with patch.object(backend,'lookup',return_value={'state':'complete','job_ids':['42.0']}) as lookup, \
             patch('produce_reference_ntuples.verify_group'),patch.object(backend,'submit',return_value=['43.0']) as submit:
            counts=tick(c,backend,groups,True,['JECUp'])
        lookup.assert_called_once()
        self.assertEqual(lookup.call_args.args[1]['analysis_systematic'],'nominal')
        submit.assert_called_once()
        self.assertEqual(submit.call_args.args[1]['analysis_systematic'],'JECUp')
        self.assertEqual(counts['complete'],1);self.assertEqual(counts['running'],1)
    def test_first_failure_blocks_new_submissions_but_preserves_other_live_groups(self):
        c,cam,inputs,frozen,fp=self.settings();c['max_parallel_groups']=2
        backend,groups=prepare_groups(c,cam,inputs,frozen,fp)
        with patch.object(backend,'submit',return_value=['42.0']):tick(c,backend,groups,True,['nominal'])
        def lookup(directory,descriptor,state):
            return {'state':'failed','reason':'worker failed'} if descriptor['trial_number']==0 else {'state':'running','job_ids':['43.0']}
        with patch.object(backend,'lookup',side_effect=lookup),patch.object(backend,'cancel') as cancel,patch.object(backend,'submit') as submit:
            counts=tick(c,backend,groups,True,['JECUp'])
        self.assertEqual(counts['failed'],1);self.assertEqual(counts['running'],1)
        submit.assert_not_called();cancel.assert_called_once()
        self.assertEqual(json.loads((groups[1][0]/'state.json').read_text())['phase'],'running')
        self.assertTrue(all(json.loads((d/'state.json').read_text())['phase']=='prepared' for d,t in groups if t['analysis_systematic']=='JECUp'))
        with patch.object(backend,'lookup',return_value={'state':'complete','job_ids':['43.0']}), \
             patch('produce_reference_ntuples.verify_group'),patch.object(backend,'submit') as submit:
            counts=tick(c,backend,groups,True,['JECUp'])
        self.assertEqual(counts['complete'],1);self.assertEqual(counts['running'],0);submit.assert_not_called()
    def test_main_drains_live_groups_then_fails_with_prepared_groups_preserved(self):
        c,cam,inputs,frozen,fp=self.settings();backend,groups=prepare_groups(c,cam,inputs,frozen,fp)
        final={'prepared':2,'failed':1,'running':0,'submitting':0,'complete':1}
        running=dict(final,running=1,complete=0)
        with patch('produce_reference_ntuples.validated_settings',return_value=(c,cam,inputs,frozen,fp)), \
             patch('produce_reference_ntuples.prepare_groups',return_value=(backend,groups)), \
             patch.object(backend,'preflight'),patch('produce_reference_ntuples.tick',side_effect=[running,final]) as poll, \
             patch('produce_reference_ntuples.time.sleep') as sleep:
            status=main(['--config',str(self.path),'--execute','--only-systematic','JECUp'])
        self.assertEqual(status,2);self.assertEqual(poll.call_count,2);sleep.assert_called_once()
    def test_completed_job_requires_identity_matched_receipt(self):
        c,cam,inputs,frozen,fp=self.settings();backend,groups=prepare_groups(c,cam,inputs,frozen,fp)
        descriptor=groups[0][1]
        for task in descriptor['tasks']:
            path=Path(task['output']);path.write_bytes(b'ROOT fixture is validated by worker in real jobs')
            atomic_json(path.parent/'worker_result.json',{'success':True,'trial_token':task['trial_token'],'sample':task['sample'],'index':task['index']})
        verify_group(descriptor)
        receipt=Path(descriptor['tasks'][0]['output']).parent/'worker_result.json';data=json.loads(receipt.read_text());data['index']=-1;atomic_json(receipt,data)
        with self.assertRaises(JobError):verify_group(descriptor)

if __name__=='__main__':unittest.main()
