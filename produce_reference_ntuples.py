#!/usr/bin/env python3
"""Produce reusable reconstruction grids and systematic shifts with durable Condor state.

This bounded validation producer shares the tested scheduler/worker protocol with
Optuna, but each job writes a small fixed grid for many cheap likelihood comparisons.
"""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,json,math,os,time
from pathlib import Path
from urllib.parse import urlsplit
from sensitivity_jobs import CondorBackend, JobError, atomic_json, immutable_json, immutable_text
from optimize_sensitivity import file_digest

REPO=Path(__file__).resolve().parent
SYSTEMATICS=('nominal','JECUp','JECDown','JERUp','JERDown')
CONFIG_KEYS={
    'run_dir','campaign','input_dir','backend','radii','systematics','scan_grid',
    'max_events_per_job','max_parallel_groups','poll_seconds','group_timeout_seconds',
    'files_per_fold','folds','seed','development_anchors','validation_anchors',
    'sample_overrides','cmssw_release','cmsrun_timeout_seconds','max_poll_errors',
}
SAMPLE_OVERRIDE_KEYS={'files_per_fold','max_events_per_job'}


def sample_settings(config,name):
    """Resolve a sample's allocation; validation is performed before preparing jobs."""
    return {key:config.get('sample_overrides',{}).get(name,{}).get(key,config[key])
            for key in sorted(SAMPLE_OVERRIDE_KEYS)}


def _positive_integer(value,name):
    if isinstance(value,bool) or not isinstance(value,int) or value<=0:
        raise JobError(f'{name} must be a positive integer')


def _event_limit(value,name):
    if isinstance(value,bool) or not isinstance(value,int) or value==0 or value < -1:
        raise JobError(f'{name} must be -1 (all events) or a positive integer')


def _source_identity(url):
    # A CMS logical file remains the same source through different redirectors.
    path=urlsplit(url).path
    return path[path.index('/store/'):] if '/store/' in path else url


def validated_settings(path):
    path=Path(path).resolve();config=json.loads(path.read_text())
    if not isinstance(config,dict):raise JobError('Production configuration must be a JSON object')
    unknown=set(config)-CONFIG_KEYS
    if unknown:raise JobError('Unknown production configuration keys: '+', '.join(sorted(unknown)))
    def absolute(value):
        value=Path(os.path.expandvars(value)).expanduser()
        return str((path.parent/value).resolve()) if not value.is_absolute() else str(value.resolve())
    for key in ('run_dir','campaign','input_dir'):config[key]=absolute(config[key])
    config['backend']['cmssw_bundle']=absolute(config['backend']['cmssw_bundle'])
    campaign=json.loads(Path(config['campaign']).read_text())
    if campaign.get('purpose')!='pilot':raise JobError('Subset reference validation must be explicitly marked pilot')
    if config['radii'] not in ([0.4],[0.8],[0.4,0.8]):raise JobError('Supported radii are 0.4 and/or 0.8')
    if not config.get('systematics') or any(v not in SYSTEMATICS for v in config['systematics']) or len(set(config['systematics']))!=len(config['systematics']):raise JobError('Invalid systematic list')
    if 'nominal' not in config['systematics']:raise JobError('Nominal production is required')
    for name in ('max_parallel_groups','poll_seconds','group_timeout_seconds','files_per_fold'):
        _positive_integer(config.get(name),name)
    _event_limit(config.get('max_events_per_job'),'max_events_per_job')
    folds=config.get('folds',['development','validation'])
    if folds!=['development','validation']:raise JobError('Independent development/validation folds required')
    from run_scripts.sensitivity.worker import scan_grid_arguments
    scan_grid_arguments(config['scan_grid'])
    from evaluate_sensitivity import validate_campaign
    validation=copy.deepcopy(campaign)
    for sample in validation['samples']:sample['files']=['planning-'+sample['name']+'.root']
    validate_campaign(validation)
    names={sample['name'] for sample in campaign['samples']}
    for field in ('sample_overrides','development_anchors','validation_anchors'):
        mapping=config.get(field,{})
        if not isinstance(mapping,dict):raise JobError(field+' must map sample names to values')
        unknown=set(mapping)-names
        if unknown:raise JobError('Unknown samples in '+field+': '+', '.join(sorted(unknown)))
    for name,overrides in config.get('sample_overrides',{}).items():
        if not isinstance(overrides,dict):raise JobError('sample_overrides.'+name+' must be an object')
        unknown=set(overrides)-SAMPLE_OVERRIDE_KEYS
        if unknown:raise JobError('Unknown sample override keys for '+name+': '+', '.join(sorted(unknown)))
        settings=sample_settings(config,name)
        _positive_integer(settings['files_per_fold'],'sample_overrides.'+name+'.files_per_fold')
        _event_limit(settings['max_events_per_job'],'sample_overrides.'+name+'.max_events_per_job')
    inputs={};catalog_owners={}
    for sample in campaign['samples']:
        if sample.get('normalization_scope')!='representative_subset':raise JobError('Subset normalization must be explicit')
        source=Path(config['input_dir'])/(sample['name']+'.txt')
        files=[s.strip() for s in source.read_text().splitlines() if s.strip() and not s.strip().startswith('#')]
        identities=[_source_identity(url) for url in files]
        if len(identities)!=len(set(identities)):raise JobError('Duplicate source files: '+sample['name'])
        for identity in identities:
            if identity in catalog_owners:
                raise JobError('Source file appears in multiple sample catalogs: '+catalog_owners[identity]+' and '+sample['name'])
            catalog_owners[identity]=sample['name']
        # Stable pseudorandom file ordering; folds and systematic shifts use exactly
        # the same selections. At least one whole distinct source file per fold.
        files.sort(key=lambda x:hashlib.sha256((str(config.get('seed',67023))+'\0'+sample['name']+'\0'+x).encode()).hexdigest())
        anchors={fold:config.get(fold+'_anchors',{}).get(sample['name'],[])
                 for fold in ('development','validation')}
        n=sample_settings(config,sample['name'])['files_per_fold']
        for fold,urls in anchors.items():
            if (not isinstance(urls,list) or any(not isinstance(url,str) or not url for url in urls)
                    or len(urls)!=len(set(urls)) or not set(urls)<=set(files)):
                raise JobError(f'{fold} anchors must be a list of unique catalog inputs: '+sample['name'])
            if len(urls)>n:raise JobError(f'files_per_fold cannot omit existing {fold} anchors: '+sample['name'])
        if set(anchors['development'])&set(anchors['validation']):
            raise JobError('Development and validation anchors overlap: '+sample['name'])
        if 2*n>len(files):raise JobError(f'Insufficient independent catalog files for {sample["name"]}: need {2*n}, have {len(files)}')
        reserved=set(anchors['development']+anchors['validation'])
        available=[url for url in files if url not in reserved]
        needed_development=n-len(anchors['development'])
        development=anchors['development']+available[:needed_development]
        validation=anchors['validation']+available[needed_development:needed_development+n-len(anchors['validation'])]
        inputs[sample['name']]={'development':development,'validation':validation}
    frozen={k:v for k,v in config.items() if k not in ('max_parallel_groups','poll_seconds','group_timeout_seconds','max_poll_errors')}
    frozen['campaign_contents']=campaign;frozen['selected_inputs']=inputs
    frozen['sample_settings']={name:sample_settings(config,name) for name in sorted(names)}
    sources=[REPO/name for name in ('produce_reference_ntuples.py','sensitivity_jobs.py','run_scripts/sensitivity/worker.py','run_scripts/sensitivity/worker.sh')]
    frozen['code_sha256']={str(p):file_digest(p) for p in sources}
    frozen['bundle_sha256']=file_digest(config['backend']['cmssw_bundle'])
    fingerprint=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    return config,campaign,inputs,frozen,fingerprint


def prepare_groups(config,campaign,inputs,frozen,fingerprint):
    root=Path(config['run_dir']);root.mkdir(parents=True,exist_ok=True)
    immutable_json(root/'definition.json',{'fingerprint':fingerprint,'frozen':frozen})
    archive=Path(config['backend']['cmssw_bundle'])
    immutable_json(root/'bundle.json',{'name':archive.name,'sha256':frozen['bundle_sha256'],'bytes':archive.stat().st_size})
    backend=CondorBackend(config['backend'],REPO);backend.state_dir=root
    groups=[]
    for systematic in config['systematics']:
        for radius in config['radii']:
            index=len(groups);name=f'ak{round(radius*10)}_{systematic}'
            directory=root/name;directory.mkdir(exist_ok=True)
            token=f'eo_{fingerprint[:20]}_{index}'
            grid=config['scan_grid']
            params={'r_ak':radius,'t_keep':grid['collection_pt_cuts'][0],'r_ca':grid['ca_radii'][0],
                    'cos_thrust':grid['cos_thrust_cuts'][0],'n_gate_jets':0,'t_gate':max(grid['gate_pt_cuts'])}
            tasks=[];campaigns={fold:copy.deepcopy(campaign) for fold in ('development','validation')}
            for fold,contents in campaigns.items():
                contents['analysis_systematic']=systematic
                contents['reference_production']={
                    'fingerprint':fingerprint,'fold':fold,
                    'sample_settings':copy.deepcopy(frozen['sample_settings']),
                    'development_anchors':copy.deepcopy(config.get('development_anchors',{})),
                    'validation_anchors':copy.deepcopy(config.get('validation_anchors',{})),
                }
            for sample in campaign['samples']:
                for fold in campaigns:
                    target=next(s for s in campaigns[fold]['samples'] if s['name']==sample['name'])
                    target['files']=[];target['input_files']=inputs[sample['name']][fold];target['complete']=True
                    for url in inputs[sample['name']][fold]:
                        task_dir=directory/'tasks'/str(len(tasks));task_dir.mkdir(parents=True,exist_ok=True)
                        task={'index':len(tasks),'sample':sample['name'],'sample_kind':sample['kind'],'fold':fold,
                              'input_files':[url],'output':str(task_dir/'output.root'),'parameters':params,
                              'scan_grid':grid,'analysis_systematic':systematic,'trial_token':token,
                              'max_events':sample_settings(config,sample['name'])['max_events_per_job'],'cmssw_release':config.get('cmssw_release','CMSSW_15_0_19'),
                              'bundle_name':archive.name,'cmsrun_timeout_seconds':config.get('cmsrun_timeout_seconds',7200)}
                        immutable_json(task_dir/'task.json',task);immutable_text(task_dir/'inputs.txt',url+'\n')
                        tasks.append(task);target['files'].append(task['output'])
            descriptor={'trial_number':index,'token':token,'fingerprint':fingerprint,'parameters':params,
                        'analysis_systematic':systematic,'tasks':tasks}
            immutable_json(directory/'trial.json',descriptor)
            for fold,contents in campaigns.items():immutable_json(directory/(fold+'_campaign.json'),contents)
            backend.prepare(directory,descriptor)
            if not (directory/'state.json').exists():atomic_json(directory/'state.json',{'phase':'prepared','created_at':time.time(),'job_ids':[]})
            groups.append((directory,descriptor))
    return backend,groups


def verify_group(descriptor):
    for task in descriptor['tasks']:
        path=Path(task['output']);receipt=path.parent/'worker_result.json'
        if not path.is_file() or path.stat().st_size==0 or not receipt.is_file():raise JobError('Missing worker output: '+str(path))
        data=json.loads(receipt.read_text())
        if data.get('success') is not True or any(data.get(key)!=task[value] for key,value in [('trial_token','trial_token'),('sample','sample'),('index','index')]):raise JobError('Invalid worker receipt: '+str(receipt))


def tick(config,backend,groups,execute,only_systematics=None):
    counts={phase:0 for phase in ('prepared','submitting','running','complete','failed')}
    for directory,descriptor in groups:
        state=json.loads((directory/'state.json').read_text());phase=state['phase']
        # A submission filter must not strand already-running groups: their
        # completion releases the shared concurrency slots, regardless of which
        # systematic a resumed controller is currently allowed to submit.
        if execute and phase in ('submitting','running'):
            try:
                status=backend.lookup(directory,descriptor,state);state['poll_errors']=0
            except JobError as exc:
                state['poll_errors']=state.get('poll_errors',0)+1;state['last_poll_error']=str(exc);atomic_json(directory/'state.json',state)
                if state['poll_errors']>=config.get('max_poll_errors',5):raise JobError('Scheduler unavailable; jobs preserved for resume') from exc
                counts[phase]+=1;continue
            if status.get('job_ids'):state['job_ids']=status['job_ids']
            reason=None
            if status['state']=='complete':
                try:verify_group(descriptor);state.update(phase='complete',finished_at=time.time())
                except (JobError,ValueError) as exc:reason=str(exc)
            elif status['state']=='failed':reason=status.get('reason','Scheduler failure')
            elif time.time()-state['submitted_at']>config['group_timeout_seconds']:reason='Group exceeded time limit'
            elif status['state']=='missing':
                state.setdefault('missing_since',time.time())
                if time.time()-state['missing_since']>180:reason='Submission absent from queue/history; not repeated automatically'
            else:state['phase']='running';state.pop('missing_since',None)
            if reason:
                backend.cancel(directory,descriptor,state);state.update(phase='failed',reason=reason,finished_at=time.time())
            atomic_json(directory/'state.json',state);phase=state['phase']
        counts[phase]+=1
    active=counts['running']+counts['submitting']
    # Preserve and reconcile jobs already in flight, but do not spend the rest
    # of the campaign after a worker/group failure invalidates this production.
    if execute and not counts['failed']:
        for directory,descriptor in groups:
            if active>=config['max_parallel_groups']:break
            if only_systematics is not None and descriptor['analysis_systematic'] not in only_systematics:continue
            state=json.loads((directory/'state.json').read_text())
            if state['phase']!='prepared':continue
            if any(Path(task['output']).exists() for task in descriptor['tasks']):raise JobError('Refusing submission over existing output')
            state.update(phase='submitting',submitted_at=time.time(),poll_errors=0);atomic_json(directory/'state.json',state)
            try:state['job_ids']=backend.submit(directory,descriptor,state);state['phase']='running'
            except JobError as exc:state['submit_error']=str(exc)
            atomic_json(directory/'state.json',state);counts['prepared']-=1;counts[state['phase']]+=1;active+=1
    return counts


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path);parser.add_argument('--execute',action='store_true');parser.add_argument('--once',action='store_true');parser.add_argument('--only-systematic',action='append',choices=SYSTEMATICS)
    args=parser.parse_args(argv)
    try:
        config,campaign,inputs,frozen,fingerprint=validated_settings(args.config)
        root=Path(config['run_dir']);root.mkdir(parents=True,exist_ok=True)
        with (root/'controller.lock').open('a+') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise JobError('Another reference producer holds the controller lock')
            backend,groups=prepare_groups(config,campaign,inputs,frozen,fingerprint)
            if args.execute:backend.preflight()
            while True:
                for path,checksum in frozen['code_sha256'].items():
                    if file_digest(path)!=checksum:raise JobError('Executable source changed during production: '+path)
                if file_digest(config['backend']['cmssw_bundle'])!=frozen['bundle_sha256']:raise JobError('Archive changed during production')
                counts=tick(config,backend,groups,args.execute,args.only_systematic)
                summary={'fingerprint':fingerprint,'states':counts,'groups':len(groups),'tasks':sum(len(g[1]['tasks']) for g in groups)}
                atomic_json(root/'production_summary.json',summary);print(json.dumps(summary),flush=True)
                selected_states=[json.loads((d/'state.json').read_text())['phase'] for d,t in groups if args.only_systematic is None or t['analysis_systematic'] in args.only_systematic]
                if not args.execute or args.once:return 2 if counts['failed'] else 0
                if counts['failed']:
                    # Prepared groups stay prepared. Finish monitoring all live
                    # groups before reporting a failed production to the caller.
                    if counts['running']+counts['submitting']==0:return 2
                elif all(s in ('complete','failed') for s in selected_states):return 0
                time.sleep(config['poll_seconds'])
    except (OSError,ValueError,KeyError,RuntimeError,JobError) as exc:parser.exit(2,'ERROR: '+str(exc)+'\n')
    except KeyboardInterrupt:return 130

if __name__=='__main__':raise SystemExit(main())
