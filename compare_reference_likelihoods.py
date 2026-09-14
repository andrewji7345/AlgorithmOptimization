#!/usr/bin/env python3
"""Compare the fast objective with blind Combine inference on independent MC folds.

Artifacts are immutable and resumable. Unsupported templates remain explicit
failures, and physicality failures never enter candidate selection. This is an
empirical proxy check for a representative likelihood, not an analysis result.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from sensitivity_jobs import atomic_json, immutable_json
from likelihood_model import export_model, DEFAULT_CONFIG
from combine_runner import (run_expected, CombineRunError, parse_expected_limits,
                            parse_expected_significance, parse_fit_diagnostics,
                            diagnostic_range_policy, significance_fit_policy)

REFERENCE = 'reference_AN2017'


def validate_binning_study(value, original_edges=None):
    """Validate a declared common coarsening before producing or fitting inputs."""
    from sensitivity_metrics import validated_edges
    if value is None:
        return None
    fields={'schema_version','name','mass_bin_edges_gev','qcd_groups','rationale',
            'selection_basis','validation_scope','qcd_group_interpretation'}
    if not isinstance(value,dict) or set(value)!=fields:
        raise ValueError('Binning study must contain exactly: '+', '.join(sorted(fields)))
    if type(value['schema_version']) is not int or value['schema_version']!=1:
        raise ValueError('Unsupported binning study schema')
    for key in ('name','rationale','selection_basis','qcd_group_interpretation'):
        if not isinstance(value[key],str) or not value[key].strip():
            raise ValueError('Binning study requires a nonempty '+key)
    if value['validation_scope'] not in ('exploratory_reuse','independent_validation'):
        raise ValueError('Declare exploratory_reuse or independent_validation scope')
    raw=value['mass_bin_edges_gev']
    if not isinstance(raw,list) or any(type(x) not in (int,float) for x in raw):
        raise ValueError('Common mass edges must be a list of finite numbers')
    edges=validated_edges(raw,'common coarse Suu bins').tolist()
    groups=value['qcd_groups']
    if not isinstance(groups,list) or any(not isinstance(g,list) or not g for g in groups):
        raise ValueError('QCD groups must contain nonempty bin-index lists')
    indices=[i for g in groups for i in g]
    if any(type(i) is not int for i in indices) or sorted(indices)!=list(range(len(edges)-1)):
        raise ValueError('QCD groups must partition the new bins exactly once')
    if original_edges is not None:
        original=validated_edges(original_edges,'original Suu bins').tolist()
        if (edges[0]!=original[0] or edges[-1]!=original[-1]
                or any(edge not in original for edge in edges)):
            raise ValueError('Common coarsening must retain endpoints and use existing boundaries')
    return dict(value,mass_bin_edges_gev=edges,qcd_groups=[list(g) for g in groups])


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def runtime_identity(wrapper, manifest_path):
    manifest_path=Path(manifest_path).resolve(strict=True)
    manifest=json.loads(manifest_path.read_text())
    release=manifest_path.parent/manifest['cmssw_version']
    for name,artifact in manifest['artifacts'].items():
        if digest(release/name)!=artifact['sha256']:raise ValueError('Combine runtime artifact differs from its manifest: '+name)
    return {'manifest_path':str(manifest_path),'manifest_sha256':digest(manifest_path),
            'manifest':manifest,'wrapper_sha256':digest(wrapper)}


def verify_saved_inference(result, directory):
    mass=result['mass_label']
    limits=parse_expected_limits(directory/f'higgsCombine.expected.AsymptoticLimits.mH{mass:g}.root')
    paths=list(directory.glob(f'higgsCombine.expected.Significance.mH{mass:g}*.root'))
    if len(paths)!=1:raise ValueError('Expected one saved significance ROOT file')
    significance=parse_expected_significance(paths[0])
    if limits!=result['expected_limit'] or significance!=result['expected_significance']:
        raise ValueError('Saved inference JSON differs from its ROOT results')
    injection=result['injected_signal_strength']
    significance_policy=significance_fit_policy(injection)
    diagnostic_policy=diagnostic_range_policy(limits['quantiles']['0.975'],injection)
    if (result.get('significance_fit_policy')!=significance_policy
            or result.get('significance_r_max')!=significance_policy['r_max']):
        raise ValueError('Saved significance fit policy is missing or differs from the current policy')
    if (result.get('fit_diagnostics_range_policy')!=diagnostic_policy
            or result.get('fit_diagnostics_r_max')!=diagnostic_policy['r_max']):
        raise ValueError('Saved diagnostic range policy is missing or differs from the current policy')
    # Metadata alone must not relabel a cached fit made with old command ranges.
    for method,options in (
            ('Significance',{'--rMin':0.,'--rMax':significance_policy['r_max'],
                             '--cminDefaultMinimizerStrategy':significance_policy['minimizer_strategy'],
                             '--cminDefaultMinimizerTolerance':significance_policy['minimizer_tolerance']}),
            ('FitDiagnostics',{'--rMin':0.,'--rMax':diagnostic_policy['r_max']})):
        commands=[command.get('argv',[]) for command in result.get('commands',[])]
        matches=[argv for argv in commands if '-M' in argv
                 and argv.index('-M')+1<len(argv) and argv[argv.index('-M')+1]==method]
        if len(matches)!=1:raise ValueError('Expected one saved '+method+' command')
        argv=matches[0]
        for option,expected in options.items():
            try:
                if argv.count(option)!=1 or float(argv[argv.index(option)+1])!=expected:
                    raise ValueError('Command option differs')
            except (ValueError,IndexError,TypeError) as exc:
                raise ValueError('Saved '+method+' command does not implement '+option+' policy') from exc
    fit=parse_fit_diagnostics(directory/'fitDiagnostics.expected.root',injection,
                              r_max=diagnostic_policy['r_max'])
    if any(result['fit_diagnostics'].get(k)!=v for k,v in fit.items()):raise ValueError('Saved fit diagnostics differ from ROOT')
    if result['fit_diagnostics'].get('covariance_quality')!=3:raise ValueError('Saved fit covariance is not valid')


def ranks(values):
    """Ascending midranks, including exact ties, without a scipy dependency."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not np.all(np.isfinite(values)):raise ValueError('Ranks require finite one-dimensional values')
    result=np.empty(len(values));order=np.argsort(values,kind='stable');start=0
    while start<len(values):
        stop=start+1
        while stop<len(values) and values[order[stop]]==values[order[start]]:stop+=1
        result[order[start:stop]]=(start+stop-1)/2+1;start=stop
    return result


def ranking_agreement(fast, target):
    """Both metrics must increase with quality; use -r95 for expected limits."""
    common=sorted(set(fast)&set(target))
    common=[k for k in common if math.isfinite(fast[k]) and math.isfinite(target[k])]
    a=np.array([fast[k] for k in common]);b=np.array([target[k] for k in common])
    rho=None
    if len(a)>1 and len(set(a))>1 and len(set(b))>1:rho=float(np.corrcoef(ranks(a),ranks(b))[0,1])
    concordant=discordant=ties=0
    for i in range(len(a)):
        for j in range(i):
            product=(a[i]-a[j])*(b[i]-b[j])
            if product>0:concordant+=1
            elif product<0:discordant+=1
            else:ties+=1
    tested=concordant+discordant
    return {'models':common,'count':len(common),'spearman':rho,
            'pairwise_agreement':concordant/tested if tested else None,
            'concordant_pairs':concordant,'rank_reversals':discordant,'tied_pairs':ties}


def _eligible(row):
    return row.get('status')=='complete' and row.get('physicality_passed') is True and row.get('mc_supported') is True and row.get('authoritative_fast_feasible') is True


def fixed_reference(rows, stage, fold, signals):
    """The predeclared anchor may normalize limits without being selectable."""
    reference={r['signal']:r for r in rows if
        (r['stage'],r['fold'],r['model'])==(stage,fold,REFERENCE) and r['signal'] in signals
        and r.get('status')=='complete' and r.get('mc_supported') is True
        and r.get('authoritative_fast_feasible') is True
        and isinstance(r.get('expected_limit'),(int,float))
        and math.isfinite(r['expected_limit']) and r['expected_limit']>0}
    return reference if set(reference)==set(signals) else None


def reference_context(reference):
    if reference is None:return {'available':False,'role':'fixed_diagnostic_normalization'}
    passed={s:r.get('physicality_passed') is True for s,r in reference.items()}
    return {'available':True,'role':'fixed_diagnostic_normalization',
            'physicality_passed':passed,'all_physicality_passed':all(passed.values()),
            'selection_eligible':all(_eligible(r) for r in reference.values())}


def global_selection(rows, stage, signals):
    """Choose one analysis configuration on development MC, across all signals."""
    folds={}
    for fold in ('development','validation'):
        models={}
        for row in rows:
            if (row['stage'],row['fold'])==(stage,fold) and _eligible(row):
                models.setdefault(row['model'],{})[row['signal']]=row
        folds[fold]={m:data for m,data in models.items() if set(data)==set(signals) and all(r.get('fast_significance') is not None for r in data.values())}
    development=folds['development'];validation=folds['validation']
    reference=fixed_reference(rows,stage,'development',signals)
    validation_reference=fixed_reference(rows,stage,'validation',signals)
    result={'required_signals':signals,'selection_metric':'mean_fast_significance',
            'validation_used_for_selection':False,'eligible_development_models':len(development),
            'reference_normalization':{'development':reference_context(reference),'validation':reference_context(validation_reference)},
            'eligibility_scope':'physicality and observed-template MC support; missing-component coverage and complete analysis validation remain separate requirements'}
    if not development:
        result['status']='inconclusive_no_globally_eligible_development_model';return result
    selected=max(development,key=lambda m:(np.mean([r['fast_significance'] for r in development[m].values()]),m))
    result.update(selected_model=selected,validation_eligible=selected in validation,
                  status='heldout_gates_passed_conditional_model' if selected in validation else 'inconclusive_selected_model_fails_validation_gates')
    if reference:
        best_limit=min(development,key=lambda m:np.mean([development[m][s]['expected_limit']/reference[s]['expected_limit'] for s in signals]))
        result['combine_model_chosen_on_development']=best_limit
        if selected in validation and best_limit in validation:
            ratios={s:validation[selected][s]['expected_limit']/validation[best_limit][s]['expected_limit'] for s in signals}
            result['validation_limit_ratio_fast_choice_to_combine_choice']=ratios
            result['validation_mean_limit_ratio_fast_choice_to_combine_choice']=float(np.mean(list(ratios.values())))
    if selected in validation and validation_reference:
        result['validation_limit_ratio_to_reference']={s:validation[selected][s]['expected_limit']/validation_reference[s]['expected_limit'] for s in signals}
    return result


def summarize(rows, required_signals=None, expected_task_count=None, binning_study=None):
    """Per-hypothesis agreement and held-out performance of development choices."""
    identities=[tuple(r[k] for k in ('fold','model','signal','stage')) for r in rows]
    if len(identities)!=len(set(identities)):raise ValueError('Duplicate comparison row identity')
    signals=sorted(set(required_signals)) if required_signals is not None else sorted({r['signal'] for r in rows})
    if any(r['signal'] not in signals for r in rows):raise ValueError('Unexpected signal benchmark')
    if expected_task_count is not None and len(rows)>expected_task_count:raise ValueError('Too many comparison rows')
    all_tasks_reported=expected_task_count is not None and len(rows)==expected_task_count
    paired={}
    for row in rows:
        if row.get('status')!='complete':continue
        identity=(row['fold'],row['model'],row['signal'])
        basis=(row.get('stage_contract_version'),row.get('finite_mc_basis_sha256'))
        if identity in paired and paired[identity]!=basis:
            raise ValueError('Likelihood stages have different finite-MC bases: '+repr(identity))
        paired[identity]=basis
    comparisons={};selection={}
    for stage in sorted({r['stage'] for r in rows}):
        for signal in signals:
            for fold in ('development','validation'):
                subset=[r for r in rows if (r['stage'],r['signal'],r['fold'])==(stage,signal,fold) and r['status']=='complete']
                for scope in ('diagnostic_all','physicality_and_mc_eligible'):
                    selected=subset if scope=='diagnostic_all' else [r for r in subset if _eligible(r)]
                    fast={r['model']:r['fast_significance'] for r in selected if r.get('fast_significance') is not None}
                    for metric in ('combine_significance','expected_limit'):
                        target={r['model']:r[metric]*(1 if metric=='combine_significance' else -1) for r in selected}
                        comparisons[f'{stage}/{signal}/{fold}/{scope}/{metric}']=ranking_agreement(fast,target)
            development={r['model']:r for r in rows if (r['stage'],r['signal'],r['fold'])==(stage,signal,'development') and _eligible(r)}
            validation={r['model']:r for r in rows if (r['stage'],r['signal'],r['fold'])==(stage,signal,'validation') and _eligible(r)}
            candidates=[r for r in development.values() if r['model']!=REFERENCE and r.get('fast_significance') is not None]
            if not candidates:continue
            best=max(candidates,key=lambda r:(r['fast_significance'],r['model']))
            selected={'model_chosen_on_development':best['model'],'selection_metric':'fast_significance',
                      'validation_used_for_selection':False,'validation_eligible':best['model'] in validation}
            anchor=fixed_reference(rows,stage,'validation',[signal])
            if best['model'] in validation and anchor:
                chosen,reference=validation[best['model']],anchor[signal]
                selected['reference_normalization']=reference_context(anchor)
                selected['validation_limit_ratio_to_reference']=chosen['expected_limit']/reference['expected_limit']
                selected['validation_significance_ratio_to_reference']=chosen['combine_significance']/reference['combine_significance'] if reference['combine_significance']>0 else None
            selection[f'{stage}/{signal}']=selected
    aggregate={}
    for stage in sorted({r['stage'] for r in rows}):
        for fold in ('development','validation'):
            group=[r for r in rows if (r['stage'],r['fold'])==(stage,fold) and _eligible(r)]
            by_model={}
            for row in group:by_model.setdefault(row['model'],{})[row['signal']]=row
            complete={model:data for model,data in by_model.items() if set(data)==set(signals) and all(r.get('fast_significance') is not None for r in data.values())}
            reference=fixed_reference(rows,stage,fold,signals)
            scores={}
            for model,data in complete.items():
                score={'mean_fast_significance':float(np.mean([r['fast_significance'] for r in data.values()]))}
                if reference:
                    ratios=[data[s]['expected_limit']/reference[s]['expected_limit'] for s in signals]
                    score.update(mean_expected_limit_ratio_to_reference=float(np.mean(ratios)),worst_expected_limit_ratio_to_reference=max(ratios))
                scores[model]=score
            # Regret uses sensitivity and a globally feasible candidate set.
            if complete:
                best={s:max(data[s]['fast_significance'] for data in complete.values()) for s in signals}
                for model,data in complete.items():
                    regrets=[(best[s]-data[s]['fast_significance'])/best[s] if best[s]>0 else 0. for s in signals]
                    scores[model].update(mean_sensitivity_regret=float(np.mean(regrets)),worst_sensitivity_regret=max(regrets))
            aggregate[f'{stage}/{fold}']={'required_signals':signals,'models':scores,'reference_normalization':reference_context(reference),'weighting':'equal weight per benchmark; limit ratios use the same-benchmark fixed diagnostic reference'}
            if reference:
                aggregate[f'{stage}/{fold}']['fast_vs_limits']=ranking_agreement({m:s['mean_fast_significance'] for m,s in scores.items()},{m:-s['mean_expected_limit_ratio_to_reference'] for m,s in scores.items()})
    provisional=not all_tasks_reported
    result={'schema_version':2,'provisional':provisional,
            'completion':{'received_tasks':len(rows),'expected_tasks':expected_task_count,'all_tasks_reported':all_tasks_reported},
            'required_signals':signals,'expected_only':True,'full_an_reproduction':False,
            'comparisons':comparisons,'development_selections':selection if all_tasks_reported else {},
            'aggregate_rankings':aggregate,
            'global_development_selections':{stage:global_selection(rows,stage,signals) for stage in sorted({r['stage'] for r in rows})} if all_tasks_reported else {},
            'counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
            'interpretation':'Correlations are descriptive on this fixed configuration set. Unsupported or unphysical points cannot establish an optimization winner.'}
    if binning_study is not None:
        result['binning_study']=binning_study
        if binning_study['validation_scope']=='exploratory_reuse':
            result['interpretation']+=' This study reuses inspected folds and changes binning and QCD grouping; any selection is exploratory and needs independent confirmation.'
            for selection in result['global_development_selections'].values():
                selection['validation_scope']='exploratory_reuse'
                if selection.get('status')=='heldout_gates_passed_conditional_model':
                    selection['status']='reused_fold_gates_passed_exploratory_model'
            for selection in result['development_selections'].values():
                selection['validation_scope']='exploratory_reuse'
    return result


def evaluate_inputs(production, output, nominal_only=False, region='SR', binning_study=None):
    from evaluate_reference import evaluate_reference_campaign, validate_disjoint_folds
    definition=json.loads((production/'definition.json').read_text())
    frozen=definition['frozen'];evaluations={};audits={}
    binning_study=validate_binning_study(binning_study,frozen['campaign_contents']['mass_bin_edges_gev'])
    for radius in frozen['radii']:
        prefix=f'ak{round(radius*10)}'
        nominal=production/(prefix+'_nominal')
        # Audit source and event identities before any model is fit or ranked.
        audits[prefix]=validate_disjoint_folds(nominal/'development_campaign.json',nominal/'validation_campaign.json')
        for fold in ('development','validation'):
            variations={}
            if not nominal_only:
                for systematic in ('JECUp','JECDown','JERUp','JERDown'):
                    group=production/(prefix+'_'+systematic)
                    if json.loads((group/'state.json').read_text())['phase']!='complete':raise ValueError(f'Incomplete systematic production: {group}')
                    variations[systematic]=group/(fold+'_campaign.json')
            if json.loads((nominal/'state.json').read_text())['phase']!='complete':raise ValueError(f'Incomplete nominal production: {nominal}')
            path=output/f'{prefix}_{fold}_evaluation.json'
            checksum=path.with_suffix('.sha256.json')
            if path.exists():
                if not checksum.exists() or json.loads(checksum.read_text())['sha256']!=digest(path):raise ValueError('Missing or mismatched evaluation cache checksum: '+str(path))
                result=json.loads(path.read_text())
            else:
                result=evaluate_reference_campaign(nominal/(fold+'_campaign.json'),variation_campaigns=variations,region=region,
                    bin_edges=binning_study['mass_bin_edges_gev'] if binning_study else None,
                    qcd_groups=binning_study['qcd_groups'] if binning_study else None,include_reference=True)
                if binning_study:
                    result['binning_study']=binning_study
                    for model in result['models'].values():
                        for info in model['per_signal'].values():
                            info['input']['provenance']['binning_study']=binning_study
                atomic_json(path,result)
                immutable_json(checksum,{'sha256':digest(path)})
            if result.get('binning_study')!=binning_study:
                raise ValueError('Cached evaluation belongs to a different binning study')
            if binning_study and (result['mass_bin_edges_gev']!=binning_study['mass_bin_edges_gev'] or result['qcd_groups']!=binning_study['qcd_groups']):
                raise ValueError('Cached evaluation has different mass edges or QCD groups')
            evaluations[(prefix,fold)]=result
    atomic_json(output/'fold_identity_audits.json',audits)
    return evaluations


def fit_one(task, output, config, wrapper, allow_unsupported_mc=False, timeout=300):
    fold,model,signal,stage,info=task
    path=output/'fits'/fold/model/signal/stage
    path.mkdir(parents=True,exist_ok=True)
    receipt=path/'comparison_row.json'
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous['status']!='complete':return previous
    authoritative_fast=info['fast_score']
    fast=(info.get('diagnostic_fast_score') or authoritative_fast) if allow_unsupported_mc and not authoritative_fast.get('feasible') else authoritative_fast
    physicality=info['physicality']
    row={'fold':fold,'model':model,'signal':signal,'stage':stage,'fast_significance':fast.get('significance'),
         'physicality_passed':physicality.get('physicality_pass',physicality.get('feasible',False)),
         'status':'preparing','path':str(path)}
    row['authoritative_fast_feasible']=authoritative_fast.get('feasible') is True
    row['authoritative_fast_failure_reasons']=authoritative_fast.get('failure_reasons',[])
    row['diagnostic_fast_score_used']=fast is not authoritative_fast
    row['background_coverage']=info['input'].get('provenance',{}).get('background_coverage')
    try:
        model_dir=path/'model'
        if (model_dir/'model.json').exists():
            manifest=json.loads((model_dir/'model.json').read_text())
            for name,checksum in manifest['artifacts_sha256'].items():
                if digest(model_dir/name)!=checksum:raise ValueError('Changed likelihood artifact on resume: '+str(model_dir/name))
        else:manifest=export_model(info['input'],model_dir,config,stage=stage,region=next(iter(info['input']['channels'])),allow_unsupported_mc=allow_unsupported_mc)
        row['mc_supported']=not manifest['unsupported_background_bins']
        row['stage_contract_version']=manifest.get('stage_contract_version')
        row['finite_mc_basis_sha256']=manifest.get('finite_mc_basis_sha256')
        row['missing_reference_nuisances']=manifest['missing_reference_nuisances']
        row['model_approximations']=manifest['approximations']
        inference=path/'inference';saved=inference/'combine_result.json'
        if saved.exists():
            result=json.loads(saved.read_text())
            if result['status']!='complete':raise ValueError('Previous inference failed/interrupted; inspect its logs and use a new comparison directory')
            if result['card_sha256']!=digest(model_dir/'datacard.txt'):raise ValueError('Datacard differs from completed inference')
            for name,artifact in result['artifacts'].items():
                if digest(inference/name)!=artifact['sha256']:raise ValueError('Changed inference artifact: '+str(inference/name))
            verify_saved_inference(result,inference)
        else:
            z=fast.get('significance') or 0.001
            result=run_expected(model_dir/'datacard.txt',inference,command_prefix=[wrapper],timeout_seconds=timeout,r_max=max(1000.,min(1e7,100./z)))
        row.update(status='complete',expected_limit=result['expected_limit']['median'],combine_significance=result['expected_significance'],combine_wall_seconds=result['wall_seconds'],fit_diagnostics=result['fit_diagnostics'])
        cross_section=info['input'].get('provenance',{}).get('signal_cross_section_pb')
        if cross_section is not None:row['expected_final_state_cross_section_limit_pb']=float(cross_section)*row['expected_limit']
    except CombineRunError as exc:row.update(status='fit_failed',error=str(exc))
    except (ValueError,KeyError,FileNotFoundError) as exc:row.update(status='unsupported_or_invalid',error=str(exc))
    atomic_json(receipt,row)
    return row


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--runtime-wrapper',required=True)
    parser.add_argument('--runtime-manifest',type=Path,help='Defaults to runtime_manifest.json beside the wrapper; hashes the actual Combine binary and library')
    parser.add_argument('--likelihood-config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--nominal-only',action='store_true',help='Explicit diagnostic comparison omitting kinematic shifts')
    parser.add_argument('--allow-unsupported-mc',action='store_true',help='Diagnostic fits only; never enables candidate selection')
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--timeout-seconds',type=float,default=300)
    parser.add_argument('--region',choices=('SR','CR','AT1b','AT0b'),default='SR')
    parser.add_argument('--binning-config',type=Path,help='Explicit immutable common coarsening and QCD grouping; use a new output directory')
    args=parser.parse_args(argv)
    if args.jobs<1 or args.jobs>4:parser.error('--jobs must be between1 and4')
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    production=args.production_dir.resolve();wrapper=str(Path(args.runtime_wrapper).resolve(strict=True))
    with (output/'comparison.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=json.loads(args.likelihood_config.read_text())
        production_definition=json.loads((production/'definition.json').read_text())
        binning_study=validate_binning_study(json.loads(args.binning_config.read_text()) if args.binning_config else None,
            production_definition['frozen']['campaign_contents']['mass_bin_edges_gev'])
        # Freeze the actual inputs, not merely the campaign filenames. An old
        # cached evaluation cannot survive regenerated/replaced ROOT files.
        input_hashes={}
        for campaign in sorted(production.glob('ak*/[dv]*_campaign.json')):
            input_hashes[str(campaign)]=digest(campaign)
            if args.nominal_only and campaign.parent.name.split('_',1)[1]!='nominal':continue
            for sample in json.loads(campaign.read_text())['samples']:
                for filename in sample['files']:input_hashes[filename]=digest(filename)
        code={name:digest(Path(__file__).parent/name) for name in ('compare_reference_likelihoods.py','evaluate_reference.py','likelihood_model.py','combine_runner.py','compact_scan_metrics.py','sensitivity_metrics.py','evaluate_sensitivity.py')}
        runtime=runtime_identity(wrapper,args.runtime_manifest or Path(wrapper).parent/'runtime_manifest.json')
        immutable_json(output/'definition.json',{'production_definition_sha256':digest(production/'definition.json'),'inputs_sha256':input_hashes,'code_sha256':code,'likelihood_config':config,'runtime_wrapper':wrapper,'runtime':runtime,'nominal_only':args.nominal_only,'allow_unsupported_mc':args.allow_unsupported_mc,'region':args.region,'timeout_seconds':args.timeout_seconds,'binning_study':binning_study})
        evaluations=evaluate_inputs(production,output,args.nominal_only,args.region,binning_study)
        tasks=[];reference={}
        for (radius,fold),evaluation in evaluations.items():
            for model,values in evaluation['models'].items():
                if model==REFERENCE:
                    # Reference reconstruction must be independent of the
                    # configurable AK radius, including its corrected weights.
                    if fold in reference:
                        for signal,info in values['per_signal'].items():
                            previous=reference[fold]['per_signal'][signal]['input']['channels']
                            if info['input']['channels']!=previous:raise ValueError('Reference templates differ across AK4/AK8 production')
                        continue
                    reference[fold]=values
                name=model if model==REFERENCE else radius+'_'+model
                for signal,info in values['per_signal'].items():
                    for stage in ('statistics_only','reference'):tasks.append((fold,name,signal,stage,info))
        required_signals=sorted(production_definition['frozen']['campaign_contents']['required_signals'])
        if any(set(values['per_signal'])!=set(required_signals) for evaluation in evaluations.values() for values in evaluation['models'].values()):raise ValueError('Evaluation lacks required signal benchmarks')
        rows=[];started=time.monotonic()
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            pending=[executor.submit(fit_one,task,output,config,wrapper,args.allow_unsupported_mc,args.timeout_seconds) for task in tasks]
            for future in as_completed(pending):
                row=future.result();rows.append(row)
                atomic_json(output/'rows.json',sorted(rows,key=lambda r:(r['fold'],r['model'],r['signal'],r['stage'])))
                atomic_json(output/'summary.json',summarize(rows,required_signals,len(tasks),binning_study))
                print(json.dumps({'finished':len(rows),'total':len(tasks),'model':row['model'],'fold':row['fold'],'signal':row['signal'],'stage':row['stage'],'status':row['status'],'elapsed_seconds':time.monotonic()-started}),flush=True)
        return 0 if all(r['status']=='complete' for r in rows) else 2


if __name__=='__main__':raise SystemExit(main())
