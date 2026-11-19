#!/usr/bin/env python3
"""Prepare, submit, monitor, and plot a fixed AN reference cosine scan."""
import argparse
import copy
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from plot_sj_tagging_efficiency import digest
from sensitivity_jobs import JobError, atomic_json, immutable_json, immutable_text

REPO = Path(__file__).resolve().parent
SOURCE_FILES = (
    'run_an_cosine_scan.py', 'plot_an_cosine_scan.py', 'plot_sj_tagging_efficiency.py',
    'sensitivity_jobs.py', 'run_scripts/sensitivity/worker.py',
    'run_scripts/sensitivity/worker.sh', 'test/validate_compact_scan.py',
    'test/validate_reference_cosine_scan.py', 'test/runCompactOptimizationScan_cfg.py',
    'test/runSensitivityScan_cfg.py', 'python/CompactOptimizationScanNtuplizer_cfi.py',
    'plugins/CompactOptimizationScanNtuplizer.cc',
)


def validate_config(config):
    """Reject short or duplicate source coverage before allocating any jobs."""
    required = {'description', 'cmssw_release', 'max_events', 'reference_cos_thrust_cuts', 'samples'}
    if not isinstance(config, dict) or set(config) != required:
        raise JobError('Scan configuration must declare exactly: ' + ', '.join(sorted(required)))
    maximum = config['max_events']
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise JobError('max_events must be a positive integer')
    cuts = config['reference_cos_thrust_cuts']
    if (not isinstance(cuts, list) or not 1 <= len(cuts) <= 32 or
            any(isinstance(c, bool) or not isinstance(c, (int, float)) or
                not math.isfinite(c) or not 0 <= c <= 1 for c in cuts) or
            cuts != sorted(set(cuts))):
        raise JobError('Expected 1 to 32 sorted, unique cosine cutoffs in [0,1]')
    if not isinstance(config['samples'], list) or not config['samples']:
        raise JobError('At least one signal sample is required')
    names, sources = set(), set()
    for sample in config['samples']:
        if set(sample) != {'name', 'input_files'}:
            raise JobError('Each sample must declare name and input_files')
        name = sample['name']
        parts = name.split('_') if isinstance(name, str) else []
        if len(parts) != 3 or parts[0] != 'WbWb' or any(not p.isdigit() or int(p) <= 0 for p in parts[1:]):
            raise JobError('Mass plots require a WbWb_<mSuu>_<mChi> sample name')
        if name in names:
            raise JobError('Duplicate sample: ' + name)
        names.add(name)
        if not isinstance(sample['input_files'], list) or not sample['input_files']:
            raise JobError('Missing input files for ' + name)
        available = 0
        for source in sample['input_files']:
            if set(source) != {'lfn', 'nevents'}:
                raise JobError('Each input must declare its CMS lfn and DAS nevents')
            lfn, count = source['lfn'], source['nevents']
            if not isinstance(lfn, str) or not lfn.startswith('/store/') or any(c.isspace() for c in lfn):
                raise JobError('Invalid CMS logical filename')
            if lfn in sources:
                raise JobError('Duplicate source file: ' + lfn)
            sources.add(lfn)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise JobError('DAS nevents must be a positive integer')
            available += count
        if available < maximum:
            raise JobError(f'Insufficient input coverage for {name}: {available} < {maximum}')
    if (not isinstance(config['cmssw_release'], str) or
            not config['cmssw_release'].startswith('CMSSW_') or '/' in config['cmssw_release']):
        raise JobError('Invalid CMSSW release name')
    return config


def backend_for(run):
    # Resume through the scheduler adapter frozen with this run.
    spec = importlib.util.spec_from_file_location('frozen_cosine_jobs', run / 'source/sensitivity_jobs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = module.CondorBackend(json.loads((run / 'backend.json').read_text()), run / 'source')
    backend.state_dir = run
    backend.resolve_schedd()
    return backend


def prepare(args):
    if args.bundle is None:
        raise JobError('prepare requires --bundle built from the scan-capable CMSSW release')
    config = validate_config(json.loads(args.config.read_text()))
    run, archive = args.run_dir, args.bundle.resolve()
    if not archive.is_file() or archive.stat().st_size == 0:
        raise JobError('Missing or empty CMSSW bundle: ' + str(archive))
    run.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in SOURCE_FILES:
        target = run / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        immutable_text(target, (REPO / name).read_text())
        hashes[name] = digest(target)
    definition = copy.deepcopy(config)
    definition.update(archive=str(archive), archive_sha256=digest(archive), code_sha256=hashes,
                      run_directory=str(run), reconstruction='AN23-067-PATAK8-CA8-Thrust-source-port-v1',
                      efficiency='N(baseline and reference SR at c) / N(baseline)',
                      primary='Unweighted counts with 68.27% Wilson intervals; failures stay in denominator',
                      mass_plots='SR-selected reconstructed masses; unit area within displayed range')
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    token = 'eo_' + fingerprint[:20] + '_0'
    grid = dict(collection_pt_cuts=[300], ca_radii=[0.8], cos_thrust_cuts=[0.85],
                gate_counts=[0], gate_pt_cuts=[300])
    tasks = []
    for index, sample in enumerate(config['samples']):
        directory = run / 'production/tasks' / str(index)
        directory.mkdir(parents=True, exist_ok=True)
        inputs = ['root://cmsxrootd.fnal.gov/' + source['lfn'] for source in sample['input_files']]
        task = dict(index=index, sample=sample['name'], sample_kind='signal', input_files=inputs,
                    output=str(directory / 'output.root'),
                    parameters=dict(r_ak=0.8, r_ca=0.8, t_keep=300, t_gate=300, cos_thrust=0.85, n_gate_jets=0),
                    scan_grid=grid, reference_cos_thrust_cuts=config['reference_cos_thrust_cuts'],
                    analysis_systematic='nominal', trial_token=token, max_events=config['max_events'],
                    require_full_event_coverage=True, cmssw_release=config['cmssw_release'],
                    bundle_name=archive.name, cmsrun_timeout_seconds=14400)
        immutable_json(directory / 'task.json', task)
        immutable_text(directory / 'inputs.txt', '\n'.join(inputs) + '\n')
        tasks.append(task)
    manifest = dict(token=token, fingerprint=fingerprint, definition=definition, tasks=tasks)
    immutable_json(run / 'manifest.json', manifest)
    immutable_json(run / 'bundle.json', dict(name=archive.name, sha256=definition['archive_sha256'],
                                            bytes=archive.stat().st_size))
    immutable_json(run / 'backend.json', dict(kind='condor', schedd=args.schedd, cmssw_bundle=str(archive),
        minimum_proxy_seconds=18000, request_cpus=1, request_memory_mb=4000,
        request_disk_mb=20000, command_timeout_seconds=45))
    # Preparing an explicit scheduler is offline. 'auto' is resolved at submit.
    backend = backend_for(run) if args.schedd != 'auto' else None
    if backend is None:
        from sensitivity_jobs import CondorBackend
        backend = CondorBackend(json.loads((run / 'backend.json').read_text()), run / 'source')
    backend.prepare(run / 'production', manifest)
    if not (run / 'status.json').exists():
        atomic_json(run / 'status.json', dict(phase='prepared', token=token, job_ids=[]))
    print(f'Prepared {len(tasks)} jobs; {config["max_events"]} events per sample', flush=True)


def load(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    definition = manifest['definition']
    for name, expected in definition['code_sha256'].items():
        if digest(run / 'source' / name) != expected:
            raise JobError('Frozen run source changed: ' + name)
    if digest(definition['archive']) != definition['archive_sha256']:
        raise JobError('Frozen CMSSW bundle changed')
    return backend_for(run), manifest


def submit(run):
    with (run / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        backend, manifest = load(run)
        state = json.loads((run / 'status.json').read_text())
        if state['phase'] != 'prepared':
            raise JobError('Refusing resubmission in phase ' + state['phase'] + '; resume with watch')
        backend.preflight()
        state.update(phase='submitting', submitted_at=time.time())
        atomic_json(run / 'status.json', state)
        # An interrupted submit response remains ambiguous: watch reconciles
        # the persisted token; this command never retries a submission.
        ids = backend.submit(run / 'production', manifest, state)
        state.update(phase='running', job_ids=ids)
        atomic_json(run / 'status.json', state)
        print('Submitted ' + ', '.join(ids), flush=True)


def watch(run, once=False):
    with (run / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        backend, manifest = load(run)
        while True:
            state = json.loads((run / 'status.json').read_text())
            if state['phase'] in ('complete', 'failed'):
                print(json.dumps(state, indent=2), flush=True)
                return
            if state['phase'] not in ('submitting', 'running', 'plotting'):
                raise JobError('Submit prepared jobs before monitoring')
            try:
                status = backend.lookup(run / 'production', manifest, state)
            except Exception as exc:
                state.update(last_poll_error=str(exc), last_poll_time=time.time())
                atomic_json(run / 'status.json', state)
                print('Scheduler query failed; preserving jobs: ' + str(exc), flush=True)
                if once:
                    raise
                time.sleep(60)
                continue
            state.pop('last_poll_error', None)
            state.update(scheduler_state=status, last_poll_time=time.time())
            if status.get('job_ids'):
                state['job_ids'] = status['job_ids']
            if status['state'] == 'failed':
                state.update(phase='failed', reason=status.get('reason'))
                atomic_json(run / 'status.json', state)
                raise JobError('Scan failed: ' + str(status))
            if status['state'] == 'complete':
                state.update(phase='plotting')
                atomic_json(run / 'status.json', state)
                try:
                    subprocess.run([sys.executable, str(run / 'source/plot_an_cosine_scan.py'),
                        '--manifest', str(run / 'manifest.json'), '--output-dir', str(run / 'plots')],
                        cwd=run, check=True)
                    if not (run / 'plots/COMPLETE.json').is_file():
                        raise JobError('Missing plot completion receipt')
                except Exception as exc:
                    state.update(phase='failed', reason='Plotting failed: ' + str(exc))
                    atomic_json(run / 'status.json', state)
                    raise
                state.update(phase='complete', finished_at=time.time(), plots=str(run / 'plots'))
                atomic_json(run / 'status.json', state)
                print('Production and plots complete', flush=True)
                return
            atomic_json(run / 'status.json', state)
            print(json.dumps(dict(phase=state['phase'], scheduler_state=status)), flush=True)
            if once:
                return
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit', 'watch', 'status'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=REPO / 'config/an_cosine_scan_2017.json')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--schedd', default='auto')
    parser.add_argument('--once', action='store_true', help='Poll once during watch')
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    override = args.run_dir / 'condor_config'
    if override.is_dir():
        os.environ['FERMIHTC_LOCALCONF_OVERRIDE'] = str(override)
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'submit':
        submit(args.run_dir)
    elif args.action == 'watch':
        watch(args.run_dir, once=args.once)
    else:
        print((args.run_dir / 'status.json').read_text(), end='')


if __name__ == '__main__':
    main()
