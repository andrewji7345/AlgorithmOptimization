#!/usr/bin/env python3
"""Export expected-inference comparison figures and a Markdown validation report.

Example (run in the project's Python environment)::

    python3 plot_reference_comparison.py --comparison-dir /path/to/comparison \\
        --output-dir /path/to/new/report

The default view is the held-out validation fold and representative likelihood.
No uncertainty bars are invented. Incomplete/unsupported fits remain explicit;
filled points pass both physicality and MC-support gates, hollow points do not.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from compare_reference_likelihoods import REFERENCE, ranking_agreement, _eligible


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(value, precision=4):
    return format(value, f'.{precision}g') if _finite(value) else '—'


def _cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def _background_support(info, minimum):
    """Summarize support in the same signal-populated bins used by the metric."""
    channels = info['input']['channels']
    if len(channels) != 1:
        raise ValueError('Expected one region per normalized model input')
    processes = next(iter(channels.values()))['processes']
    signal = np.asarray(processes['signal']['sumw'], dtype=float)
    background = np.zeros_like(signal)
    variance = np.zeros_like(signal)
    for process, values in processes.items():
        if process == 'signal':
            continue
        y, v = (np.asarray(values[key], dtype=float) for key in ('sumw', 'sumw2'))
        if y.shape != signal.shape or v.shape != signal.shape:
            raise ValueError('Inconsistent normalized process-bin shapes')
        background += y
        variance += v
    if signal.ndim != 1 or not np.all(np.isfinite(signal)) or not np.all(np.isfinite(background)) or not np.all(np.isfinite(variance)):
        raise ValueError('Nonfinite or nonvector normalized templates in evaluation')
    if np.any(variance < 0):
        raise ValueError('Negative sumw2 in normalized templates')
    populated = signal > 0
    valid = populated & (background > 0)
    neff = np.zeros_like(signal)
    neff[valid & (variance == 0)] = np.inf
    np.divide(background ** 2, variance, out=neff, where=valid & (variance > 0))
    values = neff[populated]
    smallest = float(np.min(values)) if len(values) else None
    return {'minimum_required': float(minimum), 'signal_populated_bins': int(populated.sum()),
            'supported_bins': int(np.count_nonzero(valid & (neff >= minimum))),
            'minimum_effective_events': smallest if smallest is not None and math.isfinite(smallest) else None,
            'all_populated_bins_zero_mc_variance': bool(len(values) and np.all(np.isinf(values))),
            'no_positive_background_bins': int(np.count_nonzero(populated & (background <= 0))),
            'background_yield': float(background.sum()),
            'failure_reasons': list(info.get('failure_reasons', [])),
            'physicality_passed': info.get('physicality', {}).get('physicality_pass') is True}


def _evaluation_models(evaluations, fold):
    """Map evaluator names to orchestrator names and deduplicate the AN anchor."""
    models, cutflows = {}, {}
    for filename, evaluation in sorted(evaluations.items()):
        match = re.fullmatch(r'(ak\d+)_(development|validation)_evaluation\.json', Path(filename).name)
        if not match or match.group(2) != fold:
            continue
        radius = match.group(1)
        for model, values in evaluation.get('models', {}).items():
            name = model if model == REFERENCE else radius + '_' + model
            for signal, info in values.get('per_signal', {}).items():
                key = (name, signal)
                support = _background_support(info, evaluation['minimum_background_effective_events'])
                if key in models and models[key] != support:
                    raise ValueError('AN anchor support differs between AK-radius evaluations')
                models[key] = support
            if model == REFERENCE:
                incoming = {sample: {'kind': info['kind'], 'cutflow': info.get('cutflow', {}),
                                     'region_counts': info.get('region_counts', {})}
                            for sample, info in values.get('samples', {}).items()}
                if cutflows and cutflows != incoming:
                    raise ValueError('AN anchor cutflows differ between AK-radius evaluations')
                cutflows = incoming
    return models, cutflows


def build_report_data(rows, evaluations=None, summary=None, fold='validation', stage='reference'):
    if not isinstance(rows, list):
        raise ValueError('rows.json must contain a list')
    identities = set()
    for row in rows:
        identity = tuple(row[key] for key in ('fold', 'model', 'signal', 'stage'))
        if identity in identities:
            raise ValueError('Duplicate comparison row identity: ' + repr(identity))
        identities.add(identity)
        if row.get('status') == 'complete':
            for metric in ('expected_limit', 'combine_significance'):
                if not _finite(row.get(metric)) or row[metric] <= 0:
                    raise ValueError('Completed row contains invalid ' + metric)
            if row.get('fast_significance') is not None and (not _finite(row['fast_significance']) or row['fast_significance'] < 0):
                raise ValueError('Completed row contains invalid fast significance')
    selected = [row for row in rows if row['fold'] == fold and row['stage'] == stage]
    support, cutflows = _evaluation_models(evaluations or {}, fold)
    required_signals = set((summary or {}).get('required_signals', []))
    signals = sorted(required_signals | {row['signal'] for row in selected} | {key[1] for key in support})
    names = sorted({row['model'] for row in selected} | {key[0] for key in support})
    labels, counts = {}, Counter()
    for name in names:
        if name == REFERENCE:
            labels[name] = 'AN'
        else:
            radius = name.split('_', 1)[0].upper()
            counts[radius] += 1
            labels[name] = f'{radius}-{counts[radius]}'
    details = {}
    for signal in signals:
        signal_rows = [row for row in selected if row['signal'] == signal]
        complete = [row for row in signal_rows if row['status'] == 'complete']
        points = [dict(row, label=labels[row['model']], inverse_expected_limit=1. / row['expected_limit'],
                       eligible=_eligible(row)) for row in complete if row.get('fast_significance') is not None]
        agreements = {}
        for scope in ('diagnostic_all', 'physicality_and_mc_eligible'):
            chosen = points if scope == 'diagnostic_all' else [row for row in points if row['eligible']]
            fast = {row['model']: row['fast_significance'] for row in chosen}
            agreements[scope] = {
                'significance': ranking_agreement(fast, {row['model']: row['combine_significance'] for row in chosen}),
                'inverse_limit': ranking_agreement(fast, {row['model']: row['inverse_expected_limit'] for row in chosen}),
            }
        signal_support = {name: values for (name, sig), values in support.items() if sig == signal}
        reasons = Counter(reason for values in signal_support.values() for reason in set(values['failure_reasons']))
        errors = Counter(row.get('error', row['status']) for row in signal_rows if row['status'] != 'complete')
        details[signal] = {'rows': signal_rows, 'points': points, 'status_counts': dict(Counter(row['status'] for row in signal_rows)),
                           'completed': len(complete), 'plotted': len(points), 'eligible': sum(_eligible(row) for row in complete),
                           'physicality_passed': (sum(values['physicality_passed'] for values in signal_support.values())
                                                   if signal_support else sum(row.get('physicality_passed') is True for row in signal_rows)),
                           'evaluated_models': len(signal_support) if signal_support else len(signal_rows), 'support': signal_support,
                           'failure_reason_counts': dict(reasons), 'fit_error_counts': dict(errors), 'agreement': agreements}
    observed_counts = dict(Counter(row['status'] for row in rows))
    completion = (summary or {}).get('completion', {})
    counts_match = summary is not None and summary.get('counts') == observed_counts
    received, expected = completion.get('received_tasks'), completion.get('expected_tasks')
    completion_matches = (type(received) is int and type(expected) is int
                          and received == expected == len(rows) and expected > 0
                          and completion.get('all_tasks_reported') is True)
    all_tasks_reported = (summary is not None and summary.get('provisional') is False
                          and counts_match and completion_matches
                          and required_signals.issubset({row['signal'] for row in rows}))
    snapshot_matches = (summary is None or (counts_match
                        and (not completion or received == len(rows))))
    return {'schema_version': 1, 'fold': fold, 'stage': stage, 'signals': details, 'labels': labels,
            'cutflows': cutflows, 'all_row_count': len(rows), 'selected_row_count': len(selected),
            'comparable_points': sum(detail['plotted'] for detail in details.values()),
            'eligible_points': sum(detail['eligible'] for detail in details.values()),
            'provisional': not all_tasks_reported,
            'completion': {'received_tasks': len(rows), 'expected_tasks': expected,
                           'all_tasks_reported': all_tasks_reported},
            'summary_matches_rows': snapshot_matches,
            'required_signals': sorted(required_signals),
            'binning_study': (summary or {}).get('binning_study'),
            'global_status_counts': observed_counts,
            'interpretation': 'Descriptive expected-inference diagnostics; this report does not select a configuration on the validation fold.'}


def _cutflow_table(samples):
    stages = ('processed', 'trigger', 'filters', 'lepton_veto', 'jet_veto', 'baseline',
              'baseline_and_gate', 'valid_reconstruction', 'selected_region')
    signal_names = sorted(name for name, sample in samples.items() if sample['kind'] == 'signal')
    backgrounds = [sample for sample in samples.values() if sample['kind'] == 'background']
    header = ['Selection', 'Background events', 'Background expected yield'] + signal_names
    lines = ['| ' + ' | '.join(map(_cell, header)) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |']
    for stage in stages:
        if not any(stage in sample['cutflow'] for sample in samples.values()):
            continue
        events = sum(sample['cutflow'].get(stage, {}).get('events', 0) for sample in backgrounds)
        y = sum(sample['cutflow'].get(stage, {}).get('yield', 0) for sample in backgrounds)
        cells = [stage, str(events), _number(y)] + [str(samples[name]['cutflow'].get(stage, {}).get('events', 0)) for name in signal_names]
        lines.append('| ' + ' | '.join(map(_cell, cells)) + ' |')
    return lines


def markdown_report(data, plot_paths):
    fold, stage = data['fold'], data['stage']
    study = data.get('binning_study')
    exploratory = bool(study and study.get('validation_scope') == 'exploratory_reuse')
    lines = ['# Expected-inference comparison', '', f'View: **{fold} fold, {stage} likelihood**. '
             f'This snapshot contains {data["selected_row_count"]} result rows for this view '
             f'({data["all_row_count"]} across all folds and stages).', '']
    if data['provisional']:
        lines += ['**Provisional snapshot: full task completion has not been established. No final selection is reported.**', '']
    if study:
        lines += [f'Binning study: **{_cell(study["name"])}**. {_cell(study["rationale"])}', '',
                  'Common Suu mass-bin edges (GeV): `' + json.dumps(study['mass_bin_edges_gev']) + '`. '
                  'QCD groups (zero-based bin indices): `' + json.dumps(study['qcd_groups']) + '`.', '',
                  _cell(study['qcd_group_interpretation']), '']
    if exploratory:
        lines += ['**Exploratory reuse of previously examined folds: this study measures stability on the existing '
                  'development and validation samples. It does not provide independent validation of the new binning '
                  'or an optimization winner.**', '']
    if not data['comparable_points']:
        lines += ['**No completed, comparable inference points are available. This snapshot cannot test ranking agreement.**', '']
    elif not data['eligible_points']:
        lines += ['**All plotted points are diagnostic: none passes both physicality and MC-support gates. '
                  'Their correlation does not validate an optimization winner.**', '']
    else:
        lines += [f'{data["eligible_points"]} completed rows pass both physicality and MC-support gates. '
                  + ('Agreement is descriptive on fixed configurations and previously examined folds.' if exploratory else
                     'Agreement is descriptive on the tested configurations; held-out results are not used to select a model.'), '']
    if not data['summary_matches_rows']:
        lines += ['The supplied summary and rows snapshots have different status counts or received-task totals. '
                  'Numbers below were recomputed from rows.json; the comparison may still be updating.', '']
    lines += ['Filled markers pass both gates; hollow markers are diagnostic. Stars identify the source-ported AN anchor. '
              'The inverse expected limit is 1/r95 for each signal benchmark; larger values indicate stronger expected exclusion. '
              'No uncertainty bars are assigned: expected CLs bands would not represent MC sampling uncertainty.', '']
    for path in plot_paths:
        lines.append(f'[{path.name}]({path.name})')
    lines += ['', '## Inference and constraint checks', '',
              '| Signal | Completed | Plotted | Eligible | Physicality passes / evaluated | Status counts |',
              '| --- | ---: | ---: | ---: | ---: | --- |']
    for signal, detail in data['signals'].items():
        cells = [signal, detail['completed'], detail['plotted'], detail['eligible'],
                 f'{detail["physicality_passed"]}/{detail["evaluated_models"]}', json.dumps(detail['status_counts'], sort_keys=True)]
        lines.append('| ' + ' | '.join(map(_cell, cells)) + ' |')
    lines += ['', 'Agreement uses increasing fast Z versus increasing Combine Z or 1/r95. '
              'Undefined correlations (too few points or all tied values) are shown as —.', '',
              '| Signal | Scope | Metric | Models | Spearman rho | Concordant / tested pairs | Rank reversals |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for signal, detail in data['signals'].items():
        for scope, metrics in detail['agreement'].items():
            for metric, values in metrics.items():
                cells = [signal, scope, metric, values['count'], _number(values['spearman']),
                         f'{values["concordant_pairs"]}/{values["concordant_pairs"]+values["rank_reversals"]}', values['rank_reversals']]
                lines.append('| ' + ' | '.join(map(_cell, cells)) + ' |')
    for signal, detail in data['signals'].items():
        lines += ['', f'### {_cell(signal)}: background support', '',
                  'n_eff = (sumw)²/sumw2 is evaluated only in signal-populated bins of the fixed mass map. '
                  'An absent positive background is unsupported; no epsilon yield is inserted.', '',
                  '| Model | Supported / signal-populated bins | Minimum n_eff | Required n_eff | Physicality |',
                  '| --- | ---: | ---: | ---: | --- |']
        for model, values in sorted(detail['support'].items()):
            minimum = '∞ (zero variance)' if values['all_populated_bins_zero_mc_variance'] else _number(values['minimum_effective_events'])
            cells = [data['labels'][model], f'{values["supported_bins"]}/{values["signal_populated_bins"]}', minimum,
                     _number(values['minimum_required']), 'pass' if values['physicality_passed'] else 'fail']
            lines.append('| ' + ' | '.join(map(_cell, cells)) + ' |')
        if detail['failure_reason_counts']:
            lines += ['', 'Evaluator failure reasons (a model can fail more than one gate):', '']
            for reason, count in sorted(detail['failure_reason_counts'].items()):
                lines.append(f'- {_cell(reason)}: {count}')
        if detail['fit_error_counts']:
            lines += ['', 'Failed or unsupported fit rows:', '']
            for error, count in sorted(detail['fit_error_counts'].items()):
                lines.append(f'- {_cell(error)}: {count}')
    if data['cutflows']:
        lines += ['', '## AN anchor cutflow', '',
                  'Signal columns show unweighted event counts. Background counts sum the MC samples; '
                  'the expected-yield column uses signed generator weights and luminosity normalization. '
                  'The anchor is counted once across AK radii.', '']
        lines += _cutflow_table(data['cutflows'])
        signal_names = sorted(name for name, sample in data['cutflows'].items() if sample['kind'] == 'signal')
        lines += ['', 'Region populations below are unweighted diagnostics. They do not establish background closure.', '',
                  '| Region | Background events | ' + ' | '.join(map(_cell, signal_names)) + ' |',
                  '| --- | ---: | ' + ' | '.join(['---:'] * len(signal_names)) + ' |']
        for region in ('SR', 'CR', 'AT1b', 'AT0b'):
            background = sum(sample['region_counts'].get(region, 0) for sample in data['cutflows'].values() if sample['kind'] == 'background')
            signals = [str(data['cutflows'][name]['region_counts'].get(region, 0)) for name in signal_names]
            lines.append(f'| {region} | {background} | ' + ' | '.join(signals) + ' |')
    lines += ['', '## Plot labels', '', '| Label | Configuration identifier |', '| --- | --- |']
    for model, label in data['labels'].items():
        lines.append(f'| {label} | {_cell(model)} |')
    lines += ['', 'This is a representative likelihood check, not a full AN reproduction or an observed discovery/exclusion result. '
              'Statistical-model and template limitations remain those recorded in the comparison/model manifests.', '']
    return '\n'.join(lines)


def make_plots(data, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    output = Path(output_dir)
    if not data['comparable_points']:
        counts = Counter()
        for detail in data['signals'].values():
            counts.update(detail['failure_reason_counts'])
            if not detail['failure_reason_counts']:
                counts.update(detail['status_counts'])
        fig, ax = plt.subplots(figsize=(10, max(4, .45 * len(counts) + 2)))
        if counts:
            labels, values = zip(*sorted(counts.items(), key=lambda item: item[1]))
            ax.barh(labels, values, color='#586b86')
            ax.set_xlabel('Affected model–signal rows (reasons can overlap)')
        else:
            ax.text(.5, .5, 'No inference rows are available in this snapshot.', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
        ax.set_title(f'{data["fold"]} / {data["stage"]}: no completed comparable fits\nRanking agreement has not been tested')
        name = 'comparison_status'
    else:
        signals = list(data['signals'])
        fig, axes = plt.subplots(len(signals), 2, figsize=(12, 4.2 * len(signals)), squeeze=False)
        for row_index, signal in enumerate(signals):
            detail = data['signals'][signal]
            for column, metric in enumerate(('combine_significance', 'inverse_expected_limit')):
                ax = axes[row_index, column]
                for point in detail['points']:
                    model = point['model']
                    color = '#1f77b4' if model.startswith('ak4_') else '#d17b0f' if model.startswith('ak8_') else '#343434'
                    marker = '*' if model == REFERENCE else 'o'
                    ax.scatter(point['fast_significance'], point[metric], s=160 if marker == '*' else 48,
                               marker=marker, edgecolors=color, facecolors=color if point['eligible'] else 'none',
                               linewidths=1.3, zorder=3)
                    ax.annotate(point['label'], (point['fast_significance'], point[metric]), xytext=(4, 4),
                                textcoords='offset points', fontsize=7)
                if not detail['points']:
                    ax.text(.5, .5, 'No completed comparable fits', ha='center', transform=ax.transAxes)
                if column == 0 and detail['points']:
                    limit = max(max(p['fast_significance'], p[metric]) for p in detail['points']) * 1.08
                    ax.plot([0, limit], [0, limit], color='.65', linestyle='--', linewidth=.8, zorder=1)
                if detail['points']:
                    xmax = max(p['fast_significance'] for p in detail['points'])
                    ymax = max(p[metric] for p in detail['points'])
                    if column == 0:
                        xmax = ymax = max(xmax, ymax)
                    ax.set_xlim(0, max(xmax * 1.15, 1.e-6))
                    ax.set_ylim(0, max(ymax * 1.15, 1.e-6))
                else:
                    ax.set_xlim(left=0)
                    ax.set_ylim(bottom=0)
                ax.set_xlabel('Fast Asimov significance Z')
                ax.set_ylabel('Combine expected significance Z' if column == 0 else 'Inverse median expected limit 1/r95')
                ax.set_title(f'{signal}\n{detail["plotted"]} points; {detail["eligible"]} eligible', fontsize=10)
                ax.grid(alpha=.2)
        handles = [Line2D([], [], marker='o', linestyle='', color='#1f77b4', label='AK4'),
                   Line2D([], [], marker='o', linestyle='', color='#d17b0f', label='AK8'),
                   Line2D([], [], marker='*', linestyle='', color='#343434', markersize=11, label='AN source-port anchor'),
                   Line2D([], [], marker='o', linestyle='', markerfacecolor='#555555', markeredgecolor='#555555', label='Eligible'),
                   Line2D([], [], marker='o', linestyle='', markerfacecolor='none', markeredgecolor='#555555', label='Diagnostic')]
        fig.legend(handles=handles, loc='upper center', ncol=5, bbox_to_anchor=(.5, .975), frameon=False)
        fig.suptitle(f'Expected-inference diagnostics: {data["fold"]} fold / {data["stage"]} likelihood', y=.997)
        name = 'validation_reference_comparison' if (data['fold'], data['stage']) == ('validation', 'reference') else f'{data["fold"]}_{data["stage"]}_comparison'
    if data['provisional']:
        fig.text(.995, .005, 'Provisional snapshot', ha='right', fontsize=9, color='#8b4513')
    if (data.get('binning_study') or {}).get('validation_scope') == 'exploratory_reuse':
        fig.text(.005, .005, 'Exploratory reuse of existing folds; no independent validation',
                 ha='left', fontsize=9, color='#8b4513')
    fig.tight_layout(rect=(0, 0, 1, .94 if data['comparable_points'] else 1))
    paths = []
    for suffix in ('png', 'pdf'):
        path = output / f'{name}.{suffix}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        paths.append(path)
    plt.close(fig)
    return paths


def generate_report(comparison_dir, output_dir, fold='validation', stage='reference'):
    comparison, output = Path(comparison_dir).resolve(), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty report output directory to preserve previous snapshots')
    inputs = {}

    def read(path):
        raw = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    rows = read(comparison / 'rows.json') if (comparison / 'rows.json').exists() else []
    summary = read(comparison / 'summary.json') if (comparison / 'summary.json').exists() else None
    evaluations = {path.name: read(path) for path in sorted(comparison.glob('ak*_evaluation.json'))}
    if not rows and not evaluations:
        raise ValueError('No comparison rows or evaluator artifacts found')
    data = build_report_data(rows, evaluations, summary, fold, stage)
    output.mkdir(parents=True, exist_ok=True)
    paths = make_plots(data, output)
    report = output / 'comparison_report.md'
    report.write_text(markdown_report(data, paths))
    (output / 'plot_data.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    manifest = {'schema_version': 1, 'inputs_sha256': inputs, 'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'fold': fold, 'stage': stage, 'artifacts': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths + [report, output / 'plot_data.json']}}
    (output / 'report_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return {'report': str(report), 'plots': [str(path) for path in paths], 'comparable_points': data['comparable_points'], 'eligible_points': data['eligible_points']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--comparison-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fold', choices=('development', 'validation'), default='validation')
    parser.add_argument('--stage', choices=('statistics_only', 'reference'), default='reference')
    args = parser.parse_args(argv)
    try:
        result = generate_report(args.comparison_dir, args.output_dir, args.fold, args.stage)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'Cannot generate comparison report: {exc}\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
