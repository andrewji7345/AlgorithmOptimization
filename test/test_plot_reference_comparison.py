import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from plot_reference_comparison import build_report_data, generate_report, markdown_report


def row(model='ak4_candidate', signal='WbWb_4000_1000', **updates):
    value = {'fold': 'validation', 'stage': 'reference', 'model': model, 'signal': signal,
             'status': 'complete', 'fast_significance': 2., 'combine_significance': 1.5,
             'expected_limit': 1.2, 'physicality_passed': True, 'mc_supported': True, 'authoritative_fast_feasible': True}
    value.update(updates)
    return value


def evaluation(signal='WbWb_4000_1000', reference=False):
    info = {'input': {'channels': {'SR': {'processes': {
        'signal': {'sumw': [2., 0.], 'sumw2': [.1, 0.]},
        'QCD': {'sumw': [20., 1.], 'sumw2': [40., 100.]},
        'TTbar': {'sumw': [0., 0.], 'sumw2': [0., 0.]},
    }}}}, 'failure_reasons': ['insufficient_background_effective_events'],
        'physicality': {'physicality_pass': True}}
    model = {'per_signal': {signal: info}, 'samples': {
        'bkg': {'kind': 'background', 'cutflow': {'processed': {'events': 100, 'yield': 400}},
                'region_counts': {'SR': 20}},
        signal: {'kind': 'signal', 'cutflow': {'processed': {'events': 80, 'yield': 30}},
                 'region_counts': {'SR': 4}},
    }}
    return {'models': {'reference_AN2017' if reference else 'candidate': model},
            'minimum_background_effective_events': 32.6530612244898}


class PlotReferenceComparisonTest(unittest.TestCase):
    def test_only_requested_fold_and_stage_enter_plot_and_rankings(self):
        rows = [row(), row('ak8_candidate', fast_significance=3., expected_limit=.7),
                row('development_only', fold='development', fast_significance=1000.),
                row('statistics_only', stage='statistics_only', fast_significance=1000.)]
        data = build_report_data(rows)
        detail = data['signals']['WbWb_4000_1000']
        self.assertEqual({point['model'] for point in detail['points']}, {'ak4_candidate', 'ak8_candidate'})
        self.assertEqual(data['selected_row_count'], 2)
        self.assertAlmostEqual(detail['agreement']['diagnostic_all']['inverse_limit']['spearman'], 1.)

    def test_failed_and_ineligible_points_are_not_ranked_as_eligible(self):
        rows = [row(), row('reference_AN2017', physicality_passed=False),
                row('ak8_sparse', mc_supported=False),
                row('ak4_failed', status='fit_failed', error='fit failed', fast_significance=None)]
        data = build_report_data(rows)
        detail = data['signals']['WbWb_4000_1000']
        self.assertEqual(len(detail['points']), 3)
        self.assertEqual(sum(point['eligible'] for point in detail['points']), 1)
        self.assertEqual(detail['agreement']['physicality_and_mc_eligible']['significance']['count'], 1)
        self.assertIsNone(detail['agreement']['physicality_and_mc_eligible']['significance']['spearman'])
        self.assertEqual(data['labels']['reference_AN2017'], 'AN')
        self.assertEqual(detail['fit_error_counts'], {'fit failed': 1})

    def test_authoritative_support_failure_is_hollow_and_report_is_provisional(self):
        data=build_report_data([row(authoritative_fast_feasible=False)])
        self.assertEqual(data['eligible_points'],0)
        self.assertFalse(data['signals']['WbWb_4000_1000']['points'][0]['eligible'])
        self.assertIn('Provisional snapshot',markdown_report(data,[]))

    def test_mc_support_uses_sumw2_and_only_signal_populated_bins(self):
        data = build_report_data([], {'ak4_validation_evaluation.json': evaluation()})
        support = data['signals']['WbWb_4000_1000']['support']['ak4_candidate']
        self.assertEqual(support['signal_populated_bins'], 1)
        self.assertEqual(support['minimum_effective_events'], 10.)
        self.assertEqual(support['supported_bins'], 0)
        self.assertEqual(data['comparable_points'], 0)
        text = markdown_report(data, [])
        self.assertIn('cannot test ranking agreement', text)
        self.assertIn('insufficient_background_effective_events: 1', text)
        self.assertIn('No uncertainty bars are assigned', text)

    def test_reference_cutflows_deduplicate_and_mismatches_fail(self):
        first, second = evaluation(reference=True), evaluation(reference=True)
        data = build_report_data([], {'ak4_validation_evaluation.json': first,
                                      'ak8_validation_evaluation.json': second})
        self.assertEqual(len(data['signals']['WbWb_4000_1000']['support']), 1)
        self.assertEqual(data['cutflows']['bkg']['cutflow']['processed']['events'], 100)
        second['models']['reference_AN2017']['samples']['bkg']['cutflow']['processed']['events'] = 101
        with self.assertRaisesRegex(ValueError, 'cutflows differ'):
            build_report_data([], {'ak4_validation_evaluation.json': first,
                                   'ak8_validation_evaluation.json': second})

    def test_invalid_complete_rows_and_duplicates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            build_report_data([row(), row()])
        for updates in ({'expected_limit': 0}, {'combine_significance': float('nan')},
                        {'fast_significance': float('inf')}):
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, 'invalid'):
                build_report_data([row(**updates)])

    def test_summary_snapshot_mismatch_is_explicit(self):
        data = build_report_data([row()], summary={'counts': {'complete': 5}})
        self.assertFalse(data['summary_matches_rows'])
        self.assertIn('different status counts', markdown_report(data, []))

    def test_required_benchmarks_remain_visible_without_rows_or_evaluations(self):
        summary = {'required_signals': ['s1', 's2', 's3'], 'provisional': True,
                   'counts': {'complete': 1}}
        data = build_report_data([row(signal='s1')], summary=summary)
        self.assertEqual(set(data['signals']), {'s1', 's2', 's3'})
        for missing in ('s2', 's3'):
            self.assertEqual(data['signals'][missing]['completed'], 0)
            self.assertEqual(data['signals'][missing]['points'], [])
            self.assertIn('| ' + missing + ' | 0 | 0 | 0 |', markdown_report(data, []))

    def test_nonprovisional_requires_matching_counts_and_completion_receipt(self):
        summary = {'required_signals': ['s1'], 'provisional': False,
                   'counts': {'complete': 1},
                   'completion': {'received_tasks': 1, 'expected_tasks': 1,
                                  'all_tasks_reported': True}}
        data = build_report_data([row(signal='s1')], summary=summary)
        self.assertFalse(data['provisional'])
        self.assertTrue(data['completion']['all_tasks_reported'])
        bad_summaries = []
        for field, value in [('counts', {'complete': 204}), ('completion', {}),
                             ('required_signals', ['s1', 's2'])]:
            changed = copy.deepcopy(summary)
            changed[field] = value
            bad_summaries.append(changed)
        for field, value in [('received_tasks', 204), ('expected_tasks', 204),
                             ('all_tasks_reported', False), ('expected_tasks', True)]:
            changed = copy.deepcopy(summary)
            changed['completion'][field] = value
            bad_summaries.append(changed)
        for changed in bad_summaries:
            with self.subTest(summary=changed):
                data = build_report_data([row(signal='s1')], summary=changed)
                self.assertTrue(data['provisional'])
                self.assertFalse(data['completion']['all_tasks_reported'])
                self.assertIn('Provisional snapshot', markdown_report(data, []))

    def test_exploratory_binning_metadata_and_validation_scope_are_reported(self):
        study = {'name': 'coarse-example', 'validation_scope': 'exploratory_reuse',
                 'rationale': 'Merge sparse neighboring bins using a common map.',
                 'qcd_group_interpretation': 'Independent QCD factors between merged bins.',
                 'mass_bin_edges_gev': [0, 3500, 12000], 'qcd_groups': [[0], [1]]}
        data = build_report_data([row()], summary={'binning_study': study})
        self.assertEqual(data['binning_study'], study)
        text = markdown_report(data, [])
        for expected in ('coarse-example', '[0, 3500, 12000]', '[[0], [1]]',
                         study['rationale'], study['qcd_group_interpretation'],
                         'does not provide independent validation', 'previously examined folds'):
            self.assertIn(expected, text)
        self.assertNotIn('held-out results are not used to select', text)

    def test_provisional_report_renders_real_images(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);comparison=base/'comparison';comparison.mkdir()
            (comparison/'rows.json').write_text(json.dumps([row(authoritative_fast_feasible=False)]))
            study = {'name': 'coarse-example', 'validation_scope': 'exploratory_reuse',
                     'rationale': 'Common sparse-bin merger.',
                     'qcd_group_interpretation': 'Independent QCD factors.',
                     'mass_bin_edges_gev': [0, 3500, 12000], 'qcd_groups': [[0], [1]]}
            (comparison/'summary.json').write_text(json.dumps({
                'required_signals': ['WbWb_4000_1000', 'missing_benchmark'], 'binning_study': study}))
            result=generate_report(comparison,base/'report')
            self.assertEqual(result['eligible_points'],0)
            self.assertEqual(len(result['plots']),2)
            for path in result['plots']:self.assertGreater(Path(path).stat().st_size,1000)
            self.assertIn('Provisional snapshot',Path(result['report']).read_text())
            exported = json.loads((base/'report/plot_data.json').read_text())
            self.assertEqual(exported['binning_study'], study)
            self.assertIn('missing_benchmark', exported['signals'])

    def test_report_snapshots_preserve_input_hashes_and_reject_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            comparison = base / 'comparison'
            comparison.mkdir()
            (comparison / 'rows.json').write_text(json.dumps([row()]))
            out = base / 'report'
            with patch('plot_reference_comparison.make_plots', return_value=[]):
                result = generate_report(comparison, out)
                with self.assertRaisesRegex(ValueError, 'empty report'):
                    generate_report(comparison, out)
            self.assertTrue(Path(result['report']).exists())
            manifest = json.loads((out / 'report_manifest.json').read_text())
            self.assertIn(str(comparison / 'rows.json'), manifest['inputs_sha256'])
            self.assertIn('comparison_report.md', manifest['artifacts'])


if __name__ == '__main__':
    unittest.main()
