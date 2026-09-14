"""AN-weighted 2D scan integration and independent numerical edge cases."""
import contextlib
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import evaluate_reference as reference
import evaluate_sensitivity as evaluator
from sensitivity_metrics import asimov_significance, weighted_histogram
from sensitivity_objective import context_from_result, resolve_objective
from freeze_sensitivity_references import freeze_references
from test_evaluate_reference import fixture, synchronize_regions


class Nominal2DWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.campaign, self.metadata, self.payloads, self.extras = fixture()
        self.campaign.update(
            mass_bin_edges_gev=list(range(2500, 10001, 500)),
            chi_mass_bin_edges_gev=list(range(750, 5001, 250)),
            histogram_flow="exclude", weight_convention="reference",
            physicality_mode="off", minimum_background_effective_events=0,
        )
        self.key = evaluator.parse_configuration("0:none:100:.8:.8:0")
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(mock.patch.object(evaluator, "load_metadata", side_effect=lambda p: self.metadata[str(p)]))
        self.stack.enter_context(mock.patch.object(evaluator, "read_event_payload", side_effect=lambda m: self.payloads[m.path]))
        self.stack.enter_context(mock.patch.object(reference, "_read_additive", side_effect=lambda m: self.extras[m.path]))
        self.addCleanup(self.stack.close)

    def test_an_weights_and_actual_two_dimensional_yields(self):
        result = evaluator.evaluate_campaign(self.campaign, self.key)
        self.assertTrue(result["feasible"])
        self.assertEqual(np.shape(result["background"]["sumw"]), (15, 17))
        self.assertAlmostEqual(np.sum(result["background"]["sumw"]), 1980.)
        self.assertAlmostEqual(result["per_signal"]["WbWb_4000_1000"]["signal_yield"], 90.)
        self.assertIsNone(result["per_signal"]["WbWb_4000_1000"]["physicality"])
        # Legacy analysisWeight=.1; the referenceWeight=.9 must enter once.
        self.assertAlmostEqual(result["background"]["sumw"][3][1], 1980.)
        self.assertEqual(result["samples"]["TTJets_fixture"]["category"], "TTbarMC")

    def test_scan_gate_changes_acceptance_not_denominator_or_reference_population(self):
        for payload in self.payloads.values():
            payload.ak_jet_pt[:20, 2:] = 100.
        result = evaluator.evaluate_campaign(self.campaign, evaluator.parse_configuration("4:400:100:.8:.8:0"))
        signal = result["samples"]["WbWb_4000_1000"]
        self.assertEqual(signal["selected_events"], 80)
        self.assertEqual(signal["sum_gen_weights"], 85.)
        self.assertAlmostEqual(np.sum(signal["histogram"]["sumw"]), 100 / 85 * .9 * 65)
        baseline = evaluator.evaluate_campaign(self.campaign, self.key)
        self.assertEqual(baseline["comparison_context"], result["comparison_context"])

    def test_mass_quality_cannot_reject_valid_asimov_when_physicality_off(self):
        for payload in self.payloads.values():
            payload.sj1_mass[:] = 2000.
            payload.sj2_mass[:] = 2000.
        result = evaluator.evaluate_campaign(self.campaign, self.key)
        self.assertTrue(result["feasible"])
        self.assertGreater(result["objective"], 0.)
        self.assertFalse(any("physicality" in r for r in result["failure_reasons"]))

    def test_selected_zero_signal_is_retained_with_zero_score(self):
        p = self.payloads["signal.root"]
        p.passes_baseline[:] = False
        synchronize_regions(p, self.extras["signal.root"][1])
        result = evaluator.evaluate_campaign(self.campaign, self.key)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["objective"], 0.)
        self.assertEqual(result["per_signal"]["WbWb_4000_1000"]["significance"], 0.)

    def test_absent_background_component_is_reported_without_inventing_yield(self):
        payload = self.payloads["ttbar.root"]
        payload.passes_baseline[:] = False
        synchronize_regions(payload, self.extras["ttbar.root"][1])
        result = evaluator.evaluate_campaign(self.campaign, self.key)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["background_coverage"]["unobserved_samples"], ["TTJets_fixture"])
        self.assertTrue(result["background_coverage"]["unobserved_components_require_review"])
        self.assertAlmostEqual(np.sum(result["background"]["sumw"]), 1800.)

    def test_outside_masses_are_not_added_to_edge_cells(self):
        p = self.payloads["qcd.root"]
        p.suu_mass[:10] = 2400.
        p.sj1_mass[10:20] = p.sj2_mass[10:20] = 5100.
        result = evaluator.evaluate_campaign(self.campaign, self.key)
        hist = result["samples"]["QCDMC_Pt_fixture"]["histogram"]
        self.assertEqual(hist["flow_entries"], 20)
        self.assertEqual(np.sum(hist["entries"]), 80)
        self.assertAlmostEqual(np.sum(hist["sumw"]) + hist["flow_sumw"], 1800.)
        self.assertEqual(result["samples"]["QCDMC_Pt_fixture"]["sum_gen_weights"], 85.)

    def test_region_metadata_disagreement_rejected_even_without_physicality(self):
        self.payloads["qcd.root"].passes_signal_region[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "disagrees"):
            evaluator.evaluate_campaign(self.campaign, self.key)

    def test_identity_context_detects_same_yields_from_different_events(self):
        first = evaluator.evaluate_campaign(self.campaign, self.key)
        self.payloads["qcd.root"].event += 1000
        second = evaluator.evaluate_campaign(self.campaign, self.key)
        self.assertEqual(first["background"], second["background"])
        self.assertNotEqual(first["comparison_context"], second["comparison_context"])

    def test_evaluated_calibration_freezes_and_scores_another_gate(self):
        self.campaign["objective"] = {"method": "mean_asimov"}
        baseline = evaluator.evaluate_campaign(self.campaign, self.key)
        self.assertEqual(context_from_result(baseline), baseline["comparison_context"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign_path, result_path = root / "campaign.json", root / "baseline.json"
            campaign_path.write_text(json.dumps(self.campaign))
            result_path.write_text(json.dumps(evaluator.finite_json(baseline), allow_nan=False))
            freeze_references(campaign_path, [result_path], root / "reference.json")
            self.campaign["objective"] = resolve_objective(
                {"method": "fixed_reference_regret", "mean_weight": .25,
                 "reference_file": "reference.json"}, root)
            same = evaluator.evaluate_campaign(self.campaign, self.key)
            self.assertAlmostEqual(same["objective"], 1.25)
            self.assertAlmostEqual(same["reference_regret_objective"], 0.)
            for payload in self.payloads.values():
                payload.ak_jet_pt[:20, 2:] = 100.
            gated = evaluator.evaluate_campaign(self.campaign, evaluator.parse_configuration("4:400:100:.8:.8:0"))
            self.assertTrue(gated["feasible"])
            self.assertNotEqual(gated["objective"], same["objective"])
            self.assertEqual(gated["objective_definition"], same["objective_definition"])
            self.payloads["qcd.root"].event += 1000
            with self.assertRaisesRegex(ValueError, "physical MC context"):
                evaluator.evaluate_campaign(self.campaign, self.key)


class AsimovCellTests(unittest.TestCase):
    def test_unusable_cell_does_not_erase_valid_cell_diagnostics(self):
        result = asimov_significance(np.array([[5., 5.], [0., 0.]]),
                                    np.array([[10., 0.], [0., 10.]]),
                                    np.array([[1., 0.], [0., 1.]]), 0)
        self.assertFalse(result["feasible"])
        self.assertIsNone(result["significance"])
        self.assertGreater(result["per_bin_q0"][0][0], 0)
        self.assertTrue(np.isnan(result["per_bin_q0"][0][1]))
        self.assertFalse(result["per_bin_valid"][0][1])
        self.assertEqual(result["per_bin_q0"][1][0], 0)

    def test_low_effective_count_policy_is_explicit_and_uncertainty_remains(self):
        s, b, v = np.array([5.]), np.array([10.]), np.array([100.])
        strict = asimov_significance(s, b, v, 10)
        inclusive = asimov_significance(s, b, v, 0)
        known = asimov_significance(s, b, None, 0)
        self.assertFalse(strict["feasible"])
        self.assertTrue(inclusive["feasible"])
        self.assertEqual(strict["per_bin_q0"], inclusive["per_bin_q0"])
        self.assertLess(inclusive["significance"], known["significance"])

    def test_flattening_does_not_change_asimov_information(self):
        s, b, v = np.array([[3., 5.], [0., 1.]]), np.array([[10., 20.], [0., 5.]]), np.array([[1., 4.], [0., .5]])
        a = asimov_significance(s, b, v, 0)
        f = asimov_significance(s.ravel(), b.ravel(), v.ravel(), 0)
        self.assertAlmostEqual(a["q0"], f["q0"])
        self.assertAlmostEqual(a["significance"] ** 2, np.sum(a["per_bin_q0"]))

    def test_signed_flow_conservation_and_inclusive_upper_edge(self):
        h = weighted_histogram(np.array([[0., 0.], [1., 1.], [2., .5], [.5, -1.]]),
                               np.array([2., -1., 3., -4.]), [np.array([0., 1.]), np.array([0., 1.])], "exclude")
        self.assertEqual(h.sumw[0, 0], 1.)
        self.assertEqual(h.sumw2[0, 0], 5.)
        self.assertEqual(h.flow_sumw, -1.)
        self.assertEqual(h.flow_sumw2, 25.)
        self.assertEqual(h.entries[0, 0], 2)


if __name__ == "__main__":
    unittest.main()
