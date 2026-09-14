#!/usr/bin/env python3
"""Analytic objective tests and frozen-reference provenance regressions."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensitivity_objective import (
    aggregate_objective, comparison_context, context_fingerprint, context_from_result,
    relative_regret_rankings, resolve_objective, validate_objective,
)
from freeze_sensitivity_references import freeze_references


REQUIRED = ["light_WBWB", "heavy_HTZT"]


def signal_scores(values):
    return {name: {"significance": z, "q0": z * z, "feasible": True, "failure_reasons": []}
            for name, z in zip(REQUIRED, values)}


def sample_definitions():
    return {name: {"kind": "signal" if name in REQUIRED else "background",
                   "cross_section_pb": 2, "normalization_scope": "representative_subset",
                   "sum_gen_weights": 100, "generated_events": 100,
                   "event_identity_sha256": hashlib.sha256((name + ":ids").encode()).hexdigest(),
                   "event_generator_sha256": hashlib.sha256((name + ":weights").encode()).hexdigest()}
            for name in [*REQUIRED, "QCD"]}


def make_context():
    return comparison_context({"luminosity_pb": 1000, "mass_bin_edges_gev": [2500, 3000, 3500],
                               "chi_mass_bin_edges_gev": [750, 1000, 1250], "histogram_flow": "exclude",
                               "weight_convention": "reference", "physicality_mode": "diagnostic",
                               "minimum_background_effective_events": 0}, sample_definitions(), REQUIRED)


class ObjectiveTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.context = make_context()
        self.document = {"schema_version": 1, "required_signals": REQUIRED,
                         "reference_significances": dict(zip(REQUIRED, [10.0, 2.0])),
                         "comparison_context": self.context, "context_sha256": context_fingerprint(self.context)}
        self.reference = self.directory / "references.json"
        self.reference.write_text(json.dumps(self.document))
        self.spec = resolve_objective({"method": "fixed_reference_regret", "mean_weight": 0.25,
                                       "reference_file": "references.json"}, self.directory)

    def test_stationary_utility_equals_complement_of_worst_plus_mean_deficit(self):
        result = aggregate_objective(signal_scores([5.0, 1.6]), REQUIRED, self.spec, self.context)
        self.assertAlmostEqual(result["objective"], 0.5 + 0.25 * 0.65)
        self.assertAlmostEqual(result["reference_regret_objective"], 0.5 + 0.25 * 0.35)
        self.assertAlmostEqual(result["reference_regret_objective"] + result["objective"], 1.25)
        self.assertEqual(result["direction"], "maximize")

    def test_reference_improvements_are_not_clipped(self):
        result = aggregate_objective(signal_scores([20., 6.]), REQUIRED, self.spec, self.context)
        self.assertEqual(result["reference_ratios"], dict(zip(REQUIRED, [2., 3.])))
        self.assertEqual(result["reference_deficits"], dict(zip(REQUIRED, [-1., -2.])))
        self.assertEqual(result["objective"], 2.625)
        self.assertLess(result["reference_regret_objective"], 0)

    def test_late_better_trials_do_not_change_old_objective(self):
        scores = signal_scores([5., 1.6])
        before = aggregate_objective(scores, REQUIRED, self.spec, self.context)
        aggregate_objective(signal_scores([500., 600.]), REQUIRED, self.spec, self.context)
        after = aggregate_objective(scores, list(reversed(REQUIRED)), self.spec, self.context)
        self.assertEqual(before, after)

    def test_zero_selected_signal_is_retained_and_missing_signal_is_infeasible(self):
        result = aggregate_objective(signal_scores([10., 0.]), REQUIRED, self.spec, self.context)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["objective"], 0.125)
        self.assertEqual(result["worst_reference_deficit"], 1)
        missing = aggregate_objective(signal_scores([10.]), REQUIRED, self.spec, self.context)
        self.assertFalse(missing["feasible"])
        self.assertEqual(missing["objective"], -1)

    def test_all_zero_signal_has_zero_feasible_utility(self):
        result = aggregate_objective(signal_scores([0., 0.]), REQUIRED, self.spec, self.context)
        self.assertEqual(result["objective"], 0)
        self.assertTrue(result["feasible"])

    def test_invalid_or_unexpected_per_signal_results_fail_closed(self):
        for value in (-1, float("nan"), float("inf"), True, None):
            scores = signal_scores([1., 1.])
            scores[REQUIRED[0]]["significance"] = value
            self.assertFalse(aggregate_objective(scores, REQUIRED, self.spec, self.context)["feasible"])
        scores = signal_scores([1., 1.])
        scores["undeclared"] = {"significance": 1000, "feasible": True}
        self.assertFalse(aggregate_objective(scores, REQUIRED, self.spec, self.context)["feasible"])
        scores = signal_scores([1., 1.])
        scores[REQUIRED[0]]["feasible"] = False
        self.assertFalse(aggregate_objective(scores, REQUIRED, self.spec, self.context)["feasible"])

    def test_reference_scales_must_be_exact_positive_finite_set(self):
        for references in ({REQUIRED[0]: 1}, {**self.document["reference_significances"], "extra": 2},
                           dict(zip(REQUIRED, [0, 1])), dict(zip(REQUIRED, [-1, 1])),
                           dict(zip(REQUIRED, [float("nan"), 1])), dict(zip(REQUIRED, [True, 1]))):
            policy = copy.deepcopy(self.spec)
            policy["_reference"]["reference_significances"] = references
            with self.assertRaises(ValueError):
                validate_objective(policy, REQUIRED, self.context)

    def test_missing_context_or_unresolved_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_objective(signal_scores([1., 1.]), REQUIRED, self.spec)
        with self.assertRaises(ValueError):
            validate_objective({"method": "fixed_reference_regret", "reference_file": "none"}, REQUIRED)

    def test_reference_context_hash_and_physical_population_drift_are_rejected(self):
        corrupt = copy.deepcopy(self.spec)
        corrupt["_reference"]["comparison_context"]["parameters"]["luminosity_pb"] = 2000
        with self.assertRaisesRegex(ValueError, "context hash"):
            validate_objective(corrupt, REQUIRED)
        for field, value in (("luminosity_pb", 2000), ("histogram_flow", "fold"),
                             ("minimum_background_effective_events", 10), ("weight_convention", "analysis")):
            changed = copy.deepcopy(self.context)
            changed["parameters"][field] = value
            with self.assertRaisesRegex(ValueError, "physics or physical MC"):
                aggregate_objective(signal_scores([1., 1.]), REQUIRED, self.spec, changed)
        changed = copy.deepcopy(self.context)
        changed["comparison_samples"]["QCD"]["event_identity_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "physics or physical MC"):
            aggregate_objective(signal_scores([1., 1.]), REQUIRED, self.spec, changed)

    def test_reference_requires_physical_identity_and_generator_digests(self):
        for field in ("event_identity_sha256", "event_generator_sha256"):
            policy = copy.deepcopy(self.spec)
            policy["_reference"]["comparison_context"]["comparison_samples"]["QCD"].pop(field)
            policy["_reference"]["context_sha256"] = context_fingerprint(policy["_reference"]["comparison_context"])
            with self.assertRaisesRegex(ValueError, field):
                validate_objective(policy, REQUIRED)

    def test_changed_reference_bytes_change_recorded_contract(self):
        before = aggregate_objective(signal_scores([1., 1.]), REQUIRED, self.spec, self.context)
        self.document["reference_significances"][REQUIRED[0]] = 11
        self.reference.write_text(json.dumps(self.document))
        new_spec = resolve_objective({"method": "fixed_reference_regret", "reference_file": "references.json"}, self.directory)
        after = aggregate_objective(signal_scores([1., 1.]), REQUIRED, new_spec, self.context)
        self.assertNotEqual(before["objective_definition"], after["objective_definition"])
        self.assertNotEqual(before["objective"], after["objective"])

    def test_output_paths_do_not_change_physics_context(self):
        samples = sample_definitions()
        samples["QCD"].update(files=["/elsewhere/reconstruction.root"], input_file_list="/moved/list.txt")
        parameters = {**self.context["parameters"], "configuration": {"ak_radius": 0.4}}
        changed = comparison_context(parameters, samples, list(reversed(REQUIRED)))
        self.assertEqual(self.context, changed)

    def test_legacy_mean_and_invalid_policy(self):
        for policy in (None, {"method": "mean_asimov"}):
            self.assertEqual(aggregate_objective(signal_scores([2., 4.]), REQUIRED, policy)["objective"], 3)
        for policy in ({"method": "running_best"}, {"method": "mean_asimov", "mean_weight": 0.25},
                       {**self.spec, "mean_weight": -1}, {**self.spec, "mean_weight": True},
                       {**self.spec, "mean_weight": float("nan")}, {**self.spec, "typo": 1}):
            with self.assertRaises(ValueError):
                validate_objective(policy, REQUIRED)
        for names in ([], REQUIRED + [REQUIRED[0]], "one"):
            with self.assertRaises(ValueError):
                validate_objective(None, names)

    def test_large_finite_mean_does_not_overflow(self):
        scores = {name: {"significance": 1e308, "feasible": True} for name in REQUIRED}
        self.assertEqual(aggregate_objective(scores, REQUIRED, None)["objective"], 1e308)

    def test_retrospective_regret_ranking_is_not_raw_mean_ranking(self):
        def trial(name, values, feasible=True):
            return {"configuration": name, "objective": sum(values) / len(values),
                    "feasible": feasible, "per_signal": signal_scores(values)}
        results = [trial("large_raw_mean", [100, 0]), trial("balanced", [60, 1]),
                   trial("infeasible", [1000, 1000], False), trial("missing", [1000])]
        ranking = relative_regret_rankings(results, REQUIRED)
        self.assertEqual([r["configuration"] for r in ranking], ["balanced", "large_raw_mean"])
        self.assertAlmostEqual(ranking[0]["regret_objective"], 0.4 + 0.25 * 0.2)
        self.assertEqual(ranking[0]["objective"], 30.5)
        self.assertEqual(ranking[0]["per_signal_best_significance"], dict(zip(REQUIRED, [100, 1])))
        zeros = relative_regret_rankings([trial("zero", [0, 0])], REQUIRED)
        self.assertEqual(zeros[0]["regret_objective"], 0)


class FreezeReferencesTest(unittest.TestCase):
    def setUp(self):
        from evaluate_sensitivity import score_parameters
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.campaign = {"required_signals": REQUIRED, "luminosity_pb": 1000,
                         "mass_bin_edges_gev": [2500, 3000, 3500],
                         "chi_mass_bin_edges_gev": [750, 1000, 1250], "histogram_flow": "exclude",
                         "analysis_selection": "test_selection", "correction_prescription": "test_corrections",
                         "weight_convention": "reference", "physicality_mode": "diagnostic",
                         "minimum_background_effective_events": 0,
                         "samples": [{"name": name, **sample} for name, sample in sample_definitions().items()]}
        self.campaign_path = self.directory / "campaign.json"
        self.campaign_path.write_text(json.dumps(self.campaign))
        self.parameters = score_parameters(self.campaign)
        self.paths = [self.make_result("first", [5., 2.]), self.make_result("second", [10., 1.])]
        self.output = self.directory / "references.json"

    def make_result(self, name, values):
        result = {"configuration": name, "feasible": True, "failure_reasons": [],
                  "required_signals": REQUIRED, "per_signal": signal_scores(values),
                  "parameters": self.parameters, "comparison_samples": sample_definitions()}
        path = self.directory / (name + ".json")
        path.write_text(json.dumps(result))
        return path

    def rewrite(self, path, mutate):
        data = json.loads(path.read_text())
        mutate(data)
        path.write_text(json.dumps(data))

    def test_freeze_maxima_exact_input_hashes_order_invariance_and_immutability(self):
        document = freeze_references(self.campaign_path, self.paths, self.output)
        self.assertEqual(document["reference_significances"], dict(zip(REQUIRED, [10., 2.])))
        self.assertEqual({r["sha256"] for r in document["calibration_inputs"]},
                         {hashlib.sha256(p.read_bytes()).hexdigest() for p in self.paths})
        self.assertEqual(document, freeze_references(self.campaign_path, list(reversed(self.paths)), self.output))
        spec = resolve_objective({"method": "fixed_reference_regret", "reference_file": str(self.output)}, self.directory)
        result = json.loads(self.paths[0].read_text())
        self.assertTrue(aggregate_objective(result["per_signal"], REQUIRED, spec, context_from_result(result))["feasible"])
        extra = self.make_result("improvement", [20., 4.])
        original_bytes = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "refusing to replace"):
            freeze_references(self.campaign_path, [*self.paths, extra], self.output)
        self.assertEqual(self.output.read_bytes(), original_bytes)

    def test_freeze_rejects_missing_infeasible_and_duplicate_calibrations(self):
        for paths in ([], [self.paths[0], self.paths[0]]):
            with self.assertRaises(ValueError):
                freeze_references(self.campaign_path, paths, self.output)
        self.rewrite(self.paths[0], lambda r: r["per_signal"].pop(REQUIRED[1]))
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            freeze_references(self.campaign_path, self.paths, self.output)
        path = self.make_result("infeasible", [1., 1.])
        self.rewrite(path, lambda r: r.update(feasible=False))
        with self.assertRaisesRegex(ValueError, "globally feasible"):
            freeze_references(self.campaign_path, [path], self.output)
        self.assertFalse(self.output.exists())

    def test_freeze_rejects_campaign_mismatch_and_population_changes(self):
        self.rewrite(self.campaign_path, lambda r: r.update(luminosity_pb=2000))
        with self.assertRaisesRegex(ValueError, "score physics parameters"):
            freeze_references(self.campaign_path, self.paths, self.output)
        self.campaign_path.write_text(json.dumps(self.campaign))
        self.rewrite(self.paths[1], lambda r: r["comparison_samples"]["QCD"].update(event_generator_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "physical MC context"):
            freeze_references(self.campaign_path, self.paths, self.output)
        self.assertFalse(self.output.exists())

    def test_zero_reference_and_inconsistent_q0_are_rejected(self):
        zero = self.make_result("zero", [1., 0.])
        with self.assertRaisesRegex(ValueError, "positive"):
            freeze_references(self.campaign_path, [zero], self.output)
        self.rewrite(zero, lambda r: r["per_signal"][REQUIRED[0]].update(q0=4.))
        with self.assertRaisesRegex(ValueError, "disagrees with q0"):
            freeze_references(self.campaign_path, [zero], self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
