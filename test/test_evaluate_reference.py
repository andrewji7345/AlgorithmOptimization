import contextlib
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import uproot

import evaluate_reference as evaluator
from evaluate_sensitivity import ANALYSIS_SELECTION, CORRECTION_PRESCRIPTION


def fixture():
    campaign = {"purpose": "pilot", "luminosity_pb": 1000., "mass_bin_edges_gev": [0, 3500, 5500, 7500, 12000],
                "analysis_selection": ANALYSIS_SELECTION, "correction_prescription": CORRECTION_PRESCRIPTION,
                "analysis_systematic": "nominal", "required_signals": ["WbWb_4000_1000"],
                "physicality": {"min_valid_events": 20}, "samples": []}
    payloads, metadata, extras = {}, {}, {}
    for name, kind, category, xsec, path in (("WbWb_4000_1000", "signal", "SuuToChiChi", .1, "signal.root"),
                                              ("QCDMC_Pt_fixture", "background", "QCDMC", 2., "qcd.root"),
                                              ("TTJets_fixture", "background", "TTbarMC", .2, "ttbar.root")):
        spec = {"name": name, "kind": kind, "category": category, "cross_section_pb": xsec,
                "files": [path], "input_files": [f"/store/{name}/input.root"], "sum_gen_weights": "metadata",
                "generated_events": "metadata", "normalization_scope": "representative_subset", "complete": True}
        if kind == "signal":
            spec["generated_chi_mass_gev"] = 1000.
        campaign["samples"].append(spec)
        n = 100
        gen = np.ones(n); gen[-10:] = -.5
        payload = SimpleNamespace(n_events=n, run=np.ones(n, dtype=np.uint32), lumi=np.ones(n, dtype=np.uint32),
            event=np.arange(1, n + 1, dtype=np.uint64), gen_weight=gen,
            passes_baseline=np.ones(n, dtype=bool), passes_signal_region=np.ones((n, 1), dtype=np.uint8),
            analysis_weight=np.full(n, .1), reco_status=np.zeros((n, 1), dtype=np.uint8),
            sj1_mass=np.full((n, 1), 1000.), sj2_mass=np.full((n, 1), 1000.), suu_mass=np.full((n, 1), 4000.),
            ak_jet_pt=np.full((n, 4), 500.))
        meta = SimpleNamespace(path=path, schema_version=3, sample_name=name, sample_kind=kind,
            analysis_selection=ANALYSIS_SELECTION, correction_prescription=CORRECTION_PRESCRIPTION,
            analysis_systematic="nominal", use_jec=True, ak_radius=.8,
            processed_events=n, sum_weights=85., sum_weights2=92.5,
            n_configurations=1, _config_lookup={(100., .8, 0.): 0}, config_base_index=[0],
            config_collection_pt_cut=[100.], config_ca_radius=[.8], config_cos_thrust=[0.])
        extra_meta = {"analysisObservableVersion": 1, "analysisSystematic": "nominal",
                      "referenceReconstruction": evaluator.REFERENCE_RECONSTRUCTION,
                      "weightVariationNames": list(evaluator.WEIGHT_NAMES)}
        weights = np.full((n, 14), .9)
        weights[:, 0] = .99; weights[:, 1] = .81
        if kind == "background" and category == "TTbarMC":
            weights[:, 12] = .72
        arrays = {"referenceWeight": np.full(n, .9), "referenceWeightVariations": weights,
                  "referenceRecoStatus": np.zeros(n, dtype=np.uint8), "referenceRegion": np.ones(n, dtype=np.uint8),
                  "referenceSJ1Mass": np.full(n, 1000.), "referenceSJ2Mass": np.full(n, 1000.),
                  "referenceSuuMass": np.full(n, 4000.), "analysisNBTags": np.ones(n, dtype=np.uint16)}
        for name2 in ("passesTrigger", "passesFilters", "passesLeptonVeto", "passesJetVeto"):
            arrays[name2] = np.ones(n, dtype=np.uint8)
        for prefix in ("referenceSJ1", "referenceSJ2", "sj1", "sj2"):
            shape = (n,) if prefix.startswith("reference") else (n, 1)
            for suffix in ("NCA4E50", "NCA4E300"):
                arrays[prefix + suffix] = np.full(shape, 2, dtype=np.uint16)
            arrays[prefix + "MassE100"] = np.full(shape, 400.)
        arrays["passesRecoJetVeto"] = np.ones((n, 1), dtype=np.uint8)
        for name2 in ("passesControlRegion", "passesAT1b", "passesAT0b"):
            arrays[name2] = np.zeros((n, 1), dtype=np.uint8)
        payloads[path], metadata[path], extras[path] = payload, meta, (extra_meta, arrays)
    return campaign, metadata, payloads, extras


def synchronize_regions(payload, arrays):
    arrays["referenceRegion"] = evaluator.region_codes(payload.passes_baseline, arrays["referenceRecoStatus"] == 0,
        True, arrays["analysisNBTags"], arrays["referenceSJ1NCA4E300"], arrays["referenceSJ2NCA4E300"],
        arrays["referenceSJ1NCA4E50"], arrays["referenceSJ2NCA4E50"], arrays["referenceSJ1MassE100"], arrays["referenceSJ2MassE100"])
    codes = evaluator.region_codes(payload.passes_baseline[:, None], payload.reco_status == 0, arrays["passesRecoJetVeto"],
        arrays["analysisNBTags"][:, None], arrays["sj1NCA4E300"], arrays["sj2NCA4E300"], arrays["sj1NCA4E50"],
        arrays["sj2NCA4E50"], arrays["sj1MassE100"], arrays["sj2MassE100"])
    payload.passes_signal_region = (codes == 1).astype(np.uint8)
    for name, branch in evaluator.REGION_BRANCHES.items():
        if name != "SR":
            arrays[branch] = (codes == evaluator.REGIONS[name]).astype(np.uint8)


class ReferenceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.campaign, self.metadata, self.payloads, self.extras = fixture()
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(mock.patch.object(evaluator, "load_metadata", side_effect=lambda path: self.metadata[str(path)]))
        self.stack.enter_context(mock.patch.object(evaluator, "read_event_payload", side_effect=lambda meta: self.payloads[meta.path]))
        self.stack.enter_context(mock.patch.object(evaluator, "_read_additive", side_effect=lambda meta: self.extras[meta.path]))
        self.addCleanup(self.stack.close)

    def test_empty_background_remains_a_coverage_limitation(self):
        self.payloads["ttbar.root"].passes_baseline[:] = False
        synchronize_regions(self.payloads["ttbar.root"], self.extras["ttbar.root"][1])
        result = evaluator.evaluate_reference_campaign(self.campaign)
        for model in result["models"].values():
            coverage = model["background_coverage"]
            self.assertEqual(coverage["unobserved_samples"], ["TTJets_fixture"])
            self.assertTrue(coverage["unobserved_components_require_review"])
            self.assertEqual(coverage["samples"]["TTJets_fixture"]["generated_events"], 100)
            self.assertIsNone(coverage["samples"]["TTJets_fixture"]["effective_events_in_fit_bins"])
            # Populated QCD can pass the conditional n_eff test; the absent
            # ttbar component must still be present in the exported provenance.
            info = model["per_signal"]["WbWb_4000_1000"]
            self.assertTrue(info["fast_score"]["feasible"])
            self.assertEqual(info["input"]["provenance"]["background_coverage"], coverage)

    def test_signed_cancellation_is_not_classified_as_unobserved_mc(self):
        report = evaluator.background_coverage({"signed": {
            "kind": "background", "generated_events": 100, "selected_events": 2,
            "histogram": {"sumw": [0., 0.], "sumw2": [2., 0.]}}})
        self.assertEqual(report["unobserved_samples"], [])
        detail = report["samples"]["signed"]
        self.assertEqual(detail["canceled_fit_bins"], [0])
        self.assertEqual(detail["empty_fit_bins"], [1])
        self.assertEqual(detail["effective_events_in_fit_bins"], 0.)

    def test_signed_all_event_normalization_reference_weight_and_shapes(self):
        result = evaluator.evaluate_reference_campaign(self.campaign)
        self.assertEqual(len(result["models"]), 2)
        self.assertAlmostEqual(result["minimum_background_effective_events"], 1 / .175 ** 2)
        for model in result["models"].values():
            signal = model["per_signal"]["WbWb_4000_1000"]
            self.assertTrue(signal["feasible"])
            self.assertAlmostEqual(signal["signal_yield"], 90.)
            self.assertAlmostEqual(signal["background_yield"], 1980.)
            processes = signal["input"]["channels"]["SR"]["processes"]
            self.assertAlmostEqual(processes["QCD"]["sumw2"][1], (2000 / 85 * .9) ** 2 * 92.5)
            self.assertAlmostEqual(processes["QCD"]["variations"]["CMS_pu"]["up"][1], 1980.)
            self.assertAlmostEqual(processes["QCD"]["variations"]["CMS_pu"]["down"][1], 1620.)
            self.assertNotIn("CMS_topPt_TTbar", processes["QCD"]["variations"])
            self.assertAlmostEqual(processes["TTbar"]["variations"]["CMS_topPt_TTbar"]["up"][1], 144.)
            self.assertAlmostEqual(processes["TTbar"]["variations"]["CMS_topPt_TTbar"]["down"][1], 180.)
        json.dumps(result, allow_nan=False)

    def test_reconstruction_and_baseline_failures_do_not_change_denominator(self):
        payload = self.payloads["signal.root"]
        arrays = self.extras["signal.root"][1]
        payload.passes_baseline[:20] = False
        arrays["referenceRecoStatus"][20:25] = 8
        for name in ("referenceSJ1Mass", "referenceSJ2Mass", "referenceSuuMass"):
            arrays[name][20:25] = np.nan
        synchronize_regions(payload, arrays)
        result = evaluator.evaluate_reference_campaign(self.campaign)
        model = result["models"][evaluator.REFERENCE_NAME]
        self.assertEqual(model["samples"]["WbWb_4000_1000"]["sum_gen_weights"], 85)
        self.assertEqual(model["samples"]["WbWb_4000_1000"]["generated_events"], 100)
        self.assertEqual(model["samples"]["WbWb_4000_1000"]["selected_events"], 75)
        self.assertEqual(model["per_signal"]["WbWb_4000_1000"]["physicality"]["n_gate_events"], 80)
        self.assertAlmostEqual(model["per_signal"]["WbWb_4000_1000"]["signal_yield"], 100 / 85 * .9 * 60)

    def test_physicality_before_region_can_reject_without_changing_fast_score(self):
        arrays = self.extras["signal.root"][1]
        arrays["referenceSJ1Mass"][:60] = 2000.
        result = evaluator.evaluate_reference_campaign(self.campaign)
        ref = result["models"][evaluator.REFERENCE_NAME]["per_signal"]["WbWb_4000_1000"]
        self.assertFalse(ref["feasible"])
        self.assertTrue(ref["fast_score"]["feasible"])
        self.assertIn("physicality_gate_failed", ref["failure_reasons"])

    def test_region_validation_and_independent_at_evaluation(self):
        for path, payload in self.payloads.items():
            arrays = self.extras[path][1]
            for prefix in ("referenceSJ2", "sj2"):
                arrays[prefix + "NCA4E50"][:] = 0
                arrays[prefix + "NCA4E300"][:] = 0
                arrays[prefix + "MassE100"][:] = 0
            synchronize_regions(payload, arrays)
        result = evaluator.evaluate_reference_campaign(self.campaign, region="AT1b")
        for model in result["models"].values():
            self.assertEqual(model["samples"]["WbWb_4000_1000"]["region_counts"]["AT1b"], 100)
            self.assertAlmostEqual(model["per_signal"]["WbWb_4000_1000"]["signal_yield"], 90.)
        self.extras["qcd.root"][1]["passesAT0b"][0, 0] = 1
        with self.assertRaisesRegex(ValueError, "disagrees"):
            evaluator.evaluate_reference_campaign(self.campaign, region="AT1b")

    def test_bad_metadata_unknown_weights_nonfinite_weights_and_duplicate_events_fail(self):
        for field, value in (("analysisObservableVersion", 0), ("analysisSystematic", "JECUp"),
                             ("referenceReconstruction", "other"), ("weightVariationNames", ["pileupUp"])):
            original = self.extras["qcd.root"][0][field]
            self.extras["qcd.root"][0][field] = value
            with self.assertRaises(ValueError):
                evaluator.evaluate_reference_campaign(self.campaign)
            self.extras["qcd.root"][0][field] = original
        self.extras["qcd.root"][1]["referenceWeightVariations"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            evaluator.evaluate_reference_campaign(self.campaign)
        self.extras["qcd.root"][1]["referenceWeightVariations"][0, 0] = .99
        self.payloads["qcd.root"].event[1] = 1
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluator.evaluate_reference_campaign(self.campaign)

    def test_weight_name_order_is_respected(self):
        for metadata, arrays in self.extras.values():
            metadata["weightVariationNames"] = metadata["weightVariationNames"][::-1]
            arrays["referenceWeightVariations"] = arrays["referenceWeightVariations"][:, ::-1]
        result = evaluator.evaluate_reference_campaign(self.campaign)
        qcd = result["models"][evaluator.REFERENCE_NAME]["per_signal"]["WbWb_4000_1000"]["input"]["channels"]["SR"]["processes"]["QCD"]
        self.assertAlmostEqual(qcd["variations"]["CMS_pu"]["up"][1], 1980.)

    def add_variation(self, name):
        varied = copy.deepcopy(self.campaign)
        varied["analysis_systematic"] = name
        for sample in varied["samples"]:
            old = sample["files"][0]; new = old.replace(".root", "_" + name + ".root")
            sample["files"] = [new]
            self.metadata[new] = copy.deepcopy(self.metadata[old]); self.metadata[new].path = new
            self.metadata[new].analysis_systematic = name
            self.payloads[new] = copy.deepcopy(self.payloads[old])
            self.extras[new] = copy.deepcopy(self.extras[old]); self.extras[new][0]["analysisSystematic"] = name
        return varied

    def test_kinematic_variations_pair_sources_ids_and_shape_yields(self):
        up, down = self.add_variation("JECUp"), self.add_variation("JECDown")
        with self.assertRaisesRegex(ValueError, "paired"):
            evaluator.evaluate_reference_campaign(self.campaign, {"JECUp": up})
        self.extras["signal_JECUp.root"][1]["referenceSuuMass"][:] = 6000
        self.payloads["signal_JECUp.root"].suu_mass[:] = 6000
        result = evaluator.evaluate_reference_campaign(self.campaign, {"JECUp": up, "JECDown": down})
        variations = result["models"][evaluator.REFERENCE_NAME]["per_signal"]["WbWb_4000_1000"]["input"]["channels"]["SR"]["processes"]["signal"]["variations"]
        self.assertAlmostEqual(variations["CMS_jec_Total_2017"]["up"][2], 90.)
        self.assertAlmostEqual(variations["CMS_jec_Total_2017"]["down"][1], 90.)
        self.payloads["qcd_JECDown.root"].event += 1000
        with self.assertRaisesRegex(ValueError, "identities"):
            evaluator.evaluate_reference_campaign(self.campaign, {"JECUp": up, "JECDown": down})

    def test_systematic_changed_source_and_generator_weight_rejected(self):
        up, down = self.add_variation("JERUp"), self.add_variation("JERDown")
        up["samples"][0]["input_files"] = ["/other.root"]
        with self.assertRaisesRegex(ValueError, "input files"):
            evaluator.evaluate_reference_campaign(self.campaign, {"JERUp": up, "JERDown": down})
        up["samples"][0]["input_files"] = self.campaign["samples"][0]["input_files"]
        # Permute gen weights between events: total denominator and MC variance
        # stay unchanged, but the paired identity/gen-weight audit must fail.
        gen = self.payloads["qcd_JERDown.root"].gen_weight
        gen[0], gen[-1] = gen[-1], gen[0]
        with self.assertRaisesRegex(ValueError, "generator weights"):
            evaluator.evaluate_reference_campaign(self.campaign, {"JERUp": up, "JERDown": down})

    def test_fixed_binning_folds_flow_and_keeps_low_mc_explicit(self):
        self.extras["signal.root"][1]["referenceSuuMass"][0] = 15000.
        result = evaluator.evaluate_reference_campaign(self.campaign)
        ref = result["models"][evaluator.REFERENCE_NAME]
        self.assertEqual(ref["samples"]["WbWb_4000_1000"]["histogram"]["flow_entries"], 1)
        self.assertFalse(ref["per_signal"]["WbWb_4000_1000"]["fast_score"]["feasible"])
        self.assertIn("signal_bin_without_positive_background", ref["per_signal"]["WbWb_4000_1000"]["failure_reasons"])
        self.assertIsNone(ref["per_signal"]["WbWb_4000_1000"]["diagnostic_fast_score"])

    def test_diagnostic_score_relaxes_only_mc_threshold_and_never_eligibility(self):
        self.payloads["qcd.root"].gen_weight[:] = 0
        self.payloads["qcd.root"].gen_weight[0] = 85
        self.metadata["qcd.root"].sum_weights2 = 85 ** 2
        result = evaluator.evaluate_reference_campaign(self.campaign)
        ref = result["models"][evaluator.REFERENCE_NAME]["per_signal"]["WbWb_4000_1000"]
        self.assertFalse(ref["feasible"])
        self.assertFalse(ref["fast_score"]["feasible"])
        self.assertTrue(ref["diagnostic_fast_score"]["feasible"])
        self.assertTrue(ref["diagnostic_fast_score"]["diagnostic_only"])
        self.assertGreater(ref["diagnostic_fast_score"]["significance"], 0)

    def test_default_ungated_grid_has_no_duplicate_gate_aliases(self):
        result = evaluator.evaluate_reference_campaign(self.campaign)
        hybrid = [name for name in result["models"] if name != evaluator.REFERENCE_NAME]
        self.assertEqual(hybrid, ["ng0_tgnone_tk100_ak0p8_ca0p8_c0"])
        result = evaluator.evaluate_reference_campaign(self.campaign, keys=[], include_reference=True)
        self.assertEqual(list(result["models"]), [evaluator.REFERENCE_NAME])


class ReferenceIdentityTests(unittest.TestCase):
    def test_identity_hash_is_order_independent_but_pairs_weights(self):
        ids = np.array([(1, 1, 2), (1, 1, 1)], dtype=evaluator._ID_DTYPE)
        first = evaluator._identity_audit(ids, np.array([2., 1.]), ["b", "a"])
        second = evaluator._identity_audit(ids[::-1], np.array([1., 2.]), ["a", "b"])
        self.assertEqual(first, second)
        different = evaluator._identity_audit(ids, np.array([1., 2.]), ["a", "b"])
        self.assertNotEqual(first["event_genweights_sha256"], different["event_genweights_sha256"])

    def test_fold_audit_reads_real_root_event_ids_and_detects_overlap(self):
        first, _, _, _ = fixture(); second = copy.deepcopy(first)
        with tempfile.TemporaryDirectory() as tmp:
            for fold, campaign in enumerate((first, second)):
                for sample in campaign["samples"]:
                    path = Path(tmp) / f"{fold}_{sample['name']}.root"
                    sample["files"] = [str(path)]; sample["input_files"] = [f"/store/{fold}/{sample['name']}.root"]
                    with uproot.recreate(path) as root:
                        root[evaluator.EVENTS_PATH] = {"run": np.ones(10, dtype=np.uint32), "lumi": np.ones(10, dtype=np.uint32),
                                                     "event": np.arange(fold * 10, fold * 10 + 10, dtype=np.uint64)}
            result = evaluator.validate_disjoint_folds(first, second)
            self.assertTrue(result["disjoint"])
            self.assertEqual(result["samples"]["WbWb_4000_1000"]["validation_events"], 10)
            path = second["samples"][0]["files"][0]
            with uproot.recreate(path) as root:
                root[evaluator.EVENTS_PATH] = {"run": np.ones(10, dtype=np.uint32), "lumi": np.ones(10, dtype=np.uint32),
                                             "event": np.arange(10, dtype=np.uint64)}
            with self.assertRaisesRegex(ValueError, "event IDs overlap"):
                evaluator.validate_disjoint_folds(first, second)

    def test_region_boundaries_and_exchange_symmetry(self):
        values = evaluator.region_codes([1] * 5, [1] * 5, [1] * 5, [1, 0, 1, 0, 1],
            [2] * 5, [2, 2, 0, 0, 0], [2] * 5, [2, 2, 0, 0, 0], [500] * 5, [500, 500, 0, 149.999, 150])
        np.testing.assert_array_equal(values, [1, 2, 3, 4, 0])


if __name__ == "__main__":
    unittest.main()
