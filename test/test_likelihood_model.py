import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from likelihood_model import DEFAULT_CONFIG, STAGE_CONTRACT_VERSION, build_model, export_model, from_objective, gaussian_domain, qcd_group_width


def example_inputs():
    """Synthetic weighted counts for software tests, never analysis results."""
    return {
        "schema_version": 1,
        "provenance": {"input_kind": "synthetic_unit_test", "signal_cross_section_pb": .01},
        "channels": {"SR": {
            "bin_edges": [0, 3500, 5500, 12000], "qcd_groups": [[0, 1], [2]],
            "processes": {
                "signal": {"sumw": [5., 8., 3.], "sumw2": [.25, .64, .09]},
                "QCD": {"sumw": [100., 40., 10.], "sumw2": [100., 64., 4.]},
                "TTbar": {"sumw": [20., 10., 5.], "sumw2": [4., 1., .25]},
                "ST": {"sumw": [2., 1., 1.], "sumw2": [.04, .01, .01]},
                "WJets": {"sumw": [5., 2., 1.], "sumw2": [.25, .04, .01]},
            }}}}


class LikelihoodModelTests(unittest.TestCase):
    def test_qcd_group_uses_max_relative_mc_not_sum_or_average(self):
        # Errors are 10% and 20%; the shared coefficient is sqrt(.2²+.15²).
        self.assertAlmostEqual(qcd_group_width([100., 40.], [100., 64.], [0, 1]), .25)
        self.assertAlmostEqual(qcd_group_width([100., 40.], [100., 64.], [0, 1],
                                               bb_relative_uncertainty=[.3, .1]), math.hypot(.3, .15))

    def test_shared_qcd_gaussian_and_no_double_mc(self):
        result = build_model(example_inputs())
        card, manifest = result["datacard"], result["manifest"]
        self.assertIn("QCD_bin_SR_group0_17 param 0 1 [-3.99999996,8]", card)
        self.assertIn("rateParam SR_bin0 QCD (1+0.25*@0) QCD_bin_SR_group0_17", card)
        self.assertIn("rateParam SR_bin1 QCD (1+0.25*@0) QCD_bin_SR_group0_17", card)
        self.assertNotIn("autoMCStats", card)
        self.assertNotIn("MCstat_SR_bin0_QCD", card)
        for process in ("signal", "TTbar", "ST", "WJets"):
            self.assertIn(f"MCstat_SR_bin0_{process}_17 param", card)
        self.assertEqual(manifest["qcd_mc_treatment"], "QCD_sumw2_only")
        self.assertFalse(manifest["full_an_reproduction"])
        self.assertFalse(manifest["production_ready"])

    def test_domains_are_strictly_positive_and_report_truncation(self):
        for relative in (.01, .2, 1., 5., 1000.):
            lo, hi = gaussian_domain(relative)
            self.assertGreater(1 + relative * lo, 0)
            self.assertGreater(1 + relative * hi, 0)
            self.assertLess(lo, 0)
        manifest = build_model(example_inputs())["manifest"]
        self.assertTrue(next(n for n in manifest["nuisances"] if n["name"] == "QCD_bin_SR_group0_17")["positivity_truncation_within_5sigma"])

    def test_an_normalizations_and_shared_shape_nuisance(self):
        inputs = example_inputs()
        for data in inputs["channels"]["SR"]["processes"].values():
            y = np.asarray(data["sumw"])
            data["variations"] = {"CMS_jer_2017": {"up": (y * 1.1).tolist(), "down": (y * .9).tolist()}}
        result = build_model(inputs)
        card = result["datacard"]
        for name in ("xs_QCD", "xs_TTbar", "xs_ST", "xs_WJets", "lumi_corr", "lumi_uncorr17"):
            self.assertIn(f"{name} lnN ", card)
        self.assertEqual(sum(line.startswith("CMS_jer_2017 shape") for line in card.splitlines()), 1)
        self.assertFalse(any(name.startswith("CMS_jer_2017:") for name in result["manifest"]["missing_reference_nuisances"]))
        self.assertTrue(any(name.startswith("CMS_pdf_QCD:") for name in result["manifest"]["missing_reference_nuisances"]))

    def test_total_jec_does_not_disguise_missing_split_sources(self):
        inputs = example_inputs()
        data = inputs["channels"]["SR"]["processes"]["QCD"]
        data["variations"] = {"CMS_jec_Total_2017": {"up": [110, 45, 12], "down": [90, 35, 8]}}
        result = build_model(inputs)["manifest"]
        self.assertIn("CMS_jec_RelativeBal:QCD", result["missing_reference_nuisances"])
        self.assertTrue(any("total JEC" in reason for reason in result["approximations"]))

    def test_statistics_only_has_mc_and_no_analysis_priors(self):
        result = build_model(example_inputs(), stage="statistics_only")
        self.assertNotIn(" lnN ", result["datacard"])
        self.assertNotIn(" shape ", result["datacard"])
        self.assertEqual(result["manifest"]["qcd_blanket_relative_uncertainty"], 0)
        self.assertEqual(result["manifest"]["qcd_groups"], [[0, 1], [2]])
        first = next(n for n in result["manifest"]["nuisances"] if n["name"] == "QCD_bin_SR_group0_17")
        self.assertAlmostEqual(first["relative_width"], .2)

    def test_stage_comparison_keeps_mc_basis_and_only_broadens_qcd_covariance(self):
        inputs = example_inputs()
        stat = build_model(inputs, stage="statistics_only")["manifest"]
        ref = build_model(inputs, stage="reference")["manifest"]
        self.assertEqual(stat["stage_contract_version"], STAGE_CONTRACT_VERSION)
        self.assertEqual(STAGE_CONTRACT_VERSION, 2)
        self.assertEqual(stat["finite_mc_basis"], ref["finite_mc_basis"])
        self.assertEqual(stat["finite_mc_basis_sha256"], ref["finite_mc_basis_sha256"])
        covariances = []
        qcd_yields = np.asarray(inputs["channels"]["SR"]["processes"]["QCD"]["sumw"])
        for manifest in (stat, ref):
            covariance = np.zeros((3, 3))
            for nuisance in manifest["nuisances"]:
                if nuisance["kind"] != "qcd_linear_gaussian":
                    continue
                direction = np.zeros(3)
                for effect in nuisance["affects"]:
                    index = effect["input_bin"]
                    direction[index] = qcd_yields[index] * nuisance["relative_width"]
                covariance += np.outer(direction, direction)
            covariances.append(covariance)
        self.assertGreater(covariances[0][0, 1], 0)
        self.assertEqual(covariances[0][0, 2], 0)
        # Broadening identical group directions cannot remove a fluctuation mode.
        self.assertGreaterEqual(np.linalg.eigvalsh(covariances[1] - covariances[0]).min(), -1e-12)
        stat_nuisances = {n["name"]: n for n in stat["nuisances"]}
        ref_nuisances = {n["name"]: n for n in ref["nuisances"]}
        for name, nuisance in stat_nuisances.items():
            self.assertEqual(nuisance["affects"], ref_nuisances[name]["affects"])
            if nuisance["kind"] != "qcd_linear_gaussian":
                self.assertEqual(nuisance, ref_nuisances[name])

    def test_reference_without_analysis_uncertainties_is_identical_for_both_mc_modes(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        config["normalization_nuisances"] = {}
        config["qcd_blanket_relative_uncertainty"] = 0
        for aggregate_bb in (False, True):
            with self.subTest(aggregate_bb=aggregate_bb):
                inputs = example_inputs()
                if aggregate_bb:
                    inputs["channels"]["SR"].update(
                        bb_relative_uncertainty=[.3, .4, .2],
                        bb_uncertainty_provenance="synthetic controlled-stage regression")
                stat = build_model(inputs, config, stage="statistics_only")
                ref = build_model(inputs, config, stage="reference")
                self.assertEqual(stat["datacard"], ref["datacard"])
                self.assertEqual(stat["manifest"]["finite_mc_basis"], ref["manifest"]["finite_mc_basis"])
                self.assertEqual(stat["manifest"]["nuisances"], ref["manifest"]["nuisances"])

    def test_zero_mc_group_allows_reference_only_blanket_nuisance(self):
        inputs = example_inputs()
        inputs["channels"]["SR"]["processes"]["QCD"]["sumw2"] = [0., 0., 0.]
        stat = build_model(inputs, stage="statistics_only")
        ref = build_model(inputs, stage="reference")
        self.assertNotIn("QCD_bin_SR_group0_17 param", stat["datacard"])
        self.assertIn("QCD_bin_SR_group0_17 param", ref["datacard"])
        self.assertEqual(stat["manifest"]["finite_mc_basis_sha256"], ref["manifest"]["finite_mc_basis_sha256"])

    def test_supplied_bb_absorbs_other_background_not_signal_statistics(self):
        inputs = example_inputs()
        channel = inputs["channels"]["SR"]
        channel["bb_relative_uncertainty"] = [.3, .4, .2]
        with self.assertRaisesRegex(ValueError, "provenance"):
            build_model(inputs)
        channel["bb_uncertainty_provenance"] = "independently audited combined-background coefficients"
        card = build_model(inputs)["datacard"]
        self.assertNotIn("MCstat_SR_bin0_TTbar", card)
        self.assertIn("MCstat_SR_bin0_signal", card)
        self.assertIn("(1+0.427200187266*@0)", card)
        stat = build_model(inputs, stage="statistics_only")
        self.assertNotIn("MCstat_SR_bin0_TTbar", stat["datacard"])
        self.assertIn("(1+0.4*@0)", stat["datacard"])
        self.assertEqual(stat["manifest"]["qcd_mc_treatment"], "supplied_aggregate_BB")

    def test_mc_guard_and_explicit_diagnostic_override(self):
        inputs = example_inputs()
        inputs["channels"]["SR"]["processes"]["QCD"]["sumw2"][0] = 100000.
        with self.assertRaisesRegex(ValueError, "effective MC"):
            build_model(inputs)
        manifest = build_model(inputs, allow_unsupported_mc=True)["manifest"]
        self.assertEqual(manifest["unsupported_background_bins"], [0])
        self.assertFalse(manifest["production_ready"])
        self.assertTrue(manifest["allow_unsupported_mc"])

    def test_negative_cancelled_missing_or_zero_background_bins_fail(self):
        for value, variance in ((-1, 1), (float("nan"), 1), (0, 1)):
            inputs = example_inputs()
            inputs["channels"]["SR"]["processes"]["ST"]["sumw"][0] = value
            inputs["channels"]["SR"]["processes"]["ST"]["sumw2"][0] = variance
            with self.assertRaises(ValueError):
                build_model(inputs)
        inputs = example_inputs()
        for process, data in inputs["channels"]["SR"]["processes"].items():
            if process != "signal":
                data["sumw"][0] = data["sumw2"][0] = 0.
        with self.assertRaisesRegex(ValueError, "positive background"):
            build_model(inputs, allow_unsupported_mc=True)

    def test_empty_bins_are_recorded_and_dropped_without_removing_signal(self):
        inputs = example_inputs()
        for data in inputs["channels"]["SR"]["processes"].values():
            data["sumw"][0] = data["sumw2"][0] = 0.
        result = build_model(inputs)
        self.assertEqual(result["manifest"]["active_input_bins"], [1, 2])
        self.assertEqual(result["manifest"]["dropped_empty_input_bins"], [0])
        self.assertIn("imax 2", result["datacard"])

    def test_invalid_partition_shape_name_and_zero_variation_fail(self):
        for groups in ([[0], [0, 1, 2]], [[0], [2]], [[True], [1], [2]], [[]]):
            inputs = example_inputs()
            inputs["channels"]["SR"]["qcd_groups"] = groups
            with self.assertRaisesRegex(ValueError, "qcd_groups"):
                build_model(inputs)
        for nuisance, up in (("evil name", [100, 40, 10]), ("validName", [0, 40, 10])):
            inputs = example_inputs()
            inputs["channels"]["SR"]["processes"]["QCD"]["variations"] = {nuisance: {"up": up, "down": [100, 40, 10]}}
            with self.assertRaises(ValueError):
                build_model(inputs)

    def test_regions_are_fit_individually_and_observed_data_rejected(self):
        inputs = example_inputs()
        inputs["channels"]["CR"] = copy.deepcopy(inputs["channels"]["SR"])
        result = build_model(inputs, region="CR")
        self.assertIn("QCD_bin_CR_group0_17", result["datacard"])
        self.assertNotIn("SR_bin", result["datacard"])
        inputs["channels"]["SR"]["observation"] = [120, 50, 10]
        with self.assertRaisesRegex(ValueError, "observed"):
            build_model(inputs)

    def test_complete_reference_cannot_be_claimed_from_nominal(self):
        with self.assertRaisesRegex(ValueError, "complete reference"):
            build_model(example_inputs(), require_complete_reference=True)

    def test_export_root_content_errors_and_asimov_observations(self):
        import uproot
        inputs = example_inputs()
        inputs["channels"]["SR"]["processes"]["QCD"]["variations"] = {
            "CMS_pu": {"up": [105., 43., 11.], "down": [96., 37., 9.]}}
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_model(inputs, directory)
            with uproot.open(Path(directory) / "templates.root") as root:
                self.assertAlmostEqual(root["SR_bin0/QCD"].values()[0], 100.)
                self.assertAlmostEqual(root["SR_bin0/QCD"].variances()[0], 100.)
                self.assertAlmostEqual(root["SR_bin0/signal"].variances()[0], .25)
                self.assertAlmostEqual(root["SR_bin0/data_obs"].values()[0], 127.)
                self.assertAlmostEqual(root["SR_bin0/QCD_CMS_puUp"].values()[0], 105.)
                self.assertAlmostEqual(root["SR_bin0/QCD_CMS_puDown"].values()[0], 96.)
            saved = json.loads((Path(directory) / "model.json").read_text())
            self.assertEqual(saved["artifacts_sha256"], manifest["artifacts_sha256"])
            with self.assertRaises(FileExistsError):
                export_model(inputs, directory)

    def test_adapter_preserves_category_yields_variances_and_one_signal(self):
        campaign = {"purpose": "pilot", "mass_bin_edges_gev": [0, 1, 2], "samples": [
            {"name": "sigA", "kind": "signal", "cross_section_pb": .01},
            {"name": "sigB", "kind": "signal", "cross_section_pb": .02},
            {"name": "qcd1", "kind": "background", "category": "QCDMC"},
            {"name": "qcd2", "kind": "background", "category": "QCDMC"}]}
        objective = {"samples": {
            "sigA": {"histogram": {"sumw": [1, 2], "sumw2": [.1, .2]}},
            "sigB": {"histogram": {"sumw": [100, 200], "sumw2": [1, 2]}},
            "qcd1": {"histogram": {"sumw": [5, 6], "sumw2": [1, 2]}},
            "qcd2": {"histogram": {"sumw": [7, 8], "sumw2": [3, 4]}}}}
        adapted = from_objective(objective, campaign, "sigA")
        processes = adapted["channels"]["SR"]["processes"]
        self.assertEqual(processes["QCD"]["sumw"], [12, 14])
        self.assertEqual(processes["QCD"]["sumw2"], [4, 6])
        self.assertEqual(processes["signal"]["sumw"], [1, 2])
        self.assertEqual(processes["signal"]["samples"], ["sigA"])
        self.assertEqual(adapted["provenance"]["signal_cross_section_pb"], .01)
        objective["samples"]["qcd1"]["histogram"]["sumw"][0] = -1
        with self.assertRaisesRegex(ValueError, "negative component"):
            from_objective(objective, campaign, "sigA")


if __name__ == "__main__":
    unittest.main()
