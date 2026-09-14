#!/usr/bin/env python3
"""Assembled full-grid pilot configuration and submission-preflight contracts."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

import optimize_sensitivity as driver
from evaluate_sensitivity import score_parameters, validate_campaign
from sensitivity_objective import resolve_objective


def read_config(name):
    return json.loads((REPOSITORY / "config" / name).read_text())


class FullGrid2DConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = read_config("sensitivity_2d_2017.json")
        cls.calibration = read_config("sensitivity_2d_calibration_2017.json")
        cls.main_driver = read_config("optuna_2d_2017.json")
        cls.calibration_driver = read_config("optuna_2d_calibration_2017.json")

    def test_all_signals_and_original_backgrounds_are_required_and_preserved(self):
        signals = {s["name"]: s for s in read_config("signal_grid_2017.json")["samples"]}
        originals = {s["name"]: s for s in read_config("sensitivity_2017.json")["samples"] if s["kind"] == "background"}
        actual = {s["name"]: s for s in self.main["samples"]}
        self.assertEqual(len(actual), 137)
        self.assertEqual(set(self.main["required_signals"]), set(signals))
        self.assertEqual(len(self.main["required_signals"]), 114)
        self.assertEqual({n: s for n, s in actual.items() if s["kind"] == "background"}, originals)
        for name, source in signals.items():
            sample = actual[name]
            for field in ("dataset", "cross_section_pb", "generated_suu_mass_gev", "generated_chi_mass_gev",
                          "nominal_suu_mass_gev", "nominal_chi_mass_gev", "input_file_list", "input_file_list_sha256"):
                self.assertEqual(sample[field], source[field], (name, field))
            self.assertEqual(sample["decay_channel"], source["decay_mode"])
            self.assertGreater(sample["cross_section_pb"], 0)
        substitute = actual["HtZt_6000_2000"]
        self.assertEqual((substitute["generated_suu_mass_gev"], substitute["generated_chi_mass_gev"]), (6200, 1950))
        self.assertIn("production_interpolation", substitute)

    def test_common_mass_grid_an_weights_and_explicit_pilot_policies(self):
        for campaign in (self.main, self.calibration):
            self.assertEqual(campaign["mass_bin_edges_gev"], list(range(2500, 10001, 500)))
            self.assertEqual(campaign["chi_mass_bin_edges_gev"], list(range(750, 5001, 250)))
            self.assertEqual(campaign["weight_convention"], "reference")
            self.assertEqual(campaign["physicality_mode"], "off")
            self.assertEqual(campaign["histogram_flow"], "exclude")
            self.assertEqual(campaign["minimum_background_effective_events"], 0)
            self.assertEqual(campaign["analysis_systematic"], "nominal")
            self.assertEqual(campaign["purpose"], "pilot")
            for sample in campaign["samples"]:
                self.assertEqual(sample["normalization_scope"], "representative_subset")
                self.assertEqual(sample["sum_gen_weights"], "metadata")
                self.assertEqual(sample["generated_events"], "metadata")
                self.assertTrue(sample["complete"])
                self.assertEqual(sample["files"], [])

    def test_calibration_and_main_freeze_identical_physics_and_input_limits(self):
        self.assertEqual(score_parameters(self.main), score_parameters(self.calibration))
        self.assertEqual(self.main["samples"], self.calibration["samples"])
        self.assertEqual(self.main["required_signals"], self.calibration["required_signals"])
        self.assertEqual(self.main["source_catalogs"], self.calibration["source_catalogs"])
        for key in ("search_space", "ak_radii", "files_per_job", "max_events_per_job", "max_files_per_sample",
                    "backend", "cmssw_release"):
            self.assertEqual(self.main_driver[key], self.calibration_driver[key], key)
        self.assertEqual(self.calibration["objective"], {"method": "mean_asimov"})
        self.assertEqual(self.main["objective"], {"method": "fixed_reference_regret", "mean_weight": .25,
                                               "reference_file": "sensitivity_2d_references_2017.json"})

    def test_studies_and_fresh_bundle_are_separate_from_legacy(self):
        legacy = read_config("optuna_2017.json")
        for field in ("study_name", "run_dir", "campaign"):
            self.assertEqual(len({legacy[field], self.main_driver[field], self.calibration_driver[field]}), 3)
        for config in (self.main_driver, self.calibration_driver):
            self.assertEqual(config["max_parallel_trials"], 1)
            self.assertEqual(config["max_files_per_sample"], 1)
            self.assertEqual(config["max_events_per_job"], 10000)
            self.assertEqual(config["search_space"]["n_gate_jets"], {"low": 0, "high": 6})
            self.assertEqual(config["search_space"]["t_gate_max"], 600)
            self.assertEqual(config["ak_radii"], [.4, .8])
            self.assertTrue(config["backend"]["cmssw_bundle"].endswith("CMSSW_15_0_19_sensitivity_2d_v1.tar.gz"))
            self.assertNotEqual(config["backend"]["cmssw_bundle"], legacy["backend"]["cmssw_bundle"])
        self.assertEqual(self.calibration_driver["n_trials"], 4)
        self.assertEqual(self.main_driver["n_trials"], 12)

    def test_calibration_driver_resolves_all_sources_without_submission(self):
        with mock.patch.dict(os.environ, {"CMSSW_BASE": "/tmp/sensitivity-config-test/CMSSW_15_0_19"}):
            config, campaign, fingerprint = driver.load_settings(REPOSITORY / "config/optuna_2d_calibration_2017.json")
        self.assertEqual(len(campaign["samples"]), 137)
        self.assertEqual(sum(len(s["input_files"]) for s in campaign["samples"]), 137)
        self.assertEqual(len(fingerprint), 64)
        validation = copy.deepcopy(campaign)
        for sample in validation["samples"]:
            sample["files"] = [f"/tmp/validation/{sample['name']}.root"]
        validate_campaign(validation)
        self.assertEqual(config["search_space"]["n_gate_jets"]["high"], 6)

    def test_missing_reference_stops_main_before_controller_or_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            campaign = copy.deepcopy(self.main)
            campaign["objective"]["reference_file"] = str(directory / "missing_references.json")
            for sample in campaign["samples"]:
                sample["input_file_list"] = str((REPOSITORY / "config" / sample["input_file_list"]).resolve())
            (directory / "campaign.json").write_text(json.dumps(campaign))
            config = copy.deepcopy(self.main_driver)
            config["campaign"] = str(directory / "campaign.json")
            config["backend"]["cmssw_bundle"] = str(directory / "missing_bundle.tar.gz")
            (directory / "config.json").write_text(json.dumps(config))
            with self.assertRaises(FileNotFoundError):
                resolve_objective(campaign["objective"], directory)
            errors = io.StringIO()
            with mock.patch.object(driver, "Controller") as controller, contextlib.redirect_stderr(errors):
                code = driver.main(["--config", str(directory / "config.json"), "--once"])
            self.assertEqual(code, 2)
            self.assertIn("reference", errors.getvalue().lower())
            controller.assert_not_called()

    def test_cli_defaults_to_plan_mode(self):
        controller = mock.Mock()
        controller.tick.return_value = {"trials": 0, "states": {"RUNNING": 0}}
        with mock.patch.dict(os.environ, {"CMSSW_BASE": "/tmp/sensitivity-config-test/CMSSW_15_0_19"}), \
             mock.patch.object(driver, "Controller", return_value=controller), \
             contextlib.redirect_stdout(io.StringIO()):
            code = driver.main(["--config", str(REPOSITORY / "config/optuna_2d_calibration_2017.json"), "--once"])
        self.assertEqual(code, 0)
        controller.tick.assert_called_once_with(False)


if __name__ == "__main__":
    unittest.main()
