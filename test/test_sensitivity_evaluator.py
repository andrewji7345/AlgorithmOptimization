#!/usr/bin/env python3
"""Campaign integration tests, including coverage and sample-normalization failures."""

import contextlib
import copy
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

try:
    import ROOT
except ImportError:
    ROOT = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compact_scan_metrics as compact
import evaluate_sensitivity as evaluator


def campaign_fixture():
    return {
        "purpose": "pilot", "luminosity_pb": 1000,
        "mass_bin_edges_gev": [0, 3000, 6000], "required_signals": ["WbWb_4000_1000"],
        "analysis_selection": evaluator.ANALYSIS_SELECTION,
        "correction_prescription": evaluator.CORRECTION_PRESCRIPTION,
        "physicality": {"min_valid_events": 20},
        "physicality_mode": "gate",  # Explicit legacy-contract regression.
        "samples": [
            {"name": "WbWb_4000_1000", "kind": "signal", "files": ["signal.root"],
             "generated_chi_mass_gev": 1000, "cross_section_pb": 0.1,
             "sum_gen_weights": "metadata", "generated_events": "metadata",
             "normalization_scope": "representative_subset", "complete": True},
            {"name": "QCD_HT2000toInf_UL2017", "kind": "background", "files": ["background.root"],
             "cross_section_pb": 2, "sum_gen_weights": "metadata", "generated_events": "metadata",
             "normalization_scope": "representative_subset", "complete": True},
        ],
    }


def payload_fixture(name, count=100, offset=0):
    signal = name.startswith("Wb")
    gen = np.ones(count)
    gen[-10:] = -0.5
    return SimpleNamespace(
        n_events=count, run=np.ones(count, dtype=np.uint32), lumi=np.ones(count, dtype=np.uint32),
        event=np.arange(offset + 1, offset + count + 1, dtype=np.uint64), gen_weight=gen,
        analysis_weight=np.full(count, 0.9), passes_baseline=np.ones(count, dtype=bool),
        passes_signal_region=np.ones((count, 1), dtype=bool), reco_status=np.zeros((count, 1), dtype=np.uint8),
        sj1_mass=np.full((count, 1), 1000.), sj2_mass=np.full((count, 1), 1000.),
        suu_mass=np.full((count, 1), 4000. if signal else 3500.),
    )


def metadata_fixture(sample, payload, path=None):
    return SimpleNamespace(
        path=path or sample["files"][0], schema_version=3, sample_name=sample["name"],
        sample_kind=sample["kind"], analysis_selection=evaluator.ANALYSIS_SELECTION,
        correction_prescription=evaluator.CORRECTION_PRESCRIPTION, use_jec=True,
        processed_events=payload.n_events, sum_weights=float(payload.gen_weight.sum()),
        sum_weights2=float(np.square(payload.gen_weight).sum()), ak_radius=0.8,
        _config_lookup={(100., 0.8, 0.): 0}, config_base_index=[0],
    )


class SensitivityCampaignTest(unittest.TestCase):
    def setUp(self):
        self.campaign = campaign_fixture()
        self.payloads = {sample["files"][0]: payload_fixture(sample["name"])
                         for sample in self.campaign["samples"]}
        self.metadata = {sample["files"][0]: metadata_fixture(sample, self.payloads[sample["files"][0]])
                         for sample in self.campaign["samples"]}
        self.key = evaluator.parse_configuration("0:none:100:0.8:0.8:0")
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(mock.patch.object(evaluator, "load_metadata", side_effect=lambda path: self.metadata[Path(path).name]))
        self.stack.enter_context(mock.patch.object(evaluator, "read_event_payload", side_effect=lambda meta: self.payloads[Path(meta.path).name]))
        self.addCleanup(self.stack.close)

    def test_campaign_yield_counts_signed_weights_and_finite_cli_outputs(self):
        result = evaluator.evaluate_campaign(self.campaign, self.key)
        self.assertTrue(result["feasible"])
        self.assertFalse(result["production_ready"])
        self.assertGreater(result["objective"], 0)
        np.testing.assert_allclose(result["background"]["sumw"], [0, 1800])
        self.assertAlmostEqual(result["per_signal"]["WbWb_4000_1000"]["signal_yield"], 90)
        self.assertEqual(result["samples"]["WbWb_4000_1000"]["sum_gen_weights"], 85)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.json"
            path.write_text(json.dumps(self.campaign))
            with contextlib.redirect_stdout(io.StringIO()):
                code = evaluator.main(["--campaign", str(path), "--configuration", "0:none:100:0.8:0.8:0",
                                       "--output-dir", str(Path(directory) / "out")])
            self.assertEqual(code, 0)
            output = Path(directory) / "out"
            raw = (output / "objective.json").read_text()
            self.assertNotIn("Infinity", raw)
            self.assertNotIn("NaN", raw)
            self.assertTrue(json.loads(raw)["feasible"])
            for filename in ("sensitivity_ranking.json", "per_signal.csv", "suu_mass.png"):
                self.assertGreater((output / filename).stat().st_size, 0)

    def test_baseline_and_sr_applied_without_changing_generator_denominator(self):
        signal = self.payloads["signal.root"]
        signal.passes_baseline[:20] = False
        signal.passes_signal_region[:30] = False
        result = evaluator.evaluate_campaign(self.campaign, self.key)
        info = result["per_signal"]["WbWb_4000_1000"]
        self.assertAlmostEqual(info["signal_yield"], 100 / 85 * 0.9 * 55)
        self.assertEqual(info["physicality"]["n_gate_events"], 80)
        self.assertEqual(result["samples"]["WbWb_4000_1000"]["selected_events"], 70)

    def test_physicality_failure_for_bad_pre_sr_mass_cannot_win(self):
        signal = self.payloads["signal.root"]
        signal.sj1_mass[:60] = 2000
        signal.passes_signal_region[:60] = False
        result = evaluator.evaluate_campaign(self.campaign, self.key)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["objective"], -1)
        self.assertIn("WbWb_4000_1000:physicality_gate_failed", result["failure_reasons"])

    def test_missing_background_and_incomplete_signal_coverage_rejected(self):
        bad = copy.deepcopy(self.campaign)
        bad["samples"] = bad["samples"][:1]
        with self.assertRaisesRegex(ValueError, "background"):
            evaluator.evaluate_campaign(bad, self.key)
        bad = copy.deepcopy(self.campaign)
        bad["required_signals"].append("missing_hypothesis")
        with self.assertRaisesRegex(ValueError, "required_signals"):
            evaluator.evaluate_campaign(bad, self.key)

    def test_production_requires_full_denominator_and_event_count(self):
        bad = copy.deepcopy(self.campaign)
        bad["purpose"] = "production"
        with self.assertRaisesRegex(ValueError, "pilot"):
            evaluator.evaluate_campaign(bad, self.key)
        for sample in bad["samples"]:
            sample.update(normalization_scope="full_dataset", sum_gen_weights=850, generated_events=1000)
        with self.assertRaisesRegex(ValueError, "incomplete generated coverage"):
            evaluator.evaluate_campaign(bad, self.key)
        for sample in bad["samples"]:
            sample.update(sum_gen_weights=85, generated_events=100)
        result = evaluator.evaluate_campaign(bad, self.key)
        self.assertTrue(result["production_ready"])

    def test_legacy_schema_wrong_corrections_and_duplicate_events_rejected(self):
        meta = self.metadata["background.root"]
        meta.schema_version = 2
        with self.assertRaisesRegex(ValueError, "schema version 3"):
            evaluator.evaluate_campaign(self.campaign, self.key)
        meta.schema_version = 3
        meta.correction_prescription = "uncorrected"
        with self.assertRaisesRegex(ValueError, "correction_prescription"):
            evaluator.evaluate_campaign(self.campaign, self.key)
        meta.correction_prescription = evaluator.CORRECTION_PRESCRIPTION
        self.payloads["background.root"].event[1] = 1
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluator.evaluate_campaign(self.campaign, self.key)

    def test_shifted_ntuple_cannot_silently_enter_nominal_objective(self):
        self.metadata["background.root"].analysis_systematic = "JECUp"
        with self.assertRaisesRegex(ValueError, "analysis_systematic"):
            evaluator.evaluate_campaign(self.campaign, self.key)
        self.campaign["analysis_systematic"] = "JECUp"
        with self.assertRaisesRegex(ValueError, "analysis_systematic"):
            evaluator.evaluate_campaign(self.campaign, self.key)
        for meta in self.metadata.values():
            meta.analysis_systematic = "JECUp"
        self.assertIn("per_signal", evaluator.evaluate_campaign(self.campaign, self.key))

    def test_declared_shards_aggregate_before_significance(self):
        sample = self.campaign["samples"][1]
        sample["files"].append("background_part2.root")
        payload = payload_fixture(sample["name"], offset=1000)
        self.payloads["background_part2.root"] = payload
        self.metadata["background_part2.root"] = metadata_fixture(sample, payload, "background_part2.root")
        result = evaluator.evaluate_campaign(self.campaign, self.key)
        self.assertEqual(result["samples"][sample["name"]]["generated_events"], 200)
        self.assertAlmostEqual(sum(result["background"]["sumw"]), 1800)
        first_variance = 2000 ** 2 / 85 ** 2 * 0.9 ** 2 * 92.5
        self.assertAlmostEqual(sum(result["background"]["sumw2"]), first_variance / 2)

    def test_two_dimensional_signal_template(self):
        self.campaign["chi_mass_bin_edges_gev"] = [0, 800, 1200, 2000]
        result = evaluator.evaluate_campaign(self.campaign, self.key)
        self.assertEqual(np.asarray(result["background"]["sumw"]).shape, (2, 3))
        self.assertTrue(result["feasible"])


def make_sensitivity_root_fixture(path, sample, radius=0.8):
    """Write a calibrated-schema fixture with genuine scalar/vector ROOT branches."""
    from array import array
    from test_compact_scan_metrics import make_compact_fixture
    make_compact_fixture(path, sample_name=sample["name"], schema_version_number=3, ak_radius=radius)
    root_file = ROOT.TFile(str(path), "UPDATE")
    root_file.cd("compactScan")
    source = root_file.Get("compactScan/Metadata")
    source.SetBranchStatus("useJEC", 0)
    metadata = source.CloneTree(0)
    jec = array("b", [1])
    selection = ROOT.std.string(evaluator.ANALYSIS_SELECTION)
    correction = ROOT.std.string(evaluator.CORRECTION_PRESCRIPTION)
    kind = ROOT.std.string(sample["kind"])
    metadata.Branch("useJEC", jec, "useJEC/O")
    metadata.Branch("analysisSelection", selection)
    metadata.Branch("correctionPrescription", correction)
    metadata.Branch("sampleKind", kind)
    source.GetEntry(0)
    metadata.Fill()
    metadata.Write("Metadata", ROOT.TObject.kOverwrite)
    events = root_file.Get("compactScan/Events")
    weight, baseline = array("f", [0.9]), array("b", [1])
    region = ROOT.std.vector("unsigned char")()
    new_branches = [events.Branch("analysisWeight", weight, "analysisWeight/F"),
                    events.Branch("passesBaseline", baseline, "passesBaseline/O"),
                    events.Branch("passesSignalRegion", region)]
    for index in range(events.GetEntries()):
        events.GetEntry(index)
        region.clear()
        for status in events.recoStatus:
            code = ord(status) if isinstance(status, str) else int(status)
            region.push_back(int(code == 0))
        for branch in new_branches:
            branch.Fill()
    events.Write("Events", ROOT.TObject.kOverwrite)
    root_file.Close()


@unittest.skipIf(ROOT is None or compact.uproot is None, "CMSSW ROOT I/O is not available")
class SensitivityRootCliTest(unittest.TestCase):
    def test_schema_v3_signal_and_arbitrarily_named_background_real_root_io(self):
        ROOT.gROOT.SetBatch(True)
        campaign = campaign_fixture()
        campaign["mass_bin_edges_gev"] = [0, 10000]
        # This four-event I/O fixture intentionally has tiny effective MC size;
        # production and pilot templates retain the default threshold of ten.
        campaign["minimum_background_effective_events"] = 0.1
        campaign["physicality"] = {"min_valid_events": 1, "invalid_limit": 1, "tail_limit": 1}
        with tempfile.TemporaryDirectory() as directory:
            for sample in campaign["samples"]:
                path = Path(directory) / sample["files"][0]
                make_sensitivity_root_fixture(path, sample)
            campaign_path = Path(directory) / "campaign.json"
            campaign_path.write_text(json.dumps(campaign))
            output = Path(directory) / "evaluation"
            with contextlib.redirect_stdout(io.StringIO()):
                code = evaluator.main(["--campaign", str(campaign_path),
                                       "--configuration", "0:none:100:0.8:0.8:0",
                                       "--output-dir", str(output)])
            self.assertEqual(code, 0)
            result = json.loads((output / "objective.json").read_text())
            self.assertTrue(result["feasible"])
            self.assertEqual(result["samples"]["QCD_HT2000toInf_UL2017"]["generated_events"], 4)
            self.assertAlmostEqual(result["per_signal"]["WbWb_4000_1000"]["signal_yield"], 18, places=5)
            self.assertTrue((output / "suu_mass.png").is_file())

    @unittest.skipUnless(importlib.util.find_spec("optuna") is not None, "Optuna is not available")
    def test_actual_optuna_controller_and_default_evaluator_subprocess(self):
        self._check_controller_radius(0.8)

    @unittest.skipUnless(importlib.util.find_spec("optuna") is not None, "Optuna is not available")
    def test_actual_ak4_controller_and_default_evaluator_subprocess(self):
        self._check_controller_radius(0.4)

    def _check_controller_radius(self, radius):
        from optimize_sensitivity import Controller, load_settings
        ROOT.gROOT.SetBatch(True)
        campaign = campaign_fixture()
        campaign.update(mass_bin_edges_gev=[0, 10000], minimum_background_effective_events=0.1,
                        physicality={"min_valid_events": 1, "invalid_limit": 1, "tail_limit": 1})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sample in campaign["samples"]:
                fixture = root / sample["files"][0]
                make_sensitivity_root_fixture(fixture, sample, radius)
                sample["input_files"] = [str(fixture)]
                sample["files"] = []
            (root / "campaign.json").write_text(json.dumps(campaign))
            # The terminal scheduler is deterministic and copies real ntuples;
            # Optuna, SQLite, CLI argument construction and evaluator ROOT I/O
            # are the production implementations. No batch jobs are launched.
            scheduler = root / "scheduler.py"
            scheduler.write_text('''import json, pathlib, shutil, sys
operation, descriptor_path = sys.argv[1:]
descriptor = json.loads(pathlib.Path(descriptor_path).read_text())
if operation == "submit":
    for task in descriptor["tasks"]:
        shutil.copyfile(task["input_files"][0], task["output"])
    print(json.dumps({"job_ids": ["fixture.0"]}))
elif operation == "lookup":
    print(json.dumps({"state": "complete", "job_ids": ["fixture.0"]}))
elif operation == "cancel":
    print("{}")
''')
            config = {
                "campaign": "campaign.json", "run_dir": "run", "n_trials": 1, "ak_radii": [radius],
                "max_parallel_trials": 1, "files_per_job": 1, "max_events_per_job": -1,
                "backend": {"kind": "command", **{
                    field: [sys.executable, str(scheduler), operation, "{trial_json}"]
                    for field, operation in (("submit_argv", "submit"), ("lookup_argv", "lookup"), ("cancel_argv", "cancel"))}},
                "search_space": {"t_keep": {"low": 100, "high": 100, "step": 20}, "t_gate_max": 100,
                                 "n_gate_jets": {"low": 0, "high": 0}, "r_ca": {"low": 0.8, "high": 0.8},
                                 "cos_thrust": {"low": 0., "high": 0.}},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            controller = Controller(*load_settings(config_path))
            try:
                controller.tick(execute=True)
                summary = controller.tick(execute=True)
                self.assertEqual(summary["states"]["COMPLETE"], 1,
                                 msg=json.dumps({"summary": summary,
                                                 "trial": controller.study.trials[0].user_attrs,
                                                 "state": controller.state(controller.directory(0))}))
                self.assertGreater(summary["best_feasible"]["objective"], 0)
                result = json.loads((controller.directory(0) / "evaluation/objective.json").read_text())
                self.assertIsInstance(result["configuration"], dict)
                self.assertEqual(result["configuration"]["ak_radius"], radius)
                self.assertFalse(result["production_ready"])
                self.assertAlmostEqual(result["per_signal"]["WbWb_4000_1000"]["signal_yield"], 18, places=5)
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
