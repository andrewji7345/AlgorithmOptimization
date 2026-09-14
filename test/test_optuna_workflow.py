#!/usr/bin/env python3
"""Real Optuna/SQLite integration with deterministic terminal scheduler fixtures."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tarfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from optimize_sensitivity import Controller, JobError, load_settings, resolve_path
from sensitivity_jobs import CondorBackend, expand_argv

HAS_OPTUNA = importlib.util.find_spec("optuna") is not None

SCHEDULER = '''import json, pathlib, sys
operation, store_path, trial_path = sys.argv[1:4]
store = pathlib.Path(store_path)
trial = json.loads(pathlib.Path(trial_path).read_text())
db = json.loads(store.read_text()) if store.exists() else {"submitted": [], "cancelled": []}
token = trial["token"]
if operation == "submit":
    db["submitted"].append(token)
    db[token] = {"state": "complete", "job_ids": [str(len(db["submitted"]))]}
    for task in trial["tasks"]:
        pathlib.Path(task["output"]).write_text("test fixture only")
    store.write_text(json.dumps(db))
    if pathlib.Path(str(store)+".lose_response").exists():
        print("lost submit response")
    else:
        print(json.dumps({"job_ids": db[token]["job_ids"]}))
elif operation == "lookup":
    if pathlib.Path(str(store)+".offline").exists():
        sys.exit(3)
    print(json.dumps(db.get(token, {"state": "missing"})))
elif operation == "cancel":
    db["cancelled"].append(token)
    store.write_text(json.dumps(db))
    print("{}")
'''

EVALUATOR = '''import json, pathlib, sys
campaign_path, configuration, out_dir = sys.argv[1:4]
campaign = json.loads(pathlib.Path(campaign_path).read_text())
output = pathlib.Path(out_dir)
output.mkdir(exist_ok=True)
mode_file = pathlib.Path(sys.argv[4])
mode = mode_file.read_text() if mode_file.exists() else "success"
if mode == "error":
    sys.exit(4)
if mode != "missing":
    feasible = mode != "infeasible"
    value = 2.5 if feasible else -1
    if mode == "nan": value = float("nan")
    identity = configuration
    if mode == "dict":
        n, gate, keep, ak, ca, thrust = configuration.split(":")
        identity = {"n_gate_jets":int(n),"gate_pt_cut":float(gate) if int(n) else None,
                    "collection_pt_cut":float(keep),"ak_radius":float(ak),"ca_radius":float(ca),"cos_thrust_cut":float(thrust)}
    result = {"objective":value,"feasible":feasible,"configuration":identity}
    if (campaign.get("objective") or {}).get("method") == "fixed_reference_regret":
        sys.path.insert(0, str(pathlib.Path.cwd()))
        from sensitivity_objective import aggregate_objective
        context = campaign["objective"]["_reference"]["comparison_context"]
        per_signal = {name:{"significance":2.5,"q0":6.25,"feasible":feasible}
                      for name in campaign["required_signals"]}
        result.update(aggregate_objective(per_signal, campaign["required_signals"], campaign["objective"], context))
        result.update(required_signals=campaign["required_signals"], per_signal=per_signal,
                      parameters=context["parameters"], comparison_samples=context["comparison_samples"])
        if mode == "wrong_reference": result["objective_definition"]["reference_sha256"] = "0"*64
        if mode == "wrong_value": result["objective"] += 1
        if mode == "wrong_context": result["parameters"]["luminosity_pb"] += 1
    (output/"objective.json").write_text(json.dumps(result))
'''

FAKE_CONDOR = '''#!/usr/bin/env python3
import json, os, pathlib, re, sys
operation = pathlib.Path(sys.argv[0]).name
store = pathlib.Path(os.environ["FAKE_CONDOR_STATE"])
db = json.loads(store.read_text()) if store.exists() else {"submitted": 0,"ads":[]}
if operation == "voms-proxy-info":
    print(100000)
elif operation == "condor_submit":
    trial_dir = pathlib.Path(sys.argv[-1]).parent
    trial = json.loads((trial_dir/"trial.json").read_text())
    db["submitted"] += 1
    cluster = db["submitted"]
    for task in trial["tasks"]:
        output = pathlib.Path(task["output"])
        output.write_text("test fixture only")
        (output.parent/"worker_result.json").write_text(json.dumps({"success":True,"trial_token":trial["token"],"sample":task["sample"],"index":task["index"]}))
        db["ads"].append({"ClusterId":cluster,"ProcId":task["index"],"JobStatus":4,"ExitCode":0,
            "ExistingOptimizationTask":task["index"],"ExistingOptimizationToken":trial["token"]})
    store.write_text(json.dumps(db))
    if os.environ.get("FAKE_CONDOR_LOSE_RESPONSE"):
        sys.exit(1)
    print(str(cluster)+".0 - "+str(cluster)+"."+str(len(trial["tasks"])-1))
elif operation in ("condor_q", "condor_history"):
    token = re.search(r'"([^\"]+)"',sys.argv[sys.argv.index("-constraint")+1]).group(1)
    print(json.dumps([ad for ad in db["ads"] if ad["ExistingOptimizationToken"] == token] if operation == "condor_history" else []))
elif operation == "condor_rm":
    print("removed")
'''


@unittest.skipUnless(HAS_OPTUNA, "Install requirements-sensitivity.txt for real Optuna tests")
class OptunaWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="optuna-workflow-")
        self.root = Path(self.temp.name)
        self.scheduler = self.root / "scheduler.py"
        self.scheduler.write_text(SCHEDULER)
        self.evaluator = self.root / "evaluator.py"
        self.evaluator.write_text(EVALUATOR)
        self.registry = self.root / "scheduler.json"
        self.mode = self.root / "evaluation_mode"
        self.campaign = {"purpose": "pilot", "luminosity_pb": 41480,
                         "mass_bin_edges_gev": [0, 2000, 4000, 6000, 10000],
                         "required_signals": ["signal_4000_1000"],
                         "analysis_selection": "AN-23-067-UL2017-cutbased-v1",
                         "correction_prescription": "UL2017-AK4CHS-baseline-AK4AK8Puppi-reco-JEC-JER-btagGuard-v3",
                         "samples": [{"name": name, "kind": kind,
                                      "input_files": [f"root://test.invalid/{name}_{i}.root" for i in range(3)],
                                      "files": [], "cross_section_pb": 1, "generated_chi_mass_gev": 1000,
                                      "sum_gen_weights": "metadata", "generated_events": "metadata",
                                      "normalization_scope": "representative_subset", "complete": True}
                                     for name, kind in (("signal_4000_1000", "signal"), ("QCD_Pt_170_300", "background"))]}
        (self.root / "campaign.json").write_text(json.dumps(self.campaign))
        command = [sys.executable, str(self.scheduler)]
        self.config = {"campaign": "campaign.json", "run_dir": "run", "n_trials": 3,
                       "max_parallel_trials": 2, "files_per_job": 2,
                       "max_events_per_job": 10, "max_poll_errors": 2,
                       "backend": {"kind": "command",
                                   "submit_argv": command + ["submit", str(self.registry), "{trial_json}"],
                                   "lookup_argv": command + ["lookup", str(self.registry), "{trial_json}"],
                                   "cancel_argv": command + ["cancel", str(self.registry), "{trial_json}"]},
                       "evaluator_argv": [sys.executable, str(self.evaluator), "{campaign}", "{configuration}",
                                          "{output_dir}", str(self.mode)]}
        self.config_path = self.root / "config.json"
        self.controllers = []

    def tearDown(self):
        for controller in self.controllers:
            controller.close()
        self.temp.cleanup()

    def controller(self):
        self.config_path.write_text(json.dumps(self.config))
        instance = Controller(*load_settings(self.config_path))
        self.controllers.append(instance)
        return instance

    def configure_reference_objective(self):
        from evaluate_sensitivity import score_parameters
        from sensitivity_objective import comparison_context, context_fingerprint
        samples = {}
        for sample in self.campaign["samples"]:
            samples[sample["name"]] = {
                "kind": sample["kind"], "normalization_scope": sample["normalization_scope"],
                "cross_section_pb": sample["cross_section_pb"], "generated_chi_mass_gev": 1000,
                "generated_events": 10, "sum_gen_weights": 10,
                "event_identity_sha256": hashlib.sha256(sample["name"].encode()).hexdigest(),
                "event_generator_sha256": hashlib.sha256((sample["name"] + "gen").encode()).hexdigest(),
            }
        context = comparison_context(score_parameters(self.campaign), samples, self.campaign["required_signals"])
        reference = {"schema_version": 1, "required_signals": self.campaign["required_signals"],
                     "reference_significances": {n: 5 for n in self.campaign["required_signals"]},
                     "comparison_context": context, "context_sha256": context_fingerprint(context)}
        directory = self.root / "calibration"
        directory.mkdir()
        path = directory / "references.json"
        path.write_text(json.dumps(reference, indent=1) + "\n")
        campaign_dir = self.root / "physics"
        campaign_dir.mkdir()
        self.campaign["objective"] = {"method": "fixed_reference_regret", "mean_weight": 0.25,
                                      "reference_file": "../calibration/references.json"}
        (campaign_dir / "campaign.json").write_text(json.dumps(self.campaign))
        self.config["campaign"] = "physics/campaign.json"
        return path

    def test_fixed_reference_resolves_original_campaign_directory_and_freezes_exact_snapshot(self):
        reference = self.configure_reference_objective()
        self.config.update(n_trials=1, max_parallel_trials=1)
        controller = self.controller()
        controller.tick(execute=False)
        self.assertFalse(self.registry.exists())
        self.assertEqual(controller.campaign["objective"]["reference_file"], str(reference))
        snapshot = controller.root / "objective.reference.json"
        self.assertEqual(snapshot.read_bytes(), reference.read_bytes())
        trial_campaign = json.loads((controller.directory(0) / "campaign.json").read_text())
        self.assertEqual(trial_campaign["objective"]["reference_file"], str(snapshot))
        for filename in ("sensitivity_objective.py", "freeze_sensitivity_references.py", "plot_sensitivity_2d.py"):
            self.assertIn(str(REPO / filename), controller.config["_code_sha256"])
        controller.close()
        resumed = self.controller()
        resumed.tick(execute=True)
        result = resumed.tick(execute=True)
        self.assertEqual(result["states"]["COMPLETE"], 1)
        self.assertEqual(result["best_feasible"]["objective"], 0.625)
        ranking = json.loads((resumed.root / "study_rankings.json").read_text())
        self.assertEqual(ranking["rankings"][0]["objective"], 0.625)
        self.assertEqual(ranking["rankings"][0]["regret_objective"], 0)
        self.assertIn("retrospective", ranking["ranking_basis"])

    def test_reference_content_changes_fingerprint_and_cannot_resume_old_study(self):
        reference = self.configure_reference_objective()
        controller = self.controller()
        original_fingerprint = controller.fingerprint
        controller.close()
        content = json.loads(reference.read_text())
        content["reference_significances"][self.campaign["required_signals"][0]] = 10
        reference.write_text(json.dumps(content))
        _, _, new_fingerprint = load_settings(self.config_path)
        self.assertNotEqual(original_fingerprint, new_fingerprint)
        with self.assertRaisesRegex(JobError, "fingerprint differs"):
            self.controller()
        self.assertFalse(self.registry.exists())

    def test_reference_source_and_snapshot_edits_stop_before_submission(self):
        reference = self.configure_reference_objective()
        controller = self.controller()
        controller.tick(execute=False)
        original = reference.read_bytes()
        reference.write_bytes(original + b"\n")
        with self.assertRaisesRegex(JobError, "Frozen objective reference changed"):
            controller.tick(execute=True)
        reference.write_bytes(original)
        (controller.root / "objective.reference.json").write_bytes(original + b"\n")
        with self.assertRaisesRegex(JobError, "Frozen objective reference changed"):
            controller.tick(execute=True)
        self.assertFalse(self.registry.exists())

    def test_reference_missing_or_physics_mismatch_fails_before_staging(self):
        reference = self.configure_reference_objective()
        self.config_path.write_text(json.dumps(self.config))
        original = reference.read_bytes()
        reference.unlink()
        with self.assertRaises(FileNotFoundError):
            load_settings(self.config_path)
        reference.write_bytes(original)
        campaign_path = self.root / self.config["campaign"]
        self.campaign["luminosity_pb"] += 1
        campaign_path.write_text(json.dumps(self.campaign))
        with self.assertRaisesRegex(ValueError, "score physics parameters"):
            load_settings(self.config_path)
        self.campaign["luminosity_pb"] -= 1
        self.campaign["samples"][0]["cross_section_pb"] *= 2
        campaign_path.write_text(json.dumps(self.campaign))
        with self.assertRaisesRegex(ValueError, "cross_section_pb"):
            load_settings(self.config_path)
        self.assertFalse((self.root / "run").exists())
        self.assertFalse(self.registry.exists())

    def test_fixed_reference_evaluator_contract_is_checked_before_tell(self):
        self.configure_reference_objective()
        self.config.update(n_trials=1, max_parallel_trials=1)
        for mode in ("wrong_reference", "wrong_value", "wrong_context"):
            with self.subTest(mode=mode):
                self.config["run_dir"] = "run_" + mode
                self.mode.write_text(mode)
                controller = self.controller()
                controller.tick(execute=True)
                result = controller.tick(execute=True)
                self.assertEqual(result["states"]["FAIL"], 1)
                self.assertIsNone(controller.study.trials[0].value)
                self.assertIn("fixed-reference", controller.state(controller.directory(0))["reason"])
                controller.close()

    def test_plan_splits_all_samples_without_submission_then_resumes(self):
        first = self.controller()
        summary = first.tick(execute=False)
        self.assertEqual(summary["states"]["RUNNING"], 2)
        self.assertFalse(self.registry.exists())
        descriptor = json.loads((first.directory(0) / "trial.json").read_text())
        self.assertEqual(len(descriptor["tasks"]), 4)
        self.assertEqual({t["sample_kind"] for t in descriptor["tasks"]}, {"signal", "background"})
        p = descriptor["parameters"]
        self.assertEqual(p["r_ak"], 0.8)
        self.assertGreaterEqual(p["t_gate"], p["t_keep"])
        frozen_bytes = (first.directory(0) / "trial.json").read_bytes()
        first.close()
        resumed = self.controller()
        for _ in range(4):
            summary = resumed.tick(execute=True)
        self.assertEqual(summary["states"]["COMPLETE"], 3)
        self.assertEqual(summary["best_feasible"]["objective"], 2.5)
        self.assertEqual((resumed.directory(0) / "trial.json").read_bytes(), frozen_bytes)
        self.assertEqual(len(json.loads(self.registry.read_text())["submitted"]), 3)
        self.assertEqual(len({tuple(sorted(t.params.items())) for t in resumed.study.trials}), 3)

    def test_ak4_and_ak8_are_categorical_choices_with_distinct_identities(self):
        import optuna
        from optimize_sensitivity import sample_configuration, DEFAULT_SPACE, configuration_token
        tokens = []
        for radius in (0.4, 0.8):
            trial = optuna.trial.FixedTrial(dict(r_ak=radius, t_keep=100, n_gate_jets=0,
                                                r_ca=0.8, cos_thrust=0.5))
            params = sample_configuration(trial, DEFAULT_SPACE, (0.4, 0.8))
            self.assertEqual(params["r_ak"], radius)
            self.assertEqual(trial.distributions["r_ak"].choices, (0.4, 0.8))
            tokens.append(configuration_token(params))
        self.assertNotEqual(*tokens)

    def test_new_jobs_reject_legacy_correction_profile_before_submission(self):
        self.campaign["correction_prescription"] = "UL2017-AK4PFchs-AK8PFPuppi-JEC-JER-nominal-v1"
        (self.root / "campaign.json").write_text(json.dumps(self.campaign))
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(JobError, "v3 correction_prescription"):
            load_settings(self.config_path)
        self.assertFalse(self.registry.exists())

    def test_radius_choices_validated_and_frozen_on_resume(self):
        self.campaign["correction_prescription"] = "UL2017-AK4CHS-baseline-AK4AK8Puppi-reco-JEC-JER-btagGuard-v3"
        (self.root / "campaign.json").write_text(json.dumps(self.campaign))
        for bad in ([], [0.6], [0.4, 0.4], [True], "0.4,0.8"):
            self.config["ak_radii"] = bad
            self.config_path.write_text(json.dumps(self.config))
            with self.assertRaisesRegex(JobError, "ak_radii"):
                load_settings(self.config_path)
        self.config["ak_radii"] = [0.4, 0.8]
        first = self.controller()
        first.tick(execute=False)
        self.assertEqual(first.config["ak_radii"], [0.4, 0.8])
        first.close()
        self.config["ak_radii"] = [0.8]
        with self.assertRaises(JobError):
            self.controller()

    def test_lost_submit_response_is_reconciled_without_resubmission(self):
        self.config.update(n_trials=1, max_parallel_trials=1)
        Path(str(self.registry) + ".lose_response").touch()
        controller = self.controller()
        controller.tick(execute=True)
        self.assertEqual(controller.state(controller.directory(0))["phase"], "submitting")
        controller.close()
        resumed = self.controller()
        result = resumed.tick(execute=True)
        self.assertEqual(result["states"]["COMPLETE"], 1)
        self.assertEqual(len(json.loads(self.registry.read_text())["submitted"]), 1)

    def test_physicality_rejection_is_not_a_best_feasible_model(self):
        self.config.update(n_trials=1, max_parallel_trials=1)
        self.mode.write_text("infeasible")
        controller = self.controller()
        controller.tick(execute=True)
        result = controller.tick(execute=True)
        self.assertEqual(result["states"]["COMPLETE"], 1)
        self.assertEqual(controller.study.trials[0].value, -1)
        self.assertIsNone(result["best_feasible"])

    def test_invalid_or_failed_evaluator_never_tells_fabricated_objective(self):
        self.config.update(n_trials=1, max_parallel_trials=1)
        self.mode.write_text("nan")
        controller = self.controller()
        controller.tick(execute=True)
        result = controller.tick(execute=True)
        self.assertEqual(result["states"]["FAIL"], 1)
        self.assertIsNone(controller.study.trials[0].value)

    def test_canonical_configuration_dict_is_accepted(self):
        self.config.update(n_trials=1, max_parallel_trials=1)
        self.mode.write_text("dict")
        controller = self.controller()
        controller.tick(execute=True)
        self.assertEqual(controller.tick(execute=True)["states"]["COMPLETE"], 1)

    def test_stale_objective_cannot_replace_failed_fresh_evaluation(self):
        self.config.update(n_trials=1, max_parallel_trials=1)
        controller = self.controller()
        controller.tick(execute=True)
        output = controller.directory(0) / "evaluation"
        output.mkdir()
        (output / "objective.json").write_text(json.dumps({"objective":999,"feasible":True}))
        self.mode.write_text("missing")
        result = controller.tick(execute=True)
        self.assertEqual(result["states"]["FAIL"], 1)
        self.assertIsNone(controller.study.trials[0].value)

    def test_bad_normalization_and_event_limits_fail_before_staging(self):
        campaign = copy.deepcopy(self.campaign)
        campaign["samples"][0]["cross_section_pb"] = 0
        (self.root / "campaign.json").write_text(json.dumps(campaign))
        with self.assertRaisesRegex(ValueError, "cross section"):
            self.controller()
        self.assertFalse(self.registry.exists())
        (self.root / "campaign.json").write_text(json.dumps(self.campaign))
        campaign["purpose"] = "production"
        (self.root / "campaign.json").write_text(json.dumps(campaign))
        with self.assertRaisesRegex(JobError, "event limit"):
            self.controller()

    def test_pilot_file_limit_freezes_selected_inputs(self):
        self.config.update(n_trials=1, max_parallel_trials=1, max_files_per_sample=1)
        controller = self.controller()
        controller.tick(execute=False)
        descriptor = json.loads((controller.directory(0) / "trial.json").read_text())
        self.assertEqual(len(descriptor["tasks"]), 2)
        for sample in controller.campaign["samples"]:
            self.assertEqual(sample["available_input_file_count"], 3)
            self.assertEqual(len(sample["input_files"]), 1)

    def test_code_edits_during_continuous_run_are_detected(self):
        controller = self.controller()
        controller.tick(execute=False)
        self.scheduler.write_text(SCHEDULER + "\n# changed implementation\n")
        with self.assertRaisesRegex(JobError, "Source changed"):
            controller.tick(execute=True)
        self.assertFalse(self.registry.exists())

    def test_edited_completed_objective_cannot_change_rankings(self):
        self.config.update(n_trials=1, max_parallel_trials=1)
        controller = self.controller()
        controller.tick(execute=True)
        controller.tick(execute=True)
        path = controller.directory(0) / "evaluation/objective.json"
        objective = json.loads(path.read_text())
        objective["objective"] = 100
        path.write_text(json.dumps(objective))
        with self.assertRaisesRegex(JobError, "provenance is invalid"):
            controller.tick(execute=True)

    def test_scheduler_failure_cancels_all_sibling_jobs(self):
        self.config.update(n_trials=1, max_parallel_trials=1)
        controller = self.controller()
        controller.tick(execute=True)
        data = json.loads(self.registry.read_text())
        token = data["submitted"][0]
        data[token]["state"] = "failed"
        self.registry.write_text(json.dumps(data))
        result = controller.tick(execute=True)
        self.assertEqual(result["states"]["FAIL"], 1)
        self.assertEqual(json.loads(self.registry.read_text())["cancelled"], [token])

    def test_poll_failure_is_bounded_and_preserves_running_jobs(self):
        self.config.update(n_trials=1, max_parallel_trials=1)
        controller = self.controller()
        controller.tick(execute=True)
        Path(str(self.registry) + ".offline").touch()
        controller.tick(execute=True)
        with self.assertRaisesRegex(JobError, "bounded retries"):
            controller.tick(execute=True)
        self.assertEqual(controller.study.trials[0].state.name, "RUNNING")
        self.assertEqual(json.loads(self.registry.read_text())["cancelled"], [])

    def test_timeout_cancels_before_failure(self):
        self.config.update(n_trials=1, max_parallel_trials=1, trial_timeout_seconds=1)
        controller = self.controller()
        controller.tick(execute=True)
        data = json.loads(self.registry.read_text())
        token = data["submitted"][0]
        data[token]["state"] = "running"
        self.registry.write_text(json.dumps(data))
        state = controller.state(controller.directory(0))
        state["submitted_at"] -= 5
        controller.save(controller.directory(0), state)
        result = controller.tick(execute=True)
        self.assertEqual(result["states"]["FAIL"], 1)
        self.assertEqual(json.loads(self.registry.read_text())["cancelled"], [token])

    def test_one_controller_lock_and_changed_campaign_rejection(self):
        controller = self.controller()
        with self.assertRaisesRegex(JobError, "Another controller"):
            self.controller()
        controller.close()
        campaign = copy.deepcopy(self.campaign)
        campaign["luminosity_pb"] = 999
        (self.root / "campaign.json").write_text(json.dumps(campaign))
        with self.assertRaisesRegex(JobError, "fingerprint differs"):
            self.controller()

    def test_run_directory_cannot_be_attached_to_a_new_study(self):
        controller = self.controller()
        controller.tick(execute=False)
        controller.close()
        self.config["study_name"] = "accidentally-new-study"
        with self.assertRaisesRegex(JobError, "immutable trial input"):
            self.controller()

    def test_missing_job_grace_expires_without_resubmitting(self):
        self.config.update(n_trials=1, max_parallel_trials=1, submission_visibility_seconds=1)
        controller = self.controller()
        controller.tick(execute=True)
        data = json.loads(self.registry.read_text())
        token = data["submitted"][0]
        data.pop(token)
        self.registry.write_text(json.dumps(data))
        controller.tick(execute=True)
        state = controller.state(controller.directory(0))
        state["missing_since"] -= 5
        controller.save(controller.directory(0), state)
        result = controller.tick(execute=True)
        self.assertEqual(result["states"]["FAIL"], 1)
        data = json.loads(self.registry.read_text())
        self.assertEqual(data["submitted"], [token])
        self.assertEqual(data["cancelled"], [token])

    def test_crash_after_tell_repairs_local_state(self):
        self.config.update(n_trials=1, max_parallel_trials=1)
        controller = self.controller()
        controller.tick(execute=True)
        controller.tick(execute=True)
        state = controller.state(controller.directory(0))
        state["phase"] = "evaluating"
        controller.save(controller.directory(0), state)
        controller.close()
        resumed = self.controller()
        resumed.tick(execute=True)
        self.assertEqual(resumed.state(resumed.directory(0))["phase"], "complete")
        self.assertEqual(len(json.loads(self.registry.read_text())["submitted"]), 1)

    def test_real_terminal_condor_protocol_and_lost_response_recovery(self):
        self.config.update(n_trials=1, max_parallel_trials=1)
        binaries = self.root / "bin"
        binaries.mkdir()
        for name in ("condor_submit", "condor_q", "condor_history", "condor_rm", "voms-proxy-info"):
            executable = binaries / name
            executable.write_text(FAKE_CONDOR)
            executable.chmod(0o755)
        archive = self.root / "bundle.tar.gz"
        archive.write_bytes(b"fixture bundle")
        self.config["backend"] = {"kind": "condor", "cmssw_bundle": str(archive), "schedd": "fake-schedd.example"}
        env = {"PATH": str(binaries) + os.pathsep + os.environ["PATH"],
               "FAKE_CONDOR_STATE": str(self.registry), "FAKE_CONDOR_LOSE_RESPONSE": "1"}
        with mock.patch.dict(os.environ, env):
            controller = self.controller()
            controller.tick(execute=True)
            self.assertEqual(controller.state(controller.directory(0))["phase"], "submitting")
            controller.close()
            resumed = self.controller()
            result = resumed.tick(execute=True)
        self.assertEqual(result["states"]["COMPLETE"], 1)
        self.assertEqual(json.loads(self.registry.read_text())["submitted"], 1)
        jdl = (resumed.directory(0) / "submit.jdl").read_text()
        self.assertIn("+ExistingOptimizationToken", jdl)
        self.assertIn("bundle.json", jdl)
        self.assertIn("transfer_output_files = output.root,worker_result.json", jdl)

    def test_changed_bundle_rejected_on_resume(self):
        self.config.update(n_trials=1, max_parallel_trials=1)
        archive = self.root / "bundle.tar.gz"
        archive.write_bytes(b"first bundle")
        self.config["backend"] = {"kind": "condor", "cmssw_bundle": str(archive)}
        with mock.patch.object(CondorBackend, "preflight"):
            controller = self.controller()
            with mock.patch.object(CondorBackend, "submit", return_value=["1.0"]):
                controller.tick(execute=True)
            controller.close()
            archive.write_bytes(b"different bundle")
            resumed = self.controller()
            with self.assertRaisesRegex(JobError, "immutable trial input"):
                resumed.tick(execute=True)


class AdapterValidationTest(unittest.TestCase):
    def test_lpc_native_rm_keeps_dispatch_basename(self):
        with mock.patch("sensitivity_jobs.shutil.which", side_effect=lambda n: "/usr/local/bin/" + n), \
                mock.patch("sensitivity_jobs.Path.is_file", return_value=True):
            backend = CondorBackend({"kind": "condor"}, REPO)
        self.assertEqual(backend.commands["condor_rm"], "/usr/libexec/condor/condor_rm")
        with mock.patch.object(backend, "run_condor", side_effect=['[{"JobStatus":2}]', "removed"]) as command:
            backend.cancel(Path("/tmp"), {"token": "eo_abc_0"}, {})
        self.assertEqual(command.call_args.args[0][0], "/usr/libexec/condor/condor_rm")

    def test_cancellation_empty_queue_and_completion_race(self):
        backend = CondorBackend({"kind": "condor"}, REPO)
        descriptor = {"token": "eo_abc_0"}
        with mock.patch.object(backend, "run_condor", return_value="") as command:
            backend.cancel(Path("/tmp"), descriptor, {})
        self.assertEqual(command.call_count, 1)
        with mock.patch.object(backend, "run_condor", side_effect=[
                '[{"JobStatus":2}]', JobError("No matching jobs"), ""]):
            backend.cancel(Path("/tmp"), descriptor, {})
        with mock.patch.object(backend, "run_condor", side_effect=[
                '[{"JobStatus":2}]', JobError("Permission denied"), '[{"JobStatus":2}]']):
            with self.assertRaisesRegex(JobError, "Permission denied"):
                backend.cancel(Path("/tmp"), descriptor, {})

    def test_successful_empty_queue_reconciles_completed_history(self):
        backend = CondorBackend({"kind": "condor"}, REPO)
        descriptor = {"token": "eo_abc_0", "tasks": [{"index": 0}]}
        done = [{"ClusterId": 1, "ProcId": 0, "JobStatus": 4,
                 "ExitCode": 0, "ExistingOptimizationTask": 0}]
        with mock.patch.object(backend, "run_condor", side_effect=[json.dumps(done), "\n"]):
            self.assertEqual(backend.lookup(Path("/tmp"), descriptor, {})["state"], "complete")
        with mock.patch.object(backend, "run_condor", side_effect=["", ""]):
            self.assertEqual(backend.lookup(Path("/tmp"), descriptor, {})["state"], "missing")
        with mock.patch.object(backend, "run_condor", side_effect=["[]", "connection failed"]):
            with self.assertRaisesRegex(JobError, "Malformed"):
                backend.lookup(Path("/tmp"), descriptor, {})

    def test_bundle_environment_expansion_requires_initialized_variable(self):
        with mock.patch.dict(os.environ, {"CMSSW_BASE": "/tmp/research/CMSSW_15_0_19"}):
            path = resolve_path("$CMSSW_BASE/../sensitivity_bundles/release.tar.gz", Path("/tmp"))
        self.assertEqual(path, Path("/tmp/research/sensitivity_bundles/release.tar.gz"))
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(JobError, "Unresolved environment variable"):
                resolve_path("$CMSSW_BASE/bundle.tar.gz", Path("/tmp"))

    def test_auto_schedd_is_persisted_and_all_commands_target_it(self):
        with tempfile.TemporaryDirectory() as raw:
            config = {"kind": "condor", "schedd": "auto"}
            backend = CondorBackend(config, REPO)
            backend.state_dir = Path(raw)
            ads = [{"Name": "busy", "RecentDaemonCoreDutyCycle": 0.9},
                   {"Name": "available", "RecentDaemonCoreDutyCycle": 0.1}]
            with mock.patch("sensitivity_jobs.run_command", return_value=json.dumps(ads)):
                backend.resolve_schedd()
            self.assertEqual(backend.target_args(), ["-name", "available"])
            resumed = CondorBackend(config, REPO)
            resumed.state_dir = Path(raw)
            with mock.patch("sensitivity_jobs.run_command") as command:
                resumed.resolve_schedd()
                command.assert_not_called()
            self.assertEqual(resumed.target_args(), ["-name", "available"])
            descriptor = {"token": "eo_abc_0", "tasks": [{"index": 0}]}
            with mock.patch("sensitivity_jobs.run_command", return_value="[]") as command:
                resumed.lookup(Path(raw), descriptor, {})
            for call in command.call_args_list:
                self.assertEqual(call.args[0][1:3], ["-name", "available"])

    def test_packaging_preserves_root_calibrations_and_cmssw_metadata(self):
        with tempfile.TemporaryDirectory(prefix="optuna-packaging-") as raw:
            base = Path(raw) / "CMSSW_15_0_19"
            package = base / "src/SuuAnalysis/ExistingOptimization"
            (package / "data/analysis_2017").mkdir(parents=True)
            (package / "test").mkdir()
            (base / "lib").mkdir()
            (base / ".SCRAM").mkdir()
            (base / ".SCRAM/metadata").write_text("needed")
            (package / "test/runSensitivityScan_cfg.py").write_text("# fixture")
            (package / "data/analysis_2017/pu.root").write_text("calibration")
            (package / "result.root").write_text("old result")
            (package / "legacy").mkdir()
            (package / "legacy/unused.py").write_text("legacy")
            output = Path(raw) / "bundle.tar.gz"
            argv = [sys.executable, str(REPO / "run_scripts/sensitivity/pack_cmssw.py"),
                    "--cmssw-base", str(base), "--output", str(output)]
            subprocess.run(argv, check=True, capture_output=True)
            with tarfile.open(output) as archive:
                names = archive.getnames()
            self.assertTrue(any(name.endswith("data/analysis_2017/pu.root") for name in names))
            self.assertTrue(any(name.endswith(".SCRAM/metadata") for name in names))
            self.assertFalse(any(name.endswith("result.root") or "/legacy" in name for name in names))
            self.assertNotEqual(subprocess.run(argv, capture_output=True).returncode, 0)

    def test_shell_metacharacters_stay_a_single_literal_argument(self):
        values = {"trial_dir": "/tmp/a;echo BAD $(whoami)"}
        argv = expand_argv(["printf", "%s", "{trial_dir}"], values)
        result = subprocess.run(argv, check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout, values["trial_dir"])
        with self.assertRaises(JobError):
            expand_argv(["submit", "{unknown}"], values)

    def test_condor_history_failure_and_duplicate_task_detection(self):
        backend = CondorBackend({"kind": "condor"}, REPO)
        descriptor = {"token": "eo_abc_0", "tasks": [{"index": 0}, {"index": 1}]}
        held = [{"ClusterId": 1, "ProcId": 0, "JobStatus": 5,
                 "ExistingOptimizationTask": 0, "HoldReason": "Input transfer failed"}]
        with mock.patch("sensitivity_jobs.run_command", side_effect=["[]", json.dumps(held)]):
            response = backend.lookup(Path("/tmp"), descriptor, {})
        self.assertEqual(response["state"], "failed")
        self.assertIn("transfer failed", response["reason"])
        duplicate = [{"ClusterId": n, "ProcId": 0, "JobStatus": 2,
                      "ExistingOptimizationTask": 0} for n in (1, 2)]
        with mock.patch("sensitivity_jobs.run_command", side_effect=["[]", json.dumps(duplicate)]):
            response = backend.lookup(Path("/tmp"), descriptor, {})
        self.assertEqual(response["state"], "failed")


if __name__ == "__main__":
    unittest.main()
