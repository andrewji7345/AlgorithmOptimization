#!/usr/bin/env python3
"""Persistent Optuna ask/tell controller for independently scheduled CMS jobs."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from sensitivity_jobs import (JobError, atomic_json, expand_argv, immutable_json,
                              immutable_text, make_backend, run_command)

REPO = Path(__file__).resolve().parent
FINAL_PHASES = {"complete", "failed"}
DEFAULT_SPACE = {
    "t_keep": {"low": 100, "high": 400, "step": 20},
    "t_gate_max": 600,
    "n_gate_jets": {"low": 0, "high": 6},
    "r_ca": {"low": 0.4, "high": 1.6, "step": 0.1},
    "cos_thrust": {"low": 0.0, "high": 0.95, "step": 0.05},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def resolve_path(value, base):
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        raise JobError(f"Unresolved environment variable in path: {value}")
    path = Path(expanded).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_settings(config_path, run_dir=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    campaign_path = resolve_path(config["campaign"], config_path.parent)
    campaign = json.loads(campaign_path.read_text())
    from sensitivity_objective import REFERENCE_METHOD, resolve_objective
    if "objective" in campaign:
        # Trial campaign files live below run_dir, so resolve against the
        # original campaign directory before copying any trial inputs.
        campaign["objective"] = resolve_objective(campaign["objective"], campaign_path.parent)
    config["campaign"] = str(campaign_path)
    config["run_dir"] = str(resolve_path(run_dir or config["run_dir"], config_path.parent))
    config.setdefault("backend", {"kind": "condor"})
    backend = config["backend"]
    if backend.get("kind", "condor") == "condor":
        backend["cmssw_bundle"] = str(resolve_path(backend["cmssw_bundle"], config_path.parent))
        if not backend["cmssw_bundle"].endswith((".tar.gz", ".tgz")):
            raise JobError("cmssw_bundle must have a .tar.gz or .tgz basename")
    for key, default in (("n_trials", 30), ("max_parallel_trials", 2), ("files_per_job", 5),
                         ("max_poll_errors", 5), ("trial_timeout_seconds", 86400),
                         ("submission_visibility_seconds", 180), ("poll_seconds", 30)):
        config.setdefault(key, default)
        if isinstance(config[key], bool) or not isinstance(config[key], (int, float)) or config[key] <= 0:
            raise JobError(f"{key} must be positive")
    for key in ("n_trials", "max_parallel_trials", "files_per_job", "max_poll_errors"):
        if not isinstance(config[key], int):
            raise JobError(f"{key} must be an integer")
    config.setdefault("max_events_per_job", -1)
    if (not isinstance(config["max_events_per_job"], int) or isinstance(config["max_events_per_job"], bool)
            or config["max_events_per_job"] == 0 or config["max_events_per_job"] < -1):
        raise JobError("max_events_per_job must be -1 (all) or positive")
    if config["max_events_per_job"] != -1 and campaign.get("purpose") != "pilot":
        raise JobError("An event limit requires a campaign explicitly marked purpose:'pilot'")
    max_files = config.get("max_files_per_sample")
    if max_files is not None and (isinstance(max_files, bool) or not isinstance(max_files, int) or max_files <= 0):
        raise JobError("max_files_per_sample must be a positive integer")
    if max_files is not None and campaign.get("purpose") != "pilot":
        raise JobError("A file limit requires a campaign explicitly marked purpose:'pilot'")
    if "ak_radius" in config and "ak_radii" in config:
        raise JobError("Specify ak_radii or the legacy fixed ak_radius, not both")
    radii = config.get("ak_radii", [config.get("ak_radius", 0.8)])
    if (not isinstance(radii, list) or not radii or
            any(isinstance(r, bool) or not isinstance(r, (int, float)) or r not in (0.4, 0.8) for r in radii)
            or len(set(radii)) != len(radii)):
        raise JobError("ak_radii must be a nonempty unique list containing only 0.4 and/or 0.8")
    config["ak_radii"] = sorted(float(r) for r in radii)
    config.pop("ak_radius", None)
    config.setdefault("cmssw_release", "CMSSW_15_0_19")
    if not re.fullmatch(r"CMSSW_[A-Za-z0-9_]+", config["cmssw_release"]):
        raise JobError("Invalid CMSSW release directory")
    space = copy.deepcopy(DEFAULT_SPACE)
    space.update(config.get("search_space", {}))
    config["search_space"] = space
    for key in ("t_keep", "n_gate_jets", "r_ca", "cos_thrust"):
        spec = space[key]
        values = [spec.get("low"), spec.get("high"), spec.get("step", 1)]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise JobError(f"Invalid numeric search space {key}")
        if values[0] > values[1] or values[2] <= 0:
            raise JobError(f"Invalid range or step for {key}")
        if key in ("t_keep", "n_gate_jets") and any(not isinstance(v, int) for v in values):
            raise JobError(f"{key} search limits and step must be integers")
    if space["t_keep"]["low"] <= 0 or space["r_ca"]["low"] <= 0 or space["n_gate_jets"]["low"] < 0:
        raise JobError("Jet radii/thresholds must be positive and gate count nonnegative")
    if space["cos_thrust"]["low"] < 0 or space["cos_thrust"]["high"] > 1:
        raise JobError("cos_thrust must stay in [0,1]")
    if not isinstance(space["t_gate_max"], int) or space["t_gate_max"] < space["t_keep"]["high"]:
        raise JobError("t_gate_max cannot be below the largest t_keep")
    samples = campaign.get("samples", [])
    names = [sample.get("name", "") for sample in samples]
    if not names or len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", n) for n in names):
        raise JobError("Campaign needs unique safe sample names")
    if {sample.get("kind") for sample in samples} != {"signal", "background"}:
        raise JobError("Campaign must contain signal and background samples")
    all_input_files = set()
    for sample in samples:
        files = sample.get("input_files")
        if files is None and sample.get("input_file_list"):
            path = resolve_path(sample["input_file_list"], campaign_path.parent)
            sample["input_file_list"] = str(path)
            files = [line.strip() for line in path.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith("#")]
        if not isinstance(files, list) or not files or not all(isinstance(f, str) and f.strip() and "\n" not in f for f in files):
            raise JobError(f"Sample {sample['name']} needs resolved input_files or input_file_list; resolve DAS inputs first")
        if len(files) != len(set(files)):
            raise JobError(f"Duplicate input files in {sample['name']}")
        if max_files is not None:
            sample["available_input_file_count"] = len(files)
            files = files[:max_files]
        if all_input_files.intersection(files):
            raise JobError(f"MiniAOD input file appears in more than one sample: {sample['name']}")
        all_input_files.update(files)
        sample["input_files"] = files
        if sample.get("complete") is not True:
            raise JobError(f"Sample {sample['name']} must declare complete:true for its intended normalization scope")
    # Validate normalization, fixed selection/weight policies, binning, objective
    # references, and complete signal coverage before jobs can be submitted.
    from evaluate_sensitivity import validate_campaign, CORRECTION_PRESCRIPTION
    if campaign.get("correction_prescription") != CORRECTION_PRESCRIPTION:
        raise JobError("New jobs require the v3 correction_prescription; use old v1/v2 campaigns only for standalone evaluation")
    validation_campaign = copy.deepcopy(campaign)
    for sample in validation_campaign["samples"]:
        sample["files"] = [str(Path(config["run_dir"]) / "validation" / (sample["name"] + ".root"))]
    validate_campaign(validation_campaign)
    reference_policy = campaign.get("objective") or {}
    config["_reference_sha256"] = {}
    if reference_policy.get("method") == REFERENCE_METHOD:
        from freeze_sensitivity_references import _campaign_matches
        _campaign_matches(campaign, reference_policy["_reference"]["comparison_context"])
        config["_reference_sha256"][reference_policy["reference_file"]] = reference_policy["_reference_sha256"]
    # Parameters affecting meaning or execution are frozen. Trial/concurrency
    # limits and polling cadence can change when resuming a study.
    frozen = {key: config.get(key) for key in ("search_space", "ak_radii", "max_events_per_job",
               "files_per_job", "max_files_per_sample", "backend", "evaluator_argv", "cmssw_release")}
    frozen["campaign"] = campaign
    source_files = [REPO / name for name in ("optimize_sensitivity.py", "sensitivity_jobs.py",
        "evaluate_sensitivity.py", "sensitivity_metrics.py", "sensitivity_objective.py",
        "freeze_sensitivity_references.py", "plot_sensitivity_2d.py", "compact_scan_metrics.py",
        "evaluate_reference.py", "likelihood_model.py",
        "run_scripts/sensitivity/worker.py", "run_scripts/sensitivity/worker.sh")]
    for template in [config.get("evaluator_argv", [])] + [backend.get(key, []) for key in
                                                        ("submit_argv", "lookup_argv", "cancel_argv", "preflight_argv")]:
        for argument in template:
            if isinstance(argument, str) and argument.endswith((".py", ".sh")) and Path(argument).is_file():
                source_files.append(Path(argument))
    frozen["code_sha256"] = {str(path): file_digest(path) for path in source_files}
    config["_code_sha256"] = frozen["code_sha256"]
    return config, campaign, digest(frozen)


def sample_configuration(trial, space, ak_radii=(0.8,)):
    radius = trial.suggest_categorical("r_ak", list(ak_radii))
    keep = trial.suggest_int("t_keep", **space["t_keep"])
    n = trial.suggest_int("n_gate_jets", **space["n_gate_jets"])
    gate = keep if n == 0 else trial.suggest_int(
        "t_gate", keep, space["t_gate_max"], step=space["t_keep"].get("step", 1))
    ca = trial.suggest_float("r_ca", **space["r_ca"])
    thrust = trial.suggest_float("cos_thrust", **space["cos_thrust"])
    return {"n_gate_jets": n, "t_gate": gate, "t_keep": keep,
            "r_ak": radius, "r_ca": ca, "cos_thrust": thrust}


def configuration_token(parameters):
    return ":".join(format(parameters[key], ".12g") for key in
                    ("n_gate_jets", "t_gate", "t_keep", "r_ak", "r_ca", "cos_thrust"))


class Controller:
    def __init__(self, config, campaign, fingerprint, optuna_module=None):
        if optuna_module is None:
            try:
                import optuna as optuna_module
            except ImportError as exc:
                raise JobError("Install requirements-sensitivity.txt in your Python environment") from exc
        self.optuna = optuna_module
        self.config, self.campaign, self.fingerprint = config, campaign, fingerprint
        self.reference_snapshot = None
        self.root = Path(config["run_dir"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".controller.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            raise JobError(f"Another controller is active in {self.root}") from exc
        self.backend = make_backend(config["backend"], REPO)
        self.backend.state_dir = self.root
        sampler = self.optuna.samplers.TPESampler(seed=config.get("seed", 67023), constant_liar=True)
        self.storage = config.get("storage_url", f"sqlite:///{self.root / 'study.db'}")
        try:
            immutable_json(self.root / "study.identity.json", {"storage_url": self.storage,
                "study_name": config.get("study_name", "suu-sensitivity-2017")})
        except JobError:
            self.close()
            raise
        self.study = self.optuna.create_study(
            study_name=config.get("study_name", "suu-sensitivity-2017"),
            storage=self.storage,
            direction="maximize", sampler=sampler, load_if_exists=True)
        for name, value in (("campaign_fingerprint", fingerprint), ("controller_directory", str(self.root))):
            previous = self.study.user_attrs.get(name)
            if previous is not None and previous != value:
                self.close()
                raise JobError(f"Study {name} differs; use a new run directory/study for changed inputs")
            self.study.set_user_attr(name, value)
        if self.study.direction.name != "MAXIMIZE":
            self.close()
            raise JobError("Existing study has the wrong optimization direction")
        immutable_json(self.root / "campaign.source.json", campaign)
        policy = campaign.get("objective") or {}
        if policy.get("method") == "fixed_reference_regret":
            try:
                self.verify_sources()
                source = Path(policy["reference_file"])
                raw = source.read_bytes()
                if hashlib.sha256(raw).hexdigest() != policy["_reference_sha256"]:
                    raise JobError("Reference bytes changed while initializing this study")
                target = self.root / "objective.reference.json"
                immutable_text(target, raw.decode("utf-8"))
                self.reference_snapshot = (target, policy["_reference_sha256"])
            except (OSError, JobError):
                self.close()
                raise
        atomic_json(self.root / "controller.settings.json", config)
        self.preflight_done = False

    def close(self):
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()

    def directory(self, number):
        return self.root / f"trial_{number:06d}"

    def state(self, directory):
        path = directory / "state.json"
        return json.loads(path.read_text()) if path.exists() else None

    def save(self, directory, state):
        atomic_json(directory / "state.json", state)

    def verify_sources(self):
        for path, expected in self.config.get("_code_sha256", {}).items():
            if file_digest(path) != expected:
                raise JobError(f"Source changed during this study: {path}; use a new study for changed code")
        references = dict(self.config.get("_reference_sha256", {}))
        if self.reference_snapshot is not None:
            path, expected = self.reference_snapshot
            references[str(path)] = expected
        for path, expected in references.items():
            try:
                observed = file_digest(path)
            except OSError as exc:
                raise JobError(f"Frozen objective reference is unavailable: {path}") from exc
            if observed != expected:
                raise JobError(f"Frozen objective reference changed: {path}; use a new study for changed references")

    def verify_objective(self, objective):
        """Check the stationary contract before telling Optuna an objective.

Legacy custom evaluators retain their existing scalar interface. A campaign
opting into fixed references must return the complete benchmark/context record.
        """
        policy = self.campaign.get("objective") or {}
        if policy.get("method") != "fixed_reference_regret":
            return
        from sensitivity_objective import aggregate_objective, context_from_result
        try:
            expected = aggregate_objective(objective["per_signal"], self.campaign["required_signals"],
                                           policy, context_from_result(objective))
        except (KeyError, TypeError, ValueError) as exc:
            raise JobError(f"Evaluator fixed-reference context is invalid: {exc}") from exc
        for field in ("objective", "objective_name", "objective_definition", "direction", "feasible"):
            if objective.get(field) != expected[field]:
                raise JobError(f"Evaluator fixed-reference {field} differs from the frozen objective contract")

    def completed_objective(self, trial):
        directory = self.directory(trial.number)
        path = directory / "evaluation/objective.json"
        receipt = json.loads((directory / "evaluation/completion.json").read_text())
        if (receipt.get("objective_sha256") != file_digest(path) or
                receipt.get("fingerprint") != self.fingerprint or
                receipt.get("trial_token") != f"eo_{self.fingerprint[:20]}_{trial.number}"):
            raise JobError(f"Stored objective provenance is invalid: {path}")
        objective = json.loads(path.read_text())
        if objective.get("objective") != trial.value or objective.get("feasible") != trial.user_attrs.get("feasible"):
            raise JobError(f"Stored objective differs from Optuna's recorded result: {path}")
        self.verify_objective(objective)
        return objective

    def prepare(self, frozen_trial):
        trial = self.optuna.trial.Trial(self.study, frozen_trial._trial_id)
        parameters = sample_configuration(trial, self.config["search_space"], self.config["ak_radii"])
        directory = self.directory(trial.number)
        directory.mkdir(exist_ok=True)
        token = f"eo_{self.fingerprint[:20]}_{trial.number}"
        tasks = []
        evaluated_campaign = copy.deepcopy(self.campaign)
        if self.reference_snapshot is not None:
            evaluated_campaign["objective"]["reference_file"] = str(self.reference_snapshot[0])
        for sample in evaluated_campaign["samples"]:
            sample["files"] = []
            files = sample["input_files"]
            for begin in range(0, len(files), self.config["files_per_job"]):
                index = len(tasks)
                task_dir = directory / "tasks" / str(index)
                task_dir.mkdir(parents=True, exist_ok=True)
                output = str(task_dir / "output.root")
                task = {"index": index, "sample": sample["name"], "sample_kind": sample["kind"],
                        "input_files": files[begin:begin + self.config["files_per_job"]],
                        "output": output, "parameters": parameters, "trial_token": token,
                        "max_events": self.config["max_events_per_job"],
                        "cmssw_release": self.config["cmssw_release"],
                        "bundle_name": Path(self.config["backend"].get("cmssw_bundle", "cmssw.tar.gz")).name,
                        "cmsrun_timeout_seconds": self.config.get("cmsrun_timeout_seconds", 43200)}
                immutable_json(task_dir / "task.json", task)
                immutable_text(task_dir / "inputs.txt", "\n".join(task["input_files"]) + "\n")
                tasks.append(task)
                sample["files"].append(output)
        descriptor = {"trial_number": trial.number, "token": token, "fingerprint": self.fingerprint,
                      "parameters": parameters, "configuration": configuration_token(parameters),
                      "tasks": tasks}
        immutable_json(directory / "trial.json", descriptor)
        immutable_json(directory / "campaign.json", evaluated_campaign)
        self.backend.prepare(directory, descriptor)
        trial.set_user_attr("trial_directory", str(directory))
        if self.state(directory) is None:
            self.save(directory, {"phase": "prepared", "created_at": time.time(), "job_ids": []})
        return descriptor

    def finish(self, number, directory, state, objective=None, reason=None):
        trial = self.study.trials[number]
        if trial.state == self.optuna.trial.TrialState.RUNNING:
            live = self.optuna.trial.Trial(self.study, trial._trial_id)
            if objective is not None:
                live.set_user_attr("feasible", objective["feasible"])
                live.set_user_attr("objective_path", str(directory / "evaluation/objective.json"))
                self.study.tell(number, objective["objective"])
            else:
                live.set_user_attr("failure_reason", str(reason))
                self.study.tell(number, state=self.optuna.trial.TrialState.FAIL)
        state.update(phase="complete" if objective is not None else "failed", finished_at=time.time())
        if reason:
            state["reason"] = str(reason)
        if objective is not None:
            state.update(objective=objective["objective"], feasible=objective["feasible"])
        self.save(directory, state)

    def evaluate(self, number, directory, descriptor, state):
        self.verify_sources()
        state["phase"] = "evaluating"
        self.save(directory, state)
        for task in descriptor["tasks"]:
            path = Path(task["output"])
            if not path.is_file() or path.stat().st_size == 0:
                raise JobError(f"Successful scheduler job has missing/empty output: {path}")
            result_path = path.parent / "worker_result.json"
            if self.config["backend"].get("kind", "condor") == "condor":
                if not result_path.exists():
                    raise JobError(f"Missing worker completion receipt: {result_path}")
                receipt = json.loads(result_path.read_text())
                if (receipt.get("success") is not True or receipt.get("trial_token") != descriptor["token"]
                        or receipt.get("sample") != task["sample"] or receipt.get("index") != task["index"]):
                    raise JobError(f"Invalid worker completion receipt: {result_path}")
        evaluation_root = directory / "evaluation"
        evaluation_root.mkdir(exist_ok=True)
        attempt = int(state.get("evaluation_attempt", 0)) + 1
        while (evaluation_root / f"attempt_{attempt:04d}").exists():
            attempt += 1
        output = evaluation_root / f"attempt_{attempt:04d}"
        output.mkdir()
        state["evaluation_attempt"] = attempt
        self.save(directory, state)
        objective_path = output / "objective.json"
        template = self.config.get("evaluator_argv", ["{python}", "{repo}/evaluate_sensitivity.py",
            "--campaign", "{campaign}", "--configuration", "{configuration}", "--output-dir", "{output_dir}"])
        argv = expand_argv(template, {"python": sys.executable, "repo": str(REPO),
            "campaign": str(directory / "campaign.json"), "configuration": descriptor["configuration"],
            "output_dir": str(output), "trial_dir": str(directory)})
        log = run_command(argv, self.config.get("evaluation_timeout_seconds", 3600), str(REPO))
        (output / "stdout.log").write_text(log)
        try:
            objective = json.loads(objective_path.read_text())
        except (OSError, ValueError) as exc:
            raise JobError(f"Evaluator did not produce valid {objective_path}") from exc
        value = objective.get("objective")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise JobError("Evaluator objective must be a finite number")
        if not isinstance(objective.get("feasible"), bool):
            raise JobError("Evaluator must declare feasible as a boolean")
        from evaluate_sensitivity import parse_configuration
        if ("configuration" not in objective or
                parse_configuration(objective["configuration"]) != parse_configuration(descriptor["configuration"])):
            raise JobError("Evaluator returned a different configuration")
        if (objective["feasible"] and value < 0) or (not objective["feasible"] and value != -1):
            raise JobError("Evaluator must return nonnegative sensitivity when feasible, -1 when infeasible")
        self.verify_sources()
        self.verify_objective(objective)
        immutable_json(evaluation_root / "objective.json", objective)
        immutable_json(evaluation_root / "completion.json", {"trial_token": descriptor["token"],
            "fingerprint": self.fingerprint, "objective_sha256": file_digest(evaluation_root / "objective.json")})
        self.finish(number, directory, state, objective=objective)

    def advance(self, frozen_trial):
        number = frozen_trial.number
        directory = self.directory(number)
        descriptor = self.prepare(frozen_trial)
        state = self.state(directory)
        if state["phase"] in FINAL_PHASES:
            # State persisted but tell interrupted: replay the durable result.
            if state["phase"] == "complete":
                self.evaluate(number, directory, descriptor, state)
            else:
                self.finish(number, directory, state, reason=state.get("reason", "Previous failure"))
            return
        if state["phase"] == "prepared":
            self.verify_sources()
            for task in descriptor["tasks"]:
                if Path(task["output"]).exists():
                    raise JobError(f"Refusing to submit over an existing task output: {task['output']}")
            state.update(phase="submitting", submitted_at=time.time(), poll_errors=0)
            self.save(directory, state)
            try:
                state["job_ids"] = self.backend.submit(directory, descriptor, state)
                state["phase"] = "running"
            except JobError as exc:
                state["submit_error"] = str(exc)
            self.save(directory, state)
            return
        if state["phase"] == "evaluating":
            try:
                self.evaluate(number, directory, descriptor, state)
            except (JobError, ValueError) as exc:
                self.finish(number, directory, state, reason=exc)
            return
        try:
            status = self.backend.lookup(directory, descriptor, state)
            state["poll_errors"] = 0
        except JobError as exc:
            state["poll_errors"] = state.get("poll_errors", 0) + 1
            state["last_poll_error"] = str(exc)
            self.save(directory, state)
            if state["poll_errors"] >= self.config["max_poll_errors"]:
                raise JobError("Scheduler unavailable after bounded retries; controller stopped, jobs preserved for resume") from exc
            return
        if status.get("job_ids"):
            state["job_ids"] = status["job_ids"]
        reason = None
        age = time.time() - state["submitted_at"]
        if status["state"] == "complete":
            try:
                self.evaluate(number, directory, descriptor, state)
            except (JobError, ValueError) as exc:
                self.finish(number, directory, state, reason=exc)
            return
        if status["state"] == "failed":
            reason = status.get("reason", "Scheduler reported failed jobs")
        elif age > self.config["trial_timeout_seconds"]:
            reason = "Trial exceeded configured timeout"
        elif status["state"] == "missing":
            state.setdefault("missing_since", time.time())
            if time.time() - state["missing_since"] > self.config["submission_visibility_seconds"]:
                reason = "Trial absent from scheduler queue and history; ambiguous submissions are never repeated"
        else:
            state.pop("missing_since", None)
            state["phase"] = "running"
        if reason:
            # Cancellation must succeed before new work replaces this trial.
            # The unique token removes all still-live tasks, including siblings.
            self.backend.cancel(directory, descriptor, state)
            self.finish(number, directory, state, reason=reason)
        else:
            self.save(directory, state)

    def summary(self):
        trials = self.study.trials
        feasible = [t for t in trials if t.state.name == "COMPLETE" and t.user_attrs.get("feasible")]
        best = max(feasible, key=lambda t: t.value) if feasible else None
        result = {"study_name": self.study.study_name, "trials": len(trials),
                  "states": {name: sum(t.state.name == name for t in trials) for name in ("RUNNING", "COMPLETE", "FAIL")},
                  "best_feasible": None if best is None else {"trial_number": best.number, "objective": best.value,
                                                               "parameters": best.params}}
        from sensitivity_objective import relative_regret_rankings
        objectives = [self.completed_objective(t) for t in trials if t.state.name == "COMPLETE"]
        mean_weight = (self.campaign.get("objective") or {}).get("mean_weight", 0.25)
        atomic_json(self.root / "study_rankings.json", {"required_signals": self.campaign["required_signals"],
            "ranking_basis": "retrospective_worst_plus_mean_relative_sensitivity_regret",
            "rankings": relative_regret_rankings(objectives, self.campaign["required_signals"], mean_weight)})
        atomic_json(self.root / "study_summary.json", result)
        return result

    def tick(self, execute=False):
        self.verify_sources()
        if execute and not self.preflight_done:
            self.backend.preflight()
            if self.config["backend"].get("kind", "condor") == "condor":
                archive = Path(self.config["backend"]["cmssw_bundle"])
                immutable_json(self.root / "bundle.json", {"name": archive.name,
                    "sha256": file_digest(archive), "bytes": archive.stat().st_size})
            self.preflight_done = True
        running_state = self.optuna.trial.TrialState.RUNNING
        # Recover the narrow crash window after tell() committed but before
        # state.json was updated. The Optuna database is authoritative here.
        for trial in self.study.trials:
            if trial.state.name not in {"COMPLETE", "FAIL"}:
                continue
            directory = self.directory(trial.number)
            state = self.state(directory)
            if state and state["phase"] not in FINAL_PHASES:
                state.update(phase="complete" if trial.state.name == "COMPLETE" else "failed",
                             objective=trial.value, feasible=trial.user_attrs.get("feasible", False))
                self.save(directory, state)
        for trial in self.study.get_trials(states=(running_state,)):
            if execute:
                self.advance(trial)
            else:
                self.prepare(trial)
        while (len(self.study.trials) < self.config["n_trials"] and
               len(self.study.get_trials(states=(running_state,))) < self.config["max_parallel_trials"]):
            # Seed each ask by trial number. Restarting --once must not replay
            # the initial random suggestions while TPE is warming up.
            self.study = self.optuna.load_study(study_name=self.study.study_name, storage=self.storage,
                sampler=self.optuna.samplers.TPESampler(
                    seed=self.config.get("seed", 67023) + len(self.study.trials), constant_liar=True))
            trial = self.study.ask()
            frozen = self.study.trials[trial.number]
            self.prepare(frozen)
            if execute:
                self.advance(frozen)
        return self.summary()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path, help="Override run directory (prefer local disk for SQLite)")
    parser.add_argument("--execute", action="store_true", help="Authorize automatic submissions and job management")
    parser.add_argument("--once", action="store_true", help="Perform one scheduling/reconciliation pass and exit")
    args = parser.parse_args(argv)
    controller = None
    try:
        config, campaign, fingerprint = load_settings(args.config, args.run_dir)
        controller = Controller(config, campaign, fingerprint)
        while True:
            summary = controller.tick(args.execute)
            print(json.dumps({"mode": "execute" if args.execute else "plan", **summary}), flush=True)
            if not args.execute or args.once or (summary["trials"] >= config["n_trials"] and summary["states"]["RUNNING"] == 0):
                break
            time.sleep(min(float(config["poll_seconds"]), 60))
    except KeyboardInterrupt:
        print("Controller stopped; submitted jobs remain tracked. Resume with the same command.", file=sys.stderr)
        return 130
    except (JobError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if controller is not None:
            controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
