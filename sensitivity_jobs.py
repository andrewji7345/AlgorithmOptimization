"""Bounded, shell-free terminal adapters for sensitivity trial jobs.

The scheduler token is persisted *before* submission. A submit timeout is
ambiguous, so recovery queries that token and never blindly submits it again.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


class JobError(RuntimeError):
    pass


def run_command(argv, timeout=60, cwd=None, env=None):
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise JobError("Commands must be nonempty JSON arrays of string arguments")
    try:
        result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True,
                                timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise JobError(f"Command failed: {argv[0]}: {exc}") from exc
    if result.returncode:
        raise JobError(f"{argv[0]} exited {result.returncode}: {result.stderr[-4000:]}")
    return result.stdout


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise JobError(f"Refusing to change immutable trial input: {path}")
    else:
        atomic_json(path, value)


def immutable_text(path, value):
    path = Path(path)
    if path.exists() and path.read_text() != value:
        raise JobError(f"Refusing to change immutable trial input: {path}")
    if not path.exists():
        path.write_text(value, encoding="utf-8")


def expand_argv(template, values):
    if not isinstance(template, list) or not template:
        raise JobError("Command templates must be nonempty argv arrays")
    try:
        return [part.format_map(values) for part in template]
    except (KeyError, ValueError, AttributeError) as exc:
        raise JobError(f"Invalid command placeholder: {exc}") from exc


def safe_condor_path(path):
    text = str(Path(path).absolute())
    if any(char.isspace() or char in '\"\'$,;\\' for char in text):
        raise JobError(f"Condor paths cannot contain whitespace or submit-language delimiters: {text}")
    return text


class CommandBackend:
    """Adapter to a site's own scheduler wrapper; see docs/optuna_workflow.md."""
    def __init__(self, config, repo):
        self.config = config
        self.repo = str(repo)
        self.timeout = float(config.get("command_timeout_seconds", 60))

    def values(self, trial_dir, descriptor, state):
        return {"trial_dir": str(trial_dir), "trial_json": str(trial_dir / "trial.json"),
                "token": descriptor["token"], "job_id": ",".join(state.get("job_ids", [])),
                "repo": self.repo}

    def preflight(self):
        if "preflight_argv" in self.config:
            run_command(self.config["preflight_argv"], self.timeout)
        for field in ("submit_argv", "lookup_argv", "cancel_argv"):
            if field not in self.config:
                raise JobError(f"command backend requires {field}")

    def prepare(self, trial_dir, descriptor):
        pass

    def invoke(self, field, trial_dir, descriptor, state):
        argv = expand_argv(self.config[field], self.values(trial_dir, descriptor, state))
        raw = run_command(argv, self.timeout, str(trial_dir))
        try:
            response = json.loads(raw)
        except ValueError as exc:
            raise JobError(f"{field} must return one JSON object") from exc
        if not isinstance(response, dict):
            raise JobError(f"{field} must return a JSON object")
        return response

    def submit(self, trial_dir, descriptor, state):
        response = self.invoke("submit_argv", trial_dir, descriptor, state)
        ids = response.get("job_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and x for x in ids):
            raise JobError("submit_argv must return nonempty job_ids:[strings]")
        return ids

    def lookup(self, trial_dir, descriptor, state):
        response = self.invoke("lookup_argv", trial_dir, descriptor, state)
        if response.get("state") not in {"running", "complete", "failed", "missing"}:
            raise JobError("lookup_argv returned an invalid state")
        if "job_ids" in response and (not isinstance(response["job_ids"], list) or
                                     not all(isinstance(x, str) for x in response["job_ids"])):
            raise JobError("lookup_argv job_ids must be strings")
        return response

    def cancel(self, trial_dir, descriptor, state):
        self.invoke("cancel_argv", trial_dir, descriptor, state)


class CondorBackend(CommandBackend):
    def __init__(self, config, repo):
        super().__init__(config, repo)
        self.schedd = None
        self.commands = {}
        for name in ("condor_submit", "condor_q", "condor_history", "condor_rm", "condor_status"):
            executable = shutil.which(name) or name
            # CMS LPC's wrappers have no shebang, choose different schedds on
            # each submission, and history/rm require -name. Their installed
            # underlying binaries support a fixed, persistent target directly.
            original = Path("/usr/bin") / ("original_" + name)
            native = Path("/usr/libexec/condor") / name
            if Path(executable).parent == Path("/usr/local/bin"):
                # condor_rm dispatches on argv[0]: calling original_condor_rm
                # directly fails even though the correctly named symlink works.
                if native.is_file():
                    executable = str(native)
                elif original.is_file():
                    executable = str(original)
            self.commands[name] = config.get(name + "_executable", executable)

    def target_args(self):
        return ["-name", self.schedd] if self.schedd else []

    def run_condor(self, argv, cwd=None):
        # As in the LPC wrappers, keep system Condor/proxy binaries and any
        # configured system-Python helpers independent of CMSSW's libraries.
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = ""
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        return run_command(argv, self.timeout, cwd, env=environment)

    def resolve_schedd(self):
        requested = self.config.get("schedd")
        record = Path(self.state_dir) / "scheduler.json" if hasattr(self, "state_dir") else None
        if record is not None and record.exists():
            saved = json.loads(record.read_text())
            self.schedd = saved["schedd"]
            if requested not in (None, "auto", self.schedd):
                raise JobError("Configured schedd differs from this study's persisted scheduler")
            return
        if requested == "auto":
            constraint = self.config.get("schedd_constraint",
                'FERMIHTC_DRAIN_LPCSCHEDD =?= FALSE && FERMIHTC_SCHEDD_TYPE =?= "CMSLPC"')
            raw = self.run_condor([self.commands["condor_status"], "-schedd", "-json", "-attributes",
                "Name,MaxJobsRunning,ShadowsRunning,TotalIdleJobs,RecentDaemonCoreDutyCycle",
                "-constraint", constraint])
            try:
                ads = json.loads(raw)
                candidates = [ad for ad in ads if isinstance(ad.get("Name"), str) and ad["Name"]]
                if not candidates:
                    raise ValueError("no matching available schedds")
                def load(ad):
                    capacity = max(float(ad.get("MaxJobsRunning", 1)), 1)
                    return (70 * float(ad.get("RecentDaemonCoreDutyCycle", 0)) +
                            20 * float(ad.get("ShadowsRunning", 0)) / capacity +
                            0.1 * float(ad.get("TotalIdleJobs", 0)), ad["Name"])
                self.schedd = min(candidates, key=load)["Name"]
            except (ValueError, TypeError, KeyError) as exc:
                raise JobError(f"Cannot select a schedd: {exc}") from exc
        elif requested:
            self.schedd = str(requested)
        if self.schedd and (any(c.isspace() for c in self.schedd) or self.schedd.startswith("-")):
            raise JobError("Invalid schedd name")
        if record is not None:
            immutable_json(record, {"schedd": self.schedd})

    def preflight(self):
        for executable in list(self.commands.values()) + ["voms-proxy-info"]:
            if shutil.which(executable) is None:
                raise JobError(f"Missing required executable: {executable}")
        archive = Path(self.config["cmssw_bundle"])
        if not archive.is_file() or archive.stat().st_size == 0:
            raise JobError(f"Build the CMSSW archive first: {archive}")
        raw = self.run_condor(["voms-proxy-info", "-timeleft"])
        try:
            seconds = int(raw.strip())
        except ValueError as exc:
            raise JobError("Cannot determine X.509 proxy lifetime") from exc
        if seconds < int(self.config.get("minimum_proxy_seconds", 3600)):
            raise JobError("X.509 proxy is missing or too close to expiry; renew it before --execute")
        self.resolve_schedd()

    def prepare(self, trial_dir, descriptor):
        base = safe_condor_path(trial_dir)
        scripts = safe_condor_path(Path(self.repo) / "run_scripts/sensitivity")
        archive = safe_condor_path(self.config["cmssw_bundle"])
        cpu = int(self.config.get("request_cpus", 1))
        memory = int(self.config.get("request_memory_mb", 4000))
        disk = int(self.config.get("request_disk_mb", 20000))
        if min(cpu, memory, disk) <= 0:
            raise JobError("Condor resource requests must be positive")
        text = f'''universe = vanilla
executable = /bin/bash
transfer_executable = False
arguments = "worker.sh"
initialdir = {base}/tasks/$(task_index)
transfer_input_files = {scripts}/worker.sh,{scripts}/worker.py,{base}/tasks/$(task_index)/task.json,{base}/tasks/$(task_index)/inputs.txt,{trial_dir.parent}/bundle.json,{archive}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_output_files = output.root,worker_result.json
output = {base}/tasks/$(task_index)/condor.out
error = {base}/tasks/$(task_index)/condor.err
log = {base}/condor.log
use_x509userproxy = True
request_cpus = {cpu}
request_memory = {memory}MB
request_disk = {disk}MB
notification = Never
+ExistingOptimizationToken = "{descriptor['token']}"
+ExistingOptimizationTask = $(task_index)
queue task_index from (
'''
        text += "\n".join(str(task["index"]) for task in descriptor["tasks"]) + "\n)\n"
        immutable_text(trial_dir / "submit.jdl", text)

    def submit(self, trial_dir, descriptor, state):
        # Do not automatically retry this command: an interrupted response may
        # still correspond to a committed submission on the schedd.
        raw = self.run_condor([self.commands["condor_submit"], *self.target_args(), "-terse",
                               str(trial_dir / "submit.jdl")], str(trial_dir))
        match = re.fullmatch(r"\s*(\d+)\.(\d+)(?:\s*-\s*(\d+)\.(\d+))?\s*", raw)
        if not match:
            raise JobError(f"Ambiguous condor_submit response: {raw[-1000:]}")
        cluster, start = int(match[1]), int(match[2])
        last_cluster, last = int(match[3] or cluster), int(match[4] or start)
        if last_cluster != cluster or last - start + 1 != len(descriptor["tasks"]):
            raise JobError("Submitted cluster does not match expected task count")
        return [f"{cluster}.{i}" for i in range(start, last + 1)]

    def constraint(self, descriptor):
        token = descriptor["token"]
        if re.fullmatch(r"eo_[a-f0-9]+_\d+", token) is None:
            raise JobError("Invalid trial token")
        return f'ExistingOptimizationToken == "{token}"'

    def lookup(self, trial_dir, descriptor, state):
        constraint = self.constraint(descriptor)
        attributes = "ClusterId,ProcId,JobStatus,ExitCode,ExitBySignal,HoldReason,ExistingOptimizationTask"
        ads = {}
        # Queue and history are both needed to reconcile a lost submit response
        # and the queue-to-history transition after a worker completes.
        for command in ("condor_history", "condor_q"):
            raw = self.run_condor([self.commands[command], *self.target_args(), "-constraint", constraint,
                                   "-json", "-attributes", attributes])
            try:
                # LPC Condor 25 emits no text for a successful empty -json query.
                # run_condor already rejects nonzero exits and transport errors.
                items = json.loads(raw) if raw.strip() else []
                if not isinstance(items, list):
                    raise ValueError("expected array")
                for ad in items:
                    ads[(int(ad["ClusterId"]), int(ad["ProcId"]))] = ad
            except (KeyError, TypeError, ValueError) as exc:
                raise JobError(f"Malformed {command} JSON: {exc}") from exc
        if not ads:
            return {"state": "missing", "job_ids": []}
        ids = [f"{cluster}.{proc}" for cluster, proc in sorted(ads)]
        task_indices = [ad.get("ExistingOptimizationTask") for ad in ads.values()]
        if len(set(task_indices)) != len(task_indices) or len(ads) > len(descriptor["tasks"]):
            return {"state": "failed", "job_ids": ids, "reason": "Duplicate jobs found for trial token"}
        for ad in ads.values():
            status = int(ad.get("JobStatus", 0))
            if status in (3, 5) or (status == 4 and (ad.get("ExitBySignal", False) or ad.get("ExitCode") != 0)):
                return {"state": "failed", "job_ids": ids,
                        "reason": ad.get("HoldReason", f"Job failed with status {status}, exit {ad.get('ExitCode')}")}
        if (len(ads) == len(descriptor["tasks"]) and
                set(task_indices) == {t["index"] for t in descriptor["tasks"]} and
                all(int(ad.get("JobStatus", 0)) == 4 for ad in ads.values())):
            return {"state": "complete", "job_ids": ids}
        return {"state": "running", "job_ids": ids}

    def cancel(self, trial_dir, descriptor, state):
        constraint = self.constraint(descriptor)

        def live_jobs():
            raw = self.run_condor([self.commands["condor_q"], *self.target_args(),
                                  "-constraint", constraint, "-json", "-attributes", "JobStatus"])
            try:
                ads = json.loads(raw) if raw.strip() else []
                if not isinstance(ads, list):
                    raise ValueError("expected array")
                return any(int(ad["JobStatus"]) != 4 for ad in ads)
            except (KeyError, TypeError, ValueError) as exc:
                raise JobError(f"Malformed cancellation queue JSON: {exc}") from exc

        # condor_rm returns nonzero when all matching jobs already left the
        # queue. This is normal for a completed failure discovered in history.
        if not live_jobs():
            return
        try:
            self.run_condor([self.commands["condor_rm"], *self.target_args(), "-constraint", constraint])
        except JobError:
            # A job may finish between the query and removal. Only accept the
            # race after another successful query proves no live siblings remain.
            if live_jobs():
                raise


def make_backend(config, repo):
    kind = config.get("kind", "condor")
    if kind == "condor":
        return CondorBackend(config, repo)
    if kind == "command":
        return CommandBackend(config, repo)
    raise JobError(f"Unknown backend: {kind}")
