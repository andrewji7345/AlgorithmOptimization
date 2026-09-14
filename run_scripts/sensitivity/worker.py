#!/usr/bin/env python3
"""Transferred worker helper; cmsRun runs only in the initialized CMSSW runtime."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import math


def load_task():
    return json.loads(Path("task.json").read_text())


def receipt(task, success, message=""):
    path = Path("worker_result.json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"success": success, "trial_token": task["trial_token"],
                                    "sample": task["sample"], "index": task["index"],
                                    "message": message}, indent=2) + "\n")
    temporary.replace(path)


def prepare(task):
    release = task["cmssw_release"]
    if Path(release).exists():
        raise RuntimeError(f"Refusing to reuse an existing release directory: {release}")
    provenance = json.loads(Path("bundle.json").read_text())
    checksum = hashlib.sha256()
    with Path(task["bundle_name"]).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    if provenance["name"] != task["bundle_name"] or checksum.hexdigest() != provenance["sha256"]:
        raise RuntimeError("CMSSW bundle differs from the controller's frozen archive")
    # Bundles are locally built, trusted CMSSW releases. Preserve their CVMFS
    # symlinks, while rejecting archive member paths outside the release.
    with tarfile.open(task["bundle_name"], "r:gz") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if not parts or parts[0] != release or ".." in parts:
                raise RuntimeError(f"Unexpected archive member: {member.name}")
        archive.extractall(".")
    cfg = Path(release) / "src/SuuAnalysis/ExistingOptimization/test/runSensitivityScan_cfg.py"
    if not cfg.is_file():
        raise RuntimeError(f"Sensitivity cfg missing from CMSSW bundle: {cfg}")
    Path("release_name.txt").write_text(release)


def scan_grid_arguments(grid):
    options = {"collection_pt_cuts": "collectionPtCuts", "ca_radii": "caRadii",
               "cos_thrust_cuts": "cosThrustCuts", "gate_counts": "defaultGateJetCounts",
               "gate_pt_cuts": "defaultGatePtCuts"}
    if not isinstance(grid, dict) or set(grid) != set(options):
        raise RuntimeError("scan_grid must explicitly declare all five grid axes")
    for name, option in options.items():
        values = grid[name]
        if (not isinstance(values, list) or not values or len(values) > 32 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                    not math.isfinite(v) for v in values) or len(set(values)) != len(values)):
            raise RuntimeError(f"Invalid scan_grid axis {name}")
        if name == "gate_counts":
            if any(not isinstance(v, int) or v < 0 for v in values):
                raise RuntimeError("Gate counts must be nonnegative integers")
        elif name == "cos_thrust_cuts":
            if any(v < 0 or v > 1 for v in values):
                raise RuntimeError("Thrust cuts must lie in [0,1]")
        elif any(v <= 0 for v in values):
            raise RuntimeError(f"Grid axis {name} must be positive")
    size = math.prod(len(grid[name]) for name in ("collection_pt_cuts", "ca_radii", "cos_thrust_cuts"))
    if size > 128:
        raise RuntimeError("Reference batch supports at most 128 reconstructions per task")
    return {option: ",".join(map(str, grid[name])) for name, option in options.items()}


def run(task):
    output = Path("output.root")
    if output.exists():
        raise RuntimeError("Refusing to overwrite output.root")
    package = Path(os.environ["CMSSW_BASE"]) / "src/SuuAnalysis/ExistingOptimization"
    params = task["parameters"]
    if params.get("r_ak") not in (0.4, 0.8):
        raise RuntimeError("Worker requires a calibrated r_ak of 0.4 or 0.8")
    counts = sorted({0, int(params["n_gate_jets"])})
    arguments = ["cmsRun", str(package / "test/runSensitivityScan_cfg.py"),
                 "inputRootFiles=" + str(Path("inputs.txt").resolve()),
                 "outputRootFile=" + str(output.resolve()),
                 "sampleName=" + task["sample"], "sampleKind=" + task["sample_kind"],
                 "maxEvents=" + str(task["max_events"]),
                 "akRadius=" + str(params["r_ak"]), "collectionPtCuts=" + str(params["t_keep"]),
                 "caRadii=" + str(params["r_ca"]), "cosThrustCuts=" + str(params["cos_thrust"]),
                 "defaultGateJetCounts=" + ",".join(map(str, counts)),
                 "defaultGatePtCuts=" + str(max(params["t_gate"], params["t_keep"])),
                 "enforceLegacyRadiusConstraint=False"]
    if "scan_grid" in task:
        for option, value in scan_grid_arguments(task["scan_grid"]).items():
            arguments = [arg for arg in arguments if not arg.startswith(option + "=")]
            arguments.append(option + "=" + value)
    if "analysis_systematic" in task:
        variation = task["analysis_systematic"]
        if variation not in ("nominal", "JECUp", "JECDown", "JERUp", "JERDown"):
            raise RuntimeError("Unknown analysis_systematic")
        arguments.append("analysisSystematic=" + variation)
    subprocess.run(arguments, check=True, timeout=task["cmsrun_timeout_seconds"])
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("cmsRun did not produce a nonempty output.root")
    subprocess.run([sys.executable, str(package / "test/validate_compact_scan.py"),
                    "--strict-branches", str(output)], check=True, timeout=600)
    receipt(task, True)


def main():
    task = load_task()
    try:
        if sys.argv[1] == "prepare":
            prepare(task)
        elif sys.argv[1] == "run":
            run(task)
        elif sys.argv[1] == "failure":
            receipt(task, False, "Worker setup or cmsRun failed; inspect condor.err")
            # Condor can transfer a failure receipt even when cmsRun never
            # created its requested output. Exit status still reports failure.
            Path("output.root").touch(exist_ok=True)
        else:
            raise RuntimeError("Expected prepare, run, or failure")
    except Exception as exc:
        receipt(task, False, str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
