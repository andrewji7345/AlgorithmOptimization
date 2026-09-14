#!/usr/bin/env python3
"""Run blind, pre-fit expected Combine calculations with auditable receipts.

Only the standard single signal-strength (r) model is supported. No observed
limit or observed significance is computed. Runtime wrappers receive argv, not
shell text, and must exec the requested program in a prepared Combine runtime.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time


class CombineRunError(RuntimeError):
    """A failed command or invalid result, with its durable failure receipt."""

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_tree(path, tree_name, required):
    import uproot
    try:
        # These are local artifacts. Explicit mmap avoids asynchronous fsspec
        # range-read hangs seen with CMSSW uproot 5.3.3 on FitDiagnostics trees.
        with uproot.open(path, handler=uproot.source.file.MemmapSource) as source:
            tree = source[tree_name]
            missing = set(required) - set(tree.keys())
            if missing:
                raise CombineRunError(f"{path}: missing branches {sorted(missing)}")
            return {name: tree[name].array(library="np").tolist() for name in required}
    except CombineRunError:
        raise
    except Exception as exc:
        raise CombineRunError(f"Cannot read {tree_name} in {path}: {exc}") from exc


def parse_expected_limits(path):
    """Require exactly five positive, ordered expected quantiles and no data."""
    rows = _read_tree(path, "limit", ("limit", "quantileExpected"))
    quantiles = (0.025, 0.16, 0.5, 0.84, 0.975)
    if len(rows["limit"]) != 5:
        raise CombineRunError("Blind AsymptoticLimits must return exactly five expected rows")
    values = {}
    for value, quantile in zip(rows["limit"], rows["quantileExpected"]):
        if not math.isfinite(value) or value <= 0 or not math.isfinite(quantile):
            raise CombineRunError("Non-finite or non-positive expected limit")
        matches = [q for q in quantiles if abs(quantile - q) < 1e-5]
        if len(matches) != 1 or str(matches[0]) in values:
            raise CombineRunError(f"Unexpected, observed, or duplicate limit quantile: {quantile}")
        values[str(matches[0])] = float(value)
    ordered = [values[str(q)] for q in quantiles]
    if any(a > b for a, b in zip(ordered, ordered[1:])):
        raise CombineRunError("Expected-limit quantiles are not ordered")
    return {"median": values["0.5"], "quantiles": values, "confidence_level": 0.95,
            "units": "signal_strength_r", "observed_computed": False}


def parse_expected_significance(path):
    rows = _read_tree(path, "limit", ("limit",))
    if len(rows["limit"]) != 1:
        raise CombineRunError("Asimov significance must contain exactly one result")
    value = float(rows["limit"][0])
    if not math.isfinite(value) or value < 0:
        raise CombineRunError("Invalid expected discovery significance")
    return value


def significance_fit_policy(signal_strength=1.0):
    """Freeze precision for all discovery fits, including very small Z.

    Strategy 2 and tolerance 1e-5 resolve weak-signal Asimov likelihood
    differences that default precision can incorrectly report as zero.
    Keeping the range near the injection also protects that resolution.
    """
    injection = float(signal_strength)
    r_max = max(20.0, 5.0 * injection)
    if not math.isfinite(injection) or injection <= 0 or not math.isfinite(r_max):
        raise ValueError("Significance injection and derived range must be positive and finite")
    return {"schema_version": 1, "name": "strategy_2_tolerance_1e-5",
            "minimizer_strategy": 2, "minimizer_tolerance": 1e-5,
            "r_min": 0.0, "r_max": r_max}


def diagnostic_range_policy(expected_limit_975, signal_strength=1.0):
    """Choose an independent diagnostic range from a validated blind limit.

    Discovery retains its narrow range to avoid loss of small-significance
    resolution. FitDiagnostics must instead contain the upper 68% MINOS
    endpoint, which can be far above the injected strength for weak signals.
    Twice the upper 95%-band expected limit is deterministic padding, not a
    guarantee of convergence; the resulting interval is checked separately.
    """
    quantile = float(expected_limit_975)
    injection = float(signal_strength)
    if not all(math.isfinite(value) and value > 0 for value in (quantile, injection)):
        raise ValueError("Diagnostic range inputs must be positive and finite")
    discovery_r_max = significance_fit_policy(injection)["r_max"]
    diagnostic_r_max = max(discovery_r_max, 2.0 * quantile)
    if not math.isfinite(diagnostic_r_max):
        raise ValueError("Derived diagnostic range must be finite")
    return {"schema_version": 1,
            "name": "twice_expected_limit_97p5_with_discovery_floor",
            "expected_limit_quantile": "0.975",
            "expected_limit_quantile_value": quantile,
            "expected_limit_multiplier": 2.0,
            "discovery_r_max": discovery_r_max,
            "r_min": 0.0, "r_max": diagnostic_r_max,
            "upper_interval_fraction_max": 0.99}


def validate_fit_diagnostics_interval(fit, r_max):
    """Require a positive MINOS upper error and 1% range headroom.

    Combine v11 FitDiagnostics.cc substitutes the parameter boundary when its
    MINOS upper error is nearly zero. Such a clipped endpoint can coexist with
    fit_status=0 and covariance quality=3, so symmetric rErr is insufficient.
    The physical lower boundary at r=0 is allowed to truncate the lower error.
    """
    r_max = float(r_max)
    if not math.isfinite(r_max) or r_max <= 0:
        raise ValueError("Diagnostic r_max must be positive and finite")
    try:
        estimate, upper_error = float(fit["r"]), float(fit["rHiErr"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CombineRunError("Asimov fit is missing a numeric MINOS upper error") from exc
    if not all(math.isfinite(value) for value in (estimate, upper_error)) or upper_error <= 0:
        raise CombineRunError("Asimov fit has invalid MINOS upper uncertainty")
    endpoint = estimate + upper_error
    if not math.isfinite(endpoint) or endpoint >= 0.99 * r_max:
        raise CombineRunError("Asimov diagnostic MINOS upper interval approaches r_max; "
                              "increase the diagnostic range and rerun")
    return {"r_min": 0.0, "r_max": r_max,
            "upper_endpoint": endpoint,
            "upper_endpoint_fraction": endpoint / r_max,
            "upper_interval_fraction_max": 0.99}


def parse_fit_diagnostics(path, signal_strength=1.0, r_max=None):
    # An omitted range preserves legacy receipt-audit behavior. New executions
    # always pass the diagnostic range and require the actual MINOS branch.
    required = ("fit_status", "r", "rErr", "numbadnll")
    if r_max is not None:
        required += ("rHiErr",)
    rows = _read_tree(path, "tree_fit_sb", required)
    if len(rows["fit_status"]) != 1:
        raise CombineRunError("Expected one signal-plus-background Asimov diagnostic fit")
    result = {key: values[0] for key, values in rows.items()}
    if result["fit_status"] != 0:
        raise CombineRunError(f"Asimov diagnostic fit failed: status {result['fit_status']}")
    if not all(math.isfinite(float(result[key])) for key in ("r", "rErr")) or result["rErr"] <= 0:
        raise CombineRunError("Asimov fit has invalid signal-strength estimate or uncertainty")
    if abs(result["r"] - signal_strength) > 0.01 * max(1.0, signal_strength):
        raise CombineRunError("Asimov diagnostic fit does not recover the injected signal strength")
    if r_max is not None:
        result["interval_validation"] = validate_fit_diagnostics_interval(result, r_max)
    result["scope"] = "independent_prefit_signal_plus_background_asimov_fit"
    # numbadnll counts invalid intermediate evaluations, not necessarily a bad minimum.
    # Individual AsymptoticLimits profile fits do not expose statuses in the limit tree.
    return result


def _run_command(argv, cwd, log_path, timeout_seconds, commands):
    started = time.monotonic()
    record = {"argv": list(argv), "cwd": str(cwd), "log": str(log_path)}
    commands.append(record)
    timed_out = False
    try:
        with Path(log_path).open("wb") as log:
            proc = subprocess.Popen(argv, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            except BaseException:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                raise
        record.update(returncode=proc.returncode, timed_out=timed_out)
    finally:
        record["wall_seconds"] = time.monotonic() - started
    if timed_out:
        raise CombineRunError(f"Command exceeded {timeout_seconds:g} seconds; see {log_path}")
    if proc.returncode != 0:
        raise CombineRunError(f"Command exited {proc.returncode}; see {log_path}")
    output = Path(log_path).read_text(errors="replace")
    record["warnings"] = [line.strip() for line in output.splitlines()
                          if re.search(r"warning|failed|error", line, re.IGNORECASE)]
    return output


def run_expected(card, output_dir, command_prefix=None, timeout_seconds=300,
                 mass=125, signal_strength=1.0, r_max=1000.0, fit_diagnostics=True):
    """Build one workspace and calculate blind expected CLs limits and Z.

    ``command_prefix=['/path/to/combine-env']`` supports an isolated runtime.
    The output directory must be empty; every command, failure, ROOT artifact,
    input hash and wall time is preserved. ``CombineRunError.result`` provides
    the failure receipt. A successful diagnostic fit is required by default.
    """
    card = Path(card).resolve(strict=True)
    output_dir = Path(output_dir).absolute()
    if isinstance(command_prefix, str):
        raise ValueError("command_prefix must be an argument sequence, not shell text")
    prefix = list(command_prefix or [])
    if any(not isinstance(arg, str) or not arg for arg in prefix):
        raise ValueError("command_prefix must contain nonempty argument strings")
    if not all(math.isfinite(float(x)) and float(x) > 0
               for x in (timeout_seconds, mass, signal_strength, r_max)):
        raise ValueError("Timeout, mass, signal strength and r_max must be positive and finite")
    if signal_strength >= r_max:
        raise ValueError("Injected signal strength must be below r_max")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing stale/nonempty Combine output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "status": "running", "card": str(card),
              "card_sha256": _sha256(card), "commands": [],
              "mass_label": float(mass), "injected_signal_strength": float(signal_strength),
              "r_max": float(r_max), "expected_only": True, "asimov": "prefit",
              "output_dir": str(output_dir)}
    receipt = output_dir / "combine_result.json"
    started = time.monotonic()

    def execute(label, program, args, cwd=output_dir):
        return _run_command(prefix + [program] + args, cwd, output_dir / (label + ".log"),
                            timeout_seconds, result["commands"])

    try:
        workspace = output_dir / "workspace.root"
        execute("workspace", "text2workspace.py", [str(card), "-m", f"{mass:g}", "-o", str(workspace)],
                cwd=card.parent)
        if not workspace.is_file() or workspace.stat().st_size == 0:
            raise CombineRunError("Workspace conversion produced no nonempty ROOT file")
        common = [str(workspace), "-m", f"{mass:g}", "--rMin", "0", "--rMax", f"{r_max:g}"]
        limit_log = execute("limits", "combine", ["-M", "AsymptoticLimits"] + common +
                            ["--run", "blind", "--cl", "0.95", "--rRelAcc", "0.001",
                             "--rAbsAcc", "0.000001", "-n", ".expected"])
        version_match = re.search(r"<<<\s*(v[^<>\s]+)\s*>>>", limit_log)
        if not version_match:
            raise CombineRunError("Combine did not report a recognizable version banner")
        result["combine_version"] = version_match.group(1)
        limits_path = output_dir / f"higgsCombine.expected.AsymptoticLimits.mH{mass:g}.root"
        result["expected_limit"] = parse_expected_limits(limits_path)
        if result["expected_limit"]["quantiles"]["0.975"] >= 0.95 * r_max:
            raise CombineRunError("Expected limit approaches r_max; increase the range and rerun")
        # A discovery fit concerns the injected r, not the potentially much larger
        # exclusion range. Very wide r ranges can make Minos silently return Z=0.
        significance_policy = significance_fit_policy(signal_strength)
        result["significance_fit_policy"] = significance_policy
        significance_r_max = significance_policy["r_max"]
        fit_common = [str(workspace), "-m", f"{mass:g}", "--rMin", "0",
                      "--rMax", f"{significance_r_max:.17g}"]
        result["significance_r_max"] = significance_r_max
        execute("significance", "combine", ["-M", "Significance"] + fit_common +
                ["-t", "-1", "--expectSignal", f"{signal_strength:g}",
                 "--cminDefaultMinimizerStrategy", str(significance_policy["minimizer_strategy"]),
                 "--cminDefaultMinimizerTolerance", f"{significance_policy['minimizer_tolerance']:.17g}",
                 "-n", ".expected"])
        # Asimov jobs have a deterministic seed suffix in their ROOT filenames.
        paths = list(output_dir.glob(f"higgsCombine.expected.Significance.mH{mass:g}*.root"))
        if len(paths) != 1:
            raise CombineRunError("Expected exactly one significance ROOT artifact")
        result["expected_significance"] = parse_expected_significance(paths[0])
        if result["expected_significance"] == 0:
            raise CombineRunError("Zero significance for a positive Asimov injection; investigate fit resolution before ranking")
        if fit_diagnostics:
            policy = diagnostic_range_policy(result["expected_limit"]["quantiles"]["0.975"],
                                             signal_strength)
            result["fit_diagnostics_range_policy"] = policy
            result["fit_diagnostics_r_max"] = policy["r_max"]
            diagnostic_common = [str(workspace), "-m", f"{mass:g}", "--rMin", "0",
                                 "--rMax", f"{policy['r_max']:.17g}"]
            fit_log = execute("fit_diagnostics", "combine", ["-M", "FitDiagnostics"] + diagnostic_common +
                              ["-t", "-1", "--expectSignal", f"{signal_strength:g}",
                               "--skipBOnlyFit", "-v", "1", "-n", ".expected"])
            diagnostics_path = output_dir / "fitDiagnostics.expected.root"
            result["fit_diagnostics"] = parse_fit_diagnostics(diagnostics_path, signal_strength,
                                                             r_max=policy["r_max"])
            covariance = re.findall(r"Fit S\+B, status\s*=\s*(-?\d+), numBadNLL\s*=\s*(\d+), covariance quality\s*=\s*(-?\d+)", fit_log)
            if not covariance:
                raise CombineRunError("FitDiagnostics did not report its covariance quality")
            if covariance:
                quality = int(covariance[-1][2])
                result["fit_diagnostics"]["covariance_quality"] = quality
                if quality != 3:
                    raise CombineRunError(f"Asimov diagnostic fit covariance quality is {quality}, expected 3")
        else:
            result["fit_diagnostics"] = {"performed": False}
        result["artifacts"] = {p.name: {"sha256": _sha256(p), "size_bytes": p.stat().st_size}
                               for p in sorted(output_dir.glob("*.root"))}
        result["status"] = "complete"
    except (KeyboardInterrupt, SystemExit) as exc:
        result["status"] = "interrupted"
        result["error"] = type(exc).__name__
        raise
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        raise CombineRunError(str(exc), result) from exc
    finally:
        result["wall_seconds"] = time.monotonic() - started
        receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-wrapper", help="Executable that prepares Combine and execs its argv")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--mass", type=float, default=125)
    parser.add_argument("--signal-strength", type=float, default=1)
    parser.add_argument("--r-max", type=float, default=1000)
    args = parser.parse_args(argv)
    try:
        result = run_expected(args.card, args.output,
                              command_prefix=[args.runtime_wrapper] if args.runtime_wrapper else None,
                              timeout_seconds=args.timeout_seconds, mass=args.mass,
                              signal_strength=args.signal_strength, r_max=args.r_max)
    except (ValueError, OSError, CombineRunError) as exc:
        parser.exit(2, f"Combine expected evaluation failed: {exc}\n")
    print(json.dumps({key: result[key] for key in
                      ("status", "expected_limit", "expected_significance", "wall_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
