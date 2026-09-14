#!/usr/bin/env python3
"""Stationary benchmark aggregation and retrospective sensitivity regrets.

Reference significances are frozen before an Optuna study starts. The optimizer
maximizes a worst-benchmark utility; best-observed regrets are reporting metrics
and must never be fed back as changing historical trial values.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


INFEASIBLE_OBJECTIVE = -1.0
MEAN_METHOD = "mean_asimov"
REFERENCE_METHOD = "fixed_reference_regret"
REFERENCE_SCHEMA = 1


def _required(names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise ValueError("required_signals must be a sequence of unique names")
    names = tuple(names)
    if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError("required_signals must be nonempty and unique")
    return names


def _number(value: Any, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def context_fingerprint(context: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(context).encode()).hexdigest()


def _mean(values: Sequence[float]) -> float:
    # Divide before summing: averaging finite large nonnegative scores must not
    # overflow merely because their unnormalized sum does.
    return math.fsum(value / len(values) for value in values)


def comparison_context(parameters: Mapping[str, Any], comparison_samples: Mapping[str, Any],
                       required_signals: Sequence[str]) -> dict[str, Any]:
    """Bind score physics and physical MC while excluding reconstruction paths.

``parameters`` contains fixed analysis definitions, not the scanned configuration.
Sample event-identity digests bind the processed population independently of the
paths/names of the ntuple files generated for different reconstruction choices.
All other sample metadata, including generator denominators, remains bound.
    """
    required = _required(required_signals)
    if not isinstance(parameters, Mapping) or not isinstance(comparison_samples, Mapping):
        raise ValueError("comparison context requires parameters and comparison_samples objects")
    excluded_parameters = {"objective", "objective_definition", "configuration", "configuration_slug",
                           "reconstruction", "campaign_path", "_campaign_path"}
    excluded_samples = {"files", "input_files", "input_file_list", "input_provenance",
                        "normalization_source", "selected_events", "histogram", "physicality"}
    samples = {}
    for name, sample in comparison_samples.items():
        if not isinstance(name, str) or not name or not isinstance(sample, Mapping):
            raise ValueError("comparison sample identities must be named objects")
        samples[name] = {key: value for key, value in sample.items() if key not in excluded_samples}
    signal_names = {name for name, sample in samples.items() if sample.get("kind") == "signal"}
    if signal_names != set(required):
        raise ValueError("comparison sample signal identities differ from required_signals")
    context = {"schema_version": 1, "required_signals": sorted(required),
               "parameters": {key: value for key, value in parameters.items() if key not in excluded_parameters},
               "comparison_samples": samples}
    # Round-trip detaches the context from caller-owned mutable dictionaries.
    return json.loads(canonical_json(context))


def context_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return comparison_context(result["parameters"], result["comparison_samples"], result["required_signals"])


def _validate_physical_identities(context: Mapping[str, Any]) -> None:
    samples = context.get("comparison_samples")
    if not isinstance(samples, Mapping) or not samples:
        raise ValueError("frozen references require physical comparison_samples")
    for name, sample in samples.items():
        for field in ("event_identity_sha256", "event_generator_sha256"):
            digest = sample.get(field) if isinstance(sample, Mapping) else None
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"{name}: frozen references require an {field} digest")


def resolve_objective(spec: Mapping[str, Any] | None, base_dir: str | Path) -> dict[str, Any] | None:
    """Resolve a reference file once while retaining its exact content hash."""
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        raise ValueError("objective must be an object")
    result = dict(spec)
    if result.get("method") == REFERENCE_METHOD:
        filename = result.get("reference_file")
        if not isinstance(filename, str) or not filename:
            raise ValueError("fixed_reference_regret requires reference_file")
        path = (Path(base_dir) / filename).resolve()
        raw = path.read_bytes()
        result.update(reference_file=str(path), _reference=json.loads(raw),
                      _reference_sha256=hashlib.sha256(raw).hexdigest())
    return result


def validate_objective(spec: Mapping[str, Any] | None, required: Sequence[str],
                       comparison_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    required = _required(required)
    if spec is None:
        return {"method": MEAN_METHOD}
    if not isinstance(spec, Mapping):
        raise ValueError("objective must be an object")
    allowed = {"method", "mean_weight", "reference_file", "_reference", "_reference_sha256"}
    if set(spec) - allowed:
        raise ValueError(f"unknown objective fields: {sorted(set(spec) - allowed)}")
    method = spec.get("method")
    if method == MEAN_METHOD:
        if set(spec) != {"method"}:
            raise ValueError("mean_asimov does not take reference or mean_weight fields")
        return {"method": method}
    if method != REFERENCE_METHOD:
        raise ValueError(f"unsupported objective method: {method!r}")
    weight = _number(spec.get("mean_weight", 0.25), "objective mean_weight")
    reference = spec.get("_reference")
    if not isinstance(reference, Mapping):
        raise ValueError("fixed_reference_regret requires a resolved reference_file; freeze calibration references first")
    if reference.get("schema_version") != REFERENCE_SCHEMA or isinstance(reference.get("schema_version"), bool):
        raise ValueError("unsupported reference schema_version")
    if set(_required(reference.get("required_signals", []))) != set(required):
        raise ValueError("frozen reference required_signals differ from this campaign")
    values = reference.get("reference_significances")
    if not isinstance(values, Mapping) or set(values) != set(required):
        raise ValueError("frozen reference significances must cover exactly required_signals")
    for name in required:
        _number(values[name], f"reference significance for {name}", positive=True)
    frozen_context = reference.get("comparison_context")
    if not isinstance(frozen_context, Mapping):
        raise ValueError("frozen references require a comparison_context")
    _validate_physical_identities(frozen_context)
    if reference.get("context_sha256") != context_fingerprint(frozen_context):
        raise ValueError("frozen reference context hash is invalid")
    if set(_required(frozen_context.get("required_signals", []))) != set(required):
        raise ValueError("frozen reference context signal identities differ")
    if comparison_context is not None and context_fingerprint(comparison_context) != reference["context_sha256"]:
        raise ValueError("frozen references differ from this evaluation's physics or physical MC context")
    digest = spec.get("_reference_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("resolved reference content hash is missing or invalid")
    return {**spec, "mean_weight": weight}


def aggregate_objective(per_signal: Mapping[str, Any], required: Sequence[str],
                        spec: Mapping[str, Any] | None,
                        context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    required = _required(required)
    policy = validate_objective(spec, required, context)
    method = policy["method"]
    definition = {"method": method}
    if method == REFERENCE_METHOD:
        if context is None:
            raise ValueError("fixed-reference aggregation requires this evaluation's comparison context")
        definition.update(mean_weight=policy["mean_weight"], reference_sha256=policy["_reference_sha256"],
                          context_sha256=policy["_reference"]["context_sha256"])
    name = "mean_binned_asimov_significance" if method == MEAN_METHOD else "fixed_reference_worst_plus_mean_asimov_utility"
    result = {"objective_name": name, "direction": "maximize", "objective_definition": definition}
    failures = []
    if not isinstance(per_signal, Mapping) or set(per_signal) != set(required):
        failures.append("incomplete_or_unexpected_signal_coverage")
    values = {}
    for signal in required:
        info = per_signal.get(signal, {}) if isinstance(per_signal, Mapping) else {}
        if not isinstance(info, Mapping) or info.get("feasible") is not True:
            failures.append(f"{signal}:infeasible_signal")
            continue
        try:
            values[signal] = _number(info.get("significance"), f"significance for {signal}")
        except ValueError:
            failures.append(f"{signal}:invalid_significance")
    if failures:
        return {**result, "feasible": False, "objective": INFEASIBLE_OBJECTIVE,
                "objective_failure_reasons": failures}
    if method == MEAN_METHOD:
        value = _mean(list(values.values()))
        return {**result, "feasible": True, "objective": value, "objective_failure_reasons": []}
    references = policy["_reference"]["reference_significances"]
    ratios = {signal: values[signal] / references[signal] for signal in required}
    if any(not math.isfinite(value) for value in ratios.values()):
        return {**result, "feasible": False, "objective": INFEASIBLE_OBJECTIVE,
                "objective_failure_reasons": ["nonfinite_reference_ratio"]}
    deficits = {signal: 1.0 - ratios[signal] for signal in required}
    mean_ratio = _mean(list(ratios.values()))
    utility = min(ratios.values()) + policy["mean_weight"] * mean_ratio
    if not math.isfinite(utility):
        return {**result, "feasible": False, "objective": INFEASIBLE_OBJECTIVE,
                "objective_failure_reasons": ["nonfinite_aggregate_objective"]}
    return {**result, "feasible": True, "objective": utility, "objective_failure_reasons": [],
            "reference_ratios": ratios, "reference_deficits": deficits,
            "mean_reference_deficit": _mean(list(deficits.values())),
            "worst_reference_deficit": max(deficits.values()),
            "reference_regret_objective": max(deficits.values()) + policy["mean_weight"] * _mean(list(deficits.values())),
            "reference_significances": dict(references)}


def relative_regret_rankings(results: Sequence[Mapping[str, Any]], required_signals: Sequence[str],
                             mean_weight: float = 0.25) -> list[dict[str, Any]]:
    """Rank complete feasible trials against final observed bests, for reporting.

These regrets are recomputed for the supplied trial cohort. They are deliberately
separate from the stationary objective stored in Optuna.
    """
    required = _required(required_signals)
    weight = _number(mean_weight, "mean_weight")
    eligible = []
    for result in results:
        info = result.get("per_signal", {})
        if result.get("feasible") is not True or not isinstance(info, Mapping) or set(info) != set(required):
            continue
        try:
            values = {n: _number(info[n]["significance"], f"significance for {n}") for n in required}
        except (KeyError, TypeError, ValueError):
            continue
        if any(info[n].get("feasible") is not True for n in required):
            continue
        eligible.append((result, values))
    if not eligible:
        return []
    best = {n: max(values[n] for _, values in eligible) for n in required}
    rows = []
    for result, values in eligible:
        regrets = {n: (best[n] - values[n]) / best[n] if best[n] > 0 else 0.0 for n in required}
        mean = math.fsum(regrets.values()) / len(required)
        worst = max(regrets.values())
        rows.append({"configuration": result["configuration"], "objective": result.get("objective"),
                     "ranking_basis": "retrospective_worst_plus_mean_relative_sensitivity_regret",
                     "mean_regret": mean, "worst_regret": worst, "regret_objective": worst + weight * mean,
                     "per_signal_regret": regrets, "per_signal_best_significance": dict(best),
                     "mean_weight": weight})
    rows.sort(key=lambda row: (row["regret_objective"], row["mean_regret"], canonical_json(row["configuration"])))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows
