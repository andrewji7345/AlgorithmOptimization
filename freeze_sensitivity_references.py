#!/usr/bin/env python3
"""Freeze Asimov benchmark scales from complete, compatible calibration results.

Run calibration with objective.method=mean_asimov, then freeze its complete
results once and select fixed_reference_regret in a new Optuna study. The
per-benchmark maximum over this declared calibration cohort is a scale, not an
assertion that the global optimum has been found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from sensitivity_objective import (
    REFERENCE_SCHEMA, _number, _required, _validate_physical_identities,
    canonical_json, context_fingerprint, context_from_result,
)


def _campaign_matches(campaign: Mapping[str, Any], context: Mapping[str, Any]) -> None:
    # Imported lazily so objective math and reference-file inspection do not
    # require the ROOT/ntuple reader's dependencies.
    from evaluate_sensitivity import score_parameters

    required = _required(campaign.get("required_signals", []))
    if set(required) != set(context["required_signals"]):
        raise ValueError("campaign required_signals differ from calibration")
    if canonical_json(score_parameters(campaign)) != canonical_json(context["parameters"]):
        raise ValueError("campaign score physics parameters differ from calibration")
    samples = campaign.get("samples", [])
    names = [sample.get("name") for sample in samples]
    if len(names) != len(set(names)) or set(names) != set(context["comparison_samples"]):
        raise ValueError("campaign sample identities differ from calibration")
    defaults = {"filter_efficiency": 1, "k_factor": 1, "generated_suu_mass_gev": None,
                "generated_chi_mass_gev": None, "decay_channel": None, "dataset": None, "category": None}
    fields = ("kind", "normalization_scope", "cross_section_pb", *defaults)
    for sample in samples:
        observed = context["comparison_samples"][sample["name"]]
        for field in fields:
            declared = sample.get(field, defaults.get(field))
            if field == "decay_channel":
                declared = sample.get("decay_channel", sample.get("decay_mode"))
            if declared != observed.get(field, defaults.get(field)):
                raise ValueError(f"{sample['name']}: campaign {field} differs from calibration")
        for field in ("sum_gen_weights", "generated_events"):
            expected = sample.get(field)
            if expected != "metadata" and expected != observed.get(field):
                raise ValueError(f"{sample['name']}: campaign {field} differs from calibration")


def freeze_references(campaign_path: str | Path, result_paths: Sequence[str | Path],
                      output_path: str | Path) -> dict[str, Any]:
    campaign_path = Path(campaign_path).resolve()
    raw_campaign = campaign_path.read_bytes()
    campaign = json.loads(raw_campaign)
    if not isinstance(campaign, Mapping):
        raise ValueError("campaign must be an object")
    required = _required(campaign.get("required_signals", []))
    if not result_paths:
        raise ValueError("at least one complete calibration objective is required")
    reference, context, provenance = {}, None, []
    seen_hashes, seen_configurations = set(), set()
    for filename in result_paths:
        path = Path(filename).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen_hashes:
            raise ValueError("duplicate calibration objective input")
        seen_hashes.add(digest)
        result = json.loads(raw)
        if not isinstance(result, Mapping) or result.get("feasible") is not True or result.get("failure_reasons"):
            raise ValueError(f"{path}: calibration objective must be globally feasible and complete")
        if set(_required(result.get("required_signals", []))) != set(required):
            raise ValueError(f"{path}: required signal identities differ")
        per_signal = result.get("per_signal")
        if not isinstance(per_signal, Mapping) or set(per_signal) != set(required):
            raise ValueError(f"{path}: calibration must cover exactly every required signal")
        if "configuration" not in result:
            raise ValueError(f"{path}: calibration configuration is missing")
        config_identity = canonical_json(result["configuration"])
        if config_identity in seen_configurations:
            raise ValueError("duplicate calibration reconstruction configuration")
        seen_configurations.add(config_identity)
        current_context = context_from_result(result)
        _validate_physical_identities(current_context)
        if context is None:
            _campaign_matches(campaign, current_context)
            context = current_context
        elif context_fingerprint(context) != context_fingerprint(current_context):
            raise ValueError(f"{path}: calibration physics or physical MC context differs")
        for name in required:
            item = per_signal[name]
            if not isinstance(item, Mapping) or item.get("feasible") is not True or item.get("failure_reasons"):
                raise ValueError(f"{path}: {name} is not feasible")
            value = _number(item.get("significance"), f"{name} significance")
            q0 = _number(item.get("q0"), f"{name} q0")
            if not math.isclose(math.sqrt(q0), value, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"{path}: {name} significance disagrees with q0")
            reference[name] = max(reference.get(name, 0), value)
        provenance.append({"path": str(path), "sha256": digest, "configuration": result["configuration"]})
    for name in required:
        _number(reference[name], f"{name} frozen reference", positive=True)
    document = {
        "schema_version": REFERENCE_SCHEMA,
        "reference_definition": "maximum_significance_over_frozen_feasible_calibration",
        "reference_interpretation": "fixed calibration scales; not known global optima; improvements may exceed one",
        "required_signals": sorted(required), "reference_significances": reference,
        "comparison_context": context, "context_sha256": context_fingerprint(context),
        "campaign": {"path": str(campaign_path), "sha256": hashlib.sha256(raw_campaign).hexdigest()},
        "calibration_inputs": sorted(provenance, key=lambda row: row["sha256"]),
    }
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        with output.open("x") as stream:
            stream.write(text)
    except FileExistsError:
        if output.read_text() != text:
            raise ValueError(f"refusing to replace different frozen references: {output}") from None
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, help="Campaign defining the same physics and complete signal set")
    parser.add_argument("--calibration-objective", action="append", required=True,
                        help="Complete calibration objective.json; repeat for every frozen calibration configuration")
    parser.add_argument("--output", required=True, help="Immutable JSON reference document to create")
    args = parser.parse_args(argv)
    try:
        result = freeze_references(args.campaign, args.calibration_objective, args.output)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(Path(args.output).resolve()), "context_sha256": result["context_sha256"],
                      "reference_significances": result["reference_significances"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
