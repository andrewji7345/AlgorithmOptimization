#!/usr/bin/env python3
"""Evaluate one compact schema-v3 configuration against an explicit MC campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from compact_scan_metrics import (
    ConfigurationKey, PhysicalityDefinition, VALID_STATUS, configuration_indices,
    configuration_key, configuration_slug, gate_mask, load_metadata,
    parse_configuration_slug, read_event_payload,
)
from sensitivity_metrics import (
    PhysicalityAccumulator, WeightedHistogram,
    asimov_significance, normalization_factor,
    validated_edges, weighted_histogram,
)
from sensitivity_objective import (
    aggregate_objective, comparison_context, resolve_objective, validate_objective,
    relative_regret_rankings,
)


ANALYSIS_SELECTION = "AN-23-067-UL2017-cutbased-v1"
LEGACY_CORRECTION_PRESCRIPTION = "UL2017-AK4PFchs-AK8PFPuppi-JEC-JER-nominal-v1"
PREVIOUS_CORRECTION_PRESCRIPTION = "UL2017-AK4CHS-baseline-AK4AK8Puppi-reco-JEC-JER-nominal-v2"
CORRECTION_PRESCRIPTION = "UL2017-AK4CHS-baseline-AK4AK8Puppi-reco-JEC-JER-btagGuard-v3"


def parse_configuration(value: str | Mapping[str, Any]) -> ConfigurationKey:
    if isinstance(value, Mapping):
        return ConfigurationKey(**value)
    if value.startswith("ng"):
        return parse_configuration_slug(value)
    parts = value.replace(",", ":").split(":")
    if len(parts) != 6:
        raise ValueError("configuration must be n:Tgate:Tkeep:RAK:RCA:c, slug, or JSON object")
    n = int(parts[0])
    gate = None if n == 0 else float(parts[1])
    return configuration_key(n, gate, *(float(part) for part in parts[2:]))


def load_campaign(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open() as stream:
        campaign = json.load(stream)
    if not isinstance(campaign, dict):
        raise ValueError("campaign must be a JSON object")
    campaign["_campaign_path"] = str(path)
    if "objective" in campaign:
        campaign["objective"] = resolve_objective(campaign["objective"], path.parent)
    for sample in campaign.get("samples", []):
        sample["files"] = [str((path.parent / item).resolve()) if "://" not in item else item
                           for item in sample.get("files", [])]
    validate_campaign(campaign)
    return campaign


def score_parameters(campaign: Mapping[str, Any]) -> dict[str, Any]:
    """Fixed definition of one score, independent of the reconstruction trial."""
    parameters = {
        "luminosity_pb": campaign["luminosity_pb"],
        "mass_bin_edges_gev": list(campaign["mass_bin_edges_gev"]),
        "chi_mass_bin_edges_gev": campaign.get("chi_mass_bin_edges_gev"),
        "histogram_flow": campaign.get("histogram_flow", "fold"),
        "background_uncertainty": "independent_bin_sumw2_auxiliary",
        "minimum_background_effective_events": campaign.get("minimum_background_effective_events", 10),
        "analysis_selection": campaign["analysis_selection"],
        "correction_prescription": campaign["correction_prescription"],
        "analysis_systematic": campaign.get("analysis_systematic", "nominal"),
        "weight_convention": campaign.get("weight_convention", "analysis"),
        "physicality_mode": campaign.get("physicality_mode", "off"),
    }
    if parameters["physicality_mode"] != "off":
        parameters.update(physicality=PhysicalityDefinition(**campaign.get("physicality", {})).as_dict(),
                          physicality_population="baseline_and_gate_before_signal_region")
    return parameters


def validate_campaign(campaign: Mapping[str, Any]) -> None:
    if campaign.get("purpose", "production") not in ("production", "pilot"):
        raise ValueError("campaign purpose must be production or pilot")
    lumi = float(campaign.get("luminosity_pb", 0))
    if not math.isfinite(lumi) or lumi <= 0:
        raise ValueError("luminosity_pb must be finite and positive")
    validated_edges(campaign.get("mass_bin_edges_gev", []), "mass_bin_edges_gev")
    if "chi_mass_bin_edges_gev" in campaign:
        validated_edges(campaign["chi_mass_bin_edges_gev"], "chi_mass_bin_edges_gev")
    if campaign.get("analysis_selection") != ANALYSIS_SELECTION:
        raise ValueError(f"analysis_selection must be {ANALYSIS_SELECTION!r}")
    if campaign.get("correction_prescription") not in (CORRECTION_PRESCRIPTION, PREVIOUS_CORRECTION_PRESCRIPTION, LEGACY_CORRECTION_PRESCRIPTION):
        raise ValueError("unsupported correction_prescription")
    if campaign.get("analysis_systematic", "nominal") not in ("nominal", "JECUp", "JECDown", "JERUp", "JERDown"):
        raise ValueError("unsupported analysis_systematic")
    threshold = float(campaign.get("minimum_background_effective_events", 10))
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("minimum_background_effective_events must be finite and nonnegative")
    if campaign.get("histogram_flow", "fold") not in ("fold", "exclude"):
        raise ValueError("histogram_flow must be fold or exclude")
    if campaign.get("weight_convention", "analysis") not in ("analysis", "reference"):
        raise ValueError("weight_convention must be analysis or reference")
    if campaign.get("physicality_mode", "off") not in ("off", "diagnostic", "gate"):
        raise ValueError("physicality_mode must be off, diagnostic or gate")
    PhysicalityDefinition(**campaign.get("physicality", {}))
    names, files, signal_names = set(), set(), set()
    samples = campaign.get("samples", [])
    if not samples or not any(sample.get("kind") == "background" for sample in samples):
        raise ValueError("campaign requires at least one background sample")
    for sample in samples:
        name = sample.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("sample names must be nonempty and unique")
        names.add(name)
        if sample.get("kind") not in ("signal", "background"):
            raise ValueError(f"{name}: kind must be signal or background")
        if sample["kind"] == "signal":
            signal_names.add(name)
            mass = float(sample.get("generated_chi_mass_gev", 0))
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError(f"{name}: generated_chi_mass_gev must be finite and positive")
        if sample.get("complete") is not True or not sample.get("files"):
            raise ValueError(f"{name}: complete:true and an explicit nonempty files list are required")
        for filename in sample["files"]:
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"{name}: invalid file path")
            if filename in files:
                raise ValueError(f"duplicate input file across campaign: {filename}")
            files.add(filename)
        scope = sample.get("normalization_scope")
        if scope not in ("full_dataset", "representative_subset"):
            raise ValueError(f"{name}: explicit normalization_scope is required")
        if scope == "representative_subset" and campaign.get("purpose") != "pilot":
            raise ValueError(f"{name}: representative_subset normalization requires purpose:pilot")
        denominator = sample.get("sum_gen_weights")
        count = sample.get("generated_events")
        if denominator == "metadata" or count == "metadata":
            if scope != "representative_subset":
                raise ValueError(f"{name}: metadata denominators are permitted only for a representative pilot subset")
        if denominator != "metadata":
            if not isinstance(denominator, (int, float)) or not math.isfinite(denominator) or denominator <= 0:
                raise ValueError(f"{name}: sum_gen_weights must be finite and positive")
        if count != "metadata":
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError(f"{name}: generated_events must be a positive integer")
        normalization_factor(float(sample.get("cross_section_pb", 0)), lumi, 1,
                             float(sample.get("filter_efficiency", 1)), float(sample.get("k_factor", 1)))
    required = campaign.get("required_signals", [])
    if not required or len(set(required)) != len(required) or set(required) != signal_names:
        raise ValueError("required_signals must list every signal sample exactly once")
    validate_objective(campaign.get("objective"), required)


def _validate_metadata(metadata: Any, sample: Mapping[str, Any], campaign: Mapping[str, Any]) -> None:
    if metadata.schema_version != 3:
        raise ValueError(f"{metadata.path}: sensitivity requires schema version 3; regenerate this ntuple")
    if metadata.sample_name != sample["name"]:
        raise ValueError(f"{metadata.path}: sampleName differs from campaign sample {sample['name']}")
    for field, expected in (("sample_kind", sample["kind"]),
                            ("analysis_systematic", campaign.get("analysis_systematic", "nominal")),
                            ("analysis_selection", campaign["analysis_selection"]),
                            ("correction_prescription", campaign["correction_prescription"])):
        if getattr(metadata, field, "nominal" if field == "analysis_systematic" else None) != expected:
            raise ValueError(f"{metadata.path}: {field} differs from campaign ({expected!r})")
    allowed = (0.8,) if metadata.correction_prescription == LEGACY_CORRECTION_PRESCRIPTION else (0.4, 0.8)
    if not any(math.isclose(metadata.ak_radius, radius, rel_tol=0, abs_tol=1e-6) for radius in allowed):
        raise ValueError(f"{metadata.path}: AK radius incompatible with correction_prescription")
    if not metadata.use_jec:
        raise ValueError(f"{metadata.path}: sensitivity requires corrected jets")


def evaluate_campaign(campaign: Mapping[str, Any], key: ConfigurationKey) -> dict[str, Any]:
    validate_campaign(campaign)
    edges = [validated_edges(campaign["mass_bin_edges_gev"], "mass_bin_edges_gev")]
    if "chi_mass_bin_edges_gev" in campaign:
        edges.append(validated_edges(campaign["chi_mass_bin_edges_gev"], "chi_mass_bin_edges_gev"))
    definition = PhysicalityDefinition(**campaign.get("physicality", {}))
    parameters = score_parameters(campaign)
    physicality_mode = parameters["physicality_mode"]
    background = WeightedHistogram.empty(edges)
    signals, samples = {}, {}
    for sample in campaign["samples"]:
        loaded = [load_metadata(path) for path in sample["files"]]
        for meta in loaded:
            _validate_metadata(meta, sample, campaign)
            configuration_indices(meta, key)  # Every declared shard must cover this trial.
        total_events = sum(meta.processed_events for meta in loaded)
        total_gen_weights = math.fsum(meta.sum_weights for meta in loaded)
        if total_events <= 0 or not math.isfinite(total_gen_weights) or total_gen_weights <= 0:
            raise ValueError(f"{sample['name']}: empty/nonpositive generated MC denominator")
        expected_events = total_events if sample["generated_events"] == "metadata" else sample["generated_events"]
        denominator = total_gen_weights if sample["sum_gen_weights"] == "metadata" else sample["sum_gen_weights"]
        if total_events != expected_events or not math.isclose(total_gen_weights, denominator, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"{sample['name']}: incomplete generated coverage: processed {total_events} / {expected_events} events, sumWeights {total_gen_weights} / {denominator}")
        scale = normalization_factor(float(sample["cross_section_pb"]), campaign["luminosity_pb"], denominator,
                                     sample.get("filter_efficiency", 1), sample.get("k_factor", 1))
        hist = WeightedHistogram.empty(edges)
        physicality = PhysicalityAccumulator.empty(definition)
        identities = set()
        identity_chunks, generator_chunks = [], []
        selected_count = 0
        for meta in loaded:
            payload = read_event_payload(meta)
            if payload.n_events != meta.processed_events:
                raise ValueError(f"{meta.path}: preselection event tree is incomplete")
            if any(getattr(payload, name, None) is None for name in
                   ("analysis_weight", "passes_baseline", "passes_signal_region", "suu_mass")):
                raise ValueError(f"{meta.path}: missing schema-v3 sensitivity branches")
            ids = set(zip(payload.run.tolist(), payload.lumi.tolist(), payload.event.tolist()))
            if len(ids) != payload.n_events or identities.intersection(ids):
                raise ValueError(f"{sample['name']}: duplicate run/lumi/event within or across shards")
            identities.update(ids)
            gen = np.asarray(payload.gen_weight, dtype=float)
            if parameters["weight_convention"] == "reference":
                # The additive contract independently validates the AN region
                # flags and the nominal weight convention for every model.
                from evaluate_reference import _read_additive, _validate_additive
                extra_meta, arrays = _read_additive(meta)
                _validate_additive(meta, payload, extra_meta, arrays, parameters["analysis_systematic"])
                correction = np.asarray(arrays["referenceWeight"], dtype=float)
            else:
                correction = np.asarray(payload.analysis_weight, dtype=float)
            if gen.shape != (payload.n_events,) or correction.shape != gen.shape:
                raise ValueError(f"{meta.path}: scalar event weight branch shape mismatch")
            if np.any(~np.isfinite(gen)) or np.any(~np.isfinite(correction)) or np.any(correction < 0):
                raise ValueError(f"{meta.path}: invalid generator or correction weight")
            # genWeight is stored as float; metadata accumulates original doubles.
            tolerance = 2e-6 * max(float(np.abs(gen).sum()), 1)
            if abs(float(gen.sum()) - meta.sum_weights) > tolerance:
                raise ValueError(f"{meta.path}: sumWeights differs from all preselection genWeight entries")
            if not math.isclose(float(np.square(gen).sum()), meta.sum_weights2, rel_tol=4e-6, abs_tol=1e-7):
                raise ValueError(f"{meta.path}: sumWeights2 differs from all preselection genWeight entries")
            identity = np.empty(payload.n_events, dtype=[("run", "<u4"), ("lumi", "<u4"), ("event", "<u8")])
            for field in identity.dtype.names:
                identity[field] = getattr(payload, field)
            identity_chunks.append(identity)
            generator_chunks.append(gen)
            config_index, _ = configuration_indices(meta, key)
            baseline = np.asarray(payload.passes_baseline)
            region = np.asarray(payload.passes_signal_region)
            if baseline.shape != (payload.n_events,) or region.shape != payload.reco_status.shape:
                raise ValueError(f"{meta.path}: selection branch shape mismatch")
            if np.any((baseline != 0) & (baseline != 1)) or np.any((region != 0) & (region != 1)):
                raise ValueError(f"{meta.path}: nonboolean selection flag")
            baseline_gate = baseline.astype(bool) & gate_mask(payload, key.n_gate_jets, key.gate_pt_cut)
            mass1, mass2 = payload.sj1_mass[:, config_index], payload.sj2_mass[:, config_index]
            valid = payload.reco_status[:, config_index] == VALID_STATUS
            if np.any(region[:, config_index].astype(bool) & (~valid | ~baseline.astype(bool))):
                raise ValueError(f"{meta.path}: signal-region flag violates baseline/reconstruction prerequisites")
            if sample["kind"] == "signal" and physicality_mode != "off":
                physicality.fill(mass1, mass2, valid, baseline_gate, sample["generated_chi_mass_gev"])
            selected = baseline_gate & valid & region[:, config_index].astype(bool)
            selected_count += int(np.count_nonzero(selected))
            coordinates = [payload.suu_mass[selected, config_index]]
            if len(edges) == 2:
                coordinates.append((mass1[selected] + mass2[selected]) / 2)
            hist.add(weighted_histogram(np.column_stack(coordinates),
                                        scale * gen[selected] * correction[selected], edges,
                                        flow_policy=parameters["histogram_flow"]))
        identity = np.concatenate(identity_chunks)
        order = np.argsort(identity, order=identity.dtype.names)
        ordered_ids = identity[order].tobytes()
        ordered_gen = np.asarray(np.concatenate(generator_chunks)[order], dtype="<f8").tobytes()
        info = {"kind": sample["kind"], "normalization_scope": sample["normalization_scope"],
                "category": sample.get("category", "Uncategorized"),
                "normalization_factor": scale, "sum_gen_weights": total_gen_weights,
                "generated_events": total_events, "selected_events": selected_count,
                "event_identity_sha256": hashlib.sha256(ordered_ids).hexdigest(),
                "event_generator_sha256": hashlib.sha256(ordered_ids + ordered_gen).hexdigest(),
                "histogram": hist.as_dict(), "files": list(sample["files"])}
        if sample["kind"] == "signal":
            info["physicality"] = physicality.result() if physicality_mode != "off" else None
            signals[sample["name"]] = (hist, info["physicality"])
        else:
            background.add(hist)
        samples[sample["name"]] = info
    per_signal, failures = {}, []
    for name in campaign["required_signals"]:
        hist, physicality = signals[name]
        result = asimov_significance(hist.sumw, background.sumw, background.sumw2,
                                     campaign.get("minimum_background_effective_events", 10))
        if physicality_mode == "gate" and not physicality["physicality_pass"]:
            result["feasible"] = False
            result["failure_reasons"].append("physicality_gate_failed")
        result["physicality"] = physicality
        result["signal_yield"] = float(hist.sumw.sum())
        result["signal_sumw2"] = float(hist.sumw2.sum())
        per_signal[name] = result
        failures.extend(f"{name}:{reason}" for reason in result["failure_reasons"])
    feasible = not failures
    comparison_samples = {
        sample["name"]: {
            "kind": sample["kind"], "normalization_scope": sample["normalization_scope"],
            "cross_section_pb": sample["cross_section_pb"],
            "filter_efficiency": sample.get("filter_efficiency", 1), "k_factor": sample.get("k_factor", 1),
            "generated_chi_mass_gev": sample.get("generated_chi_mass_gev"),
            "generated_suu_mass_gev": sample.get("generated_suu_mass_gev"),
            "decay_channel": sample.get("decay_channel", sample.get("decay_mode")), "dataset": sample.get("dataset"),
            "category": sample.get("category"),
            "sum_gen_weights": samples[sample["name"]]["sum_gen_weights"],
            "generated_events": samples[sample["name"]]["generated_events"],
            "event_identity_sha256": samples[sample["name"]]["event_identity_sha256"],
            "event_generator_sha256": samples[sample["name"]]["event_generator_sha256"],
            "input_files": sample.get("input_files"), "input_file_list": sample.get("input_file_list"),
        } for sample in campaign["samples"]
    }
    context = comparison_context(parameters, comparison_samples, campaign["required_signals"])
    aggregate = aggregate_objective(per_signal, campaign["required_signals"], campaign.get("objective"), context=context)
    feasible = feasible and aggregate["feasible"]
    from evaluate_reference import background_coverage
    coverage = background_coverage(samples)
    return {"schema_version": 2, "configuration": key.as_dict(), "configuration_slug": configuration_slug(key),
            **aggregate,
            "direction": "maximize", "feasible": feasible, "failure_reasons": failures,
            "purpose": campaign.get("purpose", "production"),
            "production_ready": campaign.get("purpose", "production") == "production" and feasible,
            "required_signals": list(campaign["required_signals"]), "per_signal": per_signal,
            "comparison_samples": comparison_samples,
            "background_coverage": coverage,
            "background": background.as_dict(), "samples": samples,
            "comparison_context": context, "parameters": parameters}


def finite_json(value: Any) -> Any:
    """Write standards-compliant JSON: unavailable diagnostics use null."""
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_outputs(result: Mapping[str, Any], output_dir: str | Path,
                  comparison_results: Sequence[Mapping[str, Any]] = ()) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "objective.json").write_text(json.dumps(finite_json(result), indent=2, allow_nan=False) + "\n")
    ranking = relative_regret_rankings([*comparison_results, result], result["required_signals"],
                                      mean_weight=result.get("objective_definition", {}).get("mean_weight", .25))
    (output / "sensitivity_ranking.json").write_text(json.dumps(finite_json(ranking), indent=2, allow_nan=False) + "\n")
    # Preserve the small-campaign projection; 114 overlaid hypotheses and a
    # 114-entry legend obscure it. Full 2D campaigns retain every separate map.
    if len(result["required_signals"]) <= 12 or result["parameters"].get("chi_mass_bin_edges_gev") is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        edges = np.asarray(result["parameters"]["mass_bin_edges_gev"])
        b = np.asarray(result["background"]["sumw"])
        variance = np.asarray(result["background"]["sumw2"])
        if b.ndim == 2:
            b, variance = b.sum(axis=1), variance.sum(axis=1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.stairs(b, edges, label="Total background", color="black")
        ax.errorbar((edges[:-1] + edges[1:]) / 2, b, yerr=np.sqrt(variance), fmt="none", color="black", capsize=2)
        for name in result["required_signals"]:
            s = np.asarray(result["samples"][name]["histogram"]["sumw"])
            if s.ndim == 2:
                s = s.sum(axis=1)
            ax.stairs(s, edges, label=name)
        ax.set(xlabel="Reconstructed Suu mass [GeV]", ylabel="Expected events / bin")
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(output / "suu_mass.png", dpi=160)
        plt.close(fig)
    if result["parameters"].get("chi_mass_bin_edges_gev") is not None:
        from plot_sensitivity_2d import plot_2d_outputs
        plot_2d_outputs(result, output)
    rows = [(name, item["significance"], item["feasible"], (item.get("physicality") or {}).get("physicality_score"))
            for name, item in result["per_signal"].items()]
    with (output / "per_signal.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("signal", "significance", "feasible", "physicality_score"))
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--configuration", help="n:Tgate:Tkeep:RAK:RCA:c or configuration slug")
    source.add_argument("--trial-json", help="JSON containing a configuration object or token")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compare-objective", "--ranking-input", dest="compare_objective",
                        action="append", default=[],
                        help="Prior objective.json from the same campaign; repeat for sensitivity regrets")
    args = parser.parse_args(argv)
    try:
        campaign = load_campaign(args.campaign)
        raw = args.configuration
        if args.trial_json:
            with Path(args.trial_json).open() as stream:
                trial = json.load(stream)
            raw = trial["configuration"]
        key = parse_configuration(raw)
        result = evaluate_campaign(campaign, key)
        comparisons = []
        for path in args.compare_objective:
            with Path(path).open() as stream:
                item = json.load(stream)
            if (item.get("parameters") != finite_json(result["parameters"])
                    or item.get("required_signals") != result["required_signals"]
                    or item.get("purpose") != result["purpose"]
                    or item.get("objective_definition") != result["objective_definition"]
                    or item.get("comparison_samples") != result["comparison_samples"]):
                raise ValueError(f"{path}: comparisons require identical campaign objective and coverage")
            comparisons.append(item)
        write_outputs(result, args.output_dir, comparisons)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.exit(2, f"sensitivity evaluation failed: {error}\n")
    print(json.dumps({"objective": result["objective"], "feasible": result["feasible"],
                      "output": str(Path(args.output_dir) / "objective.json")}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
