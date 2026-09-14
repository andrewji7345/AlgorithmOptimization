#!/usr/bin/env python3
"""Evaluate the source-ported AN anchor and cached reconstruction grids together.

Uses referenceWeight (without the nominal top-pT factor) for every model.  Each
generated event remains in the normalization denominator, even if reconstruction
or selection fails.  Only fixed common mass bins are supported; no candidate can
gain sensitivity by changing its bin map or inserting an epsilon background.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import awkward as ak
import numpy as np

from compact_scan_metrics import (
    EVENTS_PATH, METADATA_PATH, VALID_STATUS, STATUS_NAMES, ConfigurationKey,
    PhysicalityDefinition, configuration_indices, configuration_key,
    configuration_slug, gate_mask, load_metadata, read_event_payload, _open_root_file,
)
from evaluate_sensitivity import _validate_metadata, finite_json, load_campaign, validate_campaign
from likelihood_model import CATEGORY_MAP, PROCESSES, REFERENCE_COMMIT
from sensitivity_metrics import PhysicalityAccumulator, WeightedHistogram, asimov_significance, normalization_factor, validated_edges, weighted_histogram

REFERENCE_NAME = "reference_AN2017"
REFERENCE_RECONSTRUCTION = "AN23-067-PATAK8-CA8-Thrust-source-port-v1"
REGIONS = {"SR": 1, "CR": 2, "AT1b": 3, "AT0b": 4}
REGION_BRANCHES = {"SR": "passesSignalRegion", "CR": "passesControlRegion", "AT1b": "passesAT1b", "AT0b": "passesAT0b"}
WEIGHT_NUISANCES = {
    "pileup": "CMS_pu", "prefiring": "CMS_L1Prefiring_2017",
    "btagHFCorrelated": "CMS_bTagSF_bc_M_corr", "btagHFUncorrelated": "CMS_bTagSF_bc_M_year17",
    "btagLFCorrelated": "CMS_bTagSF_light_M_corr", "btagLFUncorrelated": "CMS_bTagSF_light_M_year17",
    "topPt": "CMS_topPt_TTbar",
}
WEIGHT_NAMES = tuple(name + direction for name in WEIGHT_NUISANCES for direction in ("Up", "Down"))
KINEMATIC_NUISANCES = {"JEC": "CMS_jec_Total_2017", "JER": "CMS_jer_2017"}
MINIMUM_REFERENCE_NEFF = 1 / .175 ** 2
_ID_DTYPE = np.dtype([("run", "<u4"), ("lumi", "<u4"), ("event", "<u8")])


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(finite_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _campaign(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = copy.deepcopy(dict(value))
        validate_campaign(result)
        return result
    return load_campaign(value)


def _identity_array(payload: Any) -> np.ndarray:
    ids = np.empty(payload.n_events, dtype=_ID_DTYPE)
    for field in _ID_DTYPE.names:
        ids[field] = getattr(payload, field)
    return ids


def _identity_audit(ids: np.ndarray, weights: np.ndarray, source_files: Sequence[str]) -> dict[str, Any]:
    order = np.argsort(ids, order=_ID_DTYPE.names)
    ordered = ids[order]
    if len(ordered) > 1 and np.any(ordered[1:] == ordered[:-1]):
        raise ValueError("duplicate run/lumi/event within or across sample shards")
    identity_hash = hashlib.sha256(ordered.tobytes()).hexdigest()
    pair_hash = hashlib.sha256(ordered.tobytes() + np.asarray(weights[order], dtype="<f8").tobytes()).hexdigest()
    if len(source_files) != len(set(source_files)):
        raise ValueError("duplicate source input files")
    return {"events": len(ids), "event_ids_sha256": identity_hash, "event_genweights_sha256": pair_hash,
            "source_files": sorted(source_files), "source_files_sha256": _hash(sorted(source_files))}


def region_codes(baseline, valid, veto, btags, n300a, n300b, n50a, n50b, mass100a, mass100b):
    """Independent array implementation of the strict AN region boundaries."""
    arrays = np.broadcast_arrays(baseline, valid, veto, btags, n300a, n300b, n50a, n50b, mass100a, mass100b)
    baseline, valid, veto, btags, n300a, n300b, n50a, n50b, mass100a, mass100b = arrays
    tag_a, tag_b = n300a >= 2, n300b >= 2
    anti_a = (n50a == 0) & np.isfinite(mass100a) & (mass100a >= 0) & (mass100a < 150)
    anti_b = (n50b == 0) & np.isfinite(mass100b) & (mass100b >= 0) & (mass100b < 150)
    allowed = baseline.astype(bool) & valid.astype(bool) & veto.astype(bool)
    tagged = allowed & tag_a & tag_b
    anti = allowed & ~tagged & ((tag_a & anti_b) | (tag_b & anti_a))
    result = np.zeros(baseline.shape, dtype=np.uint8)
    result[tagged] = np.where(btags[tagged] > 0, 1, 2)
    result[anti] = np.where(btags[anti] > 0, 3, 4)
    return result


def _read_additive(meta: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_names = ("analysisObservableVersion", "analysisSystematic", "referenceReconstruction", "weightVariationNames")
    scalar_names = ("referenceWeight", "referenceRecoStatus", "referenceRegion", "referenceSJ1Mass", "referenceSJ2Mass", "referenceSuuMass",
                    "referenceSJ1NCA4E50", "referenceSJ2NCA4E50", "referenceSJ1NCA4E300", "referenceSJ2NCA4E300",
                    "referenceSJ1MassE100", "referenceSJ2MassE100", "analysisNBTags", "passesTrigger", "passesFilters",
                    "passesLeptonVeto", "passesJetVeto")
    vector_names = ("referenceWeightVariations", "passesControlRegion", "passesAT1b", "passesAT0b", "passesRecoJetVeto",
                    "sj1NCA4E50", "sj2NCA4E50", "sj1NCA4E300", "sj2NCA4E300", "sj1MassE100", "sj2MassE100")
    with _open_root_file(meta.path) as root:
        tree = root[METADATA_PATH]
        missing = set(metadata_names) - set(tree.keys())
        if missing or tree.num_entries != 1:
            raise ValueError(f"{meta.path}: missing additive metadata {sorted(missing)} or metadata entries != 1")
        fields = tree.arrays(metadata_names, library="ak")
        metadata = {name: ak.to_list(fields[name][0]) for name in metadata_names}
        event_tree = root[EVENTS_PATH]
        missing = set(scalar_names + vector_names) - set(event_tree.keys())
        if missing:
            raise ValueError(f"{meta.path}: missing additive event branches {sorted(missing)}")
        raw = event_tree.arrays(scalar_names + vector_names, library="ak")
        arrays = {}
        for name in scalar_names + vector_names:
            try:
                arrays[name] = np.asarray(ak.to_numpy(raw[name]))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{meta.path}: {name} is not a regular event array") from exc
    return metadata, arrays


def _validate_additive(meta: Any, payload: Any, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray], systematic: str) -> None:
    if metadata["analysisObservableVersion"] != 1 or metadata["referenceReconstruction"] != REFERENCE_RECONSTRUCTION:
        raise ValueError(f"{meta.path}: incompatible reference observable version/reconstruction")
    if metadata["analysisSystematic"] != systematic:
        raise ValueError(f"{meta.path}: analysisSystematic differs from requested campaign {systematic}")
    names = metadata["weightVariationNames"]
    if len(names) != len(set(names)) or set(names) != set(WEIGHT_NAMES):
        raise ValueError(f"{meta.path}: incomplete or duplicate weightVariationNames")
    n, width = payload.n_events, meta.n_configurations
    for name, values in arrays.items():
        expected = ((n, len(names)) if name == "referenceWeightVariations" else
                    (n, width) if name.startswith("sj") or name.startswith("passesControl") or name in ("passesAT1b", "passesAT0b", "passesRecoJetVeto") else (n,))
        if values.shape != expected:
            raise ValueError(f"{meta.path}: {name} shape {values.shape} != {expected}")
        if name.startswith("passes") and np.any((values != 0) & (values != 1)):
            raise ValueError(f"{meta.path}: nonboolean {name}")
        if name in ("referenceWeight", "referenceWeightVariations") and (np.any(~np.isfinite(values)) or np.any(values < 0)):
            raise ValueError(f"{meta.path}: nonfinite/negative {name}")
    if np.any(~np.isin(arrays["referenceRecoStatus"], np.arange(len(STATUS_NAMES)))):
        raise ValueError(f"{meta.path}: unknown reference reconstruction status")
    if np.any(~np.isin(payload.reco_status, np.arange(len(STATUS_NAMES)))):
        raise ValueError(f"{meta.path}: unknown hybrid reconstruction status")
    if np.any(~np.isin(arrays["referenceRegion"], np.arange(5))):
        raise ValueError(f"{meta.path}: invalid referenceRegion")
    if np.any(arrays["analysisNBTags"] < 0):
        raise ValueError(f"{meta.path}: negative btag count")
    for prefix in ("referenceSJ1", "referenceSJ2", "sj1", "sj2"):
        for suffix in ("NCA4E300", "NCA4E50"):
            counts = arrays[prefix + suffix]
            if np.any(counts < 0) or np.any(counts != np.floor(counts)) or np.any(~np.isfinite(counts)):
                raise ValueError(f"{meta.path}: invalid CA4 count {prefix + suffix}")
        if np.any(arrays[prefix + "NCA4E300"] > arrays[prefix + "NCA4E50"]):
            raise ValueError(f"{meta.path}: inconsistent nested CA4 energy counts")
    ref_valid = arrays["referenceRecoStatus"] == VALID_STATUS
    for name in ("referenceSJ1Mass", "referenceSJ2Mass", "referenceSuuMass"):
        values = arrays[name]
        if np.any(ref_valid & (~np.isfinite(values) | (values < 0))) or np.any(~ref_valid & ~np.isnan(values)):
            raise ValueError(f"{meta.path}: reference masses violate reconstruction status")
    hybrid_valid = payload.reco_status == VALID_STATUS
    for values in (payload.sj1_mass, payload.sj2_mass, payload.suu_mass):
        if np.any(hybrid_valid & (~np.isfinite(values) | (values < 0))) or np.any(~hybrid_valid & ~np.isnan(values)):
            raise ValueError(f"{meta.path}: hybrid masses violate reconstruction status")
    expected_reference = region_codes(payload.passes_baseline, ref_valid, True, arrays["analysisNBTags"],
                                     arrays["referenceSJ1NCA4E300"], arrays["referenceSJ2NCA4E300"],
                                     arrays["referenceSJ1NCA4E50"], arrays["referenceSJ2NCA4E50"],
                                     arrays["referenceSJ1MassE100"], arrays["referenceSJ2MassE100"])
    if np.any(expected_reference != arrays["referenceRegion"]):
        raise ValueError(f"{meta.path}: reference region disagrees with independently recomputed AN cuts")
    expected_hybrid = region_codes(payload.passes_baseline[:, None], payload.reco_status == VALID_STATUS,
                                  arrays["passesRecoJetVeto"], arrays["analysisNBTags"][:, None],
                                  arrays["sj1NCA4E300"], arrays["sj2NCA4E300"], arrays["sj1NCA4E50"], arrays["sj2NCA4E50"],
                                  arrays["sj1MassE100"], arrays["sj2MassE100"])
    for region, code in REGIONS.items():
        actual = payload.passes_signal_region if region == "SR" else arrays[REGION_BRANCHES[region]]
        if np.any(actual.astype(bool) != (expected_hybrid == code)):
            raise ValueError(f"{meta.path}: {region} flag disagrees with independently recomputed AN cuts")


def _default_keys(metadata: Any) -> tuple[ConfigurationKey, ...]:
    return tuple(configuration_key(0, None, pt, metadata.ak_radius, ca, cos)
                 for pt, ca, cos in zip(metadata.config_collection_pt_cut, metadata.config_ca_radius, metadata.config_cos_thrust))


def _validate_same_campaign(nominal: Mapping[str, Any], varied: Mapping[str, Any], *, same_sources: bool = True) -> None:
    for name in ("luminosity_pb", "required_signals", "analysis_selection", "correction_prescription", "physicality", "purpose", "mass_bin_edges_gev"):
        if nominal.get(name) != varied.get(name):
            raise ValueError(f"systematic campaign differs in {name}")
    nominal_samples = {s["name"]: s for s in nominal["samples"]}
    varied_samples = {s["name"]: s for s in varied["samples"]}
    if set(nominal_samples) != set(varied_samples):
        raise ValueError("systematic campaign sample coverage differs")
    for name, sample in nominal_samples.items():
        other = varied_samples[name]
        for field in ("kind", "category", "dataset", "cross_section_pb", "filter_efficiency", "k_factor", "normalization_scope",
                      "sum_gen_weights", "generated_events", "generated_chi_mass_gev", "generated_suu_mass_gev"):
            if sample.get(field) != other.get(field):
                raise ValueError(f"{name}: systematic campaign differs in {field}")
        if same_sources and (not sample.get("input_files") or sorted(sample["input_files"]) != sorted(other.get("input_files", []))):
            raise ValueError(f"{name}: systematic source input files are missing or differ")


def _evaluate_one(campaign: Mapping[str, Any], *, systematic: str, region: str, edges: np.ndarray,
                  keys: Sequence[ConfigurationKey] | None, include_reference: bool) -> dict[str, Any]:
    definition = PhysicalityDefinition(**campaign.get("physicality", {}))
    model_keys = None
    samples_out, audits, metadata_audits = {}, {}, {}
    for sample in campaign["samples"]:
        metadata = [load_metadata(path) for path in sample["files"]]
        for meta in metadata:
            _validate_metadata(meta, sample, campaign)
        if not metadata:
            raise ValueError(f"{sample['name']}: no shards")
        chosen = tuple(keys) if keys is not None else _default_keys(metadata[0])
        slugs = [configuration_slug(key) for key in chosen]
        if len(slugs) != len(set(slugs)):
            raise ValueError("duplicate hybrid configuration keys")
        wanted = {slug: key for slug, key in zip(slugs, chosen)}
        if include_reference:
            wanted = {REFERENCE_NAME: None, **wanted}
        if not wanted:
            raise ValueError("at least one reconstruction model is required")
        if model_keys is not None and model_keys != wanted:
            raise ValueError("nominal sample grids do not contain the same configuration keys")
        model_keys = wanted
        for meta in metadata:
            if keys is None and _default_keys(meta) != chosen:
                raise ValueError(f"{meta.path}: inconsistent reconstruction grid")
            for key in chosen:
                configuration_indices(meta, key)
        n_events = sum(meta.processed_events for meta in metadata)
        gen_total = math.fsum(meta.sum_weights for meta in metadata)
        denominator = gen_total if sample["sum_gen_weights"] == "metadata" else sample["sum_gen_weights"]
        expected_events = n_events if sample["generated_events"] == "metadata" else sample["generated_events"]
        if n_events <= 0 or n_events != expected_events or not math.isclose(gen_total, denominator, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"{sample['name']}: incomplete generated coverage")
        scale = normalization_factor(sample["cross_section_pb"], campaign["luminosity_pb"], denominator,
                                     sample.get("filter_efficiency", 1), sample.get("k_factor", 1))
        records = {name: {"histogram": WeightedHistogram.empty([edges]),
                          "weight_variations": {n: WeightedHistogram.empty([edges]) for n in WEIGHT_NAMES},
                          "physicality": PhysicalityAccumulator.empty(definition), "cutflow": {}, "region_counts": {r: 0 for r in REGIONS}}
                   for name in wanted}
        ids, gen_weights = [], []
        for meta in metadata:
            payload = read_event_payload(meta)
            extra_meta, arrays = _read_additive(meta)
            _validate_additive(meta, payload, extra_meta, arrays, systematic)
            if payload.n_events != meta.processed_events:
                raise ValueError(f"{meta.path}: incomplete preselection event tree")
            gen = np.asarray(payload.gen_weight, dtype=float)
            if gen.shape != (payload.n_events,) or np.any(~np.isfinite(gen)):
                raise ValueError(f"{meta.path}: invalid generator weights")
            tolerance = 2e-6 * max(float(np.abs(gen).sum()), 1.)
            if abs(float(gen.sum()) - meta.sum_weights) > tolerance or not math.isclose(float(np.square(gen).sum()), meta.sum_weights2, rel_tol=4e-6, abs_tol=1e-7):
                raise ValueError(f"{meta.path}: all-event generator sum/sumw2 disagrees with metadata")
            ids.append(_identity_array(payload)); gen_weights.append(gen)
            weights = scale * gen * arrays["referenceWeight"]
            varied_weights = scale * gen[:, None] * arrays["referenceWeightVariations"]
            weight_index = {name: i for i, name in enumerate(extra_meta["weightVariationNames"])}
            for model_name, key in wanted.items():
                record = records[model_name]
                if key is None:
                    valid = arrays["referenceRecoStatus"] == VALID_STATUS
                    mass1, mass2, mass = (arrays[name] for name in ("referenceSJ1Mass", "referenceSJ2Mass", "referenceSuuMass"))
                    gate = np.ones(payload.n_events, dtype=bool)
                    regions = {r: arrays["referenceRegion"] == code for r, code in REGIONS.items()}
                else:
                    index, _ = configuration_indices(meta, key)
                    valid = payload.reco_status[:, index] == VALID_STATUS
                    mass1, mass2, mass = payload.sj1_mass[:, index], payload.sj2_mass[:, index], payload.suu_mass[:, index]
                    gate = gate_mask(payload, key.n_gate_jets, key.gate_pt_cut)
                    regions = {r: ((payload.passes_signal_region if r == "SR" else arrays[REGION_BRANCHES[r]])[:, index]).astype(bool) for r in REGIONS}
                base_gate = payload.passes_baseline.astype(bool) & gate
                selected = base_gate & valid & regions[region]
                if sample["kind"] == "signal":
                    record["physicality"].fill(mass1, mass2, valid, base_gate, sample["generated_chi_mass_gev"])
                record["histogram"].add(weighted_histogram(mass[selected], weights[selected], [edges]))
                if systematic == "nominal":
                    for name, index in weight_index.items():
                        record["weight_variations"][name].add(weighted_histogram(mass[selected], varied_weights[selected, index], [edges]))
                cumulative = np.ones(payload.n_events, dtype=bool)
                masks = [("processed", cumulative.copy())]
                for stage, branch in (("trigger", "passesTrigger"), ("filters", "passesFilters"), ("lepton_veto", "passesLeptonVeto"), ("jet_veto", "passesJetVeto")):
                    cumulative = cumulative & arrays[branch].astype(bool)
                    masks.append((stage, cumulative.copy()))
                masks.extend((("baseline", payload.passes_baseline.astype(bool)), ("baseline_and_gate", base_gate),
                              ("valid_reconstruction", base_gate & valid), ("selected_region", selected)))
                for stage, mask in masks:
                    count = record["cutflow"].setdefault(stage, {"events": 0, "sum_gen_weights": 0., "yield": 0., "sumw2": 0.})
                    count["events"] += int(np.count_nonzero(mask))
                    count["sum_gen_weights"] += float(gen[mask].sum())
                    count["yield"] += float(weights[mask].sum())
                    count["sumw2"] += float(np.square(weights[mask]).sum())
                for name, mask in regions.items():
                    record["region_counts"][name] += int(np.count_nonzero(base_gate & valid & mask))
        identity = _identity_audit(np.concatenate(ids), np.concatenate(gen_weights), sample.get("input_files", []))
        audits[sample["name"]] = identity
        metadata_audits[sample["name"]] = {"analysis_systematic": systematic, "reference_reconstruction": REFERENCE_RECONSTRUCTION,
                                          "observable_version": 1, "weight_variations": sorted(WEIGHT_NAMES),
                                          "files": list(sample["files"]), "grid": slugs}
        for model_name, record in records.items():
            info = {"kind": sample["kind"], "category": sample.get("category"), "normalization_scope": sample["normalization_scope"],
                    "normalization_factor": scale, "sum_gen_weights": gen_total, "generated_events": n_events,
                    "selected_events": record["cutflow"]["selected_region"]["events"], "histogram": record["histogram"].as_dict(),
                    "weight_variations": {name: h.as_dict() for name, h in record["weight_variations"].items()} if systematic == "nominal" else {},
                    "cutflow": record["cutflow"], "region_counts": record["region_counts"], "files": list(sample["files"])}
            if sample["kind"] == "signal":
                info["physicality"] = record["physicality"].result()
            samples_out.setdefault(model_name, {})[sample["name"]] = info
    return {"models": samples_out, "keys": {name: key.as_dict() if key else None for name, key in model_keys.items()},
            "identity_audit": audits, "metadata_audit": metadata_audits}


def _process_histograms(samples: Mapping[str, Any], catalog: Mapping[str, Any], signal: str, n_bins: int,
                        variation: str | None = None) -> dict[str, Any]:
    result = {p: {"sumw": np.zeros(n_bins), "sumw2": np.zeros(n_bins), "samples": []} for p in PROCESSES}
    for name, info in samples.items():
        if info["kind"] == "signal" and name != signal:
            continue
        process = "signal" if name == signal else CATEGORY_MAP.get(catalog[name].get("category"))
        if process is None:
            raise ValueError(f"{name}: unknown background category {catalog[name].get('category')}")
        # top-pT is only a ttbar nuisance, even though every event stores the slot.
        hist = info["histogram"] if variation is None or (variation.startswith("topPt") and process != "TTbar") else info["weight_variations"][variation]
        for key in ("sumw", "sumw2"):
            result[process][key] += np.asarray(hist[key], dtype=float)
        result[process]["samples"].append(name)
    for data in result.values():
        for key in ("sumw", "sumw2"):
            data[key] = data[key].tolist()
    return result


def background_coverage(samples: Mapping[str, Any]) -> dict[str, Any]:
    """Describe measured MC support without interpreting empty components as absent.

    Total-background n_eff cannot detect an important component that has never
    survived selection. Keep its exposure and empty bins visible separately.
    A zero sumw with positive sumw2 is signed cancellation, not an empty sample.
    """
    details = {}
    for name, info in samples.items():
        if info["kind"] != "background":
            continue
        yields = np.asarray(info["histogram"]["sumw"], dtype=float)
        variances = np.asarray(info["histogram"]["sumw2"], dtype=float)
        total, variance = float(yields.sum()), float(variances.sum())
        selected = info["selected_events"]
        details[name] = {
            "generated_events": info["generated_events"], "selected_events": selected,
            "yield_in_fit_bins": total, "mc_variance_in_fit_bins": variance,
            "effective_events_in_fit_bins": total ** 2 / variance if variance > 0 else None,
            "empty_fit_bins": np.flatnonzero((yields == 0) & (variances == 0)).tolist(),
            "canceled_fit_bins": np.flatnonzero((yields == 0) & (variances > 0)).tolist(),
            "status": "unobserved_after_selection" if selected == 0 else
                      "unobserved_in_fit_bins" if not np.any(variances > 0) else "observed",
        }
    unobserved = [name for name, detail in details.items() if detail["status"] != "observed"]
    return {"samples": details, "unobserved_samples": unobserved,
            "unobserved_components_require_review": bool(unobserved),
            "absence_of_unobserved_samples_proves_completeness": False,
            "interpretation": "MC support is conditional on observed components. Zero selected MC gives neither a zero physical rate nor a calibrated uncertainty on the missing contribution; no yield floor or invented upper bound is applied."}


def evaluate_reference_campaign(campaignpath: Any, variation_campaigns: Mapping[str, Any] | None = None,
                                *, region: str = "SR", bin_edges: Sequence[float] | None = None,
                                qcd_groups: list[list[int]] | None = None, keys: Sequence[ConfigurationKey] | None = None,
                                include_reference: bool = True) -> dict[str, Any]:
    """Return builder inputs, fast scores and audit information for every model.

    ``variation_campaigns`` maps JECUp/Down and/or JERUp/Down to complete paired
    campaigns.  Nominal weight nuisances are always read from additive branches.
    Default keys are the stored ungated grid, avoiding equivalent gate aliases.
    """
    nominal = _campaign(campaignpath)
    if region not in REGIONS:
        raise ValueError("region must be SR, CR, AT1b or AT0b")
    if nominal.get("analysis_systematic", "nominal") != "nominal":
        raise ValueError("the central reference campaign must be nominal")
    edges = validated_edges(bin_edges if bin_edges is not None else nominal["mass_bin_edges_gev"], "common Suu bins")
    n_bins = len(edges) - 1
    if qcd_groups is None:
        qcd_groups = [[0, 1], [2, 3]] if n_bins == 4 else [[i] for i in range(n_bins)]
    flat = [i for group in qcd_groups for i in group]
    if any(not group for group in qcd_groups) or any(type(i) is not int for i in flat) or sorted(flat) != list(range(n_bins)):
        raise ValueError("qcd_groups must partition the fixed common bins exactly once")
    variations = dict(variation_campaigns or {})
    if set(variations) - {source + direction for source in KINEMATIC_NUISANCES for direction in ("Up", "Down")}:
        raise ValueError("unsupported kinematic variation campaign name")
    for source in KINEMATIC_NUISANCES:
        if (source + "Up" in variations) != (source + "Down" in variations):
            raise ValueError(f"{source} requires paired Up and Down campaigns")
    central = _evaluate_one(nominal, systematic="nominal", region=region, edges=edges, keys=keys, include_reference=include_reference)
    shifted, variation_hashes = {}, {}
    for name, path in variations.items():
        campaign = _campaign(path)
        variation_hashes[name] = _hash(campaign)
        _validate_same_campaign(nominal, campaign)
        if campaign.get("analysis_systematic", name) != name:
            raise ValueError(f"{name}: campaign declares a different systematic")
        shifted[name] = _evaluate_one(campaign, systematic=name, region=region, edges=edges, keys=keys, include_reference=include_reference)
        if shifted[name]["keys"] != central["keys"] or shifted[name]["identity_audit"] != central["identity_audit"]:
            raise ValueError(f"{name}: systematic source files/event identities/generator weights or grid differ from nominal")
    catalog = {sample["name"]: sample for sample in nominal["samples"]}
    minimum = max(MINIMUM_REFERENCE_NEFF, float(nominal.get("minimum_background_effective_events", 10)))
    result = {"schema_version": 1, "purpose": "representative_likelihood_comparison", "region": region,
              "expected_only": True, "production_ready": False, "models": {}, "campaign_sha256": _hash(nominal),
              "campaign_path": nominal.get("_campaign_path"), "mass_bin_edges_gev": edges.tolist(), "qcd_groups": qcd_groups,
              "minimum_background_effective_events": minimum, "required_signals": nominal["required_signals"],
              "identity_audit": central["identity_audit"], "metadata_audit": central["metadata_audit"],
              "kinematic_variations": list(shifted), "missing_kinematic_variations": [source for source in KINEMATIC_NUISANCES if source + "Up" not in shifted],
              "weight_convention": "referenceWeight for source-ported anchor and all hybrid models; generator weight multiplied once",
              "physicality_population": "baseline and optional trial gate before region selection",
              "binning_scope": "fixed common one-dimensional Suu bins; not the AN two-dimensional map"}
    for model_name, samples in central["models"].items():
        model = {"configuration": central["keys"][model_name], "samples": samples, "per_signal": {}, "feasible": True,
                 "background_coverage": background_coverage(samples),
                 "model_type": "source_ported_AN" if model_name == REFERENCE_NAME else "hybrid_reconstruction"}
        for signal in nominal["required_signals"]:
            processes = _process_histograms(samples, catalog, signal, n_bins)
            for source, nuisance in WEIGHT_NUISANCES.items():
                changed = {direction.lower(): _process_histograms(samples, catalog, signal, n_bins, source + direction) for direction in ("Up", "Down")}
                for process in PROCESSES:
                    if source == "topPt" and process != "TTbar":
                        continue
                    processes[process].setdefault("variations", {})[nuisance] = {direction: changed[direction][process]["sumw"] for direction in ("up", "down")}
            for source, nuisance in KINEMATIC_NUISANCES.items():
                if source + "Up" not in shifted:
                    continue
                changed = {direction.lower(): _process_histograms(shifted[source + direction]["models"][model_name], catalog, signal, n_bins) for direction in ("Up", "Down")}
                for process in PROCESSES:
                    processes[process].setdefault("variations", {})[nuisance] = {direction: changed[direction][process]["sumw"] for direction in ("up", "down")}
            background = np.sum([processes[p]["sumw"] for p in PROCESSES[1:]], axis=0)
            variance = np.sum([processes[p]["sumw2"] for p in PROCESSES[1:]], axis=0)
            fast = asimov_significance(np.asarray(processes["signal"]["sumw"]), background, variance, minimum)
            diagnostic_fast = None
            if not fast["feasible"] and set(fast["failure_reasons"]) == {"insufficient_background_effective_events"}:
                diagnostic_fast = asimov_significance(np.asarray(processes["signal"]["sumw"]), background, variance, 1e-12)
                diagnostic_fast["diagnostic_only"] = True
                diagnostic_fast["relaxed_background_effective_events_threshold"] = 1e-12
            physicality = samples[signal]["physicality"]
            reasons = list(fast["failure_reasons"])
            if not physicality["physicality_pass"]:
                reasons.append("physicality_gate_failed")
            if sum(processes["signal"]["sumw"]) <= 0:
                reasons.append("zero_signal_yield")
            if any(np.any(np.asarray(p["sumw"]) < 0) or np.any((np.asarray(p["sumw"]) == 0) & (np.asarray(p["sumw2"]) > 0)) for p in processes.values()):
                reasons.append("invalid_signed_process_template")
            normalization = {name: {key: info.get(key) for key in ("normalization_scope", "normalization_factor", "sum_gen_weights", "generated_events", "files")} for name, info in samples.items()}
            builder = {"schema_version": 1,
                       "provenance": {"input_kind": "evaluated_reference_ntuples", "reference_commit": REFERENCE_COMMIT,
                                      "reference_reconstruction": REFERENCE_RECONSTRUCTION, "model_name": model_name,
                                      "configuration": central["keys"][model_name], "signal": signal,
                                      "signal_cross_section_pb": catalog[signal]["cross_section_pb"],
                                      "cross_section_definition": "catalog production times specified final-state branching fractions",
                                      "analysis_selection": nominal["analysis_selection"], "correction_prescription": nominal["correction_prescription"],
                                      "campaign_sha256": result["campaign_sha256"], "source_purpose": nominal["purpose"],
                                      "reference_weight_convention": result["weight_convention"], "physicality": physicality,
                                      "sample_normalization": normalization, "identity_audit": central["identity_audit"],
                                      "background_coverage": model["background_coverage"],
                                      "kinematic_variation_campaigns": variation_hashes,
                                      "reference_audits": {"original_selection": model_name == REFERENCE_NAME and nominal.get("reference_audits", {}).get("original_selection") is True,
                                                           "normalization": True,
                                                           "systematic_templates": False, "bin_map_and_groups": False, "control_region_closure": False}},
                       "channels": {region: {"bin_edges": edges.tolist(), "bin_axes_gev": [edges.tolist()], "original_shape": [n_bins],
                                             "qcd_groups": qcd_groups, "processes": processes}}}
            model["per_signal"][signal] = {"input": builder, "fast_score": fast, "diagnostic_fast_score": diagnostic_fast,
                                            "physicality": physicality,
                                            "feasible": not reasons, "failure_reasons": reasons,
                                            "signal_yield": float(sum(processes["signal"]["sumw"])), "background_yield": float(background.sum())}
            model["feasible"] &= not reasons
        zs = [entry["fast_score"]["significance"] for entry in model["per_signal"].values()]
        model["fast_mean_significance"] = float(np.mean(zs)) if all(z is not None for z in zs) else None
        result["models"][model_name] = model
    return finite_json(result)


def validate_disjoint_folds(development_campaign: Any, validation_campaign: Any) -> dict[str, Any]:
    """Verify both input-file separation and event-ID disjointness per MC sample."""
    development, validation = _campaign(development_campaign), _campaign(validation_campaign)
    _validate_same_campaign(development, validation, same_sources=False)
    first = {s["name"]: s for s in development["samples"]}
    second = {s["name"]: s for s in validation["samples"]}
    if set(first) != set(second):
        raise ValueError("development and validation sample coverage differs")
    result = {}
    for name, sample in first.items():
        sources1, sources2 = set(sample.get("input_files", [])), set(second[name].get("input_files", []))
        if not sources1 or not sources2:
            raise ValueError(f"{name}: source input files are required for fold audit")
        if sources1 & sources2 or set(sample["files"]) & set(second[name]["files"]):
            raise ValueError(f"{name}: development/validation source files overlap")
        identity_sets = []
        for entry in (sample, second[name]):
            arrays = []
            for path in entry["files"]:
                with _open_root_file(path) as root:
                    raw = root[EVENTS_PATH].arrays(["run", "lumi", "event"], library="np")
                ids = np.empty(len(raw["event"]), dtype=_ID_DTYPE)
                for key in _ID_DTYPE.names:
                    ids[key] = raw[key]
                arrays.append(ids)
            joined = np.concatenate(arrays)
            unique = np.unique(joined)
            if len(unique) != len(joined):
                raise ValueError(f"{name}: duplicate events inside a fold")
            identity_sets.append(unique)
        overlap = np.intersect1d(*identity_sets, assume_unique=True)
        if len(overlap):
            raise ValueError(f"{name}: development/validation event IDs overlap ({len(overlap)})")
        result[name] = {"development_events": len(identity_sets[0]), "validation_events": len(identity_sets[1]), "overlap_events": 0,
                        "development_sources": sorted(sources1), "validation_sources": sorted(sources2)}
    return {"disjoint": True, "samples": result}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--variation", action="append", default=[], metavar="NAME=CAMPAIGN")
    parser.add_argument("--region", choices=tuple(REGIONS), default="SR")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        pairs = [item.split("=", 1) for item in args.variation]
        if any(len(pair) != 2 for pair in pairs) or len(dict(pairs)) != len(pairs):
            raise ValueError("variations must have unique NAME=CAMPAIGN entries")
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        result = evaluate_reference_campaign(args.campaign, dict(pairs), region=args.region)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(output.resolve()), "models": len(result["models"]),
                          "feasible_models": sum(model["feasible"] for model in result["models"].values())}))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
