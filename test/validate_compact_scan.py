#!/usr/bin/env python3
"""Validate a compact ExistingOptimization ROOT output.

Run after ``cmsenv``. This checks storage/schema invariants only; physics
scoring belongs in the evaluator. Exit zero means all checked invariants pass.
Unexpected branches warn unless ``--strict-branches`` is supplied.
"""

from __future__ import print_function

import argparse
import math
import os
import sys
from collections import Counter

METADATA_PATH = "compactScan/Metadata"
EVENTS_PATH = "compactScan/Events"
SUPPORTED_SCHEMA_VERSIONS = {1}
STATUS_NAMES = (
    "valid",
    "no_selected_jets",
    "no_selected_constituents",
    "invalid_com",
    "invalid_thrust",
    "no_ca_jets",
    "complexity_guard",
    "no_valid_partition",
    "numerical_failure",
)
VALID_STATUS = 0
NO_CA_JETS_STATUS = 5
COMPLEXITY_GUARD_STATUS = 6

METADATA_BRANCHES = {
    "schemaVersion", "sampleName", "akRadius", "maxAmbiguousCAJets",
    "puppiWeighted", "useJEC", "caAlgorithm", "massObjective",
    "gateRecoConstraint", "enforceLegacyRadiusConstraint", "collectionPtCuts",
    "caRadii", "cosThrustCuts", "defaultGateJetCounts", "defaultGatePtCuts",
    "statusCodes", "statusNames", "processedEvents", "sumWeights",
    "sumWeights2", "baseCollectionPtCut", "baseCaRadius", "configId",
    "configBaseIndex", "configCollectionPtCut", "configCaRadius",
    "configCosThrust",
}
EVENT_BRANCHES = {
    "run", "lumi", "event", "genWeight", "akJetPt", "nCAJets",
    "recoStatus", "nAmbiguous", "sj1Mass", "sj2Mass",
}


class Findings(object):
    """Collect bounded diagnostics while retaining true issue counts."""

    def __init__(self, max_details=50):
        self.max_details = max_details
        self.errors = []
        self.warnings = []
        self.error_count = 0
        self.warning_count = 0

    def error(self, message):
        self.error_count += 1
        if len(self.errors) < self.max_details:
            self.errors.append(str(message))

    def warning(self, message):
        self.warning_count += 1
        if len(self.warnings) < self.max_details:
            self.warnings.append(str(message))

    def print_messages(self):
        for message in self.errors:
            print("ERROR: {}".format(message), file=sys.stderr)
        if self.error_count > len(self.errors):
            print("ERROR: {} additional error(s) omitted".format(
                self.error_count - len(self.errors)), file=sys.stderr)
        for message in self.warnings:
            print("WARNING: {}".format(message), file=sys.stderr)
        if self.warning_count > len(self.warnings):
            print("WARNING: {} additional warning(s) omitted".format(
                self.warning_count - len(self.warnings)), file=sys.stderr)


def _normalise_cpp_type(type_name):
    return str(type_name).replace("std::", "").replace(" ", "").lower()


def _branch_type(tree, name):
    branch = tree.GetBranch(name)
    if not branch:
        return ""
    class_name = str(branch.GetClassName())
    if class_name:
        return _normalise_cpp_type(class_name)
    leaf = branch.GetLeaf(name)
    if not leaf:
        leaves = branch.GetListOfLeaves()
        leaf = leaves.At(0) if leaves and leaves.GetEntries() == 1 else None
    return _normalise_cpp_type(leaf.GetTypeName()) if leaf else ""


def _check_type(tree, name, allowed, findings):
    actual = _branch_type(tree, name)
    allowed = {_normalise_cpp_type(value) for value in allowed}
    if actual not in allowed:
        findings.error("{} branch {!r} has type {!r}; expected one of {}".format(
            tree.GetName(), name, actual or "<unknown>", sorted(allowed)))


def _branch_names(tree):
    branches = tree.GetListOfBranches()
    return {str(branch.GetName()) for branch in branches} if branches else set()


def _check_branch_set(tree, required, strict, findings):
    actual = _branch_names(tree)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing:
        findings.error("{} is missing required branch(es): {}".format(
            tree.GetName(), ", ".join(missing)))
    if extra:
        message = "{} has unexpected branch(es): {}".format(
            tree.GetName(), ", ".join(extra))
        (findings.error if strict else findings.warning)(message)
    return not missing


def _scalar(tree, name):
    value = getattr(tree, name)
    try:
        return value.item()
    except AttributeError:
        return value


def _vector(tree, name, converter):
    values = getattr(tree, name)
    if converter is int:
        return [_as_int(value) for value in values]
    return [converter(value) for value in values]


def _as_int(value):
    # Legacy PyROOT exposes std::vector<unsigned char> elements as one-byte
    # Python strings, while newer cppyy releases expose ordinary integers.
    if isinstance(value, str):
        if len(value) != 1:
            raise ValueError("cannot convert multi-character value to integer")
        return ord(value)
    if isinstance(value, bytes):
        if len(value) != 1:
            raise ValueError("cannot convert multi-byte value to integer")
        return value[0]
    return int(value)


def _is_finite(value):
    return math.isfinite(float(value))


def _float_key(value):
    return round(float(value), 7)


def _same_float(left, right):
    return math.isclose(float(left), float(right), rel_tol=1.0e-6, abs_tol=1.0e-6)


def _format_bytes(size):
    value = float(size)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return "{:.2f} {}".format(value, unit)
        value /= 1024.0
    return "{:.2f} TiB".format(value)


def _tree_bytes(tree):
    try:
        return int(tree.GetZipBytes()), int(tree.GetTotBytes())
    except Exception:
        return 0, 0


def _largest_branches(tree, limit=8):
    sizes = []
    branches = tree.GetListOfBranches()
    if not branches:
        return sizes
    for branch in branches:
        try:
            sizes.append((int(branch.GetZipBytes()), str(branch.GetName())))
        except Exception:
            continue
    return sorted(sizes, reverse=True)[:limit]


def _check_tree_types(metadata, events, findings):
    scalar_unsigned = {
        "uint_t", "ulong_t", "ulong64_t", "unsignedint", "unsignedlong",
        "unsignedlonglong",
    }
    scalar_float = {"float_t", "double_t", "float", "double"}
    scalar_bool = {"bool_t", "bool"}
    string_types = {"string", "basic_string<char>"}
    vector_float = {"vector<float>", "vector<double>"}
    vector_u8 = {"vector<unsignedchar>", "vector<uchar_t>", "vector<uint8_t>"}
    vector_u16 = {
        "vector<unsignedshort>", "vector<ushort_t>", "vector<uint16_t>"
    }
    vector_u32 = {"vector<unsignedint>", "vector<uint_t>", "vector<uint32_t>"}
    vector_string = {"vector<string>", "vector<basic_string<char>>"}

    for name in ("schemaVersion", "maxAmbiguousCAJets", "processedEvents"):
        _check_type(metadata, name, scalar_unsigned, findings)
    for name in ("akRadius", "sumWeights", "sumWeights2"):
        _check_type(metadata, name, scalar_float, findings)
    for name in ("puppiWeighted", "useJEC", "enforceLegacyRadiusConstraint"):
        _check_type(metadata, name, scalar_bool, findings)
    for name in ("sampleName", "caAlgorithm", "massObjective", "gateRecoConstraint"):
        _check_type(metadata, name, string_types, findings)
    for name in (
        "collectionPtCuts", "caRadii", "cosThrustCuts", "defaultGatePtCuts",
        "baseCollectionPtCut", "baseCaRadius", "configCollectionPtCut",
        "configCaRadius", "configCosThrust",
    ):
        _check_type(metadata, name, vector_float, findings)
    _check_type(metadata, "defaultGateJetCounts", vector_u16, findings)
    for name in ("configId", "configBaseIndex"):
        _check_type(metadata, name, vector_u32, findings)
    _check_type(metadata, "statusCodes", vector_u8 | vector_u32, findings)
    _check_type(metadata, "statusNames", vector_string, findings)

    for name in ("run", "lumi"):
        _check_type(events, name, {"uint_t", "unsignedint"}, findings)
    _check_type(events, "event", {
        "ulong_t", "ulong64_t", "unsignedlong", "unsignedlonglong"
    }, findings)
    _check_type(events, "genWeight", {"float_t", "float"}, findings)
    for name in ("akJetPt", "sj1Mass", "sj2Mass"):
        _check_type(events, name, {"vector<float>"}, findings)
    _check_type(events, "nCAJets", vector_u16, findings)
    _check_type(events, "recoStatus", vector_u8, findings)
    _check_type(events, "nAmbiguous", vector_u16, findings)


def _validate_metadata_header(metadata, event_entries, expected_max, findings):
    """Validate scalar metadata and return values needed by later checks."""
    if int(metadata.GetEntries()) != 1:
        findings.error("{} must contain exactly one entry, found {}".format(
            METADATA_PATH, int(metadata.GetEntries())))
        return None
    if metadata.GetEntry(0) <= 0:
        findings.error("could not read the sole {} entry".format(METADATA_PATH))
        return None

    schema_version = int(_scalar(metadata, "schemaVersion"))
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        findings.error("unsupported schemaVersion {}; supported: {}".format(
            schema_version, sorted(SUPPORTED_SCHEMA_VERSIONS)))
    sample_name = str(_scalar(metadata, "sampleName"))
    if not sample_name.strip():
        findings.error("sampleName is empty")
    ak_radius = float(_scalar(metadata, "akRadius"))
    if not _is_finite(ak_radius) or ak_radius <= 0.0:
        findings.error("akRadius must be finite and positive, found {!r}".format(ak_radius))

    max_ambiguous = int(_scalar(metadata, "maxAmbiguousCAJets"))
    if max_ambiguous < 0 or max_ambiguous > 20:
        findings.error("maxAmbiguousCAJets must lie in [0, 20]")
    if expected_max is not None and max_ambiguous != expected_max:
        findings.error("maxAmbiguousCAJets is {}, expected {}".format(
            max_ambiguous, expected_max))
    if not bool(_scalar(metadata, "puppiWeighted")):
        findings.error("puppiWeighted is false; weighted four-vectors are required")
    if bool(_scalar(metadata, "useJEC")):
        findings.error("useJEC is true; this study requires uncorrected AK-jet pT")

    ca_algorithm = str(_scalar(metadata, "caAlgorithm"))
    if ca_algorithm != "cambridge_y_phi":
        findings.error("caAlgorithm={!r}; expected 'cambridge_y_phi'".format(
            ca_algorithm))
    mass_objective = str(_scalar(metadata, "massObjective"))
    if mass_objective != "abs(m1-m2)/(m1+m2)":
        findings.error("massObjective={!r} is not the bounded symmetric objective".format(
            mass_objective))
    gate_constraint = str(_scalar(metadata, "gateRecoConstraint"))
    expected_gate_constraint = "collectionPtCut<=gatePtCut; nGateJets=0 is ungated"
    if gate_constraint != expected_gate_constraint:
        findings.error("gateRecoConstraint={!r}; expected {!r}".format(
            gate_constraint, expected_gate_constraint))
    legacy_radius_constraint = bool(_scalar(
        metadata, "enforceLegacyRadiusConstraint"))
    processed_events = int(_scalar(metadata, "processedEvents"))
    if processed_events != event_entries:
        findings.error("processedEvents={} but {} has {} entries".format(
            processed_events, EVENTS_PATH, event_entries))
    sum_weights = float(_scalar(metadata, "sumWeights"))
    sum_weights2 = float(_scalar(metadata, "sumWeights2"))
    if not _is_finite(sum_weights):
        findings.error("sumWeights is not finite")
    if not _is_finite(sum_weights2) or sum_weights2 < 0.0:
        findings.error("sumWeights2 must be finite and nonnegative")

    return {
        "schema_version": schema_version,
        "sample_name": sample_name,
        "ak_radius": ak_radius,
        "ca_algorithm": ca_algorithm,
        "legacy_radius_constraint": legacy_radius_constraint,
        "max_ambiguous": max_ambiguous,
        "declared_sum_weights": sum_weights,
        "declared_sum_weights2": sum_weights2,
    }


def _validate_metadata_grids(metadata, info, findings):
    collection_pt_cuts = _vector(metadata, "collectionPtCuts", float)
    ca_radii = _vector(metadata, "caRadii", float)
    cos_thrust_cuts = _vector(metadata, "cosThrustCuts", float)
    for name, values in (
        ("collectionPtCuts", collection_pt_cuts),
        ("caRadii", ca_radii),
        ("cosThrustCuts", cos_thrust_cuts),
    ):
        if not values:
            findings.error("{} is empty".format(name))
        if values != sorted(values):
            findings.error("{} is not sorted".format(name))
        if len({_float_key(value) for value in values}) != len(values):
            findings.error("{} contains duplicate values".format(name))
    if any(not _is_finite(value) or value <= 0.0 for value in collection_pt_cuts):
        findings.error("collectionPtCuts must contain finite positive values")
    if any(not _is_finite(value) or value <= 0.0 for value in ca_radii):
        findings.error("caRadii must contain finite positive values")
    if any(not _is_finite(value) or value < 0.0 or value > 1.0
           for value in cos_thrust_cuts):
        findings.error("cosThrustCuts must lie in [0, 1]")
    if info["legacy_radius_constraint"]:
        minimum_ca_radius = max(0.4, info["ak_radius"] - 0.2)
        if any(radius + 1.0e-6 < minimum_ca_radius for radius in ca_radii):
            findings.error(
                "caRadii violates enabled legacy radius constraint CA>=max(0.4, AK-0.2)")

    gate_counts = _vector(metadata, "defaultGateJetCounts", int)
    gate_pt_cuts = _vector(metadata, "defaultGatePtCuts", float)
    if not gate_counts:
        findings.error("defaultGateJetCounts is empty")
    if any(value < 0 for value in gate_counts):
        findings.error("defaultGateJetCounts contains a negative multiplicity")
    if len(set(gate_counts)) != len(gate_counts):
        findings.error("defaultGateJetCounts contains duplicate values")
    if gate_counts != sorted(gate_counts):
        findings.warning("defaultGateJetCounts is not sorted")
    if gate_counts and gate_counts[0] != 0:
        findings.error("defaultGateJetCounts must include canonical ungated value 0")
    if not gate_pt_cuts:
        findings.error("defaultGatePtCuts is empty")
    if any(not _is_finite(value) or value <= 0.0 for value in gate_pt_cuts):
        findings.error("defaultGatePtCuts must contain finite positive values")
    if len({_float_key(value) for value in gate_pt_cuts}) != len(gate_pt_cuts):
        findings.error("defaultGatePtCuts contains duplicate values")
    if gate_pt_cuts != sorted(gate_pt_cuts):
        findings.warning("defaultGatePtCuts is not sorted")
    if (gate_pt_cuts and collection_pt_cuts
            and gate_pt_cuts[0] + 1.0e-6 < collection_pt_cuts[0]):
        findings.error("defaultGatePtCuts contains a cut below the minimum collection pT cut")

    status_codes = _vector(metadata, "statusCodes", int)
    status_names = _vector(metadata, "statusNames", str)
    if status_codes != list(range(len(STATUS_NAMES))):
        findings.error("statusCodes is {}; expected {}".format(
            status_codes, list(range(len(STATUS_NAMES)))))
    if status_names != list(STATUS_NAMES):
        findings.error("statusNames is {}; expected {}".format(
            status_names, list(STATUS_NAMES)))

    base_pt = _vector(metadata, "baseCollectionPtCut", float)
    base_ca = _vector(metadata, "baseCaRadius", float)
    config_id = _vector(metadata, "configId", int)
    config_base = _vector(metadata, "configBaseIndex", int)
    config_pt = _vector(metadata, "configCollectionPtCut", float)
    config_ca = _vector(metadata, "configCaRadius", float)
    config_cos = _vector(metadata, "configCosThrust", float)

    if len(base_pt) != len(base_ca):
        findings.error("base mapping lengths differ: pT={} CA={}".format(
            len(base_pt), len(base_ca)))
    nbase = min(len(base_pt), len(base_ca))
    if nbase == 0:
        findings.error("base reconstruction mapping is empty")

    config_lengths = {
        "configId": len(config_id),
        "configBaseIndex": len(config_base),
        "configCollectionPtCut": len(config_pt),
        "configCaRadius": len(config_ca),
        "configCosThrust": len(config_cos),
    }
    if len(set(config_lengths.values())) != 1:
        findings.error("configuration mapping lengths differ: {}".format(
            ", ".join("{}={}".format(name, length)
                      for name, length in sorted(config_lengths.items()))))
    nconfig = min(config_lengths.values()) if config_lengths else 0
    if nconfig == 0:
        findings.error("reconstruction configuration mapping is empty")

    if any(not _is_finite(value) or value <= 0.0 for value in base_pt):
        findings.error("baseCollectionPtCut has a non-finite/non-positive value")
    if any(not _is_finite(value) or value <= 0.0 for value in base_ca):
        findings.error("baseCaRadius has a non-finite/non-positive value")
    base_keys = [(_float_key(pt), _float_key(radius))
                 for pt, radius in zip(base_pt, base_ca)]
    if len(set(base_keys)) != len(base_keys):
        findings.error("base mapping contains duplicate (pT cut, CA radius) pairs")
    expected_bases = {
        (_float_key(pt), _float_key(radius))
        for pt in collection_pt_cuts for radius in ca_radii
    }
    if set(base_keys) != expected_bases:
        findings.error("base reconstruction mapping is not a complete Cartesian grid")

    if config_id[:nconfig] != list(range(nconfig)):
        findings.error("configId must be contiguous and ordered from 0 to Nconfig-1")
    config_keys = []
    for index in range(nconfig):
        base_index = config_base[index]
        if base_index < 0 or base_index >= nbase:
            findings.error("configuration {} has out-of-range base index {}".format(
                index, base_index))
            continue
        if not _same_float(config_pt[index], base_pt[base_index]):
            findings.error("configuration {} pT cut does not match base {}".format(
                index, base_index))
        if not _same_float(config_ca[index], base_ca[base_index]):
            findings.error("configuration {} CA radius does not match base {}".format(
                index, base_index))
        cos_cut = config_cos[index]
        if not _is_finite(cos_cut) or cos_cut < 0.0 or cos_cut > 1.0:
            findings.error("configuration {} cosThrust {} is outside [0, 1]".format(
                index, cos_cut))
        config_keys.append((_float_key(config_pt[index]),
                            _float_key(config_ca[index]), _float_key(cos_cut)))
    if len(set(config_keys)) != len(config_keys):
        findings.error("configuration mapping contains duplicate triples")
    expected_configs = {(pt, radius, cos_cut)
                        for pt, radius in expected_bases
                        for cos_cut in map(_float_key, cos_thrust_cuts)}
    if set(config_keys) != expected_configs:
        findings.error("configuration mapping is not a complete Cartesian grid")

    info.update({
        "nbase": nbase,
        "nconfig": nconfig,
        "config_base": config_base[:nconfig],
    })
    return info


def _validate_metadata(metadata, event_entries, expected_max, findings):
    info = _validate_metadata_header(metadata, event_entries, expected_max, findings)
    return _validate_metadata_grids(metadata, info, findings) if info else None


def _validate_event_results(context, n_ca, statuses, n_ambiguous,
                            sj1_mass, sj2_mass, metadata_info, findings,
                            status_counts):
    nbase = metadata_info["nbase"]
    nconfig = metadata_info["nconfig"]
    config_base = metadata_info["config_base"]
    cap = metadata_info["max_ambiguous"]

    if len(n_ca) != nbase:
        findings.error("{} nCAJets length {} does not equal Nbase {}".format(
            context, len(n_ca), nbase))
    if any(value < 0 for value in n_ca):
        findings.error("{} nCAJets contains a negative value".format(context))
    result_lengths = {
        "recoStatus": len(statuses),
        "nAmbiguous": len(n_ambiguous),
        "sj1Mass": len(sj1_mass),
        "sj2Mass": len(sj2_mass),
    }
    for name, length in result_lengths.items():
        if length != nconfig:
            findings.error("{} {} length {} does not equal Nconfig {}".format(
                context, name, length, nconfig))

    safe_nconfig = min([nconfig, len(config_base)] + list(result_lengths.values()))
    for config_index in range(safe_nconfig):
        status = statuses[config_index]
        status_counts[status] += 1
        if status < 0 or status >= len(STATUS_NAMES):
            findings.error("{} configuration {} has out-of-range status {}".format(
                context, config_index, status))
            continue

        n_amb = n_ambiguous[config_index]
        if n_amb < 0:
            findings.error("{} configuration {} has negative nAmbiguous".format(
                context, config_index))
        base_index = config_base[config_index]
        mapped_n_ca = n_ca[base_index] if 0 <= base_index < len(n_ca) else None
        if mapped_n_ca is not None and n_amb > mapped_n_ca:
            findings.error(
                "{} configuration {} has nAmbiguous={} greater than nCAJets={}".format(
                    context, config_index, n_amb, mapped_n_ca))

        mass1 = sj1_mass[config_index]
        mass2 = sj2_mass[config_index]
        if status == VALID_STATUS:
            if not (_is_finite(mass1) and _is_finite(mass2)):
                findings.error("{} valid configuration {} has non-finite masses".format(
                    context, config_index))
            elif mass1 < 0.0 or mass2 < 0.0:
                findings.error("{} valid configuration {} has negative masses".format(
                    context, config_index))
            if n_amb > cap:
                findings.error("{} valid configuration {} exceeds ambiguity cap {}".format(
                    context, config_index, cap))
            if mapped_n_ca is not None and mapped_n_ca < 2:
                findings.error("{} valid configuration {} has fewer than two CA jets".format(
                    context, config_index))
        elif not (math.isnan(mass1) and math.isnan(mass2)):
            findings.error(
                "{} invalid configuration {} must have two NaN mass sentinels".format(
                    context, config_index))

        if status == COMPLEXITY_GUARD_STATUS:
            if n_amb <= cap:
                findings.error(
                    "{} complexity-guard configuration {} has nAmbiguous={} <= cap {}".format(
                        context, config_index, n_amb, cap))
        elif n_amb > cap:
            findings.error(
                "{} configuration {} exceeds cap without complexity-guard status".format(
                    context, config_index))
        if (status == NO_CA_JETS_STATUS and mapped_n_ca is not None
                and mapped_n_ca != 0):
            findings.error("{} no_ca_jets configuration {} maps to nCAJets={}".format(
                context, config_index, mapped_n_ca))


def _validate_events(events, metadata_info, max_events, findings):
    total_entries = int(events.GetEntries())
    if total_entries == 0:
        findings.error("{} has no entries".format(EVENTS_PATH))
        return {"checked": 0, "sum_weights": 0.0, "sum_weights2": 0.0,
                "status_counts": Counter(), "is_full_scan": True}
    entries_to_check = total_entries if max_events is None else min(
        total_entries, max_events)
    event_ids = set()
    weights = []
    weights2 = []
    status_counts = Counter()
    checked = 0

    for entry in range(entries_to_check):
        if events.GetEntry(entry) <= 0:
            findings.error("failed to read event entry {}".format(entry))
            continue
        checked += 1
        context = "event entry {}".format(entry)
        run = int(_scalar(events, "run"))
        lumi = int(_scalar(events, "lumi"))
        event = int(_scalar(events, "event"))
        if run <= 0 or lumi <= 0 or event <= 0:
            findings.error("{} has invalid event identifier ({}, {}, {})".format(
                context, run, lumi, event))
        event_id = (run, lumi, event)
        if event_id in event_ids:
            findings.error("{} duplicates event identifier {}".format(context, event_id))
        event_ids.add(event_id)

        gen_weight = float(_scalar(events, "genWeight"))
        if not _is_finite(gen_weight):
            findings.error("{} has non-finite genWeight".format(context))
        else:
            weights.append(gen_weight)
            weights2.append(gen_weight * gen_weight)
        ak_pt = _vector(events, "akJetPt", float)
        if any(not _is_finite(value) or value < 0.0 for value in ak_pt):
            findings.error("{} has non-finite/negative AK-jet pT".format(context))
        if any(ak_pt[index] < ak_pt[index + 1]
               for index in range(len(ak_pt) - 1)):
            findings.error("{} akJetPt is not sorted descending".format(context))

        _validate_event_results(
            context,
            _vector(events, "nCAJets", int),
            _vector(events, "recoStatus", int),
            _vector(events, "nAmbiguous", int),
            _vector(events, "sj1Mass", float),
            _vector(events, "sj2Mass", float),
            metadata_info, findings, status_counts)

    return {
        "checked": checked,
        "sum_weights": math.fsum(weights),
        "sum_weights2": math.fsum(weights2),
        "status_counts": status_counts,
        "is_full_scan": entries_to_check == total_entries,
    }


def _print_size_report(path, metadata, events, project_events):
    file_size = os.path.getsize(path)
    entries = int(events.GetEntries())
    metadata_zip, metadata_total = _tree_bytes(metadata)
    events_zip, events_total = _tree_bytes(events)
    print("Storage:")
    print("  file: {}".format(_format_bytes(file_size)))
    print("  metadata tree: {} compressed / {} uncompressed".format(
        _format_bytes(metadata_zip), _format_bytes(metadata_total)))
    print("  events tree: {} compressed / {} uncompressed".format(
        _format_bytes(events_zip), _format_bytes(events_total)))
    if entries > 0:
        compressed_per_event = float(events_zip) / entries
        file_per_event = float(file_size) / entries
        print("  per event: {} tree-compressed; {} whole-file amortized".format(
            _format_bytes(compressed_per_event), _format_bytes(file_per_event)))
        fixed_bytes = max(file_size - events_zip, 0)
        for target in project_events:
            projected = fixed_bytes + compressed_per_event * target
            print("  projected at {:,} events: {}".format(
                target, _format_bytes(projected)))
    largest = _largest_branches(events)
    if largest:
        print("  largest compressed event branches:")
        for size, name in largest:
            print("    {:<20} {}".format(name, _format_bytes(size)))


def _compare_weight_sums(metadata_info, event_info, findings):
    if not event_info["is_full_scan"]:
        return
    for name, observed, declared in (
        ("sumWeights", event_info["sum_weights"],
         metadata_info["declared_sum_weights"]),
        ("sumWeights2", event_info["sum_weights2"],
         metadata_info["declared_sum_weights2"]),
    ):
        if not math.isclose(observed, declared, rel_tol=1.0e-5, abs_tol=1.0e-5):
            findings.error("{} metadata={} but event sum={}".format(
                name, declared, observed))


def _print_content_summary(metadata_info, event_info, event_entries):
    if metadata_info is not None:
        print("Schema v{}; sample={!r}; AK R={:g}; {} base reconstruction(s); "
              "{} hybrid configuration(s)".format(
                  metadata_info["schema_version"], metadata_info["sample_name"],
                  metadata_info["ak_radius"], metadata_info["nbase"],
                  metadata_info["nconfig"]))
    if event_info is None:
        return
    print("Checked {:,} of {:,} event(s)".format(
        event_info["checked"], event_entries))
    if event_info["status_counts"]:
        summary = []
        for code, name in enumerate(STATUS_NAMES):
            count = event_info["status_counts"].get(code, 0)
            if count:
                summary.append("{}={:,}".format(name, count))
        unknown = sum(count for code, count in event_info["status_counts"].items()
                      if code < 0 or code >= len(STATUS_NAMES))
        if unknown:
            summary.append("unknown={:,}".format(unknown))
        print("Statuses: {}".format(", ".join(summary)))


def validate_file(args):
    findings = Findings(args.max_error_details)
    try:
        import ROOT  # pylint: disable=import-error,import-outside-toplevel
    except ImportError:
        print("ERROR: PyROOT is unavailable. Run after entering CMSSW (cmsenv).",
              file=sys.stderr)
        return 2
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)

    if not os.path.isfile(args.root_file):
        print("ERROR: file does not exist: {}".format(args.root_file), file=sys.stderr)
        return 2
    root_file = ROOT.TFile.Open(args.root_file, "READ")
    if not root_file or root_file.IsZombie():
        print("ERROR: could not open ROOT file: {}".format(args.root_file),
              file=sys.stderr)
        return 2

    try:
        metadata = root_file.Get(METADATA_PATH)
        events = root_file.Get(EVENTS_PATH)
        if not metadata or not metadata.InheritsFrom("TTree"):
            findings.error("missing TTree {}".format(METADATA_PATH))
        if not events or not events.InheritsFrom("TTree"):
            findings.error("missing TTree {}".format(EVENTS_PATH))
        if findings.error_count:
            findings.print_messages()
            return 1

        metadata_complete = _check_branch_set(
            metadata, METADATA_BRANCHES, args.strict_branches, findings)
        events_complete = _check_branch_set(
            events, EVENT_BRANCHES, args.strict_branches, findings)
        if not (metadata_complete and events_complete):
            findings.print_messages()
            return 1

        _check_tree_types(metadata, events, findings)
        event_entries = int(events.GetEntries())
        metadata_info = _validate_metadata(
            metadata, event_entries, args.expected_max_ambiguous, findings)
        event_info = (_validate_events(events, metadata_info, args.max_events, findings)
                      if metadata_info else None)
        if metadata_info and event_info:
            _compare_weight_sums(metadata_info, event_info, findings)

        _print_content_summary(metadata_info, event_info, event_entries)
        _print_size_report(args.root_file, metadata, events, args.project_events)
        if args.max_bytes_per_event is not None and event_entries > 0:
            events_zip, _ = _tree_bytes(events)
            payload = float(events_zip) / event_entries
            if payload > args.max_bytes_per_event:
                findings.error(
                    "compressed event payload {:.1f} B/event exceeds {:.1f} B/event".format(
                        payload, args.max_bytes_per_event))

        findings.print_messages()
        if findings.error_count:
            print("FAILED: {} invariant violation(s), {} warning(s)".format(
                findings.error_count, findings.warning_count), file=sys.stderr)
            return 1
        print("PASS: compact scan schema and checked invariants are valid")
        return 0
    finally:
        root_file.Close()


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return parsed


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_file", help="compact-scan ROOT file to validate")
    parser.add_argument(
        "--strict-branches", action="store_true",
        help="treat any branch outside the compact schema as an error")
    parser.add_argument(
        "--max-events", type=_positive_int, default=None,
        help="check only the first N events (structural checks still cover the file)")
    parser.add_argument(
        "--project-events", type=_positive_int, nargs="+", default=[1000, 10000],
        metavar="N", help="event counts for size projections (default: 1000 10000)")
    parser.add_argument(
        "--expected-max-ambiguous", type=int, default=12, metavar="N",
        help="required ambiguity cap (default: 12; negative disables the check)")
    parser.add_argument(
        "--max-bytes-per-event", type=_nonnegative_float, default=None,
        metavar="BYTES", help="optionally cap compressed Events bytes per event")
    parser.add_argument(
        "--max-error-details", type=_positive_int, default=50, metavar="N",
        help="maximum error and warning messages to print (default: 50)")
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    if args.expected_max_ambiguous < 0:
        args.expected_max_ambiguous = None
    return validate_file(args)


if __name__ == "__main__":
    sys.exit(main())
