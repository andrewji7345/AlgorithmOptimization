#!/usr/bin/env python3
"""Export explicit, auditable Combine models for sensitivity cross-checks.

Each input histogram bin becomes a one-bin Combine channel.  This permits the
AN's shared linear Gaussian QCD factors using ordinary ``rateParam`` formulae,
without private RooParametricHist code.  No observed data are accepted: data_obs
is always the nominal background Asimov expectation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REFERENCE_COMMIT = "8ef9b506ef4229a2110ec8e662fd5ae93b73420a"
PROCESSES = ("signal", "QCD", "TTbar", "ST", "WJets")
REGIONS = ("SR", "CR", "AT1b", "AT0b")
CATEGORY_MAP = {"QCDMC": "QCD", "TTbarMC": "TTbar", "STMC": "ST", "WJetsMC": "WJets"}
DEFAULT_CONFIG = Path(__file__).parent / "config" / "likelihood_reference_2017.json"
STAGE_CONTRACT_VERSION = 2
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _name(value: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"unsafe Combine identifier: {value!r}")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def _array(value: Any, label: str, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or (size is not None and len(array) != size):
        raise ValueError(f"{label}: expected a one-dimensional array" + (f" of length {size}" if size else ""))
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{label}: negative or nonfinite bins are not supported; merge or repair the templates")
    return array


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _fmt(value: float) -> str:
    return format(value, ".12g")


def gaussian_domain(relative: float, maximum_sigma: float = 8.0) -> tuple[float, float]:
    """Strictly positive linear scale throughout a recorded nuisance domain.

    The lower bound implements the physical yield domain, not clipping the
    expectation inside a fit.  For large relative errors this truncation can be
    substantial and is explicitly reported in the model manifest.
    """
    relative = _number(relative, "Gaussian relative width", positive=True)
    maximum_sigma = _number(maximum_sigma, "maximum Gaussian sigma", positive=True)
    return max(-maximum_sigma, -(1.0 - 1e-8) / relative), maximum_sigma


def qcd_group_width(sumw: Any, sumw2: Any, group: list[int], blanket: float = 0.15,
                    bb_relative_uncertainty: Any = None) -> float:
    y, v = np.asarray(sumw, dtype=float), np.asarray(sumw2, dtype=float)
    if bb_relative_uncertainty is None:
        relative = np.divide(np.sqrt(v), y, out=np.zeros_like(y), where=y > 0)
    else:
        relative = np.asarray(bb_relative_uncertainty, dtype=float)
    return math.hypot(float(np.max(relative[group])), blanket)


def from_objective(objective: Mapping[str, Any], campaign: Mapping[str, Any], signal: str,
                   *, region: str = "SR") -> dict[str, Any]:
    """Aggregate normalized sample histograms without renormalizing or losing sumw2.

    The campaign supplies the process category and benchmark cross section.
    Only its chosen signal enters a card; other mass hypotheses are alternatives,
    never additive signal processes. Arrays flatten in C order, and the original
    axes and shape are retained for provenance.
    """
    if region not in REGIONS:
        raise ValueError(f"unsupported region {region}")
    catalog = {sample["name"]: sample for sample in campaign["samples"]}
    if len(catalog) != len(campaign["samples"]) or set(catalog) != set(objective["samples"]):
        raise ValueError("campaign and objective must contain the same unique samples")
    if signal not in catalog or catalog[signal]["kind"] != "signal":
        raise ValueError("selected signal must name a signal sample in the campaign")
    parameters = objective.get("parameters", {})
    for key in ("luminosity_pb", "mass_bin_edges_gev", "chi_mass_bin_edges_gev", "analysis_selection", "correction_prescription"):
        if key in parameters and parameters[key] != campaign.get(key):
            raise ValueError(f"campaign {key} differs from the evaluated objective")
    template = np.asarray(objective["samples"][signal]["histogram"]["sumw"], dtype=float)
    if template.ndim not in (1, 2):
        raise ValueError("objective histograms must have one or two dimensions")
    processes = {name: {"sumw": np.zeros(template.size), "sumw2": np.zeros(template.size), "samples": []}
                 for name in PROCESSES}
    for name, info in objective["samples"].items():
        spec = catalog[name]
        if "kind" in info and info["kind"] != spec["kind"]:
            raise ValueError(f"{name}: campaign sample kind differs from the objective")
        if spec["kind"] == "signal" and name != signal:
            continue
        if "normalization_factor" in info and "sum_gen_weights" in info and "luminosity_pb" in campaign:
            denominator = _number(info["sum_gen_weights"], f"{name} generator denominator", positive=True)
            expected_scale = (float(spec["cross_section_pb"]) * float(campaign["luminosity_pb"])
                              * float(spec.get("filter_efficiency", 1)) * float(spec.get("k_factor", 1)) / denominator)
            if not math.isclose(float(info["normalization_factor"]), expected_scale, rel_tol=1e-10):
                raise ValueError(f"{name}: campaign cross section/normalization differs from the evaluated objective")
        process = "signal" if name == signal else CATEGORY_MAP.get(spec.get("category"))
        if process is None:
            raise ValueError(f"{name}: unsupported background category {spec.get('category')!r}")
        hist = info["histogram"]
        for key in ("sumw", "sumw2"):
            values = np.asarray(hist[key], dtype=float)
            if values.shape != template.shape:
                raise ValueError(f"{name}: inconsistent histogram shape")
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"{name}: invalid {key}; negative component bins cannot be hidden by aggregation")
            if key == "sumw" and np.any((values == 0) & (np.asarray(hist["sumw2"]) > 0)):
                raise ValueError(f"{name}: signed cancellation with zero yield and nonzero variance")
            processes[process][key] += values.ravel(order="C")
        processes[process]["samples"].append(name)
    for process in processes.values():
        for key in ("sumw", "sumw2"):
            process[key] = process[key].tolist()
    axes = [list(campaign["mass_bin_edges_gev"])]
    if template.ndim == 2:
        axes.append(list(campaign["chi_mass_bin_edges_gev"]))
    if tuple(len(axis) - 1 for axis in axes) != template.shape:
        raise ValueError("campaign binning does not match the objective histograms")
    xsec = _number(catalog[signal]["cross_section_pb"], "signal cross section", positive=True)
    return {
        "schema_version": 1,
        "provenance": {"input_kind": "evaluated_ntuples", "objective_sha256": _hash(objective),
                       "campaign_sha256": _hash(campaign), "signal": signal,
                       "signal_cross_section_pb": xsec,
                       "cross_section_definition": "catalog production times specified decay/filter branching fractions",
                       "source_purpose": objective.get("purpose", campaign.get("purpose", "unknown")),
                       "source_feasible": bool(objective.get("feasible", False)),
                       "analysis_selection": campaign.get("analysis_selection"),
                       "correction_prescription": campaign.get("correction_prescription"),
                       "physicality": objective.get("per_signal", {}).get(signal, {}).get("physicality"),
                       "sample_normalization": {name: {key: info.get(key) for key in
                           ("normalization_scope", "normalization_factor", "sum_gen_weights", "generated_events", "files")}
                           for name, info in objective["samples"].items()},
                       "reference_commit": REFERENCE_COMMIT},
        "channels": {region: {"bin_edges": list(range(template.size + 1)), "bin_axes_gev": axes,
                              "original_shape": list(template.shape), "flatten_order": "C",
                              "processes": processes}},
    }


def _normalization_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("asymmetric lnN values must be [down, up]")
        return "/".join(_fmt(_number(x, "lnN factor", positive=True)) for x in value)
    return _fmt(_number(value, "lnN factor", positive=True))


def build_model(inputs: Mapping[str, Any], config: Mapping[str, Any] | None = None,
                *, region: str = "SR", stage: str = "reference",
                allow_unsupported_mc: bool = False, require_complete_reference: bool = False) -> dict[str, Any]:
    """Build a single-region expected model; exporting multiple regions is explicit.

    Stages are ``statistics_only`` and ``reference``.  The latter includes the
    AN normalization priors and QCD blanket term plus every supplied variation.
    Both stages use the same grouped finite-MC nuisance directions; changing
    stage must not change the assumed MC correlation or aggregate-BB treatment.
    It does not synthesize missing experimental/theory templates.
    """
    inputs = copy.deepcopy(dict(inputs))
    config = copy.deepcopy(dict(config)) if config is not None else json.loads(DEFAULT_CONFIG.read_text())
    if inputs.get("schema_version") != 1 or config.get("schema_version") != 1:
        raise ValueError("unsupported likelihood input/config schema")
    if stage not in ("statistics_only", "reference"):
        raise ValueError("stage must be statistics_only or reference")
    if region not in REGIONS or region not in inputs.get("channels", {}):
        raise ValueError("select one supplied SR, CR, AT1b or AT0b region")
    if inputs.get("observed_data") is not None or any("data_obs" in c or "observation" in c for c in inputs["channels"].values()):
        raise ValueError("this expected-only builder does not accept observed data")
    channel = inputs["channels"][region]
    edges = np.asarray(channel["bin_edges"], dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be finite and strictly increasing")
    n_bins = len(edges) - 1
    process_input = channel["processes"]
    if set(process_input) - set(PROCESSES) or "signal" not in process_input:
        raise ValueError(f"processes must be drawn from {PROCESSES} and include signal")
    process_data: dict[str, dict[str, Any]] = {}
    for name in PROCESSES:
        raw = process_input.get(name, {"sumw": [0] * n_bins, "sumw2": [0] * n_bins})
        y = _array(raw["sumw"], f"{name} sumw", n_bins)
        v = _array(raw["sumw2"], f"{name} sumw2", n_bins)
        if np.any((y == 0) & (v > 0)):
            raise ValueError(f"{name}: zero expected yield with nonzero variance; do not floor signed cancellations")
        variations = {}
        for nuisance, shifts in raw.get("variations", {}).items():
            _name(nuisance)
            up, down = (_array(shifts[key], f"{name}/{nuisance}/{key}", n_bins) for key in ("up", "down"))
            if np.any((y == 0) & ((up != 0) | (down != 0))):
                raise ValueError(f"{name}/{nuisance}: migration into nominally empty process bins requires common bin merging")
            if np.any((y > 0) & ((up == 0) | (down == 0))):
                raise ValueError(f"{name}/{nuisance}: zero shifted yield requires common bin merging")
            variations[nuisance] = {"up": up, "down": down}
        process_data[name] = {"sumw": y, "sumw2": v, "variations": variations}
    if process_data["signal"]["sumw"].sum() <= 0:
        raise ValueError("selected signal hypothesis has no positive yield")
    b = sum(process_data[p]["sumw"] for p in PROCESSES[1:])
    v = sum(process_data[p]["sumw2"] for p in PROCESSES[1:])
    s = process_data["signal"]["sumw"]
    if np.any((s > 0) & (b <= 0)):
        raise ValueError("signal-populated bin lacks positive background; no epsilon background is inserted")
    neff = np.divide(b * b, v, out=np.full(n_bins, np.inf), where=v > 0)
    neff[(b == 0) & (v == 0)] = 0
    minimum = _number(config.get("minimum_background_effective_events", 10), "minimum effective MC", positive=True)
    unsupported = np.flatnonzero((s > 0) & (neff < minimum)).tolist()
    if unsupported and not allow_unsupported_mc:
        raise ValueError(f"insufficient background effective MC in bins {unsupported}; use diagnostic override only for software checks")

    groups = channel.get("qcd_groups", [[i] for i in range(n_bins)])
    if not isinstance(groups, list) or any(not isinstance(g, list) or not g for g in groups):
        raise ValueError("qcd_groups must be a nonempty partition of bin-index lists")
    flat = [index for group in groups for index in group]
    if any(type(index) is not int for index in flat) or sorted(flat) != list(range(n_bins)):
        raise ValueError("qcd_groups must partition every bin exactly once")
    supplied_bb = channel.get("bb_relative_uncertainty")
    if supplied_bb is not None:
        supplied_bb = _array(supplied_bb, "bb_relative_uncertainty", n_bins)
        if not channel.get("bb_uncertainty_provenance"):
            raise ValueError("supplied BB uncertainty requires bb_uncertainty_provenance")
        # These coefficients already absorb all background MC.  Do not add it again.
        if np.any((process_data["QCD"]["sumw"] == 0) & (v > 0)):
            raise ValueError("a QCD-absorbed BB model cannot encode background MC uncertainty in a zero-QCD bin")

    active_bins = np.flatnonzero((s + b) > 0).tolist()
    columns = [(f"{region}_bin{i}", p, i) for i in active_bins for p in PROCESSES if process_data[p]["sumw"][i] > 0]
    process_ids = {p: i for i, p in enumerate(PROCESSES)}
    lines = ["# Expected-only, representative AN-23-067 UL2017 model; see model.json",
             f"imax {len(active_bins)}", "jmax *", "kmax *", "------------",
             "shapes * * templates.root $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC",
             "------------", "bin " + " ".join(f"{region}_bin{i}" for i in active_bins),
             "observation " + " ".join("-1" for _ in active_bins), "------------",
             "bin " + " ".join(c[0] for c in columns),
             "process " + " ".join(c[1] for c in columns),
             "process " + " ".join(str(process_ids[c[1]]) for c in columns),
             "rate " + " ".join("-1" for _ in columns), "------------"]
    nuisances, shape_names, used_names = [], set(), set()

    def reserve(name: str) -> None:
        _name(name)
        if name in used_names:
            raise ValueError(f"nuisance identifier collision: {name}")
        used_names.add(name)

    normalizations = config.get("normalization_nuisances", {}) if stage == "reference" else {}
    for name, values in sorted(normalizations.items()):
        reserve(name)
        if set(values) - set(PROCESSES):
            raise ValueError(f"{name}: unknown normalization process")
        factors = {p: _normalization_value(value) for p, value in values.items()}
        effects = [factors.get(p, "-") for _, p, _ in columns]
        if any(effect != "-" for effect in effects):
            lines.append(f"{name} lnN " + " ".join(effects))
            nuisances.append({"name": name, "kind": "lnN", "process_factors": values})
    if stage == "reference":
        shape_names = {name for data in process_data.values() for name in data["variations"]}
        for name in sorted(shape_names):
            reserve(name)
            lines.append(f"{name} shape " + " ".join("1" if name in process_data[p]["variations"] else "-" for _, p, _ in columns))
            nuisances.append({"name": name, "kind": "shape", "processes": [p for p in PROCESSES if name in process_data[p]["variations"]]})

    maximum_sigma = _number(config.get("gaussian_maximum_sigma", 8), "maximum Gaussian sigma", positive=True)

    def gaussian(name: str, width: float, affected: list[tuple[str, str, int]], kind: str) -> None:
        if width <= 0 or not affected:
            return
        reserve(name)
        low, high = gaussian_domain(width, maximum_sigma)
        lines.append(f"{name} param 0 1 [{_fmt(low)},{_fmt(high)}]")
        for bin_name, process, _ in affected:
            factor_name = f"scale_{name}_{bin_name}_{process}"
            reserve(factor_name)
            lines.append(f"{factor_name} rateParam {bin_name} {process} (1+{_fmt(width)}*@0) {name}")
        nuisances.append({"name": name, "kind": kind, "relative_width": width, "range": [low, high],
                          "positivity_truncation_within_5sigma": low > -5,
                          "affects": [{"channel": c, "process": p, "input_bin": i} for c, p, i in affected]})

    qcd = process_data["QCD"]
    blanket = _number(config.get("qcd_blanket_relative_uncertainty", 0.15), "QCD blanket") if stage == "reference" else 0.0
    if blanket < 0:
        raise ValueError("QCD blanket uncertainty cannot be negative")
    qcd_groups = groups
    mc_widths = [qcd_group_width(qcd["sumw"], qcd["sumw2"], group, 0.0, supplied_bb)
                 for group in qcd_groups]
    finite_mc_basis = {
        "qcd_treatment": "supplied_aggregate_BB" if supplied_bb is not None else "QCD_sumw2_only",
        "qcd_groups": qcd_groups,
        "qcd_group_mc_relative_widths": mc_widths,
        "qcd_correlation": "one common relative fluctuation within each group; independent between groups",
        "independent_process_bin_factors": [],
    }
    for group_index, group in enumerate(qcd_groups):
        width = math.hypot(mc_widths[group_index], blanket)
        gaussian(f"QCD_bin_{region}_group{group_index}_17", width,
                 [c for c in columns if c[1] == "QCD" and c[2] in group], "qcd_linear_gaussian")
    for c in columns:
        bin_name, process, index = c
        if process == "QCD" or (supplied_bb is not None and process != "signal"):
            continue
        variance, value = process_data[process]["sumw2"][index], process_data[process]["sumw"][index]
        width = math.sqrt(variance) / value
        name = f"MCstat_{bin_name}_{process}_17"
        if width > 0:
            finite_mc_basis["independent_process_bin_factors"].append(
                {"name": name, "process": process, "input_bin": index, "relative_width": width})
        gaussian(name, width, [c], "mc_linear_gaussian")

    missing = []
    if stage == "reference":
        for name, required in config.get("required_shape_nuisances", {}).items():
            for process in required:
                if process_data[process]["sumw"].sum() > 0 and name not in process_data[process]["variations"]:
                    missing.append(f"{name}:{process}")
        for name, required in config.get("required_additional_normalizations", {}).items():
            for process in required:
                if process_data[process]["sumw"].sum() > 0 and process not in normalizations.get(name, {}):
                    missing.append(f"{name}:{process}")
    else:
        missing = ["statistics_only_stage_omits_analysis_systematics"]
    provenance = inputs.get("provenance", {})
    audits = provenance.get("reference_audits", {})
    required_audits = ("original_selection", "normalization", "systematic_templates", "bin_map_and_groups", "control_region_closure")
    missing_audits = [name for name in required_audits if audits.get(name) is not True]
    if require_complete_reference and (missing or missing_audits or supplied_bb is None or unsupported):
        raise ValueError("complete reference requested but required templates, audits, BB coefficients or MC support are missing: "
                         + ", ".join(missing + missing_audits + (["audited_BB_coefficients"] if supplied_bb is None else [])
                                     + (["background_MC_support"] if unsupported else [])))
    approximations = []
    if supplied_bb is None:
        approximations.append("QCD-only sumw2 is absorbed into its group coefficient; other background and signal MC errors are separate Gaussian factors. This is not the AN's private Beeston-Barlow implementation.")
    if any(len(group) > 1 for group in qcd_groups):
        approximations.append("Both likelihood stages use fully correlated QCD fluctuations within each group at its maximum relative MC error. This changes the independent-bin MC covariance and can inflate individual bin variances; it is an AN-inspired approximation.")
    if "qcd_groups" not in channel:
        approximations.append("Independent input-bin QCD factors; the AN uses fixed neighboring 2D superbin groups.")
    if channel.get("original_shape", [n_bins]) == [n_bins]:
        approximations.append("One-dimensional mass binning, rather than the AN's linearized two-dimensional superjet/disuperjet bin map.")
    if "CMS_jec_Total_2017" in shape_names:
        approximations.append("Supplied total JEC variation replaces neither the seven source-specific templates nor their correlation structure.")
    approximations.append("Every input bin is a separate one-bin shape channel. Supplied shape variations retain correlated up/down bin yields with Combine's one-bin normalization interpolation; its interpolation between shifts is not the original multibin morph.")
    manifest = {"schema_version": 1, "model_kind": "representative_AN2017_expected", "stage": stage,
                "stage_contract_version": STAGE_CONTRACT_VERSION,
                "stage_comparison": "Common grouped MC basis; reference broadens the QCD coefficient in quadrature with the blanket term and adds analysis priors. The blanket term is not a separately fitted nuisance.",
                "region": region, "region_fit_mode": "individual", "expected_only": True,
                "full_an_reproduction": False, "production_ready": False,
                "input_sha256": _hash(inputs), "configuration_sha256": _hash(config),
                "provenance": provenance, "bin_edges": edges.tolist(),
                "bin_axes_gev": channel.get("bin_axes_gev"), "original_shape": channel.get("original_shape"),
                "active_input_bins": active_bins, "dropped_empty_input_bins": [i for i in range(n_bins) if i not in active_bins],
                "processes": {p: {"sumw": d["sumw"].tolist(), "sumw2": d["sumw2"].tolist(),
                                  "variations": {n: {k: a.tolist() for k, a in shifts.items()} for n, shifts in d["variations"].items()}}
                              for p, d in process_data.items()},
                "background": b.tolist(), "background_sumw2": v.tolist(),
                "background_effective_events": [float(x) if math.isfinite(x) else None for x in neff],
                "minimum_background_effective_events": minimum,
                "unsupported_background_bins": unsupported,
                "allow_unsupported_mc": allow_unsupported_mc,
                "qcd_groups": qcd_groups, "qcd_blanket_relative_uncertainty": blanket,
                "qcd_mc_treatment": finite_mc_basis["qcd_treatment"],
                "finite_mc_basis": finite_mc_basis, "finite_mc_basis_sha256": _hash(finite_mc_basis),
                "bb_uncertainty_provenance": channel.get("bb_uncertainty_provenance"),
                "finite_mc": "explicit Gaussian linear rates; no autoMCStats or second QCD MC term",
                "nuisances": nuisances, "missing_reference_nuisances": missing,
                "missing_reference_audits": missing_audits, "approximations": approximations,
                "reference_input_complete": not (missing or missing_audits or supplied_bb is None or unsupported)}
    return {"datacard": "\n".join(lines) + "\n", "manifest": manifest, "inputs": inputs, "config": config}


def export_model(inputs: Mapping[str, Any], output_dir: str | Path, config: Mapping[str, Any] | None = None,
                 **kwargs: Any) -> dict[str, Any]:
    """Write card, ROOT histograms, normalized inputs and a reproducibility manifest.

    No existing model artifact is replaced; use a distinct output directory when
    changing a model. ROOT errors retain sumw2 but are *not* consumed implicitly:
    every statistical factor is explicit in the card, and autoMCStats is absent.
    """
    import uproot
    from uproot.writing.identify import to_TAxis, to_TH1x

    built = build_model(inputs, config, **kwargs)
    out = Path(output_dir)
    artifacts = [out / name for name in ("datacard.txt", "templates.root", "model.json", "inputs.json", "model_config.json")]
    if any(path.exists() for path in artifacts):
        raise FileExistsError("likelihood output artifacts already exist; use a distinct output directory")
    out.mkdir(parents=True, exist_ok=True)
    manifest = built["manifest"]

    def hist(value: float, variance: float) -> Any:
        return to_TH1x(None, "expected yield", np.array([0., value, 0.]),
                       value * value / variance if variance else 0., value, variance, value * .5, value * .25,
                       np.array([0., variance, 0.]), to_TAxis("xaxis", "input bin", 1, 0., 1.))

    with uproot.recreate(out / "templates.root") as root:
        for index in manifest["active_input_bins"]:
            channel_name = f"{manifest['region']}_bin{index}"
            background = manifest["background"][index]
            root[f"{channel_name}/data_obs"] = hist(background, background)
            for process, data in manifest["processes"].items():
                value = data["sumw"][index]
                if value == 0:
                    continue
                root[f"{channel_name}/{process}"] = hist(value, data["sumw2"][index])
                if manifest["stage"] == "reference":
                    for nuisance, shifts in data["variations"].items():
                        for direction, suffix in (("up", "Up"), ("down", "Down")):
                            root[f"{channel_name}/{process}_{nuisance}{suffix}"] = hist(shifts[direction][index], 0.)
    (out / "datacard.txt").write_text(built["datacard"])
    (out / "inputs.json").write_text(json.dumps(built["inputs"], indent=2, allow_nan=False) + "\n")
    (out / "model_config.json").write_text(json.dumps(built["config"], indent=2, allow_nan=False) + "\n")
    manifest["artifacts_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts if path.name != "model.json"}
    (out / "model.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="normalized histogram model JSON")
    source.add_argument("--objective", help="evaluate_sensitivity objective.json")
    parser.add_argument("--campaign", help="campaign JSON for --objective")
    parser.add_argument("--signal", help="signal sample name for --objective")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--region", choices=REGIONS, default="SR")
    parser.add_argument("--stage", choices=("statistics_only", "reference"), default="reference")
    parser.add_argument("--allow-unsupported-mc", action="store_true", help="diagnostic cards only; does not make candidates eligible")
    parser.add_argument("--require-complete-reference", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.objective:
        if not args.campaign or not args.signal:
            parser.error("--objective requires --campaign and --signal")
        model_input = from_objective(json.loads(Path(args.objective).read_text()),
                                     json.loads(Path(args.campaign).read_text()), args.signal, region=args.region)
    else:
        model_input = json.loads(Path(args.input).read_text())
    manifest = export_model(model_input, args.output_dir, json.loads(Path(args.config).read_text()),
                            region=args.region, stage=args.stage, allow_unsupported_mc=args.allow_unsupported_mc,
                            require_complete_reference=args.require_complete_reference)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), "stage": manifest["stage"],
                      "missing_reference_nuisances": manifest["missing_reference_nuisances"],
                      "reference_input_complete": manifest["reference_input_complete"]}, allow_nan=False))


if __name__ == "__main__":
    main()
