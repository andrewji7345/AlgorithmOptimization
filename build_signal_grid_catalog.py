#!/usr/bin/env python3
"""Reproduce the complete UL17 signal catalog from local, audited inputs.

Normalization is evaluated from a hash-locked public source function. Dataset
identifiers are reconstructed from existing LFNs, not advertised as DAS-audited.
The documented HTZT 6200/1950 substitute uses log-linear interpolation of the
LO production cross section only, followed by the actual-mass branching fraction.
This emits a catalog and audit, not a production-ready campaign.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

SOURCE_SHA256 = "269a27a22496726a5b8e7cae85c8e9a3dc4fa3af83904dc69726907f72901f52"
SOURCE_COMMIT = "8ef9b506ef4229a2110ec8e662fd5ae93b73420a"
REPOSITORY = Path(__file__).resolve().parent
SOURCE_RELATIVE = "data/analysis_2017/signal_normalization_source.py"
SOURCE_URL = f"https://github.com/emcannaert/SuuToChiChi-analysis-software/blob/{SOURCE_COMMIT}/postprocess/return_signal_SF/return_signal_SF.py"
PRODUCTION_FB = {4000: 3200., 5000: 695., 6000: 137., 7000: 23.1, 8000: 3.03}
MODES = {"WbWb": "WBWB", "WbHt": "WBHT", "WbZt": "WBZT", "HtHt": "HTHT", "HtZt": "HTZT", "ZtZt": "ZTZT"}
GRID = {4000: (1000, 1500), 5000: (1000, 1500, 2000), 6000: (1000, 1500, 2000, 2500), 7000: (1000, 1500, 2000, 2500, 3000), 8000: (1000, 1500, 2000, 2500, 3000)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def production_cross_section_fb(mass, anchors=PRODUCTION_FB):
    """Positive LO cross sections: exact anchors or log-linear interpolation.

    No extrapolation is allowed. The catalog invokes interpolation only for the
    documented 6200-GeV substitute, never for a whole decay cross section.
    """
    if not math.isfinite(mass) or not anchors or any(not math.isfinite(v) or v <= 0 for v in anchors.values()):
        raise ValueError("Production mass and positive cross-section anchors must be finite")
    if mass in anchors:
        return anchors[mass]
    points = sorted(anchors)
    for low, high in zip(points, points[1:]):
        if low < mass < high:
            fraction = (mass - low) / (high - low)
            return math.exp((1 - fraction) * math.log(anchors[low]) + fraction * math.log(anchors[high]))
    raise ValueError("Production cross-section extrapolation is not allowed")


def build(repo=REPOSITORY, source=None):
    repo = Path(repo)
    source = Path(source) if source is not None else repo / SOURCE_RELATIVE
    if sha(source) != SOURCE_SHA256:
        raise ValueError("Public normalization source differs from audited commit")
    # The two reviewed pure functions need only math when quiet=True. Do not
    # import the unrelated source module or require numpy just for its logging.
    parsed = ast.parse(source.read_text())
    selected = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in ("return_Suu_to_chi_chi_xs", "calculate_Suu_to_chi_chi_BR")]
    if len(selected) != 2:
        raise ValueError("Required source functions not found")
    scope = {"math": math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), scope)
    xs = scope["return_Suu_to_chi_chi_xs"]
    br = scope["calculate_Suu_to_chi_chi_BR"]
    hadronic = {
        "WBWB": (.5 * .6741) ** 2,
        "HTHT": (.25 * .58 * .6741) ** 2,
        "ZTZT": (.25 * .6991 * .6741) ** 2,
        "WBHT": 2 * (.5 * .6741) * (.25 * .58 * .6741),
        "WBZT": 2 * (.5 * .6741) * (.25 * .6991 * .6741),
        "HTZT": 2 * (.25 * .58 * .6741) * (.25 * .6991 * .6741),
    }
    production = PRODUCTION_FB
    rows, seen_files, grid = [], set(), defaultdict(set)
    for path in sorted((repo / "test/signalMCFiles").glob("*.txt")):
        match = re.fullmatch(r"(WbWb|WbHt|WbZt|HtHt|HtZt|ZtZt)_(\d+)_(\d+)", path.stem)
        if not match:
            raise ValueError(f"Unexpected sample-list name: {path}")
        mode, suu, chi = match[1], int(match[2]), int(match[3])
        mode_source = MODES[mode]
        grid[mode].add((suu, chi))
        lines = [s.strip() for s in path.read_text().splitlines() if s.strip() and not s.lstrip().startswith("#")]
        datasets, lfns, actual_masses = set(), [], set()
        for uri in lines:
            marker = uri.find("/store/mc/")
            if marker < 0:
                raise ValueError(f"Not a recognized MC LFN: {uri}")
            lfn = uri[marker:]
            fields = lfn.strip("/").split("/")
            if len(fields) != 8 or fields[4] != "MINIAODSIM" or fields[2] != "RunIISummer20UL17MiniAODv2":
                raise ValueError(f"Unexpected UL17 LFN structure: {uri}")
            primary = re.fullmatch(r"SuuToChiChiTo([A-Z]+)ToJets_MSuu-(\d+)_MChi-(\d+)_TuneCP5_13TeV-madgraph-pythia8", fields[3])
            if primary is None or primary[1] != mode_source:
                raise ValueError(f"LFN decay mode does not match list name: {uri}")
            actual_masses.add((int(primary[2]), int(primary[3])))
            datasets.add(f"/{fields[3]}/{fields[2]}-{fields[5]}/{fields[4]}")
            if lfn in seen_files:
                raise ValueError(f"Duplicate LFN within/across lists: {lfn}")
            seen_files.add(lfn)
            lfns.append(lfn)
        if len(datasets) != 1 or len(actual_masses) != 1 or not lfns:
            raise ValueError(f"Empty or mixed-dataset list: {path}")
        generated_suu, generated_chi = next(iter(actual_masses))
        mass_mismatch = (generated_suu, generated_chi) != (suu, chi)
        if mass_mismatch and (path.stem, generated_suu, generated_chi) != ("HtZt_6000_2000", 6200, 1950):
            raise ValueError(f"Undocumented generated-mass mismatch: {path}")
        chi_label = f"{chi/1000:g}".replace(".", "p")
        mass_name = f"Suu{suu//1000}_chi{chi_label}"
        source_xs_fb = xs(mass_name, mode_source, quiet=True)
        independent_xs_fb = production[suu] * br(suu, chi) * hadronic[mode_source]
        if not math.isclose(source_xs_fb, independent_xs_fb, rel_tol=2e-14):
            raise ValueError(f"Source/formula normalization mismatch: {path.stem}")
        actual_production_fb = production_cross_section_fb(generated_suu)
        actual_br = br(generated_suu, generated_chi)
        actual_xs_pb = (actual_production_fb * actual_br * hadronic[mode_source] / 1000.
                        if mass_mismatch else source_xs_fb / 1000.)
        if not math.isfinite(actual_xs_pb) or actual_xs_pb <= 0:
            raise ValueError(f"Invalid final cross section: {path.stem}")
        rows.append({
            "name": path.stem, "benchmark_catalog_name": f"{mass_name}_{mode_source}",
            "kind": "signal", "category": "SuuToChiChi", "decay_mode": mode,
            "suu_mass_gev": generated_suu, "chi_mass_gev": generated_chi,
            "nominal_suu_mass_gev": suu, "nominal_chi_mass_gev": chi,
            "generated_suu_mass_gev": generated_suu, "generated_chi_mass_gev": generated_chi,
            "cross_section_pb": actual_xs_pb,
            "nominal_grid_hypothesis_cross_section_pb": source_xs_fb / 1000.,
            "normalization_status": "interpolated_LO_production_actual_generated_masses" if mass_mismatch else "exact_public_source_grid_normalization",
            "production_cross_section_fb": actual_production_fb,
            "suu_to_chichi_branching_fraction": actual_br,
            "fully_hadronic_decay_branching_fraction": hadronic[mode_source],
            "filter_efficiency": 1., "k_factor": 1.,
            "dataset": next(iter(datasets)), "dbs_instance": "prod/global",
            "dataset_identifier_provenance": "reconstructed from all local-list LFNs; not DAS-verified",
            "input_file_list": f"../test/signalMCFiles/{path.name}",
            "input_file_list_sha256": sha(path), "local_list_file_count": len(lfns),
            "full_dataset_file_coverage_verified": False,
            "full_dataset_generated_events": None,
            "full_dataset_sum_gen_weights": None,
            "mass_mismatch_note": ("The sample name is a nominal-grid alias. Existing list header and docs/compact_scan_evaluation.md document the generated6200/1950 substitute for unavailable HTZT6000/2000. Event normalization uses actual generated masses, not the nominal alias." if mass_mismatch else None),
            "production_interpolation": ({"method": "linear_in_log_cross_section_and_mass", "mass_gev": 6200,
                "anchor_masses_gev": [6000, 7000], "anchor_cross_sections_fb": [production[6000], production[7000]],
                "fraction": .2, "formula": "exp(0.8*log(sigma_LO(6000)) + 0.2*log(sigma_LO(7000)))",
                "scope": "LO production cross section only; actual6200/1950 branching fraction evaluated separately",
                "limitation": "Approximation between published source grid values; no additional theory-uncertainty estimate is assigned by this nominal catalog."} if mass_mismatch else None),
        })
    expected = {(s, c) for s, cs in GRID.items() for c in cs}
    if set(grid) != set(MODES) or any(points != expected for points in grid.values()):
        raise ValueError("Signal inventory is not the complete six-mode, nineteen-mass-point grid")
    old = json.loads((repo / "config/signal_benchmarks_2017.json").read_text())
    new_by_benchmark = {s["benchmark_catalog_name"]: s for s in rows}
    matches = []
    for prior in old["samples"]:
        current = new_by_benchmark[prior["name"]]
        for key in ("cross_section_pb", "production_cross_section_fb", "suu_to_chichi_branching_fraction", "fully_hadronic_decay_branching_fraction"):
            if not math.isclose(prior[key], current[key], rel_tol=2e-14):
                raise ValueError(f"Existing benchmark changed: {prior['name']}/{key}")
        matches.append(prior["name"])
    catalog = {
        "schema_version": 1, "year": 2017, "luminosity_fb": 41.48,
        "benchmark": {"y_uu": 2., "y_chi": 2., "chi_to_Wb_BR": .5, "chi_to_Zt_BR": .25, "chi_to_Ht_BR": .25,
                      "W_to_hadrons_BR": .6741, "Z_to_hadrons_BR": .6991, "H_hadronic_source_BR": .58,
                      "top_hadronic_source_BR": .6741, "production_order": "source LO; no extra NLO K factor"},
        "source_repository": "https://github.com/emcannaert/SuuToChiChi-analysis-software",
        "source_commit": SOURCE_COMMIT,
        "source": "postprocess/return_signal_SF/return_signal_SF.py:122-194,221-232",
        "source_url": SOURCE_URL, "vendored_source": SOURCE_RELATIVE,
        "source_sha256": SOURCE_SHA256,
        "all_samples_normalized": True,
        "notes": [
            "114 decay-specific samples: six channels at each of nineteen mass points.",
            "Cross sections include the selected chi branching fractions, source hadronic factors and mixed-channel factor two. Do not multiply these again.",
            "Source H factor 0.58 is preserved literally; it is not a new estimate of the total inclusive hadronic Higgs branching fraction.",
            "Local MiniAOD URLs establish identifiers, not current accessibility, complete DAS coverage or generated-event denominators.",
            "Use actual signed generator weights and all processed preselection denominators. The legacy source scale-factor function's hardcoded event counts are not used.",
            "HtZt_6000_2000 is a nominal-grid alias for generated masses6200/1950. Its LO production cross section uses approved log-linear interpolation between6000 and7000 GeV; its Suu branching fraction uses actual6200/1950 masses. No whole-decay cross section is interpolated.",
        ],
        "samples": rows,
    }
    audit = {"passed": True, "inventory_validation_passed": True, "all_samples_normalized": True,
             "unresolved_samples": [s["name"] for s in rows if s["cross_section_pb"] is None],
             "interpolated_samples": [{"name": s["name"], "cross_section_pb": s["cross_section_pb"],
                 "production_cross_section_fb": s["production_cross_section_fb"],
                 "generated_suu_mass_gev": s["generated_suu_mass_gev"], "generated_chi_mass_gev": s["generated_chi_mass_gev"]}
                 for s in rows if s["production_interpolation"] is not None],
             "actual_generated_mass_pair_count": len({(s["generated_suu_mass_gev"], s["generated_chi_mass_gev"]) for s in rows}),
             "sample_count": len(rows), "nominal_mass_pair_count": len(expected),
             "mode_counts": dict(Counter(s["decay_mode"] for s in rows)), "file_count": len(seen_files),
             "file_count_range": [min(s["local_list_file_count"] for s in rows), max(s["local_list_file_count"] for s in rows)],
             "mass_grid_gev": {str(k): list(v) for k, v in GRID.items()},
             "source_sha256": SOURCE_SHA256, "existing_benchmarks_preserved": matches,
             "source_function_formula_agreement_relative_tolerance": 2e-14,
             "duplicate_lfns": 0, "mixed_dataset_lists": 0, "DAS_queries": 0, "event_files_opened": 0}
    return catalog, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPOSITORY)
    parser.add_argument("--source", type=Path, help="Default: vendored source in the repository")
    parser.add_argument("--catalog-output", type=Path, help="Default: config/signal_grid_2017.json")
    parser.add_argument("--audit-output", type=Path, help="Default: data/analysis_2017/signal_grid_inventory_audit.json")
    parser.add_argument("--check", action="store_true", help="Require existing catalog and audit to reproduce exactly; write nothing")
    args = parser.parse_args()
    catalog, audit = build(args.repo, args.source)
    outputs = ((args.catalog_output or args.repo / "config/signal_grid_2017.json", catalog),
               (args.audit_output or args.repo / "data/analysis_2017/signal_grid_inventory_audit.json", audit))
    for path, data in outputs:
        content = json.dumps(data, indent=2, allow_nan=False) + "\n"
        if args.check:
            if path.read_text() != content:
                parser.error(f"Catalog reproduction differs: {path}")
        else:
            path.write_text(content)
    print(json.dumps({"checked": args.check, "samples": audit["sample_count"], "input_files": audit["file_count"],
                      "interpolated_samples": audit["interpolated_samples"]}, indent=2))


if __name__ == "__main__":
    main()
