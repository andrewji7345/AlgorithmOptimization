"""Evaluate legacy and two-threshold multi-sample clustering scans.

The accepted input filenames are

    <decay>_<M_Suu>_<M_chi>_pt<pt>_ak<10R_AK>_ca<10R_CA>_th<th>.root
    <decay>_<M_Suu>_<M_chi>_pt<low>_ak<10R_AK>_ca<10R_CA>_th<th>_two_threshold_<high>_<low>.root

For example,

    WbWb_4000_1000_pt380_ak4_ca6_th85.root
    WbWb_4000_1000_pt100_ak4_ca6_th85_two_threshold_300_100.root

The script preserves the per-sample diagnostics from
evaluate_ntuplizer_pt_ak_ca.py and adds comparisons across every discovered
mass point and decay channel.  Sample names and true chi masses are inferred
from filenames unless --samples is supplied. Legacy and two-threshold
configurations are kept distinct in global rankings. Family-specific regime
summaries are produced separately, so two-threshold inputs never contaminate
the legacy heatmaps.
"""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import uproot

from resolution_metrics import relative_fwhm_resolution


TREE_NAME = "existingOptimizationNtuplizer/Events"
LABEL_BRANCH_CANDIDATES = ("particle_newAlgoLabel",)
EVENT_SELECTION_BRANCH = "passesEventSelection"

PT_CUTS = np.arange(100, 401, 20, dtype=int)
TWO_THRESHOLD_EVENT_PT_CUT = 300
TWO_THRESHOLD_LOW_PT_CUTS = np.arange(100, 301, 20, dtype=int)
AK_RADII = np.arange(4, 17, 2, dtype=int) / 10.0
CA_RADII = np.arange(4, 17, 2, dtype=int) / 10.0

LEGACY_FAMILY = "single_threshold"
TWO_THRESHOLD_FAMILY = "two_threshold"
FAMILY_ORDER = (LEGACY_FAMILY, TWO_THRESHOLD_FAMILY)
FAMILY_LABELS = {
    LEGACY_FAMILY: "Single-threshold (legacy)",
    TWO_THRESHOLD_FAMILY: "Two-threshold",
}

DECAY_ORDER = ("WbWb", "WbZt", "WbHt", "ZtZt", "HtZt", "HtHt")
SAMPLE_RE = re.compile(r"^(?P<decay>[A-Za-z]+)_(?P<suu_mass>\d+)_(?P<chi_mass>\d+)$")
LEGACY_FILENAME_RE = re.compile(
    r"^(?P<sample>[A-Za-z]+_\d+_\d+)"
    r"_pt(?P<pt_cut>\d+)_ak(?P<ak_code>\d+)_ca(?P<ca_code>\d+)"
    r"_th(?P<th_code>\d+)\.root$"
)
TWO_THRESHOLD_FILENAME_RE = re.compile(
    r"^(?P<sample>[A-Za-z]+_\d+_\d+)"
    r"_pt(?P<filename_low_pt_cut>\d+)_ak(?P<ak_code>\d+)"
    r"_ca(?P<ca_code>\d+)_th(?P<th_code>\d+)"
    r"_two_threshold_(?P<event_pt_cut>\d+)_(?P<pt_cut>\d+)\.root$"
)


def parse_number_list(text, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def open_root_file(filename):
    """Open local/FUSE files without fsspec's asynchronous local reader."""
    options = {}
    if "://" not in str(filename):
        options["handler"] = uproot.source.file.MemmapSource
    return uproot.open(filename, **options)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate jet-clustering scans across mass points and decay channels.")
    parser.add_argument("--input-dir", default="/eos/uscms/store/user/aji/rootfiles_existingOptimization")
    parser.add_argument("--samples", nargs="+", default=None, help=("Samples such as WbWb_4000_1000. If omitted, discover samples from input filenames matching the requested cos thrust."))
    parser.add_argument("--output-dir", default="results/evaluate_ntuplizer_dc_mp_pt_ak_ca_th")
    parser.add_argument("--tree", default=TREE_NAME)
    parser.add_argument("--label-branch", default=None)
    parser.add_argument("--th-code", type=int, default=85)
    parser.add_argument("--max-events", type=int, default=1000, help="Maximum events read from each file; use -1 for all events.")
    parser.add_argument("--peak-response-min", type=float, default=0.70, help="Lower edge of the good-mass window in m_reco/M_chi.")
    parser.add_argument("--peak-response-max", type=float, default=1.30, help="Upper edge of the good-mass window in m_reco/M_chi.")
    parser.add_argument("--low-response-max", type=float, default=0.30, help="Legacy diagnostic only; it is not used in the score.")
    parser.add_argument("--mass-plot-max-factor", type=float, default=2.0, help="Upper mass-histogram edge as a multiple of the true chi mass.")
    parser.add_argument(
        "--minimum-efficiency",
        type=float,
        default=0.90,
        help=(
            "Minimum event-selection efficiency for an eligible optimum."
        ),
    )
    parser.add_argument("--nominal-pt-cut", type=int, default=300)
    parser.add_argument("--nominal-ak-radius", type=float, default=0.8)
    parser.add_argument("--nominal-ca-radius", type=float, default=0.8)
    parser.add_argument("--top-configurations", type=int, default=15, help="Number of globally ranked configurations shown in comparison plots.")
    parser.add_argument("--global-mean-weight", type=float, default=0.25, help="Weight of mean regret added to worst-case regret for global ranking.")
    parser.add_argument("--minimum-regime-coverage", type=float, default=1.0, help="Required fraction of regimes for global configuration ranking.")
    parser.add_argument("--coverage-deltas", default="0.02,0.05,0.10", help="Comma-separated additive-regret tolerances summarized in CSV output.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-detailed-histograms", action="store_true", help="Skip rereading selected ROOT files for mass-distribution overlays.")
    args = parser.parse_args()

    if not 0.0 <= args.peak_response_min < args.peak_response_max:
        parser.error("The response window must satisfy 0 <= min < max")
    if not 0.0 <= args.minimum_efficiency <= 1.0:
        parser.error("--minimum-efficiency must lie in [0, 1]")
    if not 0.0 < args.minimum_regime_coverage <= 1.0:
        parser.error("--minimum-regime-coverage must lie in (0, 1]")
    args.coverage_deltas = parse_number_list(args.coverage_deltas, float)
    return args


def sample_metadata(sample):
    match = SAMPLE_RE.fullmatch(sample)
    if match is None:
        raise ValueError(
            f"Sample '{sample}' must have form <decay>_<M_Suu>_<M_chi>"
        )
    result = match.groupdict()
    return {
        "sample": sample,
        "decay": result["decay"],
        "suu_mass": int(result["suu_mass"]),
        "chi_mass": int(result["chi_mass"]),
    }


def decay_sort_key(decay):
    try:
        return (0, DECAY_ORDER.index(decay))
    except ValueError:
        return (1, decay)


def parse_scan_filename(filename, requested_th_code=None):
    """Return metadata for one canonical legacy or two-threshold filename."""
    path = Path(filename)
    match = TWO_THRESHOLD_FILENAME_RE.fullmatch(path.name)
    algorithm_family = TWO_THRESHOLD_FAMILY
    if match is None:
        match = LEGACY_FILENAME_RE.fullmatch(path.name)
        algorithm_family = LEGACY_FAMILY
    if match is None:
        return None

    fields = match.groupdict()
    th_code = int(fields["th_code"])
    if requested_th_code is not None and th_code != requested_th_code:
        return None

    pt_cut = int(fields["pt_cut"])
    event_pt_cut = None
    if algorithm_family == TWO_THRESHOLD_FAMILY:
        filename_low_pt_cut = int(fields["filename_low_pt_cut"])
        if filename_low_pt_cut != pt_cut:
            raise ValueError(
                f"Inconsistent two-threshold filename '{path.name}': "
                f"pt{filename_low_pt_cut} does not match suffix low cut {pt_cut}"
            )
        event_pt_cut = int(fields["event_pt_cut"])

    # The strict sample expression prevents an algorithm suffix from leaking
    # into a discovered sample identifier.
    sample_metadata(fields["sample"])
    return {
        "sample": fields["sample"],
        "algorithm_family": algorithm_family,
        "event_pt_cut": event_pt_cut,
        "pt_cut": pt_cut,
        "ak_radius": int(fields["ak_code"]) / 10.0,
        "ca_radius": int(fields["ca_code"]) / 10.0,
        "th_code": th_code,
        "filename": str(path),
    }


def scan_sort_key(scan):
    family_index = FAMILY_ORDER.index(scan["algorithm_family"])
    event_pt_cut = scan["event_pt_cut"] if scan["event_pt_cut"] is not None else -1
    return (
        family_index,
        event_pt_cut,
        scan["pt_cut"],
        scan["ak_radius"],
        scan["ca_radius"],
    )


def valid_radius_pair(ak_radius, ca_radius):
    return ca_radius + 1e-9 >= max(0.4, ak_radius - 0.2)


def value_in_grid(value, grid):
    return any(math.isclose(value, candidate) for candidate in grid)


def scan_on_supported_grid(scan):
    pt_grid = (
        TWO_THRESHOLD_LOW_PT_CUTS
        if scan["algorithm_family"] == TWO_THRESHOLD_FAMILY
        else PT_CUTS
    )
    if scan["pt_cut"] not in pt_grid:
        return False
    if not value_in_grid(scan["ak_radius"], AK_RADII):
        return False
    if not value_in_grid(scan["ca_radius"], CA_RADII):
        return False
    if (
        scan["algorithm_family"] == TWO_THRESHOLD_FAMILY
        and scan["event_pt_cut"] != TWO_THRESHOLD_EVENT_PT_CUT
    ):
        return False
    return valid_radius_pair(scan["ak_radius"], scan["ca_radius"])


def discover_scan_files(input_dir, th_code, strict=False):
    """Discover present inputs without manufacturing absent grid points."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    scans = []
    seen = set()
    for filename in sorted(input_dir.glob("*.root")):
        try:
            scan = parse_scan_filename(filename, requested_th_code=th_code)
        except ValueError as error:
            if strict:
                raise
            print(f"WARNING: ignoring malformed scan filename: {error}")
            continue
        if scan is None:
            continue
        if not scan_on_supported_grid(scan):
            message = f"Scan point outside the requested grid: {filename.name}"
            if strict:
                raise ValueError(message)
            print(f"WARNING: ignoring {message}")
            continue
        identity = (
            scan["sample"],
            scan["algorithm_family"],
            scan["event_pt_cut"],
            scan["pt_cut"],
            scan["ak_radius"],
            scan["ca_radius"],
            scan["th_code"],
        )
        if identity in seen:
            message = f"Duplicate scan configuration represented by {filename.name}"
            if strict:
                raise ValueError(message)
            print(f"WARNING: {message}; keeping the first file")
            continue
        seen.add(identity)
        scans.append(scan)

    return sorted(
        scans,
        key=lambda scan: (
            sample_metadata(scan["sample"])["suu_mass"],
            sample_metadata(scan["sample"])["chi_mass"],
            decay_sort_key(sample_metadata(scan["sample"])["decay"]),
            scan_sort_key(scan),
        ),
    )


def discover_samples(input_dir, th_code):
    samples = {scan["sample"] for scan in discover_scan_files(input_dir, th_code)}
    return sorted(
        samples,
        key=lambda sample: (
            sample_metadata(sample)["suu_mass"],
            sample_metadata(sample)["chi_mass"],
            decay_sort_key(sample_metadata(sample)["decay"]),
        ),
    )


def scan_filename(input_dir, sample, pt_cut, ak_radius, ca_radius, th_code):
    ak_code = int(round(10.0 * ak_radius))
    ca_code = int(round(10.0 * ca_radius))
    name = f"{sample}_pt{pt_cut}_ak{ak_code}_ca{ca_code}_th{th_code}.root"
    return Path(input_dir) / name


def two_threshold_scan_filename(
    input_dir, sample, event_pt_cut, pt_cut, ak_radius, ca_radius, th_code
):
    ak_code = int(round(10.0 * ak_radius))
    ca_code = int(round(10.0 * ca_radius))
    name = (
        f"{sample}_pt{pt_cut}_ak{ak_code}_ca{ca_code}_th{th_code}"
        f"_two_threshold_{event_pt_cut}_{pt_cut}.root"
    )
    return Path(input_dir) / name


def choose_label_branch(tree, requested_branch):
    keys = set(tree.keys())
    if requested_branch is not None:
        if requested_branch not in keys:
            raise KeyError(f"Requested label branch '{requested_branch}' is absent")
        return requested_branch
    for branch in LABEL_BRANCH_CANDIDATES:
        if branch in keys:
            return branch
    raise KeyError(
        "Could not find a reconstruction-label branch. Available label-like branches: "
        + ", ".join(sorted(key for key in keys if "label" in key.lower()))
    )


def invariant_mass(px, py, pz, energy):
    mass_squared = energy**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.maximum(mass_squared, 0.0))


def read_chi_masses(filename, tree_name, requested_label_branch, max_events):
    return read_chi_masses_with_selection(
        filename,
        tree_name,
        requested_label_branch,
        max_events,
        offline_event_selection=None,
    )


def read_chi_masses_with_selection(
    filename,
    tree_name,
    requested_label_branch,
    max_events,
    offline_event_selection,
):
    entry_stop = None if max_events < 0 else max_events
    with open_root_file(filename) as root_file:
        tree = root_file[tree_name]
        tree_keys = set(tree.keys())
        label_branch = choose_label_branch(tree, requested_label_branch)
        kinematic_names = (
            "particle_px",
            "particle_py",
            "particle_pz",
            "particle_energy",
        )
        read_names = list(kinematic_names)
        if offline_event_selection is not None:
            if "ak_pt" not in tree_keys:
                raise KeyError(
                    f"Cannot reconstruct the two-threshold event gate from "
                    f"{filename}: branch 'ak_pt' is absent"
                )
            read_names.append("ak_pt")
        elif EVENT_SELECTION_BRANCH in tree_keys:
            read_names.append(EVENT_SELECTION_BRANCH)
        elif "minEventJets" in tree_keys:
            read_names.append("minEventJets")
        events = tree.arrays(
            [*read_names, label_branch], entry_stop=entry_stop, library="ak"
        )

    n_events = len(events[label_branch])
    if offline_event_selection is not None:
        event_pt_cut, min_event_jets = offline_event_selection
        passes_selection = np.asarray(
            ak.to_numpy(
                ak.sum(events["ak_pt"] >= event_pt_cut, axis=1)
                >= min_event_jets
            ),
            dtype=bool,
        )
    elif EVENT_SELECTION_BRANCH in events.fields:
        passes_selection = np.asarray(
            ak.to_numpy(events[EVENT_SELECTION_BRANCH]), dtype=bool
        )
    elif "minEventJets" in events.fields:
        min_event_jets = np.asarray(
            ak.to_numpy(events["minEventJets"]), dtype=int
        )
        if np.any(min_event_jets > 0):
            raise RuntimeError(
                f"{filename} is an old two-threshold ntuple without the "
                f"'{EVENT_SELECTION_BRANCH}' branch. Its rejected events are "
                "absent; use the scan-aware evaluator with the corresponding "
                "all-event single-threshold ntuple."
            )
        passes_selection = np.ones(n_events, dtype=bool)
    else:
        # Ntuples made before the two-threshold mode existed had no event gate.
        passes_selection = np.ones(n_events, dtype=bool)

    labels = events[label_branch]
    components = []
    counts = []
    for chi_label in (1, 2):
        mask = labels == chi_label
        counts.append(np.asarray(ak.to_numpy(ak.sum(mask, axis=1)), dtype=int))
        components.append(
            tuple(
                np.asarray(ak.to_numpy(ak.sum(events[name][mask], axis=1)), dtype=float)
                for name in kinematic_names
            )
        )

    px0, py0, pz0, energy0 = components[0]
    px1, py1, pz1, energy1 = components[1]
    mass0 = invariant_mass(px0, py0, pz0, energy0)
    mass1 = invariant_mass(px1, py1, pz1, energy1)
    valid0 = (counts[0] > 0) & np.isfinite(mass0) & np.isfinite(energy0) & (energy0 > 0)
    valid1 = (counts[1] > 0) & np.isfinite(mass1) & np.isfinite(energy1) & (energy1 > 0)
    valid_reconstruction = valid0 & valid1
    return mass0, mass1, passes_selection, valid_reconstruction, label_branch


def inspect_event_selection_bookkeeping(filename, tree_name):
    """Describe how one ntuple represents its event selection."""
    with open_root_file(filename) as root_file:
        tree = root_file[tree_name]
        tree_keys = set(tree.keys())
        has_stored_selection = EVENT_SELECTION_BRANCH in tree_keys
        if "minEventJets" not in tree_keys:
            return ("stored_branch", None) if has_stored_selection else ("no_gate", 0)
        values = tree["minEventJets"].array(
            entry_start=0, entry_stop=1, library="np"
        )
    if len(values) == 0:
        raise RuntimeError(f"Cannot read minEventJets from empty ntuple: {filename}")
    min_event_jets = int(values[0])
    if has_stored_selection:
        return "stored_branch", min_event_jets
    return ("old_filtered_ntuple", min_event_jets) if min_event_jets > 0 else ("no_gate", 0)


def legacy_companion_filename(scan):
    return scan_filename(
        Path(scan["filename"]).parent,
        scan["sample"],
        scan["pt_cut"],
        scan["ak_radius"],
        scan["ca_radius"],
        scan["th_code"],
    )


def read_scan_chi_masses(scan, args):
    """Read one scan point, reconstructing old two-threshold gates offline."""
    filename = Path(scan["filename"])
    selection_mode, min_event_jets = inspect_event_selection_bookkeeping(
        filename, args.tree
    )

    evaluation_filename = filename
    offline_event_selection = None
    if scan["algorithm_family"] == TWO_THRESHOLD_FAMILY:
        if selection_mode == "old_filtered_ntuple":
            evaluation_filename = legacy_companion_filename(scan)
            if not evaluation_filename.is_file():
                raise FileNotFoundError(
                    "Old two-threshold ntuple requires its all-event "
                    f"single-threshold companion, which is missing: {evaluation_filename}"
                )
            offline_event_selection = (scan["event_pt_cut"], min_event_jets)
            selection_mode = "offline_from_single_threshold"
        elif selection_mode != "stored_branch":
            raise RuntimeError(
                f"Two-threshold ntuple has no usable event-selection "
                f"bookkeeping: {filename}"
            )

    result = read_chi_masses_with_selection(
        evaluation_filename,
        args.tree,
        args.label_branch,
        args.max_events,
        offline_event_selection,
    )
    read_info = {
        "evaluation_filename": str(evaluation_filename),
        "selection_evaluation_mode": selection_mode,
        "min_event_jets": min_event_jets,
    }
    return (*result, read_info)


def safe_fraction(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else np.nan


def binomial_error(fraction, denominator):
    if denominator <= 0 or not np.isfinite(fraction):
        return np.nan
    return float(np.sqrt(fraction * (1.0 - fraction) / denominator))


def calculate_metrics(
    mass0,
    mass1,
    passes_selection,
    valid_reconstruction,
    true_chi_mass,
    args,
):
    """Calculate metrics relative to every analyzed event.

    selection_efficiency is the event-gate acceptance requested by the scan:
    selected events divided by all analyzed events. Finite two-chi
    reconstruction is tracked separately and defines the population for the
    mass-quality metrics. reconstruction_efficiency is retained as a deprecated
    output alias so existing CSV consumers do not break.
    """
    n_events = len(passes_selection)
    n_selected = int(np.count_nonzero(passes_selection))
    valid_event = passes_selection & valid_reconstruction
    n_valid = int(np.count_nonzero(valid_event))
    efficiency = safe_fraction(n_selected, n_events)
    empty = {
        "n_events": n_events,
        "n_selected_events": n_selected,
        "n_valid_events": n_valid,
        "selection_efficiency": efficiency,
        "selection_efficiency_err": binomial_error(efficiency, n_events),
        "reconstruction_efficiency": efficiency,
        "reconstruction_efficiency_err": binomial_error(efficiency, n_events),
        "f_low_event": np.nan,
        "f_low_event_err": np.nan,
        "f_low_chi": np.nan,
        "f_peak_chi": np.nan,
        "f_peak_event_both": np.nan,
        "f_peak_event_both_all": 0.0 if n_events else np.nan,
        "off_peak_event_fraction": np.nan,
        "off_peak_event_fraction_err": np.nan,
        "median_mass": np.nan,
        "median_mass_response": np.nan,
        "fwhm_mass_response": np.nan,
        "fwhm_peak_mass_response": np.nan,
        "fwhm_resolution": np.nan,
        "median_mass_asymmetry": np.nan,
    }
    if n_valid == 0:
        return empty

    m0 = mass0[valid_event]
    m1 = mass1[valid_event]
    r0 = m0 / true_chi_mass
    r1 = m1 / true_chi_mass
    masses = np.concatenate((m0, m1))
    responses = np.concatenate((r0, r1))

    low0 = r0 < args.low_response_max
    low1 = r1 < args.low_response_max
    peak0 = (r0 >= args.peak_response_min) & (r0 <= args.peak_response_max)
    peak1 = (r1 >= args.peak_response_min) & (r1 <= args.peak_response_max)
    low_event = low0 | low1
    peak_event_both = peak0 & peak1

    f_low_event = float(np.mean(low_event))
    f_peak_event_both = float(np.mean(peak_event_both))
    off_peak = 1.0 - f_peak_event_both
    median_mass = np.median(masses)
    median_response = np.median(responses)
    resolution, fwhm_response, fwhm_peak_response = relative_fwhm_resolution(
        responses
    )
    denominator = m0 + m1
    asymmetry = np.divide(
        np.abs(m0 - m1),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )

    return {
        "n_events": n_events,
        "n_selected_events": n_selected,
        "n_valid_events": n_valid,
        "selection_efficiency": efficiency,
        "selection_efficiency_err": binomial_error(efficiency, n_events),
        "reconstruction_efficiency": efficiency,
        "reconstruction_efficiency_err": binomial_error(efficiency, n_events),
        "f_low_event": f_low_event,
        "f_low_event_err": binomial_error(f_low_event, n_valid),
        "f_low_chi": float(np.mean(np.concatenate((low0, low1)))),
        "f_peak_chi": float(np.mean(np.concatenate((peak0, peak1)))),
        "f_peak_event_both": f_peak_event_both,
        "f_peak_event_both_all": safe_fraction(np.count_nonzero(peak_event_both), n_events),
        "off_peak_event_fraction": off_peak,
        "off_peak_event_fraction_err": binomial_error(off_peak, n_valid),
        "median_mass": float(median_mass),
        "median_mass_response": float(median_response),
        "fwhm_mass_response": fwhm_response,
        "fwhm_peak_mass_response": fwhm_peak_response,
        "fwhm_resolution": resolution,
        "median_mass_asymmetry": float(np.nanmedian(asymmetry)),
    }


def balanced_score_components(row):
    required = (
        row["off_peak_event_fraction"],
        row["median_mass_response"],
        row["fwhm_resolution"],
        row["selection_efficiency"],
    )
    if not all(np.isfinite(value) for value in required):
        return {
            "score_off_peak_component": np.nan,
            "score_mass_bias_component": np.nan,
            "score_resolution_component": np.nan,
            "score_inefficiency_component": np.nan,
        }
    return {
        "score_off_peak_component": float(row["off_peak_event_fraction"]),
        "score_mass_bias_component": float(
            0.1 * abs(row["median_mass_response"] - 1.0)
        ),
        "score_resolution_component": float(row["fwhm_resolution"]),
        "score_inefficiency_component": float(
            1.0 - row["selection_efficiency"]
        ),
    }


def balanced_score(row, args):
    del args  # Retained in the signature for compatibility with earlier callers.
    components = balanced_score_components(row)
    values = np.asarray(list(components.values()), dtype=float)
    return float(np.sum(values)) if np.all(np.isfinite(values)) else np.nan


def radius_edges(values):
    values = np.asarray(values, dtype=float)
    edges = np.empty(len(values) + 1)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def metric_cube(rows, metric, pt_cuts=PT_CUTS):
    cube = np.full((len(pt_cuts), len(AK_RADII), len(CA_RADII)), np.nan)
    pt_index = {value: i for i, value in enumerate(pt_cuts)}
    ak_index = {round(value, 2): i for i, value in enumerate(AK_RADII)}
    ca_index = {round(value, 2): i for i, value in enumerate(CA_RADII)}
    for row in rows:
        if row["status"] != "ok":
            continue
        pt_position = pt_index.get(row["pt_cut"])
        ak_position = ak_index.get(round(row["ak_radius"], 2))
        ca_position = ca_index.get(round(row["ca_radius"], 2))
        if pt_position is None or ak_position is None or ca_position is None:
            continue
        cube[
            pt_position,
            ak_position,
            ca_position,
        ] = row[metric]
    return cube


def draw_heatmap(
    values,
    title,
    colorbar_label,
    output_file,
    vmin=None,
    vmax=None,
    cmap="viridis",
    annotation_format=".2f",
):
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    mesh = ax.pcolormesh(
        radius_edges(AK_RADII),
        radius_edges(CA_RADII),
        values.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
    )
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(colorbar_label)
    ax.set_xticks(AK_RADII)
    ax.set_yticks(CA_RADII)
    ax.set_xlabel(r"AK clustering radius $R_{\mathrm{AK}}$")
    ax.set_ylabel(r"CA clustering radius $R_{\mathrm{CA}}$")
    ax.set_title(title)
    for i, ak_radius in enumerate(AK_RADII):
        for j, ca_radius in enumerate(CA_RADII):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(
                    ak_radius,
                    ca_radius,
                    format(value, annotation_format),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def finite_upper_limit(values, fallback):
    finite = np.asarray(values)[np.isfinite(values)]
    if len(finite) == 0:
        return fallback
    return max(fallback, float(np.quantile(finite, 0.98)))


def sample_title(metadata):
    return (
        rf"{metadata['decay']}, $M_{{S_{{uu}}}}={metadata['suu_mass']}$ GeV, "
        rf"$M_\chi={metadata['chi_mass']}$ GeV"
    )


def make_per_threshold_heatmaps(
    rows, output_dir, args, metadata, pt_cuts=PT_CUTS, family_title=None
):
    """Retained per-sample, per-threshold heatmaps from the original script."""
    specifications = {
        "off_peak_event_fraction": {
            "label": (
                rf"Event fraction outside {args.peak_response_min:.2f}--"
                rf"{args.peak_response_max:.2f} $m_\chi/M_\chi$ window"
            ),
            "vmin": 0.0,
            "vmax": 1.0,
            "cmap": "magma",
            "format": ".2f",
        },
        "f_peak_event_both": {
            "label": "Valid-event fraction with both chis in peak window",
            "vmin": 0.0,
            "vmax": 1.0,
            "cmap": "viridis",
            "format": ".2f",
        },
        "f_peak_chi": {
            "label": "Individual-chi fraction in peak window",
            "vmin": 0.0,
            "vmax": 1.0,
            "cmap": "viridis",
            "format": ".2f",
        },
        "median_mass": {
            "label": r"Median reconstructed $m_\chi$ [GeV]",
            "vmin": 0.0,
            "vmax": None,
            "cmap": "viridis",
            "format": ".0f",
        },
        "median_mass_response": {
            "label": r"Median reconstructed $m_\chi/M_\chi$",
            "vmin": 0.0,
            "vmax": None,
            "cmap": "viridis",
            "format": ".2f",
        },
        "fwhm_resolution": {
            "label": r"FWHM / histogram peak",
            "vmin": 0.0,
            "vmax": None,
            "cmap": "plasma",
            "format": ".2f",
        },
        "selection_efficiency": {
            "label": "Event-selection efficiency",
            "vmin": 0.0,
            "vmax": 1.0,
            "cmap": "viridis",
            "format": ".2f",
        },
        "balanced_score": {
            "label": "Balanced score",
            "vmin": 0.0,
            "vmax": None,
            "cmap": "viridis_r",
            "format": ".2f",
        },
        # Kept as a legacy diagnostic, but deliberately excluded from the score.
        "f_low_event": {
            "label": rf"Event fraction with $m_\chi/M_\chi < {args.low_response_max:.2f}$",
            "vmin": 0.0,
            "vmax": 1.0,
            "cmap": "magma_r",
            "format": ".2f",
        },
    }
    for metric, settings in specifications.items():
        cube = metric_cube(rows, metric, pt_cuts)
        metric_dir = output_dir / "heatmaps" / metric
        metric_dir.mkdir(parents=True, exist_ok=True)
        vmax = settings["vmax"]
        if metric == "median_mass":
            vmax = finite_upper_limit(cube, 1.5 * metadata["chi_mass"])
        elif metric == "median_mass_response":
            vmax = finite_upper_limit(cube, 1.5)
        elif metric in ("fwhm_resolution", "balanced_score"):
            vmax = finite_upper_limit(cube, 0.5)
        for i, pt_cut in enumerate(pt_cuts):
            title = sample_title(metadata)
            if family_title:
                title += "\n" + family_title
            draw_heatmap(
                cube[i],
                title + "\n" + rf"Jet $p_T$ threshold = {pt_cut} GeV",
                settings["label"],
                metric_dir / f"{metric}_pt{pt_cut}.png",
                vmin=settings["vmin"],
                vmax=vmax,
                cmap=settings["cmap"],
                annotation_format=settings["format"],
            )


def optimize_over_pt(metric_values, efficiency_values, minimum_efficiency, pt_cuts):
    best_value = np.full((len(AK_RADII), len(CA_RADII)), np.nan)
    best_pt = np.full_like(best_value, np.nan)
    for i in range(len(AK_RADII)):
        for j in range(len(CA_RADII)):
            values = metric_values[:, i, j]
            efficiencies = efficiency_values[:, i, j]
            allowed = (
                np.isfinite(values)
                & np.isfinite(efficiencies)
                & (efficiencies >= minimum_efficiency)
            )
            if not np.any(allowed):
                continue
            indices = np.flatnonzero(allowed)
            winner = indices[np.argmin(values[allowed])]
            best_value[i, j] = values[winner]
            best_pt[i, j] = pt_cuts[winner]
    return best_value, best_pt


def make_summary_heatmaps(
    rows, output_dir, args, metadata, pt_cuts=PT_CUTS, family_title=None
):
    """Retained AK/CA summaries after optimizing over pT."""
    summary_dir = output_dir / "summary_heatmaps"
    summary_dir.mkdir(parents=True, exist_ok=True)
    efficiency = metric_cube(rows, "selection_efficiency", pt_cuts)
    for metric, label, cmap in (
        ("off_peak_event_fraction", "Minimum off-peak event fraction", "magma"),
        ("balanced_score", "Minimum balanced score", "viridis_r"),
    ):
        values = metric_cube(rows, metric, pt_cuts)
        best_value, best_pt = optimize_over_pt(
            values, efficiency, args.minimum_efficiency, pt_cuts
        )
        title = sample_title(metadata)
        if family_title:
            title += "\n" + family_title
        draw_heatmap(
            best_value,
            title
            + "\n"
            + rf"Best over $p_T$ ($\epsilon_{{\rm sel}} \geq {100 * args.minimum_efficiency:.0f}\%$)",
            label,
            summary_dir / f"best_{metric}_over_pt.png",
            vmin=0.0,
            vmax=finite_upper_limit(best_value, 0.25),
            cmap=cmap,
            annotation_format=".2f",
        )
        draw_heatmap(
            best_pt,
            title + "\n" + f"Threshold minimizing {label.lower()}",
            r"Optimal jet $p_T$ threshold [GeV]",
            summary_dir / f"optimal_pt_for_{metric}.png",
            vmin=float(pt_cuts.min()),
            vmax=float(pt_cuts.max()),
            cmap="turbo",
            annotation_format=".0f",
        )


def write_csv(rows, output_file, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with output_file.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_ranked_csv(rows, output_file):
    ordered = sorted(
        rows,
        key=lambda row: (
            row["status"] != "ok",
            not np.isfinite(row["balanced_score"]),
            row["balanced_score"] if np.isfinite(row["balanced_score"]) else np.inf,
        ),
    )
    write_csv(ordered, output_file)


def same_point(row, pt_cut, ak_radius, ca_radius):
    return (
        row["pt_cut"] == pt_cut
        and math.isclose(row["ak_radius"], ak_radius)
        and math.isclose(row["ca_radius"], ca_radius)
    )


def eligible_rows(rows, args):
    valid = [
        row
        for row in rows
        if row["status"] == "ok" and np.isfinite(row["balanced_score"])
    ]
    efficient = [
        row for row in valid if row["selection_efficiency"] >= args.minimum_efficiency
    ]
    return efficient if efficient else valid # returns efficient rows if they exist; otherwise return valid rows


def select_interesting_points(rows, args):
    valid = [
        row
        for row in rows
        if row["status"] == "ok" and np.isfinite(row["balanced_score"])
    ]
    pool = eligible_rows(rows, args)
    candidates = []
    nominal = next(
        (
            row
            for row in valid
            if same_point(
                row,
                args.nominal_pt_cut,
                args.nominal_ak_radius,
                args.nominal_ca_radius,
            )
        ),
        None,
    )
    if nominal is not None:
        candidates.append(("nominal", nominal))
    if pool:
        candidates.extend(
            [
                ("best_balanced", min(pool, key=lambda row: row["balanced_score"])),
                ("best_peak_fraction", max(pool, key=lambda row: row["f_peak_event_both"])),
                ("worst_balanced", max(pool, key=lambda row: row["balanced_score"])),
                (
                    "largest_mass_bias",
                    max(valid, key=lambda row: abs(row["median_mass_response"] - 1.0)),
                ),
                ("worst_resolution", max(valid, key=lambda row: row["fwhm_resolution"])),
                (
                    "lowest_efficiency",
                    min(valid, key=lambda row: row["selection_efficiency"]),
                ),
            ]
        )
    unique = []
    seen = set()
    for description, row in candidates:
        key = configuration_key(row)
        if key not in seen:
            seen.add(key)
            unique.append((description, row))
    return unique


def make_detailed_histograms(rows, output_dir, args, metadata, family_title=None):
    selected = select_interesting_points(rows, args)
    histogram_dir = output_dir / "selected_mass_histograms"
    histogram_dir.mkdir(parents=True, exist_ok=True)
    true_mass = metadata["chi_mass"]
    mass_max = args.mass_plot_max_factor * true_mass
    bins = np.linspace(0.0, mass_max, 81)
    overlay_entries = []
    for description, row in selected:
        (
            mass0,
            mass1,
            passes_selection,
            valid_reconstruction,
            _,
            _,
        ) = read_scan_chi_masses(row, args)
        valid_event = passes_selection & valid_reconstruction
        m0 = mass0[valid_event]
        m1 = mass1[valid_event]
        combined = np.concatenate((m0, m1))
        overlay_entries.append((description, row, combined))
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        ax.axvspan(
            args.peak_response_min * true_mass,
            args.peak_response_max * true_mass,
            color="tab:green",
            alpha=0.10,
        )
        ax.hist(m0, bins=bins, histtype="step", linewidth=1.8, label=r"$\chi_0$")
        ax.hist(m1, bins=bins, histtype="step", linewidth=1.8, label=r"$\chi_1$")
        ax.axvline(true_mass, color="black", linestyle="--", linewidth=1.5, label="Generated mass")
        ax.set_xlim(0.0, mass_max)
        ax.set_xlabel(r"Reconstructed $m_\chi$ [GeV]")
        ax.set_ylabel("Chis / bin")
        title = sample_title(metadata)
        if family_title:
            title += "\n" + family_title
        ax.set_title(
            title
            + "\n"
            + f"{description.replace('_', ' ').title()}: "
            + rf"$p_T^{{\rm cut}}={row['pt_cut']}$ GeV, "
            + rf"$R_{{\rm AK}}={row['ak_radius']:.1f}$, "
            + rf"$R_{{\rm CA}}={row['ca_radius']:.1f}$"
        )
        ax.text(
            0.98,
            0.95,
            rf"$f_{{\rm both\ peak}}={row['f_peak_event_both']:.3f}$"
            + "\n"
            + rf"$\epsilon_{{\rm sel}}={row['selection_efficiency']:.3f}$"
            + "\n"
            + rf"median $m_\chi/M_\chi={row['median_mass_response']:.3f}$"
            + "\n"
            + rf"FWHM/peak $={row['fwhm_resolution']:.3f}$"
            + "\n"
            + rf"score $={row['balanced_score']:.3f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(histogram_dir / f"{description}.png", dpi=180)
        plt.close(fig)

    if overlay_entries:
        fig, ax = plt.subplots(figsize=(9.0, 6.5))
        for description, row, masses in overlay_entries:
            label = (
                f"{description.replace('_', ' ')}: "
                f"pT={row['pt_cut']}, AK={row['ak_radius']:.1f}, CA={row['ca_radius']:.1f}"
            )
            ax.hist(masses, bins=bins, density=True, histtype="step", linewidth=1.5, label=label)
        ax.axvspan(
            args.peak_response_min * true_mass,
            args.peak_response_max * true_mass,
            color="tab:green",
            alpha=0.08,
        )
        ax.axvline(true_mass, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlim(0.0, mass_max)
        ax.set_xlabel(r"Reconstructed $m_\chi$ [GeV]")
        ax.set_ylabel("Normalized chis / bin")
        title = sample_title(metadata)
        if family_title:
            title += "\n" + family_title
        ax.set_title(title + "\nSelected scan points")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(histogram_dir / "selected_points_overlay.png", dpi=180)
        plt.close(fig)


def configuration_key(row):
    """A globally comparable key; algorithm family is intentionally first."""
    return (
        row["algorithm_family"],
        row["event_pt_cut"],
        row["pt_cut"],
        round(row["ak_radius"], 2),
        round(row["ca_radius"], 2),
    )


def regime_key(row):
    return (row["suu_mass"], row["chi_mass"], row["decay"])


def configuration_label(config):
    family, event_pt_cut, pt_cut, ak_radius, ca_radius = config
    if family == TWO_THRESHOLD_FAMILY:
        threshold_label = f"two-threshold={event_pt_cut}/{pt_cut}"
    else:
        threshold_label = f"pT={pt_cut}"
    return f"{threshold_label}, AK={ak_radius:.1f}, CA={ca_radius:.1f}"


def short_configuration_label(config):
    family, event_pt_cut, pt_cut, ak_radius, ca_radius = config
    if family == TWO_THRESHOLD_FAMILY:
        return f"2T:{event_pt_cut}/{pt_cut}/{ak_radius:.1f}/{ca_radius:.1f}"
    return f"1T:{pt_cut}/{ak_radius:.1f}/{ca_radius:.1f}"


def get_regime_axes(rows):
    mass_points = sorted({(row["suu_mass"], row["chi_mass"]) for row in rows})
    decays = sorted({row["decay"] for row in rows}, key=decay_sort_key)
    return mass_points, decays


def regime_matrix(mapping, mass_points, decays):
    matrix = np.full((len(mass_points), len(decays)), np.nan)
    for i, (suu_mass, chi_mass) in enumerate(mass_points):
        for j, decay in enumerate(decays):
            matrix[i, j] = mapping.get((suu_mass, chi_mass, decay), np.nan)
    return matrix


def draw_regime_heatmap(
    values,
    mass_points,
    decays,
    title,
    colorbar_label,
    output_file,
    cmap="viridis",
    vmin=None,
    vmax=None,
    annotation_format=".2f",
):
    width = max(8.0, 1.35 * len(decays) + 2.0)
    height = max(4.5, 1.15 * len(mass_points) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    ax.set_xticks(np.arange(len(decays)))
    ax.set_xticklabels(decays)
    ax.set_yticks(np.arange(len(mass_points)))
    ax.set_yticklabels([rf"{suu}/{chi}" for suu, chi in mass_points])
    ax.set_xlabel("Decay channel")
    ax.set_ylabel(r"$M_{S_{uu}}/M_\chi$ [GeV]")
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, format(values[i, j], annotation_format), ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_file, dpi=190)
    plt.close(fig)


def find_best_rows(rows, args):
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "ok" and np.isfinite(row["balanced_score"]):
            grouped[regime_key(row)].append(row)
    best = {}
    for regime, regime_rows in grouped.items():
        efficient = [
            row
            for row in regime_rows
            if row["selection_efficiency"] >= args.minimum_efficiency
        ]
        pool = efficient if efficient else regime_rows
        winner = min(pool, key=lambda row: row["balanced_score"])
        winner["regime_efficiency_constraint_met"] = bool(efficient)
        best[regime] = winner
    return best


def make_best_regime_heatmaps(
    best_rows,
    mass_points,
    decays,
    output_dir,
    args,
    algorithm_family=LEGACY_FAMILY,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    family_title = FAMILY_LABELS[algorithm_family]
    if algorithm_family == TWO_THRESHOLD_FAMILY:
        high_cuts = sorted(
            {
                row["event_pt_cut"]
                for row in best_rows.values()
                if row["event_pt_cut"] is not None
            }
        )
        if len(high_cuts) == 1:
            family_title += rf" ($p_T^{{\rm event}}={high_cuts[0]}$ GeV)"
    metric_specs = [
        ("balanced_score", "Best achievable balanced score", "Balanced score", "viridis_r", ".3f", 0.0, None),
        ("off_peak_event_fraction", "Off-peak fraction at regime optimum", "Off-peak event fraction", "magma", ".3f", 0.0, 1.0),
        ("f_peak_event_both", "Both-chi peak fraction at regime optimum", "Both-chi peak fraction", "viridis", ".3f", 0.0, 1.0),
        (
            "selection_efficiency",
            "Event-selection efficiency at regime optimum",
            "Event-selection efficiency",
            "viridis",
            ".3f",
            0.0,
            1.0,
        ),
        ("median_mass_response", "Median response at regime optimum", r"Median $m_\chi/M_\chi$", "coolwarm", ".3f", 0.5, 1.5),
        ("fwhm_resolution", "Resolution at regime optimum", "Relative FWHM resolution", "plasma", ".3f", 0.0, None),
    ]
    if algorithm_family == TWO_THRESHOLD_FAMILY:
        # These four terms sum exactly to balanced_score. Keep them explicit so
        # the family-specific 3x6 summaries explain why each regime won.
        metric_specs.extend(
            [
                ("score_off_peak_component", "Off-peak score term at regime optimum", "Off-peak score term", "magma", ".3f", 0.0, 1.0),
                ("score_mass_bias_component", "Mass-bias score term at regime optimum", r"$0.1\,|\mathrm{median}(m_\chi/M_\chi)-1|$", "magma", ".3f", 0.0, None),
                ("score_resolution_component", "Resolution score term at regime optimum", "FWHM-resolution score term", "magma", ".3f", 0.0, None),
                ("score_inefficiency_component", "Inefficiency score term at regime optimum", r"$1-\epsilon_{\rm sel}$", "magma", ".3f", 0.0, 1.0),
            ]
        )
    for metric, title, label, cmap, fmt, vmin, vmax in metric_specs:
        mapping = {regime: row[metric] for regime, row in best_rows.items()}
        matrix = regime_matrix(mapping, mass_points, decays)
        if vmax is None:
            vmax = finite_upper_limit(matrix, 0.25)
        draw_regime_heatmap(
            matrix,
            mass_points,
            decays,
            title
            + ("\n" + family_title if algorithm_family == TWO_THRESHOLD_FAMILY else "")
            + "\n"
            + rf"Peak: {args.peak_response_min:.2f}--{args.peak_response_max:.2f}, $\epsilon_{{\rm sel}}\geq{100*args.minimum_efficiency:.0f}\%$",
            label,
            output_dir / f"best_regime_{metric}.png",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annotation_format=fmt,
        )

    threshold_title = "Optimal jet pT threshold"
    threshold_label = r"$p_T^{\rm cut}$ [GeV]"
    threshold_limits = (PT_CUTS.min(), PT_CUTS.max())
    if algorithm_family == TWO_THRESHOLD_FAMILY:
        threshold_title = "Optimal lower (collection) jet pT threshold"
        threshold_label = r"$p_T^{\rm collection}$ [GeV]"
        threshold_limits = (
            TWO_THRESHOLD_LOW_PT_CUTS.min(),
            TWO_THRESHOLD_LOW_PT_CUTS.max(),
        )
    parameter_specs = [
        ("pt_cut", threshold_title, threshold_label, "turbo", ".0f", *threshold_limits),
        ("ak_radius", "Optimal AK radius", r"$R_{\rm AK}$", "viridis", ".1f", AK_RADII.min(), AK_RADII.max()),
        ("ca_radius", "Optimal CA radius", r"$R_{\rm CA}$", "viridis", ".1f", CA_RADII.min(), CA_RADII.max()),
    ]
    if algorithm_family == TWO_THRESHOLD_FAMILY:
        parameter_specs.insert(
            1,
            (
                "event_pt_cut",
                "Event-qualification jet pT threshold",
                r"$p_T^{\rm event}$ [GeV]",
                "turbo",
                ".0f",
                PT_CUTS.min(),
                PT_CUTS.max(),
            ),
        )
    for metric, title, label, cmap, fmt, vmin, vmax in parameter_specs:
        matrix = regime_matrix(
            {regime: row[metric] for regime, row in best_rows.items()},
            mass_points,
            decays,
        )
        draw_regime_heatmap(
            matrix,
            mass_points,
            decays,
            title
            + ("\n" + family_title if algorithm_family == TWO_THRESHOLD_FAMILY else ""),
            label,
            output_dir / f"best_parameter_{metric}.png",
            cmap=cmap,
            vmin=float(vmin),
            vmax=float(vmax),
            annotation_format=fmt,
        )


def profiled_scores(regime_rows, variable, args, pt_cuts=PT_CUTS):
    values = {"pt_cut": pt_cuts, "ak_radius": AK_RADII, "ca_radius": CA_RADII}[variable]
    result = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        candidates = [
            row
            for row in regime_rows
            if row["status"] == "ok"
            and np.isfinite(row["balanced_score"])
            and row["selection_efficiency"] >= args.minimum_efficiency
            and (
                row[variable] == value
                if variable == "pt_cut"
                else math.isclose(row[variable], value)
            )
        ]
        if candidates:
            result[i] = min(row["balanced_score"] for row in candidates)
    return np.asarray(values), result


def make_profiled_score_landscapes(
    rows, mass_points, decays, output_dir, args, pt_cuts=PT_CUTS,
    family_title=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[regime_key(row)].append(row)
    variables = (
        ("pt_cut", r"Jet $p_T$ threshold [GeV]", args.nominal_pt_cut),
        ("ak_radius", r"$R_{\rm AK}$", args.nominal_ak_radius),
        ("ca_radius", r"$R_{\rm CA}$", args.nominal_ca_radius),
    )
    for variable, xlabel, nominal in variables:
        fig, axes = plt.subplots(
            len(mass_points),
            len(decays),
            figsize=(3.0 * len(decays), 2.55 * len(mass_points)),
            sharey=True,
            squeeze=False,
        )
        all_profiles = []
        for i, mass_point in enumerate(mass_points):
            for j, decay in enumerate(decays):
                regime = (*mass_point, decay)
                ax = axes[i, j]
                x, y = profiled_scores(
                    grouped.get(regime, []), variable, args, pt_cuts
                )
                all_profiles.extend(y[np.isfinite(y)])
                if np.any(np.isfinite(y)):
                    ax.plot(x, y, marker="o", markersize=3, linewidth=1.2)
                    winner = np.nanargmin(y)
                    ax.scatter([x[winner]], [y[winner]], color="tab:red", s=22, zorder=3)
                else:
                    ax.text(0.5, 0.5, "No eligible data", transform=ax.transAxes, ha="center", va="center", fontsize=8)
                ax.axvline(nominal, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
                if i == 0:
                    ax.set_title(decay)
                if j == 0:
                    ax.set_ylabel(f"{mass_point[0]}/{mass_point[1]}\nScore")
                if i == len(mass_points) - 1:
                    ax.set_xlabel(xlabel)
                ax.grid(alpha=0.2)
        if all_profiles:
            upper = max(0.25, float(np.quantile(all_profiles, 0.98)) * 1.08)
            for ax in axes.flat:
                ax.set_ylim(0.0, upper)
        title = f"Profiled balanced score versus {variable.replace('_', ' ')}"
        if family_title:
            title += "\n" + family_title
        fig.suptitle(
            title + "\nEach point is minimized over the other two scan parameters",
            y=1.01,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"profiled_score_vs_{variable}.png", dpi=190, bbox_inches="tight")
        plt.close(fig)


def compute_global_statistics(rows, best_rows, args, expected_regimes=None):
    regimes = sorted(
        expected_regimes if expected_regimes is not None else best_rows,
        key=lambda regime: (regime[0], regime[1], decay_sort_key(regime[2])),
    )
    best_scores = {
        regime: row["balanced_score"] for regime, row in best_rows.items()
    }
    by_configuration = defaultdict(dict)
    for row in rows:
        if (
            row["status"] == "ok"
            and np.isfinite(row["balanced_score"])
            and row["selection_efficiency"] >= args.minimum_efficiency
            and regime_key(row) in best_scores
        ):
            by_configuration[configuration_key(row)][regime_key(row)] = row

    statistics = []
    n_regimes = len(regimes)
    for config, regime_rows in by_configuration.items():
        regrets = {
            regime: regime_rows[regime]["balanced_score"] - best_scores[regime]
            for regime in regime_rows
        }
        values = np.asarray(list(regrets.values()), dtype=float)
        coverage = safe_fraction(len(values), n_regimes)
        if len(values) == 0:
            continue
        representative = next(iter(regime_rows.values()))
        result = {
            "configuration": config,
            "algorithm_family": representative["algorithm_family"],
            "event_pt_cut": representative["event_pt_cut"],
            "pt_cut": representative["pt_cut"],
            "ak_radius": representative["ak_radius"],
            "ca_radius": representative["ca_radius"],
            "n_regimes": len(values),
            "regime_coverage": coverage,
            "mean_regret": float(np.mean(values)),
            "median_regret": float(np.median(values)),
            "q90_regret": float(np.quantile(values, 0.90)),
            "worst_regret": float(np.max(values)),
            "mean_score": float(np.mean([row["balanced_score"] for row in regime_rows.values()])),
            "regrets": regrets,
            "rows": regime_rows,
        }
        # Worst-case performance is primary; mean regret breaks near-ties.
        result["global_objective"] = (
            result["worst_regret"] + args.global_mean_weight * result["mean_regret"]
        )
        for delta in args.coverage_deltas:
            result[f"coverage_regret_le_{delta:g}"] = safe_fraction(
                np.count_nonzero(values <= delta), n_regimes
            )
        statistics.append(result)

    eligible = [
        item
        for item in statistics
        if item["regime_coverage"] >= args.minimum_regime_coverage
    ]
    if not eligible and statistics:
        maximum_coverage = max(item["regime_coverage"] for item in statistics)
        print(
            "WARNING: no configuration reaches the requested regime coverage; "
            f"maximum available coverage is {maximum_coverage:.1%}. "
            "Global rankings will not be produced. Lower "
            "--minimum-regime-coverage explicitly to allow an incomplete comparison."
        )
    eligible.sort(key=lambda item: (item["global_objective"], item["mean_regret"]))
    return regimes, statistics, eligible


def write_global_ranking(statistics, output_file, args):
    serializable = []
    for rank, item in enumerate(statistics, start=1):
        row = {
            "rank": rank,
            "algorithm_family": item["algorithm_family"],
            "event_pt_cut": item["event_pt_cut"],
            "pt_cut": item["pt_cut"],
            "ak_radius": item["ak_radius"],
            "ca_radius": item["ca_radius"],
            "n_regimes": item["n_regimes"],
            "regime_coverage": item["regime_coverage"],
            "global_objective": item["global_objective"],
            "mean_score": item["mean_score"],
            "mean_regret": item["mean_regret"],
            "median_regret": item["median_regret"],
            "q90_regret": item["q90_regret"],
            "worst_regret": item["worst_regret"],
        }
        for delta in args.coverage_deltas:
            row[f"coverage_regret_le_{delta:g}"] = item[f"coverage_regret_le_{delta:g}"]
        serializable.append(row)
    write_csv(serializable, output_file)


def regime_display_label(regime):
    suu_mass, chi_mass, decay = regime
    return f"{suu_mass}/{chi_mass}\n{decay}"


def make_regret_heatmap(top, regimes, output_file):
    if not top:
        return
    matrix = np.full((len(top), len(regimes)), np.nan)
    for i, item in enumerate(top):
        for j, regime in enumerate(regimes):
            matrix[i, j] = item["regrets"].get(regime, np.nan)
    fig, ax = plt.subplots(figsize=(max(12.0, 0.78 * len(regimes)), max(6.0, 0.48 * len(top))))
    vmax = finite_upper_limit(matrix, 0.10)
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Additive regret relative to regime optimum")
    ax.set_xticks(np.arange(len(regimes)))
    ax.set_xticklabels(
        [regime_display_label(r) for r in regimes], rotation=45, ha="right"
    )
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(
        [short_configuration_label(item["configuration"]) for item in top]
    )
    ax.set_xlabel(r"Regime: $M_{S_{uu}}/M_\chi$ and decay")
    ax.set_ylabel("Configuration (1T or 2T thresholds / AK / CA)")
    ax.set_title("Regret of top globally ranked configurations")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def make_mean_worst_scatter(statistics, top, output_file):
    if not statistics:
        return
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    marker_cycle = ("o", "s", "^", "D", "v", "P", "X")
    marker_by_ak = {radius: marker_cycle[i % len(marker_cycle)] for i, radius in enumerate(AK_RADII)}
    edge_by_family = {
        LEGACY_FAMILY: "black",
        TWO_THRESHOLD_FAMILY: "deeppink",
    }
    pt_norm = plt.Normalize(float(PT_CUTS.min()), float(PT_CUTS.max()))
    cmap = plt.get_cmap("turbo")
    for item in statistics:
        ax.scatter(
            item["mean_regret"],
            item["worst_regret"],
            c=[cmap(pt_norm(item["pt_cut"]))],
            marker=marker_by_ak[item["ak_radius"]],
            s=30 + 75 * (item["ca_radius"] - CA_RADII.min()) / (CA_RADII.max() - CA_RADII.min()),
            alpha=0.55,
            linewidths=1.0 if item["algorithm_family"] == TWO_THRESHOLD_FAMILY else 0.3,
            edgecolors=edge_by_family[item["algorithm_family"]],
        )
    for rank, item in enumerate(top[:5], start=1):
        ax.scatter(item["mean_regret"], item["worst_regret"], facecolors="none", edgecolors="black", s=170, linewidths=1.5)
        ax.annotate(str(rank), (item["mean_regret"], item["worst_regret"]), xytext=(5, 5), textcoords="offset points")
    scalar = plt.cm.ScalarMappable(norm=pt_norm, cmap=cmap)
    fig.colorbar(scalar, ax=ax, label=r"Collection/single jet $p_T$ threshold [GeV]")
    handles = [Line2D([0], [0], marker=marker_by_ak[r], color="none", markerfacecolor="gray", label=f"AK {r:.1f}", markersize=7) for r in AK_RADII]
    ak_legend = ax.legend(handles=handles, title="Marker: AK radius", fontsize=8, ncol=2, loc="upper right")
    ax.add_artist(ak_legend)
    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="gray",
            markeredgecolor=edge_by_family[family],
            markeredgewidth=1.4 if family == TWO_THRESHOLD_FAMILY else 0.5,
            label=FAMILY_LABELS[family],
            markersize=7,
        )
        for family in FAMILY_ORDER
        if any(item["algorithm_family"] == family for item in statistics)
    ]
    ax.legend(handles=family_handles, title="Marker edge: family", fontsize=8, loc="lower right")
    ax.set_xlabel("Mean additive regret")
    ax.set_ylabel("Worst-case additive regret")
    ax.set_title("Global configuration tradeoff\nMarker size increases with CA radius; circled points are ranks 1--5")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def make_global_distribution_plots(top, ranking_pool, regimes, output_dir):
    if not top:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [short_configuration_label(item["configuration"]) for item in top]
    regret_values = [list(item["regrets"].values()) for item in top]
    fig, ax = plt.subplots(figsize=(max(10.0, 0.7 * len(top)), 6.5))
    ax.boxplot(regret_values, labels=labels, showmeans=True)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Configuration (1T or 2T thresholds / AK / CA)")
    ax.set_ylabel("Additive regret across regimes")
    ax.set_title("Global regret distributions for top configurations")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "global_regret_distributions.png", dpi=200)
    plt.close(fig)

    rank_arrays = []
    for item in top:
        ranks = []
        for regime in regimes:
            comparable = [
                other for other in ranking_pool if regime in other["regrets"]
            ]
            comparable.sort(key=lambda other: other["regrets"][regime])
            rank_lookup = {other["configuration"]: rank + 1 for rank, other in enumerate(comparable)}
            if item["configuration"] in rank_lookup:
                ranks.append(rank_lookup[item["configuration"]])
        rank_arrays.append(ranks)
    fig, ax = plt.subplots(figsize=(max(10.0, 0.7 * len(top)), 6.5))
    ax.boxplot(rank_arrays, labels=labels, showmeans=True)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Configuration (1T or 2T thresholds / AK / CA)")
    ax.set_ylabel("Rank among displayed configurations")
    ax.invert_yaxis()
    ax.set_title("Per-regime rank distributions for top configurations")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "global_rank_distributions.png", dpi=200)
    plt.close(fig)


def make_near_optimal_coverage(top, n_regimes, output_file, requested_deltas):
    if not top or n_regimes == 0:
        return
    maximum = max(max(item["regrets"].values(), default=0.0) for item in top)
    upper = max(0.12, min(0.5, maximum))
    thresholds = np.unique(np.concatenate((np.linspace(0.0, upper, 121), requested_deltas)))
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    for item in top[:10]:
        regrets = np.asarray(list(item["regrets"].values()))
        coverage = [np.count_nonzero(regrets <= delta) / n_regimes for delta in thresholds]
        ax.plot(thresholds, coverage, linewidth=1.8, label=short_configuration_label(item["configuration"]))
    for delta in requested_deltas:
        ax.axvline(delta, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Allowed additive regret")
    ax.set_ylabel("Fraction of all regimes within tolerance")
    ax.set_title("Near-optimal regime coverage")
    ax.grid(alpha=0.25)
    ax.legend(title="1T or 2T thresholds / AK / CA", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def make_global_ranking_suite(
    rows, best_rows, output_dir, args, expected_regimes, suite_label
):
    """Write one self-contained regret/ranking suite for a row pool."""
    regimes, statistics, eligible = compute_global_statistics(
        rows, best_rows, args, expected_regimes=expected_regimes
    )
    if not eligible:
        print(f"WARNING: insufficient common configurations for {suite_label} rankings")
        return regimes, statistics, eligible

    output_dir.mkdir(parents=True, exist_ok=True)
    write_global_ranking(
        eligible, output_dir / "global_configuration_ranking.csv", args
    )
    top = eligible[: args.top_configurations]
    make_regret_heatmap(
        top, regimes, output_dir / "top_configuration_regret_heatmap.png"
    )
    make_mean_worst_scatter(
        eligible, top, output_dir / "mean_vs_worst_case_regret.png"
    )
    make_global_distribution_plots(top, eligible, regimes, output_dir)
    make_near_optimal_coverage(
        top,
        len(regimes),
        output_dir / "near_optimal_coverage.png",
        args.coverage_deltas,
    )
    winner = eligible[0]
    print(f"\nSelected {suite_label} configuration:")
    print(f"  {configuration_label(winner['configuration'])}")
    print(f"  mean regret  = {winner['mean_regret']:.5f}")
    print(f"  worst regret = {winner['worst_regret']:.5f}")
    print(f"  coverage     = {winner['regime_coverage']:.1%}")
    return regimes, statistics, eligible


def find_config_row(rows_by_regime, regime, config):
    return next(
        (row for row in rows_by_regime.get(regime, []) if configuration_key(row) == config and row["status"] == "ok"),
        None,
    )


def make_nominal_comparison(rows, best_rows, global_winner, mass_points, decays, output_dir, args):
    if global_winner is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_regime = defaultdict(list)
    for row in rows:
        rows_by_regime[regime_key(row)].append(row)
    nominal = (
        LEGACY_FAMILY,
        None,
        args.nominal_pt_cut,
        round(args.nominal_ak_radius, 2),
        round(args.nominal_ca_radius, 2),
    )
    selected = global_winner["configuration"]
    mappings = {"nominal": {}, "global": {}, "improvement": {}}
    for regime in best_rows:
        nominal_row = find_config_row(rows_by_regime, regime, nominal)
        selected_row = find_config_row(rows_by_regime, regime, selected)
        if nominal_row is not None:
            mappings["nominal"][regime] = nominal_row["balanced_score"]
        if selected_row is not None:
            mappings["global"][regime] = selected_row["balanced_score"]
        if nominal_row is not None and selected_row is not None:
            mappings["improvement"][regime] = nominal_row["balanced_score"] - selected_row["balanced_score"]

    for name, title, label, cmap in (
        ("nominal", "Nominal-configuration score", "Balanced score", "viridis_r"),
        ("global", "Selected global-configuration score", "Balanced score", "viridis_r"),
        ("improvement", "Score improvement: nominal minus selected global", "Positive means global configuration is better", "coolwarm"),
    ):
        matrix = regime_matrix(mappings[name], mass_points, decays)
        if name == "improvement":
            finite = np.abs(matrix[np.isfinite(matrix)])
            limit = max(0.05, float(np.quantile(finite, 0.98))) if len(finite) else 0.05
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = 0.0, finite_upper_limit(matrix, 0.5)
        draw_regime_heatmap(
            matrix,
            mass_points,
            decays,
            title
            + "\n"
            + (configuration_label(nominal) if name == "nominal" else configuration_label(selected)),
            label,
            output_dir / f"{name}_score_by_regime.png",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annotation_format=".3f",
        )

    nominal_scores = []
    selected_scores = []
    labels = []
    for regime in sorted(
        best_rows,
        key=lambda regime: (regime[0], regime[1], decay_sort_key(regime[2])),
    ):
        nominal_row = find_config_row(rows_by_regime, regime, nominal)
        selected_row = find_config_row(rows_by_regime, regime, selected)
        if nominal_row is not None and selected_row is not None:
            nominal_scores.append(nominal_row["balanced_score"])
            selected_scores.append(selected_row["balanced_score"])
            labels.append(regime_display_label(regime).replace("\n", " "))
    if labels:
        x = np.arange(len(labels))
        width = 0.42
        fig, ax = plt.subplots(figsize=(max(13.0, 0.72 * len(labels)), 6.5))
        ax.bar(x - width / 2, nominal_scores, width, label=configuration_label(nominal))
        ax.bar(x + width / 2, selected_scores, width, label=configuration_label(selected))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Balanced score")
        ax.set_title("Nominal versus selected global configuration")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "nominal_vs_global_score_bars.png", dpi=200)
        plt.close(fig)


def evaluate_sample(metadata, args, output_dir, scan_files):
    sample = metadata["sample"]
    rows = []
    sample_scans = sorted(
        (scan for scan in scan_files if scan["sample"] == sample),
        key=scan_sort_key,
    )
    total_files = len(sample_scans)
    detected_branches = set()
    family_counts = {
        family: sum(scan["algorithm_family"] == family for scan in sample_scans)
        for family in FAMILY_ORDER
    }
    count_text = ", ".join(
        f"{family}={count}" for family, count in family_counts.items() if count
    )
    print(f"\n=== {sample}: {total_files} discovered configurations ({count_text}) ===")
    for file_number, scan in enumerate(sample_scans, start=1):
        filename = Path(scan["filename"])
        print(f"[{file_number:4d}/{total_files}] {filename.name}", flush=True)
        row = {
            **metadata,
            **scan,
            "collection_pt_cut": scan["pt_cut"],
            "evaluation_filename": scan["filename"],
            "selection_evaluation_mode": "unresolved",
            "min_event_jets": np.nan,
        }
        try:
            (
                mass0,
                mass1,
                passes_selection,
                valid_reconstruction,
                label_branch,
                read_info,
            ) = read_scan_chi_masses(row, args)
            row.update(read_info)
            detected_branches.add(label_branch)
            row["status"] = "ok"
            row["error"] = ""
            row.update(
                calculate_metrics(
                    mass0,
                    mass1,
                    passes_selection,
                    valid_reconstruction,
                    metadata["chi_mass"],
                    args,
                )
            )
            row.update(balanced_score_components(row))
            row["balanced_score"] = balanced_score(row, args)
        except Exception as error:
            if args.strict:
                raise
            print(f"    WARNING: {error}")
            row["status"] = "failed"
            row["error"] = str(error)
            for metric in (
                "n_events",
                "n_selected_events",
                "n_valid_events",
                "selection_efficiency",
                "selection_efficiency_err",
                "reconstruction_efficiency",
                "reconstruction_efficiency_err",
                "f_low_event",
                "f_low_event_err",
                "f_low_chi",
                "f_peak_chi",
                "f_peak_event_both",
                "f_peak_event_both_all",
                "off_peak_event_fraction",
                "off_peak_event_fraction_err",
                "median_mass",
                "median_mass_response",
                "fwhm_mass_response",
                "fwhm_peak_mass_response",
                "fwhm_resolution",
                "median_mass_asymmetry",
                "score_off_peak_component",
                "score_mass_bias_component",
                "score_resolution_component",
                "score_inefficiency_component",
                "balanced_score",
            ):
                row[metric] = np.nan
        rows.append(row)

    successful = sum(row["status"] == "ok" for row in rows)
    sample_output = output_dir / "per_sample" / sample
    sample_output.mkdir(parents=True, exist_ok=True)
    write_ranked_csv(rows, sample_output / "ranked_all_families_scan_metrics.csv")

    legacy_rows = [row for row in rows if row["algorithm_family"] == LEGACY_FAMILY]
    if legacy_rows:
        # Preserve every pre-existing per-sample output path and keep these
        # diagnostics strictly legacy-only.
        write_ranked_csv(legacy_rows, sample_output / "ranked_scan_metrics.csv")
        if any(row["status"] == "ok" for row in legacy_rows):
            make_per_threshold_heatmaps(legacy_rows, sample_output, args, metadata)
            make_summary_heatmaps(legacy_rows, sample_output, args, metadata)
            if not args.skip_detailed_histograms:
                make_detailed_histograms(
                    legacy_rows,
                    sample_output,
                    args,
                    metadata,
                    FAMILY_LABELS[LEGACY_FAMILY],
                )

    two_threshold_rows = [
        row for row in rows if row["algorithm_family"] == TWO_THRESHOLD_FAMILY
    ]
    if two_threshold_rows:
        for event_pt_cut in sorted(
            {row["event_pt_cut"] for row in two_threshold_rows}
        ):
            family_rows = [
                row
                for row in two_threshold_rows
                if row["event_pt_cut"] == event_pt_cut
            ]
            two_threshold_output = (
                sample_output / f"two_threshold_{event_pt_cut}"
            )
            two_threshold_output.mkdir(parents=True, exist_ok=True)
            family_title = (
                f"{FAMILY_LABELS[TWO_THRESHOLD_FAMILY]} "
                f"(event threshold {event_pt_cut} GeV)"
            )
            write_ranked_csv(
                family_rows,
                two_threshold_output / "ranked_scan_metrics.csv",
            )
            if any(row["status"] == "ok" for row in family_rows):
                make_per_threshold_heatmaps(
                    family_rows,
                    two_threshold_output,
                    args,
                    metadata,
                    TWO_THRESHOLD_LOW_PT_CUTS,
                    family_title,
                )
                make_summary_heatmaps(
                    family_rows,
                    two_threshold_output,
                    args,
                    metadata,
                    TWO_THRESHOLD_LOW_PT_CUTS,
                    family_title,
                )
                if not args.skip_detailed_histograms:
                    make_detailed_histograms(
                        family_rows,
                        two_threshold_output,
                        args,
                        metadata,
                        family_title,
                    )

    if successful == 0:
        print(f"WARNING: no files were read successfully for {sample}")
        return rows
    print(
        "Using reconstruction label branch(es): "
        + ", ".join(sorted(detected_branches))
    )
    print(f"Finished {sample}: {successful}/{total_files} files read successfully")
    return rows


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_files = discover_scan_files(args.input_dir, args.th_code, strict=args.strict)
    discovered_samples = {scan["sample"] for scan in scan_files}
    samples = args.samples or sorted(
        discovered_samples,
        key=lambda sample: (
            sample_metadata(sample)["suu_mass"],
            sample_metadata(sample)["chi_mass"],
            decay_sort_key(sample_metadata(sample)["decay"]),
        ),
    )
    if not samples:
        raise RuntimeError(
            "No canonical legacy or two-threshold scan files for "
            f"th{args.th_code} were found in {args.input_dir}"
        )
    metadata = [sample_metadata(sample) for sample in samples]
    print(f"Found {len(samples)} samples for th{args.th_code}:")
    for sample in samples:
        sample_scans = [scan for scan in scan_files if scan["sample"] == sample]
        counts = {
            family: sum(scan["algorithm_family"] == family for scan in sample_scans)
            for family in FAMILY_ORDER
        }
        count_text = ", ".join(
            f"{family}={count}" for family, count in counts.items() if count
        )
        print(f"  {sample}: {count_text or 'no matching files'}")

    all_rows = []
    for sample_info in metadata:
        all_rows.extend(evaluate_sample(sample_info, args, output_dir, scan_files))
    successful_rows = [row for row in all_rows if row["status"] == "ok"]
    if not successful_rows:
        raise RuntimeError("No scan files were read successfully")
    write_ranked_csv(all_rows, output_dir / "all_scan_metrics.csv")

    mass_points, decays = get_regime_axes(successful_rows)
    higher = output_dir / "across_regimes"
    higher.mkdir(parents=True, exist_ok=True)
    expected_regimes = {
        (item["suu_mass"], item["chi_mass"], item["decay"]) for item in metadata
    }

    legacy_rows = [
        row for row in successful_rows if row["algorithm_family"] == LEGACY_FAMILY
    ]
    legacy_best_rows = {}
    legacy_eligible_statistics = []
    if legacy_rows:
        legacy_best_rows = find_best_rows(legacy_rows, args)
        # These paths and plot inputs intentionally match the legacy evaluator.
        make_best_regime_heatmaps(
            legacy_best_rows,
            mass_points,
            decays,
            higher / "best_regime",
            args,
            algorithm_family=LEGACY_FAMILY,
        )
        make_profiled_score_landscapes(
            legacy_rows, mass_points, decays, higher / "score_landscapes", args
        )
        legacy_best_summary = []
        for regime in sorted(legacy_best_rows):
            row = dict(legacy_best_rows[regime])
            row["regime"] = regime_display_label(regime).replace("\n", "_")
            legacy_best_summary.append(row)
        write_csv(
            legacy_best_summary,
            higher / "best_configuration_by_regime.csv",
        )
        _, _, legacy_eligible_statistics = make_global_ranking_suite(
            legacy_rows,
            legacy_best_rows,
            higher / "single_threshold" / "global_configuration",
            args,
            expected_regimes,
            "single-threshold",
        )

    two_threshold_rows = [
        row
        for row in successful_rows
        if row["algorithm_family"] == TWO_THRESHOLD_FAMILY
    ]
    for event_pt_cut in sorted(
        {row["event_pt_cut"] for row in two_threshold_rows}
    ):
        family_rows = [
            row for row in two_threshold_rows if row["event_pt_cut"] == event_pt_cut
        ]
        family_best_rows = find_best_rows(family_rows, args)
        family_dir = higher / f"two_threshold_{event_pt_cut}"
        family_dir.mkdir(parents=True, exist_ok=True)
        make_best_regime_heatmaps(
            family_best_rows,
            mass_points,
            decays,
            family_dir / "best_regime",
            args,
            algorithm_family=TWO_THRESHOLD_FAMILY,
        )
        make_profiled_score_landscapes(
            family_rows,
            mass_points,
            decays,
            family_dir / "score_landscapes",
            args,
            pt_cuts=TWO_THRESHOLD_LOW_PT_CUTS,
            family_title=(
                f"{FAMILY_LABELS[TWO_THRESHOLD_FAMILY]} "
                f"(event threshold {event_pt_cut} GeV)"
            ),
        )
        family_best_summary = []
        for regime in sorted(family_best_rows):
            row = dict(family_best_rows[regime])
            row["regime"] = regime_display_label(regime).replace("\n", "_")
            family_best_summary.append(row)
        write_csv(
            family_best_summary,
            family_dir / "best_configuration_by_regime.csv",
        )
        make_global_ranking_suite(
            family_rows,
            family_best_rows,
            family_dir / "global_configuration",
            args,
            expected_regimes,
            f"two-threshold-{event_pt_cut}",
        )

    # The cross-family optimum defines regret only for the explicitly combined
    # ranking/regret/coverage comparison. No combined-family heatmaps are made.
    best_rows = find_best_rows(successful_rows, args)
    regimes, all_statistics, eligible_statistics = make_global_ranking_suite(
        successful_rows,
        best_rows,
        higher / "combined" / "global_configuration",
        args,
        expected_regimes,
        "combined",
    )
    if eligible_statistics:
        global_dir = higher / "combined" / "global_configuration"
        if legacy_eligible_statistics:
            make_nominal_comparison(
                legacy_rows,
                legacy_best_rows,
                legacy_eligible_statistics[0],
                mass_points,
                decays,
                global_dir / "nominal_comparison",
                args,
            )

    print(f"\nOutputs written under: {output_dir.resolve()}")


if __name__ == "__main__":
    main(parse_args())
