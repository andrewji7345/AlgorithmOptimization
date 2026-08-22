"""Evaluate a multi-sample jet-pT / AK / CA / thrust scan.

The expected input filename is

    <decay>_<M_Suu>_<M_chi>_pt<pt>_ak<10R_AK>_ca<10R_CA>_th<th>.root

For example

    WbWb_4000_1000_pt380_ak4_ca6_th85.root

The script preserves the per-sample diagnostics from
evaluate_ntuplizer_pt_ak_ca.py and adds comparisons across every discovered
mass point and decay channel.  Sample names and true chi masses are inferred
from filenames unless --samples is supplied.
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


TREE_NAME = "existingOptimizationNtuplizer/Events"
LABEL_BRANCH_CANDIDATES = ("particle_newAlgoLabel",)

PT_CUTS = np.arange(100, 401, 20, dtype=int)
AK_RADII = np.arange(4, 17, 2, dtype=int) / 10.0
CA_RADII = np.arange(4, 17, 2, dtype=int) / 10.0

DECAY_ORDER = ("WbWb", "WbZt", "WbHt", "ZtZt", "HtZt", "HtHt")
SAMPLE_RE = re.compile(r"^(?P<decay>[A-Za-z]+)_(?P<suu_mass>\d+)_(?P<chi_mass>\d+)$")


def parse_number_list(text, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


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
    parser.add_argument("--minimum-efficiency", type=float, default=0.90, help="Minimum two-chi efficiency for an eligible optimum.")
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


def discover_samples(input_dir, th_code):
    input_dir = Path(input_dir)
    suffix = re.compile(rf"^(?P<sample>.+)_pt\d+_ak\d+_ca\d+_th{th_code}\.root$")
    samples = set()
    for filename in input_dir.glob(f"*_th{th_code}.root"):
        match = suffix.fullmatch(filename.name)
        if match is None:
            continue
        sample = match.group("sample")
        sample_metadata(sample) # Check correct form
        samples.add(sample)

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
    entry_stop = None if max_events < 0 else max_events
    with uproot.open(filename) as root_file:
        tree = root_file[tree_name]
        label_branch = choose_label_branch(tree, requested_label_branch)
        names = (
            "particle_px",
            "particle_py",
            "particle_pz",
            "particle_energy",
        )
        events = tree.arrays([*names, label_branch], entry_stop=entry_stop, library="ak")

    labels = events[label_branch]
    components = []
    counts = []
    for chi_label in (1, 2):
        mask = labels == chi_label
        counts.append(np.asarray(ak.to_numpy(ak.sum(mask, axis=1)), dtype=int))
        components.append(
            tuple(
                np.asarray(ak.to_numpy(ak.sum(events[name][mask], axis=1)), dtype=float)
                for name in names
            )
        )

    px0, py0, pz0, energy0 = components[0]
    px1, py1, pz1, energy1 = components[1]
    mass0 = invariant_mass(px0, py0, pz0, energy0)
    mass1 = invariant_mass(px1, py1, pz1, energy1)
    valid0 = (counts[0] > 0) & np.isfinite(mass0) & np.isfinite(energy0) & (energy0 > 0)
    valid1 = (counts[1] > 0) & np.isfinite(mass1) & np.isfinite(energy1) & (energy1 > 0)
    return mass0, mass1, valid0 & valid1, label_branch


def safe_fraction(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else np.nan


def binomial_error(fraction, denominator):
    if denominator <= 0 or not np.isfinite(fraction):
        return np.nan
    return float(np.sqrt(fraction * (1.0 - fraction) / denominator))


def calculate_metrics(mass0, mass1, valid_event, true_chi_mass, args):
    n_events = len(valid_event)
    n_valid = int(np.count_nonzero(valid_event))
    efficiency = safe_fraction(n_valid, n_events)
    empty = {
        "n_events": n_events,
        "n_valid_events": n_valid,
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
        "q16_mass": np.nan,
        "q84_mass": np.nan,
        "median_mass_response": np.nan,
        "q16_mass_response": np.nan,
        "q84_mass_response": np.nan,
        "robust_resolution": np.nan,
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
    q16_mass, median_mass, q84_mass = np.quantile(masses, [0.16, 0.50, 0.84])
    q16_response, median_response, q84_response = np.quantile(responses, [0.16, 0.50, 0.84])
    resolution = (
        (q84_response - q16_response) / (2.0 * median_response)
        if median_response > 0
        else np.nan
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
        "n_valid_events": n_valid,
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
        "q16_mass": float(q16_mass),
        "q84_mass": float(q84_mass),
        "median_mass_response": float(median_response),
        "q16_mass_response": float(q16_response),
        "q84_mass_response": float(q84_response),
        "robust_resolution": float(resolution),
        "median_mass_asymmetry": float(np.nanmedian(asymmetry)),
    }


def balanced_score(row, args):
    required = (
        row["off_peak_event_fraction"],
        row["median_mass_response"],
        row["robust_resolution"],
        row["reconstruction_efficiency"],
    )
    if not all(np.isfinite(value) for value in required):
        return np.nan
    mass_bias = abs(row["median_mass_response"] - 1.0)
    inefficiency = 1.0 - row["reconstruction_efficiency"]
    return float(
        row["off_peak_event_fraction"]
        + 0.1 * mass_bias
        + row["robust_resolution"]
        + inefficiency
    )


def radius_edges(values):
    values = np.asarray(values, dtype=float)
    edges = np.empty(len(values) + 1)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def metric_cube(rows, metric):
    cube = np.full((len(PT_CUTS), len(AK_RADII), len(CA_RADII)), np.nan)
    pt_index = {value: i for i, value in enumerate(PT_CUTS)}
    ak_index = {round(value, 2): i for i, value in enumerate(AK_RADII)}
    ca_index = {round(value, 2): i for i, value in enumerate(CA_RADII)}
    for row in rows:
        if row["status"] != "ok":
            continue
        cube[
            pt_index[row["pt_cut"]],
            ak_index[round(row["ak_radius"], 2)],
            ca_index[round(row["ca_radius"], 2)],
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


def make_per_threshold_heatmaps(rows, output_dir, args, metadata):
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
        "robust_resolution": {
            "label": r"$(Q_{84}-Q_{16})/(2Q_{50})$",
            "vmin": 0.0,
            "vmax": None,
            "cmap": "plasma",
            "format": ".2f",
        },
        "reconstruction_efficiency": {
            "label": "Two-chi reconstruction efficiency",
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
        cube = metric_cube(rows, metric)
        metric_dir = output_dir / "heatmaps" / metric
        metric_dir.mkdir(parents=True, exist_ok=True)
        vmax = settings["vmax"]
        if metric == "median_mass":
            vmax = finite_upper_limit(cube, 1.5 * metadata["chi_mass"])
        elif metric == "median_mass_response":
            vmax = finite_upper_limit(cube, 1.5)
        elif metric in ("robust_resolution", "balanced_score"):
            vmax = finite_upper_limit(cube, 0.5)
        for i, pt_cut in enumerate(PT_CUTS):
            draw_heatmap(
                cube[i],
                sample_title(metadata) + "\n" + rf"Jet $p_T$ threshold = {pt_cut} GeV",
                settings["label"],
                metric_dir / f"{metric}_pt{pt_cut}.png",
                vmin=settings["vmin"],
                vmax=vmax,
                cmap=settings["cmap"],
                annotation_format=settings["format"],
            )


def optimize_over_pt(metric_values, efficiency_values, minimum_efficiency):
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
            best_pt[i, j] = PT_CUTS[winner]
    return best_value, best_pt


def make_summary_heatmaps(rows, output_dir, args, metadata):
    """Retained AK/CA summaries after optimizing over pT."""
    summary_dir = output_dir / "summary_heatmaps"
    summary_dir.mkdir(parents=True, exist_ok=True)
    efficiency = metric_cube(rows, "reconstruction_efficiency")
    for metric, label, cmap in (
        ("off_peak_event_fraction", "Minimum off-peak event fraction", "magma"),
        ("balanced_score", "Minimum balanced score", "viridis_r"),
    ):
        values = metric_cube(rows, metric)
        best_value, best_pt = optimize_over_pt(
            values, efficiency, args.minimum_efficiency
        )
        draw_heatmap(
            best_value,
            sample_title(metadata)
            + "\n"
            + rf"Best over $p_T$ ($\epsilon_{{2\chi}} \geq {100 * args.minimum_efficiency:.0f}\%$)",
            label,
            summary_dir / f"best_{metric}_over_pt.png",
            vmin=0.0,
            vmax=finite_upper_limit(best_value, 0.25),
            cmap=cmap,
            annotation_format=".2f",
        )
        draw_heatmap(
            best_pt,
            sample_title(metadata) + "\n" + f"Threshold minimizing {label.lower()}",
            r"Optimal jet $p_T$ threshold [GeV]",
            summary_dir / f"optimal_pt_for_{metric}.png",
            vmin=float(PT_CUTS.min()),
            vmax=float(PT_CUTS.max()),
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
        row for row in valid if row["reconstruction_efficiency"] >= args.minimum_efficiency
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
                ("worst_resolution", max(valid, key=lambda row: row["robust_resolution"])),
                (
                    "lowest_efficiency",
                    min(valid, key=lambda row: row["reconstruction_efficiency"]),
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


def make_detailed_histograms(rows, output_dir, args, metadata):
    selected = select_interesting_points(rows, args)
    histogram_dir = output_dir / "selected_mass_histograms"
    histogram_dir.mkdir(parents=True, exist_ok=True)
    true_mass = metadata["chi_mass"]
    mass_max = args.mass_plot_max_factor * true_mass
    bins = np.linspace(0.0, mass_max, 81)
    overlay_entries = []
    for description, row in selected:
        mass0, mass1, valid_event, _ = read_chi_masses(
            row["filename"], args.tree, args.label_branch, args.max_events
        )
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
        ax.set_title(
            sample_title(metadata)
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
            + rf"$\epsilon_{{2\chi}}={row['reconstruction_efficiency']:.3f}$"
            + "\n"
            + rf"median $m_\chi/M_\chi={row['median_mass_response']:.3f}$"
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
        ax.set_title(sample_title(metadata) + "\nSelected scan points")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(histogram_dir / "selected_points_overlay.png", dpi=180)
        plt.close(fig)


def configuration_key(row):
    return (row["pt_cut"], round(row["ak_radius"], 2), round(row["ca_radius"], 2))


def regime_key(row):
    return (row["suu_mass"], row["chi_mass"], row["decay"])


def configuration_label(config):
    pt_cut, ak_radius, ca_radius = config
    return f"pT={pt_cut}, AK={ak_radius:.1f}, CA={ca_radius:.1f}"


def short_configuration_label(config):
    pt_cut, ak_radius, ca_radius = config
    return f"{pt_cut}/{ak_radius:.1f}/{ca_radius:.1f}"


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
    ax.set_xticks(np.arange(len(decays)), labels=decays)
    ax.set_yticks(
        np.arange(len(mass_points)),
        labels=[rf"{suu}/{chi}" for suu, chi in mass_points],
    )
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
            if row["reconstruction_efficiency"] >= args.minimum_efficiency
        ]
        pool = efficient if efficient else regime_rows
        winner = min(pool, key=lambda row: row["balanced_score"])
        winner["regime_efficiency_constraint_met"] = bool(efficient)
        best[regime] = winner
    return best


def make_best_regime_heatmaps(best_rows, mass_points, decays, output_dir, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_specs = (
        ("balanced_score", "Best achievable balanced score", "Balanced score", "viridis_r", ".3f", 0.0, None),
        ("off_peak_event_fraction", "Off-peak fraction at regime optimum", "Off-peak event fraction", "magma", ".3f", 0.0, 1.0),
        ("f_peak_event_both", "Both-chi peak fraction at regime optimum", "Both-chi peak fraction", "viridis", ".3f", 0.0, 1.0),
        ("reconstruction_efficiency", "Efficiency at regime optimum", "Two-chi efficiency", "viridis", ".3f", 0.0, 1.0),
        ("median_mass_response", "Median response at regime optimum", r"Median $m_\chi/M_\chi$", "coolwarm", ".3f", 0.5, 1.5),
        ("robust_resolution", "Resolution at regime optimum", "Robust relative resolution", "plasma", ".3f", 0.0, None),
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
            title + "\n" + rf"Peak: {args.peak_response_min:.2f}--{args.peak_response_max:.2f}, $\epsilon_{{2\chi}}\geq{100*args.minimum_efficiency:.0f}\%$",
            label,
            output_dir / f"best_regime_{metric}.png",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annotation_format=fmt,
        )

    parameter_specs = (
        ("pt_cut", "Optimal jet pT threshold", r"$p_T^{\rm cut}$ [GeV]", "turbo", ".0f", PT_CUTS.min(), PT_CUTS.max()),
        ("ak_radius", "Optimal AK radius", r"$R_{\rm AK}$", "viridis", ".1f", AK_RADII.min(), AK_RADII.max()),
        ("ca_radius", "Optimal CA radius", r"$R_{\rm CA}$", "viridis", ".1f", CA_RADII.min(), CA_RADII.max()),
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
            title,
            label,
            output_dir / f"best_parameter_{metric}.png",
            cmap=cmap,
            vmin=float(vmin),
            vmax=float(vmax),
            annotation_format=fmt,
        )


def profiled_scores(regime_rows, variable, args):
    values = {"pt_cut": PT_CUTS, "ak_radius": AK_RADII, "ca_radius": CA_RADII}[variable]
    result = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        candidates = [
            row
            for row in regime_rows
            if row["status"] == "ok"
            and np.isfinite(row["balanced_score"])
            and row["reconstruction_efficiency"] >= args.minimum_efficiency
            and (
                row[variable] == value
                if variable == "pt_cut"
                else math.isclose(row[variable], value)
            )
        ]
        if candidates:
            result[i] = min(row["balanced_score"] for row in candidates)
    return np.asarray(values), result


def make_profiled_score_landscapes(rows, mass_points, decays, output_dir, args):
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
                x, y = profiled_scores(grouped.get(regime, []), variable, args)
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
        fig.suptitle(
            f"Profiled balanced score versus {variable.replace('_', ' ')}\n"
            + "Each point is minimized over the other two scan parameters",
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
            and row["reconstruction_efficiency"] >= args.minimum_efficiency
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
        result = {
            "configuration": config,
            "pt_cut": config[0],
            "ak_radius": config[1],
            "ca_radius": config[2],
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
    ax.set_xticks(np.arange(len(regimes)), labels=[regime_display_label(r) for r in regimes], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(top)), labels=[short_configuration_label(item["configuration"]) for item in top])
    ax.set_xlabel(r"Regime: $M_{S_{uu}}/M_\chi$ and decay")
    ax.set_ylabel(r"Configuration: $p_T^{\rm cut}/R_{\rm AK}/R_{\rm CA}$")
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
            linewidths=0.2,
            edgecolors="black",
        )
    for rank, item in enumerate(top[:5], start=1):
        ax.scatter(item["mean_regret"], item["worst_regret"], facecolors="none", edgecolors="black", s=170, linewidths=1.5)
        ax.annotate(str(rank), (item["mean_regret"], item["worst_regret"]), xytext=(5, 5), textcoords="offset points")
    scalar = plt.cm.ScalarMappable(norm=pt_norm, cmap=cmap)
    fig.colorbar(scalar, ax=ax, label=r"Jet $p_T$ threshold [GeV]")
    handles = [Line2D([0], [0], marker=marker_by_ak[r], color="none", markerfacecolor="gray", label=f"AK {r:.1f}", markersize=7) for r in AK_RADII]
    ax.legend(handles=handles, title="Marker: AK radius", fontsize=8, ncols=2)
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
    ax.set_xlabel(r"Configuration: $p_T^{\rm cut}/R_{\rm AK}/R_{\rm CA}$")
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
    ax.set_xlabel(r"Configuration: $p_T^{\rm cut}/R_{\rm AK}/R_{\rm CA}$")
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
    ax.legend(title=r"$p_T^{\rm cut}/R_{\rm AK}/R_{\rm CA}$", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


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
        ax.set_xticks(x, labels=labels, rotation=45, ha="right")
        ax.set_ylabel("Balanced score")
        ax.set_title("Nominal versus selected global configuration")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "nominal_vs_global_score_bars.png", dpi=200)
        plt.close(fig)


def evaluate_sample(metadata, args, output_dir):
    sample = metadata["sample"]
    rows = []
    total_files = len(PT_CUTS) * len(AK_RADII) * len(CA_RADII)
    detected_branch = None
    print(f"\n=== {sample}: {total_files} configurations ===")
    file_number = 0
    for pt_cut in PT_CUTS:
        for ak_radius in AK_RADII:
            for ca_radius in CA_RADII:
                file_number += 1
                filename = scan_filename(
                    args.input_dir,
                    sample,
                    pt_cut,
                    ak_radius,
                    ca_radius,
                    args.th_code,
                )
                print(f"[{file_number:4d}/{total_files}] {filename.name}", flush=True)
                row = {
                    **metadata,
                    "th_code": args.th_code,
                    "pt_cut": int(pt_cut),
                    "ak_radius": float(ak_radius),
                    "ca_radius": float(ca_radius),
                    "filename": str(filename),
                }
                try:
                    if not filename.is_file():
                        raise FileNotFoundError(filename)
                    mass0, mass1, valid_event, label_branch = read_chi_masses(
                        filename, args.tree, args.label_branch, args.max_events
                    )
                    detected_branch = label_branch
                    row["status"] = "ok"
                    row["error"] = ""
                    row.update(
                        calculate_metrics(
                            mass0, mass1, valid_event, metadata["chi_mass"], args
                        )
                    )
                    row["balanced_score"] = balanced_score(row, args)
                except Exception as error:
                    if args.strict:
                        raise
                    print(f"    WARNING: {error}")
                    row["status"] = "failed"
                    row["error"] = str(error)
                    for metric in (
                        "n_events",
                        "n_valid_events",
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
                        "q16_mass",
                        "q84_mass",
                        "median_mass_response",
                        "q16_mass_response",
                        "q84_mass_response",
                        "robust_resolution",
                        "median_mass_asymmetry",
                        "balanced_score",
                    ):
                        row[metric] = np.nan
                rows.append(row)

    successful = sum(row["status"] == "ok" for row in rows)
    if successful == 0:
        print(f"WARNING: no files were read successfully for {sample}")
        return rows
    sample_output = output_dir / "per_sample" / sample
    sample_output.mkdir(parents=True, exist_ok=True)
    print(f"Using reconstruction label branch: {detected_branch}")
    write_ranked_csv(rows, sample_output / "ranked_scan_metrics.csv")
    make_per_threshold_heatmaps(rows, sample_output, args, metadata)
    make_summary_heatmaps(rows, sample_output, args, metadata)
    if not args.skip_detailed_histograms:
        make_detailed_histograms(rows, sample_output, args, metadata)
    print(f"Finished {sample}: {successful}/{total_files} files read successfully")
    return rows


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = args.samples or discover_samples(args.input_dir, args.th_code)
    if not samples:
        raise RuntimeError(
            f"No samples matching '*_th{args.th_code}.root' were found in {args.input_dir}"
        )
    metadata = [sample_metadata(sample) for sample in samples]
    print(f"Found {len(samples)} samples for th{args.th_code}:")
    for sample in samples:
        print(f"  {sample}")

    all_rows = []
    for sample_info in metadata:
        all_rows.extend(evaluate_sample(sample_info, args, output_dir))
    successful_rows = [row for row in all_rows if row["status"] == "ok"]
    if not successful_rows:
        raise RuntimeError("No scan files were read successfully")
    write_ranked_csv(all_rows, output_dir / "all_scan_metrics.csv")

    mass_points, decays = get_regime_axes(successful_rows)
    best_rows = find_best_rows(successful_rows, args)
    higher = output_dir / "across_regimes"
    make_best_regime_heatmaps(best_rows, mass_points, decays, higher / "best_regime", args)
    make_profiled_score_landscapes(
        successful_rows, mass_points, decays, higher / "score_landscapes", args
    )

    best_summary = []
    for regime in sorted(best_rows):
        row = dict(best_rows[regime])
        row["regime"] = regime_display_label(regime).replace("\n", "_")
        best_summary.append(row)
    write_csv(best_summary, higher / "best_configuration_by_regime.csv")

    expected_regimes = {
        (item["suu_mass"], item["chi_mass"], item["decay"]) for item in metadata
    }
    regimes, all_statistics, eligible_statistics = compute_global_statistics(
        successful_rows, best_rows, args, expected_regimes=expected_regimes
    )
    if not eligible_statistics:
        print("WARNING: insufficient common configurations for global comparison plots")
    else:
        top = eligible_statistics[: args.top_configurations]
        global_dir = higher / "global_configuration"
        global_dir.mkdir(parents=True, exist_ok=True)
        write_global_ranking(eligible_statistics, global_dir / "global_configuration_ranking.csv", args)
        make_regret_heatmap(top, regimes, global_dir / "top_configuration_regret_heatmap.png")
        make_mean_worst_scatter(
            eligible_statistics,
            top,
            global_dir / "mean_vs_worst_case_regret.png",
        )
        make_global_distribution_plots(top, eligible_statistics, regimes, global_dir)
        make_near_optimal_coverage(
            top,
            len(regimes),
            global_dir / "near_optimal_coverage.png",
            args.coverage_deltas,
        )
        make_nominal_comparison(
            successful_rows,
            best_rows,
            eligible_statistics[0],
            mass_points,
            decays,
            global_dir / "nominal_comparison",
            args,
        )
        winner = eligible_statistics[0]
        print("\nSelected global configuration:")
        print(f"  {configuration_label(winner['configuration'])}")
        print(f"  mean regret  = {winner['mean_regret']:.5f}")
        print(f"  worst regret = {winner['worst_regret']:.5f}")
        print(f"  coverage     = {winner['regime_coverage']:.1%}")

    print(f"\nOutputs written under: {output_dir.resolve()}")


if __name__ == "__main__":
    main(parse_args())