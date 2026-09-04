"""Summarize the jet-pT / AK-radius / CA-radius reconstruction scan.

The script reads only PF-candidate four-vectors and stored reconstruction
labels.  It deliberately does not run the full evaluate_ntuplizer.py workflow.
"""

import argparse
import csv
import math
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

from resolution_metrics import relative_fwhm_resolution


TREE_NAME = "existingOptimizationNtuplizer/Events"

# The first name found is used unless --label-branch is supplied.
LABEL_BRANCH_CANDIDATES = (
    "particle_newAlgoLabel",
)

PT_CUTS = np.arange(100, 401, 20, dtype=int)
AK_RADII = np.arange(4, 17, 2, dtype=int) / 10.0
CA_RADII = np.arange(4, 17, 2, dtype=int) / 10.0


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the jet clustering parameter scan.")
    parser.add_argument("--input-dir", default="/eos/uscms/store/user/aji/rootfiles_existingOptimization")
    parser.add_argument("--sample", default="WbWb_4000_1000")
    parser.add_argument("--output-dir", default="evaluate_ntuplizer_pt_ak_ca")
    parser.add_argument("--tree", default=TREE_NAME)
    parser.add_argument("--label-branch", default=None, help="PF reconstruction-label branch. If omitted, detect it automatically.")
    parser.add_argument("--max-events", type=int, default=1000, help="Maximum events read from each file; use -1 for all events.")
    parser.add_argument("--true-chi-mass", type=float, default=1000.0)
    parser.add_argument("--low-mass-max", type=float, default=300.0)
    parser.add_argument("--peak-mass-min", type=float, default=800.0)
    parser.add_argument("--peak-mass-max", type=float, default=1100.0)
    parser.add_argument("--mass-plot-max", type=float, default=2000.0)
    parser.add_argument("--minimum-efficiency", type=float, default=0.90, help="Minimum two-chi efficiency for selecting an optimum scan point.")
    parser.add_argument("--nominal-pt-cut", type=int, default=300)
    parser.add_argument("--nominal-ak-radius", type=float, default=0.8)
    parser.add_argument("--nominal-ca-radius", type=float, default=0.8)
    parser.add_argument("--strict", action="store_true", help="Stop on a missing/bad file.")
    return parser.parse_args()


def scan_filename(input_dir, sample, pt_cut, ak_radius, ca_radius):
    ak_code = int(round(10.0 * ak_radius))
    ca_code = int(round(10.0 * ca_radius))
    name = f"{sample}_jetPtCut{pt_cut}_ak{ak_code}_ca{ca_code}.root"
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
        branches = [
            "particle_px",
            "particle_py",
            "particle_pz",
            "particle_energy",
            label_branch,
        ]
        events = tree.arrays(branches, entry_stop=entry_stop, library="ak")

    labels = events[label_branch]
    components = []
    counts = []

    for chi_label in (1, 2):
        mask = labels == chi_label
        counts.append(np.asarray(ak.to_numpy(ak.sum(mask, axis=1)), dtype=int)) # how many particles from the event are tagged as chi0 or chi1
        components.append(
            tuple(
                np.asarray(ak.to_numpy(ak.sum(events[name][mask], axis=1)), dtype=float)
                for name in (
                    "particle_px",
                    "particle_py",
                    "particle_pz",
                    "particle_energy",
                )
            )
        )

    px0, py0, pz0, energy0 = components[0]
    px1, py1, pz1, energy1 = components[1]
    mass0 = invariant_mass(px0, py0, pz0, energy0)
    mass1 = invariant_mass(px1, py1, pz1, energy1)

    valid0 = (counts[0] > 0) & np.isfinite(mass0) & np.isfinite(energy0) & (energy0 > 0)
    valid1 = (counts[1] > 0) & np.isfinite(mass1) & np.isfinite(energy1) & (energy1 > 0)
    valid_event = valid0 & valid1

    return mass0, mass1, valid_event, label_branch


def safe_fraction(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else np.nan


def binomial_error(fraction, denominator):
    if denominator <= 0 or not np.isfinite(fraction):
        return np.nan
    return float(np.sqrt(fraction * (1.0 - fraction) / denominator))


def calculate_metrics(mass0, mass1, valid_event, args):
    n_events = len(valid_event)
    n_valid = int(np.count_nonzero(valid_event))
    efficiency = safe_fraction(n_valid, n_events)

    if n_valid == 0:
        return {
            "n_events": n_events,
            "n_valid_events": 0,
            "reconstruction_efficiency": efficiency,
            "reconstruction_efficiency_err": binomial_error(efficiency, n_events),
            "f_low_event": np.nan,
            "f_low_event_err": np.nan,
            "f_low_event_all": 0.0 if n_events else np.nan,
            "f_low_chi": np.nan,
            "f_peak_chi": np.nan,
            "f_peak_event_both": np.nan,
            "median_mass": np.nan,
            "fwhm_mass": np.nan,
            "fwhm_peak_mass": np.nan,
            "fwhm_resolution": np.nan,
            "median_mass_asymmetry": np.nan,
        }

    m0 = mass0[valid_event]
    m1 = mass1[valid_event]
    masses = np.concatenate((m0, m1))

    low0 = m0 < args.low_mass_max
    low1 = m1 < args.low_mass_max
    low_event = low0 | low1

    peak0 = (m0 >= args.peak_mass_min) & (m0 <= args.peak_mass_max)
    peak1 = (m1 >= args.peak_mass_min) & (m1 <= args.peak_mass_max)

    f_low_event = float(np.mean(low_event))
    f_low_chi = float(np.mean(np.concatenate((low0, low1))))
    f_peak_chi = float(np.mean(np.concatenate((peak0, peak1))))
    f_peak_event_both = float(np.mean(peak0 & peak1))

    median = np.median(masses)
    resolution, fwhm_mass, fwhm_peak_mass = relative_fwhm_resolution(masses)
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
        "f_low_event_all": safe_fraction(np.count_nonzero(low_event), n_events),
        "f_low_chi": f_low_chi,
        "f_peak_chi": f_peak_chi,
        "f_peak_event_both": f_peak_event_both,
        "median_mass": float(median),
        "fwhm_mass": fwhm_mass,
        "fwhm_peak_mass": fwhm_peak_mass,
        "fwhm_resolution": resolution,
        "median_mass_asymmetry": float(np.nanmedian(asymmetry)),
    }


def balanced_score(row, args):
    required = (
        row["f_low_event"],
        row["median_mass"],
        row["fwhm_resolution"],
        row["reconstruction_efficiency"],
    )
    if not all(np.isfinite(value) for value in required):
        return np.nan

    mass_bias = abs(row["median_mass"] - args.true_chi_mass) / args.true_chi_mass
    inefficiency = 1.0 - row["reconstruction_efficiency"]
    return float(row["f_low_event"] + 0.1 * mass_bias + row["fwhm_resolution"] + inefficiency)


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
                    color="black",
                )

    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def finite_upper_limit(cube, fallback):
    finite = cube[np.isfinite(cube)]
    if len(finite) == 0:
        return fallback
    return max(fallback, float(np.quantile(finite, 0.98))) # set upper limit as 98% quantile


def make_per_threshold_heatmaps(rows, output_dir, args):
    specifications = {
        "f_low_event": {
            "label": rf"Event fraction with $m_\chi < {args.low_mass_max:g}$ GeV",
            "vmin": 0.0,
            "vmax": 1.0,
            "cmap": "magma_r",
            "format": ".2f",
        },
        "f_peak_chi": {
            "label": rf"Chi fraction in {args.peak_mass_min:g}--{args.peak_mass_max:g} GeV",
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
        "fwhm_resolution": {
            "label": r"FWHM / histogram peak",
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
    }

    for metric, settings in specifications.items():
        cube = metric_cube(rows, metric)
        metric_dir = output_dir / "heatmaps" / metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        vmax = settings["vmax"]
        if metric == "median_mass":
            vmax = finite_upper_limit(cube, 1.5 * args.true_chi_mass)
        elif metric == "fwhm_resolution":
            vmax = finite_upper_limit(cube, 0.5)

        for i, pt_cut in enumerate(PT_CUTS):
            draw_heatmap(
                cube[i],
                rf"Jet $p_T$ threshold = {pt_cut} GeV",
                settings["label"],
                metric_dir / f"{metric}_jetPtCut{pt_cut}.png",
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
            allowed_indices = np.flatnonzero(allowed)
            winner = allowed_indices[np.argmin(values[allowed])]
            best_value[i, j] = values[winner]
            best_pt[i, j] = PT_CUTS[winner]

    return best_value, best_pt


def make_summary_heatmaps(rows, output_dir, args):
    summary_dir = output_dir / "summary_heatmaps"
    summary_dir.mkdir(parents=True, exist_ok=True)

    efficiency = metric_cube(rows, "reconstruction_efficiency")

    for metric, label, cmap in (
        ("f_low_event", "Minimum low-mass event fraction", "magma_r"),
        ("balanced_score", "Minimum balanced score", "viridis_r"),
    ):
        values = metric_cube(rows, metric)
        best_value, best_pt = optimize_over_pt(
            values, efficiency, args.minimum_efficiency
        )
        draw_heatmap(
            best_value,
            rf"Best over $p_T$ thresholds ($\epsilon_{{2\chi}} \geq {100.0 * args.minimum_efficiency:.0f}\%$)",
            label,
            summary_dir / f"best_{metric}_over_pt.png",
            vmin=0.0,
            vmax=finite_upper_limit(best_value, 0.25),
            cmap=cmap,
            annotation_format=".2f",
        )
        draw_heatmap(
            best_pt,
            rf"Threshold minimizing {metric.replace('_', ' ')}",
            r"Optimal jet $p_T$ threshold [GeV]",
            summary_dir / f"optimal_pt_for_{metric}.png",
            vmin=float(PT_CUTS.min()),
            vmax=float(PT_CUTS.max()),
            cmap="turbo",
            annotation_format=".0f",
        )


def write_ranked_csv(rows, output_file):
    good_rows = sorted(
        (row for row in rows if row["status"] == "ok"),
        key=lambda row: (
            not np.isfinite(row["balanced_score"]),
            row["balanced_score"] if np.isfinite(row["balanced_score"]) else np.inf,
        ),
    )
    bad_rows = [row for row in rows if row["status"] != "ok"]
    ordered_rows = good_rows + bad_rows
    fieldnames = list(ordered_rows[0].keys())

    with output_file.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered_rows)


def same_point(row, pt_cut, ak_radius, ca_radius):
    return (
        row["pt_cut"] == pt_cut
        and math.isclose(row["ak_radius"], ak_radius)
        and math.isclose(row["ca_radius"], ca_radius)
    )


def select_interesting_points(rows, args):
    valid = [
        row
        for row in rows
        if row["status"] == "ok" and np.isfinite(row["balanced_score"])
    ]
    efficient = [
        row
        for row in valid
        if row["reconstruction_efficiency"] >= args.minimum_efficiency
    ]
    selection_pool = efficient if efficient else valid

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

    if selection_pool:
        candidates.extend(
            [
                ("lowest_failure", min(selection_pool, key=lambda row: row["f_low_event"])),
                ("best_balanced", min(selection_pool, key=lambda row: row["balanced_score"])),
                ("highest_failure", max(selection_pool, key=lambda row: row["f_low_event"])),
                ("largest_mass_bias", max(valid, key=lambda row: abs(row["median_mass"] - args.true_chi_mass))),
                ("worst_resolution", max(valid, key=lambda row: row["fwhm_resolution"])),
                ("lowest_efficiency", min(valid, key=lambda row: row["reconstruction_efficiency"])),
            ]
        )

    # Do not reread or redraw the same scan point under several names.
    unique = []
    seen = set()
    for description, row in candidates:
        key = (row["pt_cut"], row["ak_radius"], row["ca_radius"])
        if key not in seen:
            seen.add(key)
            unique.append((description, row))
    return unique


def make_detailed_histograms(rows, output_dir, args):
    selected = select_interesting_points(rows, args)
    histogram_dir = output_dir / "selected_mass_histograms"
    histogram_dir.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0.0, args.mass_plot_max, 81)
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
        ax.axvspan(0.0, args.low_mass_max, color="tab:red", alpha=0.10)
        ax.axvspan(
            args.peak_mass_min, args.peak_mass_max, color="tab:green", alpha=0.10
        )
        ax.hist(m0, bins=bins, histtype="step", linewidth=1.8, label=r"$\chi_0$")
        ax.hist(m1, bins=bins, histtype="step", linewidth=1.8, label=r"$\chi_1$")
        ax.axvline(
            args.true_chi_mass,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Generated mass",
        )
        ax.set_xlim(0.0, args.mass_plot_max)
        ax.set_xlabel(r"Reconstructed $m_\chi$ [GeV]")
        ax.set_ylabel("Chis / bin")
        ax.set_title(
            f"{description.replace('_', ' ').title()}\n"
            rf"$p_T^{{\rm cut}}={row['pt_cut']}$ GeV, "
            rf"$R_{{\rm AK}}={row['ak_radius']:.1f}$, "
            rf"$R_{{\rm CA}}={row['ca_radius']:.1f}$"
        )
        ax.text(
            0.98,
            0.95,
            rf"$f_{{\rm low}}^{{\rm event}}={row['f_low_event']:.3f}$" + "\n"
            + rf"$\epsilon_{{2\chi}}={row['reconstruction_efficiency']:.3f}$" + "\n"
            + rf"median $m_\chi={row['median_mass']:.0f}$ GeV" + "\n"
            + rf"FWHM/$m_{{\rm peak}}={row['fwhm_resolution']:.3f}$",
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
            ax.hist(
                masses,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                label=label,
            )
        ax.axvline(args.true_chi_mass, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlim(0.0, args.mass_plot_max)
        ax.set_xlabel(r"Reconstructed $m_\chi$ [GeV]")
        ax.set_ylabel("Normalized chis / bin")
        ax.set_title("Selected scan points")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(histogram_dir / "selected_points_overlay.png", dpi=180)
        plt.close(fig)


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total_files = len(PT_CUTS) * len(AK_RADII) * len(CA_RADII)
    file_number = 0
    detected_branch = None

    for pt_cut in PT_CUTS:
        for ak_radius in AK_RADII:
            for ca_radius in CA_RADII:
                file_number += 1
                filename = scan_filename(
                    args.input_dir, args.sample, pt_cut, ak_radius, ca_radius
                )
                print(f"[{file_number:4d}/{total_files}] {filename.name}", flush=True)

                row = {
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
                    row.update(calculate_metrics(mass0, mass1, valid_event, args))
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
                        "f_low_event_all",
                        "f_low_chi",
                        "f_peak_chi",
                        "f_peak_event_both",
                        "median_mass",
                        "fwhm_mass",
                        "fwhm_peak_mass",
                        "fwhm_resolution",
                        "median_mass_asymmetry",
                        "balanced_score",
                    ):
                        row[metric] = np.nan
                rows.append(row)

    if not any(row["status"] == "ok" for row in rows):
        raise RuntimeError("No scan files were read successfully")

    print(f"Using reconstruction label branch: {detected_branch}")
    write_ranked_csv(rows, output_dir / "ranked_scan_metrics.csv")
    make_per_threshold_heatmaps(rows, output_dir, args)
    make_summary_heatmaps(rows, output_dir, args)
    make_detailed_histograms(rows, output_dir, args)

    successful = sum(row["status"] == "ok" for row in rows)
    print(f"Finished: {successful}/{total_files} files read successfully")
    print(f"Outputs written under: {output_dir.resolve()}")


if __name__ == "__main__":
    main(parse_args())
