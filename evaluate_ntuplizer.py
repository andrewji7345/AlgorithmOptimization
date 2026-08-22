"""Evaluate old/new six-step reconstruction methods from one ntuple.

Usage:
  python3 evaluate_ntuplizer_old_new.py --old
  python3 evaluate_ntuplizer_old_new.py --new
  python3 evaluate_ntuplizer_old_new.py --both

Comparable 1D distributions are overplotted in --both mode.  Two-dimensional
diagnostics are saved separately for each method so their bin counts stay clear.
"""

import argparse
import os

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot


################################################################################
# Helper functions
################################################################################

def invariant_mass(px, py, pz, E):
    """Calculate invariant mass for scalars, numpy arrays, or awkward arrays."""
    return np.sqrt(np.maximum(E**2 - px**2 - py**2 - pz**2, 0.0))


def delta_r(eta1, phi1, eta2, phi2):
    """Calculate delta R between two objects."""
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt((eta1 - eta2)**2 + dphi**2)


def sum_p4(mask, px, py, pz, E):
    """Sum four-vectors of particles passing a boolean mask."""
    return np.sum(px[mask]), np.sum(py[mask]), np.sum(pz[mask]), np.sum(E[mask])


def event_mass(mask, px, py, pz, E):
    """Reconstruct invariant mass from selected particles."""
    return invariant_mass(*sum_p4(mask, px, py, pz, E))


def build_jets(indices, px, py, pz, E):
    """Build jet four-vectors from PF candidates sharing an index."""
    jets = {}
    indices = np.asarray(indices)

    for index in np.unique(indices):
        if index < 0:
            continue
        mask = indices == index
        jpx, jpy, jpz, jE = sum_p4(mask, px, py, pz, E)
        pt = np.hypot(jpx, jpy)
        eta = np.arcsinh(jpz / pt) if pt > 0 else 0.0
        phi = np.arctan2(jpy, jpx) if pt > 0 else 0.0
        jets[int(index)] = (jpx, jpy, jpz, jE, eta, phi)

    return jets


def simplify_truth_labels(truth_labels):
    """Convert detailed labels to 0=background, 1=chi0, 2=chi1."""
    simple = np.zeros(len(truth_labels), dtype=np.int32)
    for i, labels in enumerate(truth_labels):
        labels = list(labels)
        has_chi0 = any(1 <= label <= 10 for label in labels)
        has_chi1 = any(11 <= label <= 20 for label in labels)
        if has_chi0 and not has_chi1:
            simple[i] = 1
        elif has_chi1 and not has_chi0:
            simple[i] = 2
    return simple


def get_method_configs(ak_radius, ca_radius):
    """Centralize the branch and presentation differences between methods."""
    return {
        "old": {
            "display": "Old (slimmedJetsAK8)",
            "label_branch": "particle_oldAlgoLabel",
            "ca_index_branch": "particle_oldAlgoCA8Index",
            "ak_index_branch": "particle_oldAlgoAK8Index",
            "ak_pt_branch": "ak8_pt",
            "ak_name": "AK8",
            "ca_name": "CA8",
            "color": "tab:blue",
            "linestyle": "-",
        },
        "new": {
            "display": f"New (reclustered AK{ak_radius}/CA{ca_radius})",
            "label_branch": "particle_newAlgoLabel",
            "ca_index_branch": "particle_newAlgoCAIndex",
            "ak_index_branch": "particle_newAlgoAKIndex",
            "ak_pt_branch": None,
            "ak_name": f"AK{ak_radius}",
            "ca_name": f"CA{ca_radius}",
            "color": "tab:orange",
            "linestyle": "--",
        },
    }


def pairwise_delta_rs(labels, jets):
    """Calculate all pairwise delta-R values for the selected jets."""
    values = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            eta_i, phi_i = jets[labels[i]][4:6]
            eta_j, phi_j = jets[labels[j]][4:6]
            values.append(delta_r(eta_i, phi_i, eta_j, phi_j))
    return values


def save_close(output, filename):
    plt.tight_layout()
    plt.savefig(os.path.join(output, filename))
    plt.close()


################################################################################
# Event reconstruction
################################################################################

def reconstruct_event(event, selected_methods, method_configs):
    """Reconstruct truth and all requested algorithm quantities for one event."""
    px = np.asarray(event["particle_px"])
    py = np.asarray(event["particle_py"])
    pz = np.asarray(event["particle_pz"])
    E = np.asarray(event["particle_energy"])
    truth = simplify_truth_labels(event["particle_truthLabel"])
    truth_chi0 = truth == 1
    truth_chi1 = truth == 2

    result = {
        "total_mass": event_mass(np.ones(len(px), dtype=bool), px, py, pz, E),
        "truth_suu_mass": event_mass(truth_chi0 | truth_chi1, px, py, pz, E),
        "truth_chi0_mass": event_mass(truth_chi0, px, py, pz, E),
        "truth_chi1_mass": event_mass(truth_chi1, px, py, pz, E),
        "n_pf": len(px),
        "n_truth_chi0": int(np.sum(truth_chi0)),
        "n_truth_chi1": int(np.sum(truth_chi1)),
        "n_truth_unassigned": int(np.sum(truth == 0)),
        "algorithms": {},
    }

    for method in selected_methods:
        config = method_configs[method]
        labels = np.asarray(event[config["label_branch"]])
        ca_indices = np.asarray(event[config["ca_index_branch"]])
        ak_indices = np.asarray(event[config["ak_index_branch"]])
        ca_jets = build_jets(ca_indices, px, py, pz, E)
        ak_jets = build_jets(ak_indices, px, py, pz, E)

        if config["ak_pt_branch"]:
            # Corrected pT from the input slimmedJetsAK8 collection.
            ak_pts = np.asarray(event[config["ak_pt_branch"]], dtype=float)
        else:
            # Custom AK jets are reconstructed directly from PF constituents.
            ak_pts = np.asarray([np.hypot(jet[0], jet[1]) for jet in ak_jets.values()])

        chi0_mask = labels == 1
        chi1_mask = labels == 2
        sj_kinematics = []

        for sj_label in [1, 2]:
            sj_mask = labels == sj_label
            if not np.any(sj_mask):
                continue
            sj_px, sj_py, sj_pz, sj_E = sum_p4(sj_mask, px, py, pz, E)
            ca_labels = [int(x) for x in np.unique(ca_indices[sj_mask])
                         if x >= 0 and int(x) in ca_jets]
            ak_labels = [int(x) for x in np.unique(ak_indices[sj_mask])
                         if x >= 0 and int(x) in ak_jets]
            sj_kinematics.append({
                "mass": float(invariant_mass(sj_px, sj_py, sj_pz, sj_E)),
                "energy": float(sj_E),
                "pt": float(np.hypot(sj_px, sj_py)),
                "n_constituents": int(np.sum(sj_mask)),
                "n_ca": len(ca_labels),
                "n_ak": len(ak_labels),
                "ca_delta_rs": pairwise_delta_rs(ca_labels, ca_jets),
                "ak_delta_rs": pairwise_delta_rs(ak_labels, ak_jets),
            })

        result["algorithms"][method] = {
            "suu_mass": event_mass(chi0_mask | chi1_mask, px, py, pz, E),
            "chi0_mass": event_mass(chi0_mask, px, py, pz, E),
            "chi1_mass": event_mass(chi1_mask, px, py, pz, E),
            "n_chi0": int(np.sum(chi0_mask)),
            "n_chi1": int(np.sum(chi1_mask)),
            "n_unassigned": int(np.sum(labels == 0)),
            "n_ak_event": len(ak_pts),
            "n_ak_event_300": int(np.count_nonzero(ak_pts >= 300.0)),
            "sj_kinematics": sj_kinematics,
            "ak_kinematics": ak_jets,
            "ak_pts": ak_pts,
        }

    return result


################################################################################
# Main evaluation
################################################################################

def main(args):
    selected_methods = ["old"] if args.old else ["new"] if args.new else ["old", "new"]

    print(f"Opening file {args.input}\n")
    root_file = uproot.open(args.input)
    tree = root_file["existingOptimizationNtuplizer/Events"]
    branches = tree.arrays(library="ak", entry_stop=args.num_events)
    n_events = len(branches)
    if n_events == 0:
        raise RuntimeError("The requested input contains no events.")

    jet_pt_cut = int(branches["jetPtCut"][0])
    ak_radius = int(round(float(branches["akRadius"][0]) * 10))
    ca_radius = int(round(float(branches["caRadius"][0]) * 10))
    configs = get_method_configs(ak_radius, ca_radius)

    required = {"particle_px", "particle_py", "particle_pz", "particle_energy",
                "particle_truthLabel", "gen_Suu_E", "gen_Suu_px", "gen_Suu_py",
                "gen_Suu_pz", "nParticles"}
    for method in selected_methods:
        config = configs[method]
        required.update([config["label_branch"], config["ca_index_branch"], config["ak_index_branch"]])
        if config["ak_pt_branch"]:
            required.add(config["ak_pt_branch"])
    missing = sorted(required - set(branches.fields))
    if missing:
        raise KeyError("Missing branches required by selected method(s): " + ", ".join(missing))

    output = (f"{args.output}_jetPtCut{jet_pt_cut}_ak{ak_radius}_ca{ca_radius}_{args.method}/")
    os.makedirs(output, exist_ok=True)
    print(f"Loaded {n_events} events, {len(branches.fields)} branches\n")

    results = []
    for i in range(n_events):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:6d} / {n_events}")
        results.append(reconstruct_event(branches[i], selected_methods, configs))

    total_mass = np.asarray([r["total_mass"] for r in results])
    truth_suu_mass = np.asarray([r["truth_suu_mass"] for r in results])
    truth_chi0_mass = np.asarray([r["truth_chi0_mass"] for r in results])
    truth_chi1_mass = np.asarray([r["truth_chi1_mass"] for r in results])
    n_pf = np.asarray([r["n_pf"] for r in results])
    n_truth_chi0 = np.asarray([r["n_truth_chi0"] for r in results])
    n_truth_chi1 = np.asarray([r["n_truth_chi1"] for r in results])
    n_truth_unassigned = np.asarray([r["n_truth_unassigned"] for r in results])

    method_data = {}
    for method in selected_methods:
        data = {
            "suu_mass": np.asarray([r["algorithms"][method]["suu_mass"] for r in results]),
            "chi0_mass": np.asarray([r["algorithms"][method]["chi0_mass"] for r in results]),
            "chi1_mass": np.asarray([r["algorithms"][method]["chi1_mass"] for r in results]),
            "n_chi0": np.asarray([r["algorithms"][method]["n_chi0"] for r in results]),
            "n_chi1": np.asarray([r["algorithms"][method]["n_chi1"] for r in results]),
            "n_unassigned": np.asarray([r["algorithms"][method]["n_unassigned"] for r in results]),
            "all_sjs": [], "low_mass_sjs": [], "high_mass_sjs": [],
            "all_sjs_per_event": [],
        }
        for r in results:
            event_sjs = r["algorithms"][method]["sj_kinematics"]
            data["all_sjs_per_event"].append(event_sjs)
            data["all_sjs"].extend(event_sjs)
            for sj in event_sjs:
                if sj["mass"] < 300:
                    data["low_mass_sjs"].append(sj)
                elif sj["mass"] > 700:
                    data["high_mass_sjs"].append(sj)
        method_data[method] = data

    ################################################################################
    # Energy accounting
    ################################################################################

    truth_frac = {"chi0": [], "chi1": []}
    algo_frac = {m: {"chi0": [], "chi1": []} for m in selected_methods}
    visible_energy_ratio, visible_mass_ratio, bad_events = [], [], []

    for iev in range(n_events):
        event = branches[iev]
        E = np.asarray(event["particle_energy"])
        truth = simplify_truth_labels(event["particle_truthLabel"])
        visible_E = np.sum(E[truth > 0])
        if visible_E <= 0:
            continue
        gen_E = float(event["gen_Suu_E"])
        gen_mass = invariant_mass(float(event["gen_Suu_px"]), float(event["gen_Suu_py"]),
                                  float(event["gen_Suu_pz"]), gen_E)
        if gen_E > 0:
            visible_energy_ratio.append(visible_E / gen_E)
        if gen_mass > 0:
            visible_mass_ratio.append(results[iev]["truth_suu_mass"] / gen_mass)
        truth_frac["chi0"].append(np.sum(E[truth == 1]) / visible_E)
        truth_frac["chi1"].append(np.sum(E[truth == 2]) / visible_E)

        for method in selected_methods:
            labels = np.asarray(event[configs[method]["label_branch"]])
            algo_frac[method]["chi0"].append(np.sum(E[labels == 1]) / visible_E)
            algo_frac[method]["chi1"].append(np.sum(E[labels == 2]) / visible_E)

        reasons = []
        for key, limit in [("truth_suu_mass", 2500), ("truth_chi0_mass", 200),
                           ("truth_chi1_mass", 200)]:
            if results[iev][key] < limit:
                reasons.append(f"low {key}")
        for method in selected_methods:
            algo = results[iev]["algorithms"][method]
            for key, limit in [("suu_mass", 2500), ("chi0_mass", 200), ("chi1_mass", 200)]:
                if algo[key] < limit:
                    reasons.append(f"low {method} {key}")
        if reasons:
            bad_events.append((iev, reasons))

    visible_energy_ratio = np.asarray(visible_energy_ratio)
    visible_mass_ratio = np.asarray(visible_mass_ratio)
    for key in truth_frac:
        truth_frac[key] = np.asarray(truth_frac[key])
    for method in selected_methods:
        for key in algo_frac[method]:
            algo_frac[method][key] = np.asarray(algo_frac[method][key])

    ################################################################################
    # Text summary
    ################################################################################

    with open(os.path.join(output, "evaluate_ntuplizer.txt"), "w") as f:
        f.write("=" * 80 + "\nNTUPLIZER SUMMARY\n" + "=" * 80 + "\n")
        f.write(f"Number of events : {n_events}\nSelected method(s): {', '.join(selected_methods)}\n")
        f.write(f"Jet pT cutoff: {jet_pt_cut}\nNew AK radius: {ak_radius / 10:.1f}\n")
        f.write(f"New CA radius: {ca_radius / 10:.1f}\n")
        f.write(f"Average PF candidates/event: {ak.mean(branches['nParticles']):.1f}\n")
        f.write(f"Minimum/maximum PF candidates: {ak.min(branches['nParticles'])}/"
                f"{ak.max(branches['nParticles'])}\n")
        f.write("\nBranch names:\n")
        for name in sorted(branches.fields):
            f.write(f"    {name}\n")

        f.write("\n" + "=" * 80 + "\nEVENT STATISTICS\n" + "=" * 80 + "\n")
        f.write(f"Events: {n_events}\nAverage PF candidates: {np.mean(n_pf):.2f}\n")
        f.write(f"Truth mean chi0/chi1/unassigned counts: {np.mean(n_truth_chi0):.2f} / "
                f"{np.mean(n_truth_chi1):.2f} / {np.mean(n_truth_unassigned):.2f}\n")
        f.write(f"Average total PF mass: {np.mean(total_mass):.2f} GeV\n")
        f.write(f"Average truth Suu mass: {np.mean(truth_suu_mass):.2f} GeV\n")
        f.write(f"Average truth chi0/chi1 mass: {np.mean(truth_chi0_mass):.2f} / "
                f"{np.mean(truth_chi1_mass):.2f} GeV\n")
        truth_total = n_truth_chi0 + n_truth_chi1 + n_truth_unassigned
        f.write(f"Truth particle-count mismatches: {np.sum(truth_total != n_pf)}\n")

        for method in selected_methods:
            d = method_data[method]
            algo_total = d["n_chi0"] + d["n_chi1"] + d["n_unassigned"]
            f.write(f"\n{configs[method]['display']}\n" + "-" * 40 + "\n")
            f.write(f"Mean chi0/chi1/unassigned counts: {np.mean(d['n_chi0']):.2f} / "
                    f"{np.mean(d['n_chi1']):.2f} / {np.mean(d['n_unassigned']):.2f}\n")
            f.write(f"Average Suu mass: {np.mean(d['suu_mass']):.2f} GeV\n")
            f.write(f"Average chi0/chi1 mass: {np.mean(d['chi0_mass']):.2f} / "
                    f"{np.mean(d['chi1_mass']):.2f} GeV\n")
            f.write(f"Events missing chi0/chi1: {np.sum(d['n_chi0'] == 0)} / "
                    f"{np.sum(d['n_chi1'] == 0)}\n")
            f.write(f"Particle-count mismatches: {np.sum(algo_total != n_pf)}\n")
            f.write(f"Low/high-mass SJs: {len(d['low_mass_sjs'])} / {len(d['high_mass_sjs'])}\n")

        f.write("\n" + "=" * 80 + "\nVISIBLE PF VS GENERATOR\n" + "=" * 80 + "\n")
        f.write(f"Visible PF energy / generator energy: {visible_energy_ratio.mean():.3f} "
                f"+/- {visible_energy_ratio.std():.3f}\n")
        f.write(f"Visible PF mass / generator mass: {visible_mass_ratio.mean():.3f} "
                f"+/- {visible_mass_ratio.std():.3f}\n")
        truth_total_frac = truth_frac["chi0"] + truth_frac["chi1"]
        f.write(f"Truth chi0/chi1 mean energy fraction: {truth_frac['chi0'].mean():.3f} / "
                f"{truth_frac['chi1'].mean():.3f}\n")
        f.write(f"Truth chi-system energy fraction: mean={truth_total_frac.mean():.3f}, "
                f"std={truth_total_frac.std():.3f}\n")
        for method in selected_methods:
            total_frac = algo_frac[method]["chi0"] + algo_frac[method]["chi1"]
            f.write(f"{method} chi0/chi1 mean energy fraction: "
                    f"{algo_frac[method]['chi0'].mean():.3f} / "
                    f"{algo_frac[method]['chi1'].mean():.3f}\n")
            f.write(f"{method} chi-system fraction: mean={total_frac.mean():.3f}, "
                    f"std={total_frac.std():.3f}, min={total_frac.min():.3f}, "
                    f"max={total_frac.max():.3f}\n")

        f.write(f"\nSuspicious events identified: {len(bad_events)}\n")
        for iev, reasons in bad_events[:20]:
            f.write("-" * 60 + f"\nEvent {iev}\nReasons: {', '.join(reasons)}\n")
            f.write(f"Truth Suu/chi0/chi1: {results[iev]['truth_suu_mass']:.1f} / "
                    f"{results[iev]['truth_chi0_mass']:.1f} / "
                    f"{results[iev]['truth_chi1_mass']:.1f}\n")
            for method in selected_methods:
                a = results[iev]["algorithms"][method]
                f.write(f"{method} Suu/chi0/chi1: {a['suu_mass']:.1f} / "
                        f"{a['chi0_mass']:.1f} / {a['chi1_mass']:.1f}\n")

    ################################################################################
    # Mass plots
    ################################################################################

    plt.figure(figsize=(8, 6))
    plt.hist(total_mass, bins=100, range=(0, 10000), histtype="step", linewidth=2,
             label="All PF candidates")
    plt.hist(truth_suu_mass, bins=100, range=(0, 10000), histtype="step", linewidth=2,
             label="Truth labels", color="grey")
    for method in selected_methods:
        c = configs[method]
        plt.hist(method_data[method]["suu_mass"], bins=100, range=(0, 10000),
                 histtype="step", linewidth=2, color=c["color"],
                 linestyle=c["linestyle"], label=c["display"])
    plt.xlabel(r"Reconstructed $m_{Suu}$ [GeV]")
    plt.ylabel("Events")
    plt.title(r"Reconstructed $m_{Suu}$")
    plt.legend()
    save_close(output, "suu_mass.png")

    plt.figure(figsize=(8, 6))
    plt.hist(np.concatenate([truth_chi0_mass, truth_chi1_mass]), bins=100,
             range=(0, 4000), histtype="step", linewidth=2, label="Truth", color="grey")
    for method in selected_methods:
        c = configs[method]
        masses = np.concatenate([method_data[method]["chi0_mass"],
                                 method_data[method]["chi1_mass"]])
        plt.hist(masses, bins=100, range=(0, 4000), histtype="step", linewidth=2,
                 color=c["color"], linestyle=c["linestyle"], label=c["display"])
    plt.xlabel(r"Reconstructed $m_{\chi}$ [GeV]")
    plt.ylabel("Events")
    plt.title(r"Reconstructed $m_{\chi}$")
    plt.legend()
    save_close(output, "chi_mass.png")

    plt.figure(figsize=(6, 6))
    plt.scatter(truth_chi0_mass, truth_chi1_mass, s=5, alpha=0.4, color="grey")
    plt.xlabel(r"Truth $m(\chi_0)$ [GeV]")
    plt.ylabel(r"Truth $m(\chi_1)$ [GeV]")
    plt.title(r"Truth $m(\chi_1)$ vs $m(\chi_0)$")
    save_close(output, "truth_chi_scatter.png")

    # In --both mode this directly compares the event-by-event method outputs.
    plt.figure(figsize=(6, 6))
    for method in selected_methods:
        c = configs[method]
        plt.scatter(method_data[method]["chi0_mass"], method_data[method]["chi1_mass"],
                    s=5, alpha=0.35, color=c["color"], label=c["display"])
    plt.xlabel(r"Algo $m(\chi_0)$ [GeV]")
    plt.ylabel(r"Algo $m(\chi_1)$ [GeV]")
    plt.xlim(0,6000)
    plt.ylim(0,6000)
    plt.title(r"Algorithm $m(\chi_1)$ vs $m(\chi_0)$")
    plt.legend()
    save_close(output, "algo_chi_scatter.png")

    ################################################################################
    # Visible PF plots
    ################################################################################

    for values, xlabel, title, filename in [
        (visible_energy_ratio, r"$E_{\rm PF}/E_{\rm gen}$", "Visible PF energy fraction",
         "visible_energy_fraction.png"),
        (visible_mass_ratio, r"$M_{\rm PF}/M_{\rm gen}$", "Visible PF mass fraction",
         "visible_mass_fraction.png"),
    ]:
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=60, range=(0, 1.2), histtype="step", linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel("Events")
        plt.title(title)
        save_close(output, filename)

    ################################################################################
    # Six-step SJ one-dimensional plots
    ################################################################################

    specs = [
        ("energy", np.linspace(0, 10000, 61), "SJ Energy [GeV]", "algo_sj_energy.png"),
        ("pt", np.linspace(0, 10000, 61), r"SJ $p_T$ [GeV]", "algo_sj_pt.png"),
        ("n_constituents", np.arange(0, 201, 2), "Number of PF constituents",
         "algo_sj_nconstituents.png"),
        ("n_ca", np.arange(-0.5, 20.5, 1), "Number of CA jets", "algo_sj_nca.png"),
    ]
    for quantity, bins, xlabel, filename in specs:
        plt.figure(figsize=(8, 6))
        for method in selected_methods:
            c = configs[method]
            for category, mass_text, alpha in [
                ("low_mass_sjs", r"$m_{\mathrm{SJ}}<300$ GeV", 1.0),
                ("high_mass_sjs", r"$m_{\mathrm{SJ}}>700$ GeV", 0.65),
            ]:
                plt.hist([sj[quantity] for sj in method_data[method][category]], bins=bins,
                         histtype="step", linewidth=2, color=c["color"],
                         linestyle=c["linestyle"], alpha=alpha,
                         label=f"{c['display']}, {mass_text}")
        plt.xlabel(xlabel)
        plt.ylabel("SJs")
        plt.title(f"Algorithm SJ {quantity.replace('_', ' ')}")
        plt.legend()
        save_close(output, filename)

    ################################################################################
    # SJ mass vs jet multiplicity (one heatmap per selected method)
    ################################################################################

    for method in selected_methods:
        c = configs[method]
        all_sjs = method_data[method]["all_sjs"]
        if not all_sjs:
            continue
        masses = np.asarray([sj["mass"] for sj in all_sjs])
        mass_bins = np.linspace(masses.min(), masses.max() + 1e-9, 50)
        for count_key, jet_name, filename in [
            ("n_ca", c["ca_name"], f"{method}_algo_sj_m_vs_nca.png"),
            ("n_ak", c["ak_name"], f"{method}_algo_sj_m_vs_nak.png"),
        ]:
            plt.figure(figsize=(6, 6))
            plt.hist2d([sj[count_key] for sj in all_sjs], masses,
                       bins=[np.arange(0.5, 6.5, 1), mass_bins])
            plt.ylim(0, 2000)
            plt.xlabel(f"Number of {jet_name} jets")
            plt.ylabel(r"$m_{\mathrm{SJ}}$ [GeV]")
            plt.title(f"{c['display']}: SJ mass vs {jet_name} multiplicity")
            plt.colorbar(label="Number of SJs")
            save_close(output, filename)

    ################################################################################
    # Delta R between jets in each SJ
    ################################################################################

    for jet_type, key, filename in [("CA", "ca_delta_rs", "algo_sj_ca_deltaR.png"),
                                    ("AK", "ak_delta_rs", "algo_sj_ak_deltaR.png")]:
        plt.figure(figsize=(8, 6))
        for method in selected_methods:
            c = configs[method]
            for category, mass_text, alpha in [
                ("low_mass_sjs", r"$m_{\mathrm{SJ}}<300$ GeV", 1.0),
                ("high_mass_sjs", r"$m_{\mathrm{SJ}}>700$ GeV", 0.65),
            ]:
                values = [dr for sj in method_data[method][category] for dr in sj[key]]
                plt.hist(values, bins=60, range=(0, 5), histtype="step", linewidth=2,
                         color=c["color"], linestyle=c["linestyle"], alpha=alpha,
                         label=f"{c['display']}, {mass_text}")
        plt.xlabel(rf"$\Delta R$({jet_type}, {jet_type})")
        plt.ylabel(f"{jet_type} jet pairs")
        plt.title(f"Angular separation of {jet_type} jets within algorithm SJs")
        plt.legend()
        save_close(output, filename)

    ################################################################################
    # Scatter: SJ mass vs max delta R
    ################################################################################

    for method in selected_methods:
        c = configs[method]
        all_sjs = method_data[method]["all_sjs"]
        for jet_name, count_key, dr_key, file_tag in [
            (c["ca_name"], "n_ca", "ca_delta_rs", "ca"),
            (c["ak_name"], "n_ak", "ak_delta_rs", "ak"),
        ]:
            drs = np.asarray([max(sj[dr_key]) if sj[dr_key] else 0 for sj in all_sjs])
            counts = np.asarray([sj[count_key] for sj in all_sjs])
            masses = np.asarray([sj["mass"] for sj in all_sjs])
            plt.figure(figsize=(6, 6))
            for n_jets, color in [(1, "red"), (2, "green"), (3, "blue")]:
                mask = counts == n_jets
                plt.scatter(drs[mask], masses[mask], s=5, alpha=0.4, color=color,
                            label=f"N({jet_name} in SJ) = {n_jets}")
            plt.xlabel(r"Max $\Delta R$ separation")
            plt.ylabel(r"$m_{\mathrm{SJ}}$ [GeV]")
            plt.title(f"{c['display']}: SJ mass vs max {jet_name} separation")
            plt.legend()
            save_close(output, f"{method}_algo_sj_m_vs_{file_tag}deltaR.png")

    ################################################################################
    # N_CA vs N_CA per SJ, split by event-wide N_AK above 300 GeV
    ################################################################################

    for method in selected_methods:
        c = configs[method]
        n_aks = np.asarray([r["algorithms"][method]["n_ak_event_300"] for r in results])
        for n_ak in range(1, int(np.max(n_aks)) + 1):
            selected_events = [event_sjs for event_sjs, keep in
                               zip(method_data[method]["all_sjs_per_event"], n_aks == n_ak)
                               if keep and len(event_sjs) == 2]
            if not selected_events:
                continue
            plt.figure(figsize=(6, 6))
            plt.hist2d([event[0]["n_ca"] for event in selected_events],
                       [event[1]["n_ca"] for event in selected_events],
                       bins=[np.arange(0.5, 6.5, 1), np.arange(0.5, 6.5, 1)])
            plt.xlim(0.5, 5.5)
            plt.ylim(0.5, 5.5)
            plt.xlabel(f"N({c['ca_name']}) in SJ 1")
            plt.ylabel(f"N({c['ca_name']}) in SJ 2")
            plt.title(f"{c['display']}, N({c['ak_name']}, pT>300 GeV)={n_ak}")
            plt.colorbar(label="Number of events")
            save_close(output, f"{method}_algo_sj_nca_vs_sj_nca_nak_{n_ak}.png")

    ################################################################################
    # N_CA in the other SJ vs event-wide N_AK
    ################################################################################

    for method in selected_methods:
        c = configs[method]
        all_sjs = method_data[method]["all_sjs"]
        if not all_sjs:
            continue
        for selected_n_ca in range(1, max(sj["n_ca"] for sj in all_sjs) + 1):
            other_n_ca, event_n_ak = [], []
            for r in results:
                a = r["algorithms"][method]
                sjs = a["sj_kinematics"]
                if len(sjs) != 2:
                    continue
                for i in range(2):
                    if sjs[i]["n_ca"] == selected_n_ca:
                        other_n_ca.append(sjs[1 - i]["n_ca"])
                        event_n_ak.append(a["n_ak_event_300"])
            if not other_n_ca:
                continue
            plt.figure(figsize=(7, 6))
            plt.hist2d(other_n_ca, event_n_ak,
                       bins=[np.arange(-0.5, 6.5, 1), np.arange(-0.5, 6.5, 1)])
            plt.xlim(-0.5, 5.5)
            plt.ylim(-0.5, 5.5)
            plt.xlabel(f"N({c['ca_name']}) in other SJ")
            plt.ylabel(f"N({c['ak_name']}) with pT>300 GeV")
            plt.title(f"{c['display']}: selected SJ N(CA)={selected_n_ca}")
            plt.colorbar(label="Number of events")
            save_close(output, f"{method}_algo_other_sj_nca_vs_event_nak_"
                               f"selected_nca_{selected_n_ca}.png")

    ################################################################################
    # Number of AK jets vs pT threshold, split by SJ topology
    ################################################################################

    pt_thresholds = np.arange(0, 401, 20)
    sj_modes = {
        "1ca_1ca": lambda n1, n2: n1 == 1 and n2 == 1,
        "2ca_2ca": lambda n1, n2: n1 == 2 and n2 == 2,
        "1ca_2ca": lambda n1, n2: {n1, n2} == {1, 2},
    }
    for method in selected_methods:
        c = configs[method]
        for mode_name, select_mode in sj_modes.items():
            selected = []
            for r in results:
                sjs = r["algorithms"][method]["sj_kinematics"]
                if len(sjs) == 2 and select_mode(sjs[0]["n_ca"], sjs[1]["n_ca"]):
                    selected.append(r)
            if not selected:
                continue
            n_ak_by_threshold = np.asarray([
                [np.count_nonzero(r["algorithms"][method]["ak_pts"] >= threshold)
                 for r in selected]
                for threshold in pt_thresholds
            ], dtype=int)
            max_n_ak = int(np.max(n_ak_by_threshold))
            counts = np.zeros((len(pt_thresholds), max_n_ak + 1), dtype=int)
            for i, row in enumerate(n_ak_by_threshold):
                for n_ak in row:
                    counts[i, n_ak] += 1
            plt.figure(figsize=(6, 6))
            plt.imshow(counts, origin="lower", aspect="auto",
                       extent=[-0.5, max_n_ak + 0.5,
                               pt_thresholds[0] - 10, pt_thresholds[-1] + 10])
            display_max = min(6, max_n_ak)
            plt.xlim(-0.5, display_max + 0.5)
            plt.xticks(np.arange(display_max + 1))
            plt.yticks(pt_thresholds)
            plt.xlabel(f"N({c['ak_name']}) with pT > threshold")
            plt.ylabel(f"{c['ak_name']} pT threshold [GeV]")
            plt.title(f"{c['display']}, SJ topology: {mode_name}")
            plt.colorbar(label="Number of events")
            save_close(output, f"{method}_algo_nak_vs_pt_threshold_{mode_name}.png")

    print(f"Finished. Results written to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/eos/uscms/store/user/aji/rootfiles_existingOptimization/"
                "WbWb_4000_1000.root",
        help="Input ROOT ntuple",
    )
    parser.add_argument("--output", default="evaluate_ntuplizer",
                        help="Base directory for plots and printed results")
    parser.add_argument("--num-events", default=1000, type=int,
                        help="Maximum number of events")

    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument("--old", action="store_true",
                              help="Use only the slimmedJetsAK8 method")
    method_group.add_argument("--new", action="store_true",
                              help="Use only the directly reclustered method")
    method_group.add_argument("--both", action="store_true",
                              help="Run both methods and overplot comparable distributions")

    args = parser.parse_args()
    args.method = "old" if args.old else "new" if args.new else "both"
    main(args)