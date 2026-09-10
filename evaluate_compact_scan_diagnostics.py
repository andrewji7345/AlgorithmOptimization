#!/usr/bin/env python3
"""Detailed diagnostics for selected compact optimization configurations.

This is the drill-down companion to the scan-wide evaluator. It reads only
the compact schema and deliberately does not emulate truth/PF-candidate plots
that require branches omitted by the compact ntuplizer. It does write a
legacy-style ``chi_mass.png`` from the stored SJ masses. An invariant
``m(Suu)`` cannot be reconstructed because the compact schema does not retain
SJ four-vectors; its deliberate absence is documented next to each diagnostic
output.

A configuration is ``n_gate:T_gate:T_keep:R_AK:R_CA:c``. For example::

  --configuration 4:300:100:0.8:0.8:0.5

Use ``n_gate=0`` for the canonical ungated point; its T_gate token is ignored.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compact_scan_metrics import (
    ConfigurationKey,
    PhysicalityDefinition,
    calculate_configuration_metrics,
    configuration_indices,
    configuration_key,
    configuration_slug,
    discover_compact_files,
    gate_jet_multiplicity,
    gate_mask,
    load_sample_definitions,
    read_event_payload,
    true_chi_mass_for_sample,
)


DEFAULT_INPUT_DIR = "/eos/uscms/store/user/aji/rootfiles_existingOptimization_compact"
DEFAULT_OUTPUT_DIR = "results/evaluate_compact_scan_diagnostics"
DEFAULT_SAMPLE_LISTS = Path(__file__).resolve().parent / "test" / "signalMCFiles"
CONFIGURATION_FIELDS = (
    "n_gate_jets", "gate_pt_cut", "collection_pt_cut", "ak_radius",
    "ca_radius", "cos_thrust_cut",
)
RANKING_ALIASES = {
    "n_gate_jets": ("n_gate_jets", "gate_jet_count", "min_gate_jets", "n_gate", "n_jets"),
    "gate_pt_cut": ("gate_pt_cut", "event_pt_cut", "threshold_1", "gate_threshold"),
    "collection_pt_cut": ("collection_pt_cut", "pt_cut", "threshold_2", "keep_pt_cut"),
    "ak_radius": ("ak_radius", "akRadius", "r_ak"),
    "ca_radius": ("ca_radius", "caRadius", "r_ca"),
    "cos_thrust_cut": ("cos_thrust_cut", "cos_thrust", "cosThrust", "hybrid_cut", "c"),
}


@dataclass(frozen=True)
class Selection:
    key: ConfigurationKey
    sources: Tuple[str, ...]


def _finite_float(text, label):
    try:
        value = float(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric, found {text!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, found {text!r}")
    return value


def _nonnegative_int(text, label):
    try:
        value = int(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer, found {text!r}") from error
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _optional_gate_pt(text, n_gate_jets):
    stripped = "" if text is None else str(text).strip()
    none_tokens = {"", "none", "null", "na", "-"}
    if n_gate_jets == 0:
        if stripped.lower() not in none_tokens:
            _finite_float(stripped, "T_gate")
        return None
    if stripped.lower() in none_tokens:
        raise ValueError("T_gate is required when n_gate_jets is positive")
    value = _finite_float(stripped, "T_gate")
    if value <= 0:
        raise ValueError("T_gate must be positive")
    return value


def parse_configuration(text):
    """Parse ``n:Tgate:Tkeep:RAK:RCA:c`` into a shared canonical key."""
    parts = [part.strip() for part in text.split(":")]
    if len(parts) != 6:
        raise ValueError(
            "configuration must have six fields n_gate:T_gate:T_keep:R_AK:R_CA:c"
        )
    n_gate_jets = _nonnegative_int(parts[0], "n_gate_jets")
    return configuration_key(
        n_gate_jets=n_gate_jets,
        gate_pt_cut=_optional_gate_pt(parts[1], n_gate_jets),
        collection_pt_cut=_finite_float(parts[2], "T_keep"),
        ak_radius=_finite_float(parts[3], "R_AK"),
        ca_radius=_finite_float(parts[4], "R_CA"),
        cos_thrust_cut=_finite_float(parts[5], "c"),
    )


def _first_present(row, logical_name, allow_blank=False):
    for candidate in RANKING_ALIASES[logical_name]:
        if candidate in row:
            value = row[candidate].strip()
            if value or allow_blank:
                return value
    raise ValueError(
        f"ranking CSV has no usable {logical_name} column; accepted names: "
        + ", ".join(RANKING_ALIASES[logical_name])
    )


def _configuration_from_row(row):
    n_gate_jets = _nonnegative_int(_first_present(row, "n_gate_jets"), "n_gate_jets")
    return configuration_key(
        n_gate_jets=n_gate_jets,
        gate_pt_cut=_optional_gate_pt(
            _first_present(row, "gate_pt_cut", allow_blank=True), n_gate_jets
        ),
        collection_pt_cut=_finite_float(_first_present(row, "collection_pt_cut"), "collection_pt_cut"),
        ak_radius=_finite_float(_first_present(row, "ak_radius"), "ak_radius"),
        ca_radius=_finite_float(_first_present(row, "ca_radius"), "ca_radius"),
        cos_thrust_cut=_finite_float(_first_present(row, "cos_thrust_cut"), "cos_thrust_cut"),
    )


def configurations_from_ranking(filename, ranks, sample):
    filename = Path(filename)
    opener = gzip.open if filename.suffix == ".gz" else open
    with opener(filename, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "rank" not in reader.fieldnames:
            raise ValueError(f"ranking CSV has no 'rank' column: {filename}")
        rows = list(reader)
    result = []
    for requested in ranks:
        matches = []
        for row in rows:
            try:
                row_rank = int(row["rank"])
            except (TypeError, ValueError):
                continue
            row_sample = row.get("sample", "").strip()
            if row_rank == requested and (not row_sample or row_sample == sample):
                matches.append(row)
        if len(matches) != 1:
            raise ValueError(
                f"rank {requested} resolves to {len(matches)} rows for sample "
                f"{sample!r} in {filename}; expected exactly one"
            )
        result.append((_configuration_from_row(matches[0]), f"ranking:{filename}:rank={requested}"))
    return result


def _key_tuple(key):
    return tuple(getattr(key, name) for name in CONFIGURATION_FIELDS)


def deduplicate_selections(requested):
    ordered = {}
    for key, source in requested:
        identity = _key_tuple(key)
        if identity not in ordered:
            ordered[identity] = [key, []]
        if source not in ordered[identity][1]:
            ordered[identity][1].append(source)
    return [Selection(item[0], tuple(item[1])) for item in ordered.values()]


def build_parser():
    defaults = PhysicalityDefinition()
    parser = argparse.ArgumentParser(
        description="Detailed per-sample diagnostics for compact-scan configurations."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    source.add_argument(
        "--input", action="append", default=None, metavar="ROOT",
        help="Explicit ROOT input; repeat for multiple AK radii.",
    )
    parser.add_argument("--pattern", default="*.root")
    parser.add_argument("--sample", required=True, help="Exact Metadata sampleName.")
    parser.add_argument(
        "--configuration", action="append", default=[],
        metavar="N:TGATE:TKEEP:RAK:RCA:C",
    )
    parser.add_argument("--ranking-csv", type=Path)
    parser.add_argument("--rank", action="append", type=int, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sample-lists-dir", type=Path, default=DEFAULT_SAMPLE_LISTS)
    parser.add_argument("--true-chi-mass", type=float)
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--max-problem-events", type=int, default=100)
    parser.add_argument("--write-event-table", action="store_true")
    parser.add_argument("--bias-limit", type=float, default=defaults.bias_limit)
    parser.add_argument("--resolution-limit", type=float, default=defaults.resolution_limit)
    parser.add_argument("--invalid-fraction-limit", type=float, default=defaults.invalid_limit)
    parser.add_argument("--tail-fraction-limit", type=float, default=defaults.tail_limit)
    parser.add_argument("--minimum-valid-events", type=int, default=defaults.min_valid_events)
    parser.add_argument("--response-min", type=float, default=defaults.tail_response_min)
    parser.add_argument("--response-max", type=float, default=defaults.tail_response_max)
    parser.add_argument("--response-hist-min", type=float, default=defaults.response_hist_min)
    parser.add_argument("--response-hist-max", type=float, default=defaults.response_hist_max)
    parser.add_argument("--response-hist-bins", type=int, default=defaults.response_hist_bins)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.configuration and not args.rank:
        parser.error("provide at least one --configuration or --rank")
    if args.rank and args.ranking_csv is None:
        parser.error("--rank requires --ranking-csv")
    if args.ranking_csv is not None and not args.rank:
        parser.error("--ranking-csv requires at least one --rank")
    if any(rank <= 0 for rank in args.rank):
        parser.error("--rank values must be positive")
    if args.max_events == 0 or args.max_events < -1:
        parser.error("--max-events must be -1 or positive")
    if args.max_problem_events < 0:
        parser.error("--max-problem-events must be nonnegative")
    if args.true_chi_mass is not None and (
        not math.isfinite(args.true_chi_mass) or args.true_chi_mass <= 0
    ):
        parser.error("--true-chi-mass must be finite and positive")
    try:
        selections = [(parse_configuration(value), f"explicit:{value}") for value in args.configuration]
        if args.rank:
            selections.extend(configurations_from_ranking(args.ranking_csv, args.rank, args.sample))
        args.physicality = PhysicalityDefinition(
            bias_limit=args.bias_limit,
            resolution_limit=args.resolution_limit,
            invalid_limit=args.invalid_fraction_limit,
            tail_limit=args.tail_fraction_limit,
            min_valid_events=args.minimum_valid_events,
            tail_response_min=args.response_min,
            tail_response_max=args.response_max,
            response_hist_min=args.response_hist_min,
            response_hist_max=args.response_hist_max,
            response_hist_bins=args.response_hist_bins,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.selections = deduplicate_selections(selections)
    return args


def _metadata_path(metadata):
    for name in ("path", "filename"):
        if hasattr(metadata, name):
            return Path(getattr(metadata, name))
    raise AttributeError("CompactMetadata exposes neither path nor filename")


def _metadata_for_key(metadata_items, key):
    matches = [
        item for item in metadata_items
        if math.isclose(float(item.ak_radius), float(key.ak_radius), abs_tol=1e-6, rel_tol=0)
    ]
    if len(matches) != 1:
        available = ", ".join(f"{float(item.ak_radius):g}" for item in metadata_items) or "none"
        raise ValueError(
            f"R_AK={key.ak_radius:g} has {len(matches)} matching sample files; available: {available}"
        )
    return matches[0]


def _column(values, index, dtype):
    try:
        selected = values[:, index]
    except (IndexError, TypeError, ValueError):
        selected = [row[index] for row in values]
    return np.asarray(selected, dtype=dtype)


def _array(values, dtype):
    return np.asarray(values, dtype=dtype)


def _fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else math.nan


def _status_name(code, names):
    return names[code] if 0 <= code < len(names) else f"unknown_{code}"


def _write_csv(path, rows, fields=None):
    if fields is None:
        if not rows:
            raise ValueError(f"field names required for empty CSV {path}")
        fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _scalar_metrics_row(metrics, selection, metadata, true_chi_mass):
    row = {
        "sample": metadata.sample_name,
        "true_chi_mass": true_chi_mass,
        "selection_source": ";".join(selection.sources),
        "input_file": str(_metadata_path(metadata)),
    }
    for name in CONFIGURATION_FIELDS:
        value = getattr(selection.key, name)
        row[name] = "" if value is None else value
    for name, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            row[name] = value
    return row


def _status_rows(status, gate, names):
    codes = sorted(set(range(len(names))) | {int(code) for code in status})
    n_events = len(status)
    n_gate = int(np.count_nonzero(gate))
    rows = []
    for code in codes:
        selected = status == code
        count_all = int(np.count_nonzero(selected))
        count_gate = int(np.count_nonzero(selected & gate))
        rows.append({
            "status_code": code,
            "status_name": _status_name(code, names),
            "count_all": count_all,
            "fraction_all": _fraction(count_all, n_events),
            "count_gate": count_gate,
            "fraction_gate": _fraction(count_gate, n_gate),
        })
    return rows


def _event_arrays(metadata, payload, key, metrics):
    config_index, base_index = configuration_indices(metadata, key)
    if "config_index" in metrics and int(metrics["config_index"]) != config_index:
        raise RuntimeError("shared metric/config index disagreement")
    if "base_index" in metrics and int(metrics["base_index"]) != base_index:
        raise RuntimeError("shared metric/base index disagreement")
    gate = np.asarray(gate_mask(payload, key.n_gate_jets, key.gate_pt_cut), dtype=bool)
    status = _column(payload.reco_status, config_index, int)
    sj1_mass = _column(payload.sj1_mass, config_index, float)
    sj2_mass = _column(payload.sj2_mass, config_index, float)
    finite_masses = (
        np.isfinite(sj1_mass) & np.isfinite(sj2_mass)
        & (sj1_mass >= 0) & (sj2_mass >= 0)
    )
    valid = (status == 0) & finite_masses
    arrays = {
        "run": _array(payload.run, np.uint64),
        "lumi": _array(payload.lumi, np.uint64),
        "event": _array(payload.event, np.uint64),
        "gen_weight": _array(payload.gen_weight, float),
        "gate": gate,
        "status": status,
        "valid": valid,
        "survivor": gate & valid,
        "n_ca": _column(payload.n_ca_jets, base_index, int),
        "n_ambiguous": _column(payload.n_ambiguous, config_index, int),
        "sj1_mass": sj1_mass,
        "sj2_mass": sj2_mass,
    }
    arrays["gate_multiplicity"] = (
        None if key.gate_pt_cut is None else
        np.asarray(gate_jet_multiplicity(payload, key.gate_pt_cut), dtype=int)
    )
    lengths = {name: len(value) for name, value in arrays.items() if value is not None}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"selected event-array lengths disagree: {lengths}")
    expected = {
        "n_events": len(gate),
        "n_gate_events": int(np.count_nonzero(gate)),
        "n_valid_all_events": int(np.count_nonzero(valid)),
        "n_valid_events": int(np.count_nonzero(gate & valid)),
    }
    for name, count in expected.items():
        if name in metrics and int(metrics[name]) != count:
            raise RuntimeError(
                f"shared metric {name}={metrics[name]} disagrees with event arrays ({count})"
            )
    return arrays


EVENT_FIELDS = (
    "run", "lumi", "event", "gen_weight", "passes_gate",
    "gate_jet_multiplicity", "reco_status", "status_name", "is_valid",
    "is_tail_event", "n_ca_jets", "n_ambiguous", "sj1_mass", "sj2_mass",
    "sj1_mass_response", "sj2_mass_response",
)


def _event_row(index, arrays, names, true_chi_mass, tail):
    status = int(arrays["status"][index])
    multiplicity = arrays["gate_multiplicity"]
    mass1 = float(arrays["sj1_mass"][index])
    mass2 = float(arrays["sj2_mass"][index])
    return {
        "run": int(arrays["run"][index]),
        "lumi": int(arrays["lumi"][index]),
        "event": int(arrays["event"][index]),
        "gen_weight": float(arrays["gen_weight"][index]),
        "passes_gate": int(arrays["gate"][index]),
        "gate_jet_multiplicity": "" if multiplicity is None else int(multiplicity[index]),
        "reco_status": status,
        "status_name": _status_name(status, names),
        "is_valid": int(arrays["valid"][index]),
        "is_tail_event": int(tail[index]),
        "n_ca_jets": int(arrays["n_ca"][index]),
        "n_ambiguous": int(arrays["n_ambiguous"][index]),
        "sj1_mass": mass1,
        "sj2_mass": mass2,
        "sj1_mass_response": mass1 / true_chi_mass if math.isfinite(mass1) else math.nan,
        "sj2_mass_response": mass2 / true_chi_mass if math.isfinite(mass2) else math.nan,
    }


def _write_event_tables(output_dir, arrays, status_names, true_chi_mass,
                        max_problem_events, write_full_table, physicality):
    response1 = arrays["sj1_mass"] / true_chi_mass
    response2 = arrays["sj2_mass"] / true_chi_mass
    tail = arrays["survivor"] & (
        (response1 < physicality.tail_response_min)
        | (response1 > physicality.tail_response_max)
        | (response2 < physicality.tail_response_min)
        | (response2 > physicality.tail_response_max)
    )
    # Prioritize gated reconstruction failures, then gated valid tail events.
    # Ungated failures remain useful for diagnosing survivor bias and follow.
    problem_indices = np.concatenate((
        np.flatnonzero(arrays["gate"] & ~arrays["valid"]),
        np.flatnonzero(tail),
        np.flatnonzero(~arrays["gate"] & ~arrays["valid"]),
    ))
    rows = [
        _event_row(index, arrays, status_names, true_chi_mass, tail)
        for index in problem_indices[:max_problem_events]
    ]
    _write_csv(output_dir / "problem_events.csv", rows, fields=EVENT_FIELDS)
    if not write_full_table:
        return
    with gzip.open(output_dir / "event_table.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        for index in range(len(arrays["event"])):
            writer.writerow(_event_row(index, arrays, status_names, true_chi_mass, tail))


def _human_configuration(key):
    gate = (
        "ungated" if key.n_gate_jets == 0 else
        f"N(AK pT > {key.gate_pt_cut:g} GeV) >= {key.n_gate_jets}"
    )
    return (
        f"{gate}; Tkeep={key.collection_pt_cut:g} GeV, RAK={key.ak_radius:g}, "
        f"RCA={key.ca_radius:g}, c={key.cos_thrust_cut:g}"
    )


def _format(value):
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.8g}"
    return str(value)


def _write_summary(output_dir, selection, metadata, metrics, status_rows,
                   true_chi_mass, physicality, sample_definition):
    key = selection.key
    lines = [
        "COMPACT OPTIMIZATION CONFIGURATION DIAGNOSTICS",
        "=" * 54,
        f"Sample: {metadata.sample_name}",
        f"Input: {_metadata_path(metadata)}",
        f"Configuration: {_human_configuration(key)}",
        f"Configuration slug: {configuration_slug(key)}",
        f"Selection source: {'; '.join(selection.sources)}",
        f"Generated chi mass used: {true_chi_mass:g} GeV",
    ]
    if sample_definition is not None:
        for attribute, label in (
            ("nominal_suu_mass", "Nominal Suu mass"),
            ("nominal_chi_mass", "Nominal chi mass"),
            ("generated_suu_mass", "Generated Suu mass"),
            ("generated_chi_mass", "Generated chi mass"),
        ):
            if hasattr(sample_definition, attribute):
                lines.append(f"{label}: {getattr(sample_definition, attribute):g} GeV")
    lines += [
        "", "Efficiency populations", "-" * 54,
        f"All analyzed events: {_format(metrics.get('n_events', math.nan))}",
        f"Gate-passing events: {_format(metrics.get('n_gate_events', math.nan))}",
        f"Valid reconstructions before gate: {_format(metrics.get('n_valid_all_events', math.nan))}",
        f"Valid reconstructions after gate: {_format(metrics.get('n_valid_events', math.nan))}",
        f"Ungated reconstruction efficiency: {_format(metrics.get('ungated_reco_efficiency', math.nan))}",
        f"Gate efficiency: {_format(metrics.get('gate_efficiency', math.nan))}",
        f"Reco efficiency given gate: {_format(metrics.get('reco_given_gate_efficiency', math.nan))}",
        f"End-to-end signal retention: {_format(metrics.get('signal_retention', math.nan))}",
        f"Peak signal retention: {_format(metrics.get('peak_signal_retention', math.nan))}",
        f"Complexity-guard events after gate: {_format(metrics.get('n_complexity_guard_events', math.nan))}",
        f"Complexity-guard fraction given gate: {_format(metrics.get('complexity_guard_fraction_given_gate', math.nan))}",
        f"Other invalid events after gate: {_format(metrics.get('n_other_invalid_events', math.nan))}",
        f"Other invalid fraction given gate: {_format(metrics.get('other_invalid_fraction_given_gate', math.nan))}",
        "", "Mass quality among gate-passing valid survivors", "-" * 54,
        f"Median mass response: {_format(metrics.get('median_mass_response', math.nan))}",
        f"Absolute mass bias: {_format(metrics.get('mass_bias', math.nan))}",
        f"FWHM response: {_format(metrics.get('fwhm_mass_response', math.nan))}",
        f"FWHM peak response: {_format(metrics.get('fwhm_peak_response', math.nan))}",
        f"FWHM / peak: {_format(metrics.get('fwhm_resolution', math.nan))}",
        f"Tail event fraction: {_format(metrics.get('tail_fraction', math.nan))}",
        f"Invalid fraction given gate: {_format(metrics.get('invalid_fraction', math.nan))}",
        "", "Physicality gate", "-" * 54,
        (f"Limits: bias={physicality.bias_limit:g}, resolution={physicality.resolution_limit:g}, "
         f"invalid={physicality.invalid_limit:g}, tail={physicality.tail_limit:g}, "
         f"minimum valid events={physicality.min_valid_events}"),
        (f"Tail response window: [{physicality.tail_response_min:g}, "
         f"{physicality.tail_response_max:g}]"),
    ]
    lines += [f"{name}: {_format(metrics[name])}" for name in sorted(metrics)
              if name.startswith("physicality_")]
    lines += ["", "Reconstruction status breakdown", "-" * 54]
    for row in status_rows:
        lines.append(
            f"{row['status_code']} {row['status_name']}: all={row['count_all']} "
            f"({_format(row['fraction_all'])}), gate={row['count_gate']} "
            f"({_format(row['fraction_gate'])})"
        )
    lines += [
        "", "Notes", "-" * 54,
        "SJ1/SJ2 are ordered by lab-frame pT, not matched to generated chi labels.",
        "Mass plots use only gate-passing events with recoStatus=valid.",
        "Status fractions retain complexity_guard and every other failure in denominators.",
        "chi_mass.png is legacy-style but contains only compact reconstructed SJs.",
        "An invariant Suu mass cannot be recovered: compact files retain SJ masses, not SJ four-vectors.",
        "Truth, slimmedJetsAK8 (old), PF-candidate, and constituent diagnostics are absent from this compact schema.",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def _title(sample, key):
    return f"{sample}\n{_human_configuration(key)}"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _survivor_responses(arrays, true_chi_mass):
    selected = arrays["survivor"]
    response1 = arrays["sj1_mass"][selected] / true_chi_mass
    response2 = arrays["sj2_mass"][selected] / true_chi_mass
    finite = np.isfinite(response1) & np.isfinite(response2)
    return response1[finite], response2[finite]


def _plot_legacy_compatible_chi_mass(output_dir, sample, key, arrays):
    """Write the old evaluator's chi-mass histogram from compact SJ masses.

    The legacy evaluator pooled its two reconstructed chi masses into one
    100-bin, 0--4000 GeV histogram. The compact representation preserves the
    two SJ masses, so that observable is available for gate-passing, valid
    reconstructions. Truth and the old slimmedJetsAK8 curve are intentionally
    not fabricated because their inputs are absent from compact files.
    """
    selected = arrays["survivor"]
    masses = np.concatenate((arrays["sj1_mass"][selected],
                             arrays["sj2_mass"][selected]))
    masses = masses[np.isfinite(masses) & (masses >= 0)]
    fig, ax = plt.subplots(figsize=(8, 6))
    if masses.size:
        ax.hist(masses, bins=100, range=(0, 4000), histtype="step",
                linewidth=2, color="tab:blue",
                label="Compact reconstruction (gate-passing, valid)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No gate-passing valid reconstructions",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel(r"Reconstructed $m_{\chi}$ [GeV]")
    ax.set_ylabel("Events")
    ax.set_title(r"Reconstructed $m_{\chi}$" + "\n" + _title(sample, key))
    _save(fig, output_dir / "chi_mass.png")


def _write_unavailable_suu_note(output_dir):
    """Document why no legacy-equivalent Suu-mass plot is written."""
    (output_dir / "suu_mass_unavailable.txt").write_text(
        "No legacy-equivalent suu_mass.png was produced.\n\n"
        "The compact schema stores sj1Mass and sj2Mass but not reconstructed "
        "SJ four-vectors (or their opening angle). Therefore invariant "
        "m(Suu) cannot be calculated. sj1Mass + sj2Mass is not an invariant "
        "Suu mass and is deliberately not plotted as a substitute.\n\n"
        "The compact schema also lacks truth labels/PF candidates and the "
        "slimmedJetsAK8 inputs, so neither the truth nor the legacy old-method "
        "mass curves can be recovered.\n"
    )


def _plot_mass_responses(output_dir, sample, key, arrays, true_chi_mass,
                         physicality, metrics):
    response1, response2 = _survivor_responses(arrays, true_chi_mass)
    bins = np.linspace(
        physicality.response_hist_min, physicality.response_hist_max,
        physicality.response_hist_bins + 1,
    )
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.axvspan(
        physicality.tail_response_min, physicality.tail_response_max,
        color="tab:green", alpha=0.10, label="Accepted response window",
    )
    ax.axvline(1, color="black", linestyle="--", linewidth=1.3)
    if response1.size:
        pooled = np.concatenate((response1, response2))
        ax.hist(pooled, bins=bins, weights=np.full(pooled.size, 1 / pooled.size),
                histtype="stepfilled",
                color="0.6", alpha=0.20, label="Pooled SJ response")
        ax.hist(response1, bins=bins, weights=np.full(response1.size, 1 / response1.size),
                histtype="step",
                linewidth=1.6, color="tab:blue", label="SJ1 (higher lab-pT)")
        ax.hist(response2, bins=bins, weights=np.full(response2.size, 1 / response2.size),
                histtype="step",
                linewidth=1.6, color="tab:orange", label="SJ2 (lower lab-pT)")
    else:
        ax.text(0.5, 0.5, "No gate-passing valid reconstructions",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlim(physicality.response_hist_min, physicality.response_hist_max)
    ax.set_xlabel(r"Reconstructed SJ mass / generated $M_\chi$")
    ax.set_ylabel("Fraction of entries / bin")
    ax.set_title(_title(sample, key))
    ax.text(
        0.98, 0.95,
        f"median={_format(metrics.get('median_mass_response', math.nan))}\n"
        f"FWHM/peak={_format(metrics.get('fwhm_resolution', math.nan))}\n"
        f"tail={_format(metrics.get('tail_fraction', math.nan))}",
        transform=ax.transAxes, ha="right", va="top",
    )
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, output_dir / "pooled_mass_response.png")

    fig, ax = plt.subplots(figsize=(7.4, 6.7))
    limits = (physicality.response_hist_min, physicality.response_hist_max)
    if response1.size:
        image = ax.hist2d(response1, response2, bins=50,
                          range=[list(limits), list(limits)], cmap="viridis", cmin=1)
        fig.colorbar(image[3], ax=ax, label="Events / bin")
    else:
        ax.text(0.5, 0.5, "No gate-passing valid reconstructions",
                transform=ax.transAxes, ha="center", va="center")
    ax.plot(limits, limits, color="tab:red", linestyle="--", linewidth=1)
    ax.axvline(1, color="tab:red", linestyle=":", linewidth=1)
    ax.axhline(1, color="tab:red", linestyle=":", linewidth=1)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("SJ1 response (higher lab-pT)")
    ax.set_ylabel("SJ2 response (lower lab-pT)")
    ax.set_title(_title(sample, key))
    _save(fig, output_dir / "paired_mass_response.png")


def _plot_asymmetry(output_dir, sample, key, arrays):
    selected = arrays["survivor"]
    mass1, mass2 = arrays["sj1_mass"][selected], arrays["sj2_mass"][selected]
    denominator = mass1 + mass2
    values = np.divide(
        np.abs(mass1 - mass2), denominator,
        out=np.full_like(denominator, np.nan), where=denominator > 0,
    )
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(8, 6))
    if values.size:
        ax.hist(values, bins=np.linspace(0, 1, 51), histtype="step", linewidth=1.8)
        median = np.median(values)
        ax.axvline(median, color="tab:red", linestyle="--", label=f"median={median:.3f}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No finite survivor masses",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$|m_{SJ1}-m_{SJ2}|/(m_{SJ1}+m_{SJ2})$")
    ax.set_ylabel("Events / bin")
    ax.set_title(_title(sample, key))
    _save(fig, output_dir / "mass_asymmetry.png")


def _plot_status(output_dir, sample, key, status_rows):
    labels = [row["status_name"] for row in status_rows]
    all_values = np.nan_to_num([row["fraction_all"] for row in status_rows], nan=0)
    gate_values = np.nan_to_num([row["fraction_gate"] for row in status_rows], nan=0)
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(labels)), 6.2))
    ax.bar(x - width / 2, all_values, width, label="All events")
    ax.bar(x + width / 2, gate_values, width, label="Gate-passing events")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, max(1, 1.08 * max(np.max(all_values), np.max(gate_values))))
    ax.set_ylabel("Fraction of population")
    ax.set_title(_title(sample, key))
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    _save(fig, output_dir / "reco_status.png")


def _multiplicity_bins(*collections):
    values = [np.asarray(item, dtype=int) for item in collections if len(item)]
    if not values:
        return np.arange(-0.5, 1.5, 1)
    maximum = max(int(np.max(item)) for item in values)
    return (np.arange(-0.5, maximum + 1.5, 1) if maximum <= 60
            else np.linspace(-0.5, maximum + 0.5, 51))


def _weights(values):
    return np.full(len(values), 1 / len(values)) if len(values) else None


def _plot_multiplicity(output_dir, sample, key, values, gate, filename, xlabel):
    gated = values[gate]
    bins = _multiplicity_bins(values, gated)
    fig, ax = plt.subplots(figsize=(8.2, 6))
    if len(values):
        ax.hist(values, bins=bins, weights=_weights(values), histtype="step",
                linewidth=1.8, label="All events")
    if len(gated):
        ax.hist(gated, bins=bins, weights=_weights(gated), histtype="step",
                linewidth=1.8, label="Gate-passing events")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of population / bin")
    ax.set_title(_title(sample, key))
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    _save(fig, output_dir / filename)


def _plot_gate_multiplicity(output_dir, sample, key, arrays):
    fig, ax = plt.subplots(figsize=(8.2, 6))
    values = arrays["gate_multiplicity"]
    if values is None:
        ax.text(0.5, 0.5, "Canonical ungated configuration\n(no T_gate multiplicity)",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.hist(values, bins=_multiplicity_bins(values), weights=_weights(values),
                histtype="step", linewidth=1.8)
        ax.axvline(key.n_gate_jets - 0.5, color="tab:red", linestyle="--",
                   label=f"gate requires at least {key.n_gate_jets}")
        ax.set_xlabel(f"Number of AK jets with pT > {key.gate_pt_cut:g} GeV")
        ax.set_ylabel("Fraction of all events / bin")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
    ax.set_title(_title(sample, key))
    _save(fig, output_dir / "gate_ak_multiplicity.png")


def _make_plots(output_dir, sample, key, arrays, status_rows, true_chi_mass,
                physicality, metrics):
    _plot_legacy_compatible_chi_mass(output_dir, sample, key, arrays)
    _write_unavailable_suu_note(output_dir)
    _plot_mass_responses(output_dir, sample, key, arrays, true_chi_mass,
                         physicality, metrics)
    _plot_asymmetry(output_dir, sample, key, arrays)
    _plot_status(output_dir, sample, key, status_rows)
    _plot_multiplicity(output_dir, sample, key, arrays["n_ca"], arrays["gate"],
                       "n_ca_jets.png", "Number of CA jets for selected base")
    _plot_multiplicity(output_dir, sample, key, arrays["n_ambiguous"], arrays["gate"],
                       "n_ambiguous_ca_jets.png", "Number of ambiguous CA jets")
    _plot_gate_multiplicity(output_dir, sample, key, arrays)


def _sample_mass(args):
    if args.true_chi_mass is not None:
        return float(args.true_chi_mass), None
    definitions = load_sample_definitions(args.sample_lists_dir, expected_count=114)
    return float(true_chi_mass_for_sample(args.sample, definitions)), definitions[args.sample]


def _discover(args):
    kwargs = {"samples": [args.sample], "pattern": args.pattern}
    if args.input:
        kwargs["paths"] = args.input
    else:
        kwargs["input_dir"] = args.input_dir
    metadata_items = discover_compact_files(**kwargs)
    metadata_items = [item for item in metadata_items if item.sample_name == args.sample]
    if not metadata_items:
        raise RuntimeError(
            f"no compact ROOT files with Metadata sampleName={args.sample!r} were found"
        )
    return metadata_items


def main(argv=None):
    args = parse_args(argv)
    true_chi_mass, sample_definition = _sample_mass(args)
    metadata_items = _discover(args)
    payload_cache = {}
    sample_output = args.output_dir / args.sample
    sample_output.mkdir(parents=True, exist_ok=True)
    print(
        f"Found {len(metadata_items)} AK-radius file(s) for {args.sample}; "
        f"evaluating {len(args.selections)} unique configuration(s)."
    )
    for number, selection in enumerate(args.selections, start=1):
        key = selection.key
        metadata = _metadata_for_key(metadata_items, key)
        path = _metadata_path(metadata)
        if path not in payload_cache:
            print(f"Reading {path}", flush=True)
            payload_cache[path] = read_event_payload(metadata, max_events=args.max_events)
        payload = payload_cache[path]
        metrics = calculate_configuration_metrics(
            metadata, payload, key, true_chi_mass, physicality=args.physicality
        )
        arrays = _event_arrays(metadata, payload, key, metrics)
        status_names = tuple(metadata.status_names)
        status_rows = _status_rows(arrays["status"], arrays["gate"], status_names)
        output_dir = sample_output / configuration_slug(key)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "configuration_metrics.csv", [
            _scalar_metrics_row(metrics, selection, metadata, true_chi_mass)
        ])
        _write_csv(output_dir / "status_breakdown.csv", status_rows)
        _write_event_tables(output_dir, arrays, status_names, true_chi_mass,
                            args.max_problem_events, args.write_event_table,
                            args.physicality)
        _write_summary(output_dir, selection, metadata, metrics, status_rows,
                       true_chi_mass, args.physicality, sample_definition)
        _make_plots(output_dir, args.sample, key, arrays, status_rows,
                    true_chi_mass, args.physicality, metrics)
        print(
            f"[{number}/{len(args.selections)}] {configuration_slug(key)}: "
            f"physicality={_format(metrics.get('physicality_score', math.nan))}, "
            f"retention={_format(metrics.get('signal_retention', math.nan))}"
        )
    print(f"Diagnostics written under: {sample_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
