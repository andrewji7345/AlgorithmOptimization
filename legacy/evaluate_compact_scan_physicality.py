#!/usr/bin/env python3
"""Evaluate the expanded compact scan across all signal regimes.

This is additive to the legacy evaluators. It reads compact metadata rather
than inferring configurations from filenames, evaluates the factorized event
gates without duplicating reconstruction work, and reports a seven-dimensional
Pareto front. Relative mass bias, relative FWHM, invalid fraction, tail
fraction, finite-statistics penalty, worst regret, and mean regret are all
minimized independently.

Default physicality limits are documented by ``--help`` and are written to
``evaluation_parameters.csv``. Generated masses come from the packaged
MiniAOD lists, including the HtZt_6000_2000 -> generated 6200/1950 alias.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import gzip
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compact_scan_metrics import (
    DEFAULT_BIAS_LIMIT,
    DEFAULT_INVALID_LIMIT,
    DEFAULT_MIN_VALID_EVENTS,
    DEFAULT_RESOLUTION_LIMIT,
    DEFAULT_RESPONSE_HIST_BINS,
    DEFAULT_RESPONSE_HIST_MAX,
    DEFAULT_RESPONSE_HIST_MIN,
    DEFAULT_TAIL_LIMIT,
    DEFAULT_TAIL_RESPONSE_MAX,
    DEFAULT_TAIL_RESPONSE_MIN,
    CompactMetadata,
    ConfigurationKey,
    PhysicalityDefinition,
    SampleDefinition,
    configuration_slug,
    discover_compact_files,
    enumerate_configuration_keys,
    iter_configuration_metrics,
    load_sample_definitions,
    metadata_manifest_row,
    parse_sample_name,
    read_event_payload,
    true_chi_mass_for_sample,
)


DEFAULT_INPUT_DIR = "/eos/uscms/store/user/aji/rootfiles_existingOptimization_compact"
DEFAULT_OUTPUT_DIR = "results/evaluate_compact_scan"
DEFAULT_SAMPLE_LISTS_DIR = Path(__file__).resolve().parents[1] / "test" / "signalMCFiles"
DECAY_ORDER = ("WbWb", "WbZt", "WbHt", "ZtZt", "HtZt", "HtHt")
ONLINE_METRIC_FIELDS = (
    "gate_efficiency",
    "reco_given_gate_efficiency",
    "ungated_reco_efficiency",
    "peak_signal_retention",
    "mass_bias",
    "fwhm_resolution",
    "invalid_fraction",
    "tail_fraction",
    "complexity_guard_fraction_given_gate",
    "other_invalid_fraction_given_gate",
    "n_valid_events",
)
PARETO_COMPONENT_FIELDS = (
    "mass_bias",
    "fwhm_resolution",
    "invalid_fraction",
    "tail_fraction",
    "physicality_statistics_term",
)
PARETO_COMPONENT_LABELS = {
    "mass_bias": "Relative mass bias",
    "fwhm_resolution": "Relative mass FWHM",
    "invalid_fraction": "Invalid fraction",
    "tail_fraction": "Tail fraction",
    "physicality_statistics_term": "Statistics penalty (50/Nvalid)",
}


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def fraction(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("must lie in (0,1]")
    return value


def parse_true_mass_overrides(values: Sequence[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"true-mass override must be SAMPLE=MASS: {value!r}")
        sample, raw_mass = value.split("=", 1)
        sample = sample.strip()
        parse_sample_name(sample)
        mass = float(raw_mass)
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"invalid true chi mass in override {value!r}")
        if sample in result:
            raise ValueError(f"duplicate true-mass override for {sample}")
        result[sample] = mass
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    source.add_argument(
        "--input-file", action="append", default=None,
        help="Explicit compact ROOT input; repeat for multiple files.",
    )
    parser.add_argument("--input-glob", default="*.root")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-lists-dir", default=str(DEFAULT_SAMPLE_LISTS_DIR))
    parser.add_argument(
        "--discovered-only", action="store_true",
        help="Pilot mode: use discovered samples, rather than all 114 lists, as the coverage denominator.",
    )
    parser.add_argument(
        "--samples", nargs="+", default=None,
        help="Explicit pilot sample set; also defines the coverage denominator.",
    )
    parser.add_argument(
        "--true-mass-override", action="append", default=[], metavar="SAMPLE=MASS",
        help="Override generated MChi for a named sample; recorded in outputs.",
    )
    parser.add_argument("--max-events", type=int, default=-1, help="-1 reads every ntuple event.")
    parser.add_argument("--chunk-size", type=positive_int, default=64)
    parser.add_argument("--temp-dir", default=None, help="Parent directory for temporary float32 matrices.")

    physicality = parser.add_argument_group("physicality definition")
    physicality.add_argument(
        "--bias-limit", type=positive_float, default=DEFAULT_BIAS_LIMIT,
        help="Maximum allowed absolute median response bias.",
    )
    physicality.add_argument(
        "--resolution-limit", type=positive_float, default=DEFAULT_RESOLUTION_LIMIT,
        help="Maximum allowed fixed-histogram FWHM/peak resolution.",
    )
    physicality.add_argument(
        "--invalid-limit", "--invalid-fraction-limit", dest="invalid_limit",
        type=positive_float, default=DEFAULT_INVALID_LIMIT,
        help="Maximum invalid-reconstruction fraction conditional on the gate.",
    )
    physicality.add_argument(
        "--tail-limit", "--tail-fraction-limit", dest="tail_limit",
        type=positive_float, default=DEFAULT_TAIL_LIMIT,
        help="Maximum valid-event response-tail fraction.",
    )
    physicality.add_argument(
        "--min-valid-events", "--minimum-valid-events", dest="min_valid_events",
        type=positive_int, default=DEFAULT_MIN_VALID_EVENTS,
        help="Minimum gated valid-event count (the statistics term is this/Nvalid).",
    )
    physicality.add_argument(
        "--tail-response-min", "--response-min", dest="tail_response_min",
        type=float, default=DEFAULT_TAIL_RESPONSE_MIN,
        help="Lower inclusive edge of the non-tail mass-response window.",
    )
    physicality.add_argument(
        "--tail-response-max", "--response-max", dest="tail_response_max",
        type=float, default=DEFAULT_TAIL_RESPONSE_MAX,
        help="Upper inclusive edge of the non-tail mass-response window.",
    )
    physicality.add_argument(
        "--response-hist-min", type=float, default=DEFAULT_RESPONSE_HIST_MIN,
        help="Lower edge of the shared fixed response histogram.",
    )
    physicality.add_argument(
        "--response-hist-max", type=float, default=DEFAULT_RESPONSE_HIST_MAX,
        help="Upper edge of the shared fixed response histogram.",
    )
    physicality.add_argument(
        "--response-hist-bins", type=positive_int, default=DEFAULT_RESPONSE_HIST_BINS,
        help="Number of equal-width bins in the shared response histogram.",
    )

    ranking = parser.add_argument_group("global ranking")
    ranking.add_argument("--mean-weight", type=float, default=0.25)
    ranking.add_argument("--minimum-regime-coverage", type=fraction, default=1.0)
    ranking.add_argument("--physicality-statistic", choices=("worst", "q90"), default="worst")
    ranking.add_argument("--top-configurations", type=positive_int, default=15)
    args = parser.parse_args(argv)
    if args.max_events == 0 or args.max_events < -1:
        parser.error("--max-events must be -1 or positive")
    if not math.isfinite(args.mean_weight) or args.mean_weight < 0:
        parser.error("--mean-weight must be finite and nonnegative")
    try:
        args.true_mass_overrides = parse_true_mass_overrides(args.true_mass_override)
        args.physicality = PhysicalityDefinition(
            bias_limit=args.bias_limit,
            resolution_limit=args.resolution_limit,
            invalid_limit=args.invalid_limit,
            tail_limit=args.tail_limit,
            min_valid_events=args.min_valid_events,
            tail_response_min=args.tail_response_min,
            tail_response_max=args.tail_response_max,
            response_hist_min=args.response_hist_min,
            response_hist_max=args.response_hist_max,
            response_hist_bins=args.response_hist_bins,
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def build_configuration_universe(
    metadata: Sequence[CompactMetadata],
) -> Tuple[Tuple[ConfigurationKey, ...], Dict[ConfigurationKey, int]]:
    # Production has 114 copies of each AK-radius grid. Enumerate each unique
    # metadata grid once instead of manufacturing ~35 million duplicate key
    # objects merely to reduce them to the ~311k-key union.
    unique_grids: Dict[Tuple[Any, ...], CompactMetadata] = {}
    for item in metadata:
        fingerprint = (
            item.ak_radius,
            item.config_collection_pt_cut,
            item.config_ca_radius,
            item.config_cos_thrust,
            item.default_gate_jet_counts,
            item.default_gate_pt_cuts,
        )
        unique_grids.setdefault(fingerprint, item)
    keys = tuple(sorted({
        key
        for item in unique_grids.values()
        for key in enumerate_configuration_keys(item)
    }))
    return keys, {key: index for index, key in enumerate(keys)}


def _rowwise_quantile(values: np.ndarray, quantile: float) -> np.ndarray:
    """Row-wise linear quantile that treats real ``+inf`` as an endpoint.

    NumPy's interpolated quantile may evaluate ``inf - inf`` and return NaN.
    Here NaNs are missing, while an upper order statistic of +inf yields +inf
    whenever it has nonzero interpolation weight.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("row-wise quantile input must be two-dimensional")
    count = np.count_nonzero(~np.isnan(values), axis=1)
    ordered = np.sort(np.where(np.isnan(values), np.inf, values), axis=1)
    position = np.maximum(count - 1, 0) * float(quantile)
    lower_index = np.floor(position).astype(int)
    upper_index = np.ceil(position).astype(int)
    row = np.arange(len(values))
    lower = ordered[row, lower_index]
    upper = ordered[row, upper_index]
    weight = position - lower_index
    result = lower.copy()
    interpolate = (count > 0) & (upper_index != lower_index) & (weight > 0.0)
    finite_pair = interpolate & np.isfinite(lower) & np.isfinite(upper)
    result[finite_pair] = lower[finite_pair] + weight[finite_pair] * (
        upper[finite_pair] - lower[finite_pair]
    )
    result[interpolate & np.isposinf(upper)] = np.inf
    result[count == 0] = np.nan
    return result


def _rowwise_statistics(
    matrix: np.ndarray, chunk_size: int = 4096,
) -> Dict[str, np.ndarray]:
    n_rows = matrix.shape[0]
    output = {
        "mean": np.full(n_rows, np.nan),
        "median": np.full(n_rows, np.nan),
        "q90": np.full(n_rows, np.nan),
        "worst": np.full(n_rows, np.nan),
    }
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        values = np.asarray(matrix[start:stop], dtype=np.float64)
        counts = np.count_nonzero(~np.isnan(values), axis=1)
        with np.errstate(invalid="ignore"):
            sums = np.nansum(values, axis=1)
            output["mean"][start:stop] = np.divide(
                sums, counts, out=np.full(len(values), np.nan), where=counts > 0
            )
            output["median"][start:stop] = _rowwise_quantile(values, 0.50)
            output["q90"][start:stop] = _rowwise_quantile(values, 0.90)
            maximum = np.max(np.where(np.isnan(values), -np.inf, values), axis=1)
            maximum[counts == 0] = np.nan
            output["worst"][start:stop] = maximum
    return output


def pareto_mask(*objectives: np.ndarray, eligible: Optional[np.ndarray] = None) -> np.ndarray:
    """Return nondominated points for two or more minimization objectives.

    Equal objective vectors are all retained. Positive infinity is allowed;
    a NaN in any objective excludes that point.
    """

    if len(objectives) == 1:
        matrix = np.asarray(objectives[0], dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
    else:
        matrix = np.column_stack([np.asarray(value, dtype=float) for value in objectives])
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("Pareto evaluation requires at least two objectives")
    candidates = ~np.any(np.isnan(matrix), axis=1)
    if eligible is not None:
        candidates &= np.asarray(eligible, dtype=bool)
    result = np.zeros(len(matrix), dtype=bool)
    indices = np.flatnonzero(candidates)
    if not len(indices):
        return result
    values = matrix[indices]
    order = indices[np.lexsort(tuple(
        values[:, column] for column in range(values.shape[1] - 1, -1, -1)
    ))]
    skyline: List[np.ndarray] = []
    position = 0
    while position < len(order):
        end = position + 1
        current = matrix[order[position]]
        while end < len(order) and np.array_equal(matrix[order[end]], current):
            end += 1
        if not any(np.all(front <= current) for front in skyline):
            result[order[position:end]] = True
            skyline.append(current)
        position = end
    return result


def aggregate_scan_matrices(
    retention: np.ndarray,
    physicality: np.ndarray,
    keys: Optional[Sequence[ConfigurationKey]] = None,
    mean_weight: float = 0.25,
    minimum_regime_coverage: float = 1.0,
    physicality_statistic: str = "worst",
    mean_metrics: Optional[Mapping[str, np.ndarray]] = None,
    pareto_components: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Aggregate per-regime axes without hiding missing or failed regimes."""

    if retention.shape != physicality.shape or retention.ndim != 2:
        raise ValueError("retention and physicality must be equal 2D matrices")
    if keys is not None and len(keys) != retention.shape[0]:
        raise ValueError("key count does not match matrix rows")
    if physicality_statistic not in ("worst", "q90"):
        raise ValueError("physicality_statistic must be 'worst' or 'q90'")
    n_configs, n_regimes = retention.shape
    observed = ~np.isnan(retention)
    coverage_count = np.count_nonzero(observed, axis=1)
    coverage = coverage_count / n_regimes if n_regimes else np.full(n_configs, np.nan)
    best_retention = np.full(n_regimes, np.nan)
    rank_sum = np.zeros(n_configs, dtype=np.float64)
    rank_count = np.zeros(n_configs, dtype=np.int32)
    rank_worst = np.full(n_configs, np.nan)
    physical_rank_sum = np.zeros(n_configs, dtype=np.float64)
    physical_rank_count = np.zeros(n_configs, dtype=np.int32)
    physical_rank_worst = np.full(n_configs, np.nan)

    def accumulate_ranks(values: np.ndarray, selected: np.ndarray,
                         sums: np.ndarray, counts: np.ndarray,
                         worst: np.ndarray) -> None:
        indices = np.flatnonzero(selected)
        if not len(indices):
            return
        order = indices[np.argsort(-values[indices], kind="stable")]
        sorted_values = values[order]
        starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
        ends = np.r_[starts[1:], len(order)]
        ranks = np.empty(len(order), dtype=np.int32)
        for start, end in zip(starts, ends):
            ranks[start:end] = start + 1
        sums[order] += ranks
        counts[order] += 1
        prior = worst[order]
        worst[order] = np.where(np.isnan(prior), ranks, np.maximum(prior, ranks))

    for regime in range(n_regimes):
        values = np.asarray(retention[:, regime], dtype=float)
        finite = np.isfinite(values)
        if np.any(finite):
            best_retention[regime] = np.max(values[finite])
            accumulate_ranks(values, finite, rank_sum, rank_count, rank_worst)
            physical = finite & np.isfinite(physicality[:, regime]) & (physicality[:, regime] <= 1.0)
            accumulate_ranks(
                values, physical, physical_rank_sum, physical_rank_count,
                physical_rank_worst,
            )

    regret_stats = {name: np.full(n_configs, np.nan) for name in ("mean", "median", "q90", "worst")}
    retention_mean = np.full(n_configs, np.nan)
    for start in range(0, n_configs, 4096):
        stop = min(start + 4096, n_configs)
        values = np.asarray(retention[start:stop], dtype=np.float64)
        regrets = best_retention[None, :] - values
        retention_summary = _rowwise_statistics(values, chunk_size=len(values))
        regret_summary = _rowwise_statistics(regrets, chunk_size=len(values))
        retention_mean[start:stop] = retention_summary["mean"]
        for name in regret_stats:
            regret_stats[name][start:stop] = regret_summary[name]

    pstats = _rowwise_statistics(physicality)
    evaluated_physicality = observed
    failed = evaluated_physicality & ((physicality > 1.0) | ~np.isfinite(physicality))
    failure_count = np.count_nonzero(failed, axis=1)
    failure_fraction = np.divide(
        failure_count, coverage_count,
        out=np.full(n_configs, np.nan), where=coverage_count > 0,
    )
    failure_fraction_expected = failure_count / n_regimes if n_regimes else np.full(n_configs, np.nan)
    objective = regret_stats["worst"] + mean_weight * regret_stats["mean"]
    eligible = (coverage >= minimum_regime_coverage) & np.isfinite(objective)
    physicality_pass_covered = eligible & (failure_count == 0)
    physicality_pass_all = (coverage_count == n_regimes) & (failure_count == 0)
    component_statistics: Dict[str, Dict[str, np.ndarray]] = {}
    if pareto_components is None:
        component_statistics["physicality"] = pstats
    else:
        for name in PARETO_COMPONENT_FIELDS:
            values = np.asarray(pareto_components[name])
            if values.shape != retention.shape:
                raise ValueError(f"Pareto component {name!r} has shape {values.shape}")
            component_statistics[name] = _rowwise_statistics(values)
    pareto_objectives = [
        component_statistics[name][physicality_statistic]
        for name in component_statistics
    ]
    pareto_objectives.extend((regret_stats["worst"], regret_stats["mean"]))
    on_pareto = pareto_mask(*pareto_objectives, eligible=eligible)
    result = {
        "n_regimes": coverage_count,
        "regime_coverage": coverage,
        "mean_signal_retention": retention_mean,
        "best_retention_by_regime": best_retention,
        "mean_regret": regret_stats["mean"],
        "median_regret": regret_stats["median"],
        "q90_regret": regret_stats["q90"],
        "worst_regret": regret_stats["worst"],
        "mean_signal_retention_rank": np.divide(
            rank_sum, rank_count, out=np.full(n_configs, np.nan), where=rank_count > 0
        ),
        "worst_signal_retention_rank": rank_worst,
        "mean_physical_signal_retention_rank": np.divide(
            physical_rank_sum, physical_rank_count,
            out=np.full(n_configs, np.nan), where=physical_rank_count > 0,
        ),
        "worst_physical_signal_retention_rank": physical_rank_worst,
        "global_objective": objective,
        "mean_physicality": pstats["mean"],
        "median_physicality": pstats["median"],
        "q90_physicality": pstats["q90"],
        "worst_physicality": pstats["worst"],
        "physicality_failure_count": failure_count,
        "physicality_failure_fraction": failure_fraction,
        "physicality_failure_fraction_expected": failure_fraction_expected,
        "eligible_coverage": eligible,
        "physicality_pass_covered_regimes": physicality_pass_covered,
        "physicality_pass_all": physicality_pass_all,
        "pareto": on_pareto,
    }
    for name, statistics in component_statistics.items():
        for statistic, values in statistics.items():
            result[f"{statistic}_{name}"] = values
    if mean_metrics:
        for name, values in mean_metrics.items():
            values = np.asarray(values)
            if values.shape != (n_configs,):
                raise ValueError(f"mean metric {name!r} has shape {values.shape}")
            result[f"mean_{name}"] = values
    return result


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str],
    compressed: bool = False,
) -> None:
    opener = gzip.open if compressed else open
    with opener(path, "wt", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


KEY_FIELDS = (
    "configuration", "n_gate_jets", "gate_pt_cut", "collection_pt_cut",
    "ak_radius", "ca_radius", "cos_thrust_cut",
)
COMPONENT_AGGREGATE_FIELDS = tuple(
    f"{statistic}_{name}"
    for name in PARETO_COMPONENT_FIELDS
    for statistic in ("mean", "median", "q90", "worst")
)
AGGREGATE_FIELDS = KEY_FIELDS + (
    "n_regimes", "regime_coverage", "mean_signal_retention",
    "global_objective", "mean_regret", "median_regret", "q90_regret",
    "worst_regret", "mean_signal_retention_rank", "worst_signal_retention_rank",
    "mean_physical_signal_retention_rank", "worst_physical_signal_retention_rank",
    "mean_gate_efficiency", "mean_reco_given_gate_efficiency",
    "mean_ungated_reco_efficiency", "mean_peak_signal_retention",
    "mean_mass_bias", "mean_fwhm_resolution", "mean_invalid_fraction",
    "mean_tail_fraction", "mean_complexity_guard_fraction_given_gate",
    "mean_other_invalid_fraction_given_gate", "mean_n_valid_events",
    "mean_physicality", "median_physicality",
    "q90_physicality", "worst_physicality", "physicality_failure_count",
    "physicality_failure_fraction", "physicality_failure_fraction_expected",
    "eligible_coverage", "physicality_pass_covered_regimes",
    "physicality_pass_all", "pareto",
) + COMPONENT_AGGREGATE_FIELDS


def aggregate_row(
    index: int, key: ConfigurationKey, aggregate: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    row = {**key.as_dict(), "configuration": configuration_slug(key)}
    for name in AGGREGATE_FIELDS:
        if name not in row and name in aggregate:
            row[name] = aggregate[name][index]
    return row


def _regime_sort_key(definition: SampleDefinition) -> Tuple[Any, ...]:
    try:
        decay = (0, DECAY_ORDER.index(definition.decay))
    except ValueError:
        decay = (1, definition.decay)
    return (definition.nominal_suu_mass, definition.nominal_chi_mass, decay)


def _fallback_definition(sample: str) -> SampleDefinition:
    decay, suu, chi = parse_sample_name(sample)
    return SampleDefinition(sample, decay, suu, chi, suu, chi, "")


def _write_manifests(
    output_dir: Path,
    metadata: Sequence[CompactMetadata],
    definitions: Mapping[str, SampleDefinition],
    expected_samples: Sequence[str],
    overrides: Mapping[str, float],
    physicality: PhysicalityDefinition,
    args: argparse.Namespace,
) -> None:
    input_fields = (
        "sample", "decay", "nominal_suu_mass", "nominal_chi_mass",
        "generated_suu_mass", "generated_chi_mass", "response_chi_mass",
        "true_mass_overridden", "ak_radius", "path", "schema_version",
        "processed_events", "sum_weights", "sum_weights2",
        "n_reconstruction_configurations", "n_full_configurations",
        "collection_pt_cuts", "ca_radii", "cos_thrust_cuts",
        "gate_jet_counts", "gate_pt_cuts",
    )
    rows = []
    for item in metadata:
        definition = definitions[item.sample_name]
        rows.append({
            **metadata_manifest_row(item), **definition.as_dict(),
            "response_chi_mass": true_chi_mass_for_sample(item.sample_name, definitions, overrides),
            "true_mass_overridden": item.sample_name in overrides,
        })
    write_csv(output_dir / "input_manifest.csv", rows, input_fields)
    by_sample = defaultdict(list)
    for item in metadata:
        by_sample[item.sample_name].append(item.ak_radius)
    regime_fields = (
        "regime_index", "sample", "decay", "nominal_suu_mass",
        "nominal_chi_mass", "generated_suu_mass", "generated_chi_mass",
        "response_chi_mass", "has_input", "n_ak_files", "ak_radii",
    )
    regime_rows = []
    for index, sample in enumerate(expected_samples):
        definition = definitions[sample]
        radii = sorted(by_sample[sample])
        regime_rows.append({
            "regime_index": index, **definition.as_dict(),
            "response_chi_mass": true_chi_mass_for_sample(sample, definitions, overrides),
            "has_input": bool(radii), "n_ak_files": len(radii),
            "ak_radii": ";".join(map(str, radii)),
        })
    write_csv(output_dir / "regime_manifest.csv", regime_rows, regime_fields)
    parameters = {
        **physicality.as_dict(),
        "mean_weight": args.mean_weight,
        "minimum_regime_coverage": args.minimum_regime_coverage,
        "physicality_statistic": args.physicality_statistic,
        "max_events": args.max_events,
        "chunk_size": args.chunk_size,
        "n_expected_regimes": len(expected_samples),
        "primary_weighting": "unweighted event counts",
        "gate_comparison": "stored float akJetPt strictly greater than Tgate",
        "signal_retention": "N(gate and valid reconstruction)/N(all)",
        "regret_reference": "unconstrained maximum signal retention per regime",
        "global_objective": "worst additive retention regret + mean_weight * mean regret",
        "pareto_objectives": (
            f"{args.physicality_statistic} relative mass bias; "
            f"{args.physicality_statistic} relative mass FWHM; "
            f"{args.physicality_statistic} invalid fraction; "
            f"{args.physicality_statistic} tail fraction; "
            f"{args.physicality_statistic} statistics penalty (min_valid_events/n_valid); "
            "worst retention regret; mean retention regret"
        ),
        "pareto_directions": "minimize every objective independently",
        "physicality_pass_all": "covered in every expected regime and score<=1 in every regime",
        "physicality_pass_covered_regimes": "requested coverage met and score<=1 wherever evaluated",
    }
    write_csv(
        output_dir / "evaluation_parameters.csv",
        ({"parameter": key, "value": value} for key, value in parameters.items()),
        ("parameter", "value"),
    )


def _best_by_regime_rows(
    retention: np.ndarray,
    physicality: np.ndarray,
    keys: Sequence[ConfigurationKey],
    expected_samples: Sequence[str],
    definitions: Mapping[str, SampleDefinition],
) -> List[Dict[str, Any]]:
    rows = []
    for regime_index, sample in enumerate(expected_samples):
        definition = definitions[sample]
        values = np.asarray(retention[:, regime_index], dtype=float)
        available = np.isfinite(values)
        unconstrained = int(np.nanargmax(values)) if np.any(available) else None
        physical = available & np.isfinite(physicality[:, regime_index]) & (physicality[:, regime_index] <= 1.0)
        physical_best = int(np.nanargmax(np.where(physical, values, np.nan))) if np.any(physical) else None
        row = {"regime_index": regime_index, **definition.as_dict()}
        for prefix, index in (("unconstrained", unconstrained), ("physical", physical_best)):
            if index is None:
                row.update({f"{prefix}_configuration": "", f"{prefix}_signal_retention": np.nan,
                            f"{prefix}_physicality_score": np.nan})
            else:
                row.update({
                    f"{prefix}_configuration": configuration_slug(keys[index]),
                    f"{prefix}_signal_retention": retention[index, regime_index],
                    f"{prefix}_physicality_score": physicality[index, regime_index],
                })
        if unconstrained is not None and physical_best is not None:
            row["retention_gain_if_unconstrained"] = values[unconstrained] - values[physical_best]
        else:
            row["retention_gain_if_unconstrained"] = np.nan
        rows.append(row)
    return rows


def _plot_tradeoff(
    output_file: Path,
    aggregate: Mapping[str, np.ndarray],
    statistic_name: str,
) -> None:
    x = np.asarray(aggregate[f"{statistic_name}_physicality"], float)
    y = np.asarray(aggregate["global_objective"], float)
    eligible = np.asarray(aggregate["eligible_coverage"], bool)
    finite = eligible & np.isfinite(x) & np.isfinite(y)
    indices = np.flatnonzero(finite)
    if len(indices) > 100000:
        indices = indices[:: int(math.ceil(len(indices) / 100000))]
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    if len(indices):
        scatter = ax.scatter(
            x[indices], y[indices],
            c=np.asarray(aggregate["physicality_failure_fraction"])[indices],
            s=8, alpha=0.35, cmap="viridis", vmin=0.0, vmax=1.0,
            rasterized=True,
        )
        fig.colorbar(scatter, ax=ax, label="Fraction of evaluated regimes failing physicality")
    overflow = np.flatnonzero(eligible & np.isposinf(x) & np.isfinite(y))
    finite_x = x[finite]
    overflow_x = (
        max(1.1, float(np.max(finite_x)) * 1.05)
        if len(finite_x) else 1.1
    )
    if len(overflow):
        shown = overflow
        if len(shown) > 10000:
            shown = shown[:: int(math.ceil(len(shown) / 10000))]
        ax.scatter(
            np.full(len(shown), overflow_x), y[shown], marker=">", s=15,
            color="0.45", alpha=0.35, rasterized=True,
            label=f"+inf physicality ({len(overflow):,}; shown at right edge)",
        )
    pareto = np.flatnonzero(np.asarray(aggregate["pareto"], bool) & np.isfinite(x) & np.isfinite(y))
    if len(pareto):
        ax.scatter(x[pareto], y[pareto], facecolors="none", edgecolors="tab:red",
                   s=30, linewidths=0.8, label="7D Pareto points")
    pareto_overflow = np.flatnonzero(
        np.asarray(aggregate["pareto"], bool) & np.isposinf(x) & np.isfinite(y)
    )
    if len(pareto_overflow):
        ax.scatter(np.full(len(pareto_overflow), overflow_x), y[pareto_overflow],
                   marker=">", s=38, color="tab:red", zorder=4,
                   label="Pareto point with +inf physicality")
    boundary_label = (
        "All-regime physicality boundary"
        if statistic_name == "worst"
        else "q90 physicality = 1 reference"
    )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0,
               label=boundary_label)
    ax.set_xlabel(f"Global {statistic_name} physicality score (lower is better)")
    ax.set_ylabel("Worst retention regret + weighted mean regret (lower is better)")
    ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
    ax.set_yscale("symlog", linthresh=0.01, linscale=1.0)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_title("Scalar projection of the seven-objective Pareto space")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_file, dpi=190)
    plt.close(fig)


def _plot_component_tradeoffs(
    output_file: Path,
    aggregate: Mapping[str, np.ndarray],
    statistic_name: str,
) -> None:
    """Project the seven-dimensional front onto each component and regret."""

    worst_regret = np.asarray(aggregate["worst_regret"], float)
    mean_regret = np.asarray(aggregate["mean_regret"], float)
    eligible = np.asarray(aggregate["eligible_coverage"], bool)
    pareto = np.asarray(aggregate["pareto"], bool)
    specifications = list(PARETO_COMPONENT_FIELDS) + ["mean_regret"]
    fig = plt.figure(figsize=(17.0, 9.5))
    grid = fig.add_gridspec(
        2, 4, width_ratios=(1.0, 1.0, 1.0, 0.045),
        left=0.065, right=0.94, bottom=0.08, top=0.91,
        wspace=0.32, hspace=0.32,
    )
    axes = np.asarray([
        [fig.add_subplot(grid[row, column]) for column in range(3)]
        for row in range(2)
    ])
    colorbar_ax = fig.add_subplot(grid[:, 3])
    color_image = None
    for ax, name in zip(axes.flat, specifications):
        x = mean_regret if name == "mean_regret" else np.asarray(
            aggregate[f"{statistic_name}_{name}"], float
        )
        drawable = np.flatnonzero(
            eligible & np.isfinite(x) & np.isfinite(worst_regret) & np.isfinite(mean_regret)
        )
        if len(drawable) > 100000:
            drawable = drawable[:: int(math.ceil(len(drawable) / 100000))]
        if len(drawable):
            color_image = ax.scatter(
                x[drawable], worst_regret[drawable], c=mean_regret[drawable],
                s=8, alpha=0.28, cmap="viridis", rasterized=True,
            )
        front = np.flatnonzero(
            pareto & np.isfinite(x) & np.isfinite(worst_regret)
        )
        if len(front):
            ax.scatter(x[front], worst_regret[front], facecolors="none",
                       edgecolors="tab:red", s=28, linewidths=0.8,
                       label="7D Pareto points")
        ax.set_xlabel(
            "Mean retention regret"
            if name == "mean_regret" else PARETO_COMPONENT_LABELS[name]
        )
        ax.set_ylabel("Worst retention regret")
        if name != "tail_fraction":
            ax.set_xscale("symlog", linthresh=0.01, linscale=1.0)
        ax.set_yscale("symlog", linthresh=0.01, linscale=1.0)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    if color_image is not None:
        fig.colorbar(
            color_image, cax=colorbar_ax, label="Mean retention regret",
        )
    else:
        colorbar_ax.set_visible(False)
    fig.suptitle(
        f"Seven-objective Pareto projections ({statistic_name} component values across regimes)"
    )
    fig.savefig(output_file, dpi=190)
    plt.close(fig)


def _plot_pareto_hyperparameters(
    output_file: Path,
    pareto_indices: Sequence[int],
    keys: Sequence[ConfigurationKey],
    aggregate: Mapping[str, np.ndarray],
    statistic_name: str,
) -> None:
    """Show which hyperparameters trace the global Pareto tradeoff."""

    indices = np.asarray(pareto_indices, dtype=int)
    if not len(indices):
        return
    x = np.asarray(aggregate[f"{statistic_name}_physicality"], float)[indices]
    y = np.asarray(aggregate["global_objective"], float)[indices]
    finite_y = np.isfinite(y)
    indices, x, y = indices[finite_y], x[finite_y], y[finite_y]
    if not len(indices):
        return
    finite_x = np.isfinite(x)
    overflow_x = (
        max(1.1, float(np.max(x[finite_x])) * 1.05)
        if np.any(finite_x) else 1.1
    )
    x = np.where(np.isposinf(x), overflow_x, x)
    specifications = (
        ("Gate multiplicity", np.asarray([keys[i].n_gate_jets for i in indices], float)),
        ("Gate pT [GeV]", np.asarray([
            np.nan if keys[i].gate_pt_cut is None else keys[i].gate_pt_cut for i in indices
        ], float)),
        ("Collection pT [GeV]", np.asarray([keys[i].collection_pt_cut for i in indices], float)),
        ("AK radius", np.asarray([keys[i].ak_radius for i in indices], float)),
        ("CA radius", np.asarray([keys[i].ca_radius for i in indices], float)),
        ("Hybrid cut c", np.asarray([keys[i].cos_thrust_cut for i in indices], float)),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.0), sharex=True, sharey=True)
    for ax, (label, values) in zip(axes.flat, specifications):
        drawable = np.isfinite(values)
        image = ax.scatter(x[drawable], y[drawable], c=values[drawable], cmap="viridis",
                           s=24, edgecolors="none")
        fig.colorbar(image, ax=ax, label=label)
        if np.any(~drawable):
            ax.scatter(x[~drawable], y[~drawable], marker="x", color="black", s=28,
                       label="ungated")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        ax.set_title(label)
    for ax in axes[-1]:
        ax.set_xlabel(f"Global {statistic_name} physicality")
    for ax in axes[:, 0]:
        ax.set_ylabel("Global retention-regret objective")
    for ax in axes.flat:
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
        ax.set_yscale("symlog", linthresh=0.01, linscale=1.0)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
    fig.suptitle("Hyperparameters on the seven-objective compact-scan Pareto front")
    if np.any(~finite_x):
        fig.text(
            0.5, 0.005,
            "+inf physicality points are shown at the right-edge overflow coordinate.",
            ha="center", fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_file, dpi=190)
    plt.close(fig)


def _plot_coverage(output_file: Path, aggregate: Mapping[str, np.ndarray]) -> None:
    coverage = np.asarray(aggregate["regime_coverage"], float)
    objective = np.asarray(aggregate["global_objective"], float)
    failure = np.asarray(aggregate["physicality_failure_fraction"], float)
    drawable = np.isfinite(coverage) & np.isfinite(objective)
    indices = np.flatnonzero(drawable)
    if len(indices) > 100000:
        indices = indices[:: int(math.ceil(len(indices) / 100000))]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    if len(indices):
        image = ax.scatter(
            coverage[indices], objective[indices], c=failure[indices],
            s=8, alpha=0.35, cmap="viridis", vmin=0.0, vmax=1.0,
            rasterized=True,
        )
        fig.colorbar(image, ax=ax, label="Physicality failure fraction")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Expected-regime coverage")
    ax.set_ylabel("Global retention-regret objective")
    ax.set_title("Configuration coverage and global performance")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_file, dpi=190)
    plt.close(fig)


def _plot_regime_best(
    output_file: Path, rows: Sequence[Mapping[str, Any]],
) -> None:
    decays = [value for value in DECAY_ORDER if any(row["decay"] == value for row in rows)]
    decays += sorted({row["decay"] for row in rows} - set(decays))
    mass_points = sorted({(row["nominal_suu_mass"], row["nominal_chi_mass"]) for row in rows})
    matrix = np.full((len(mass_points), len(decays)), np.nan)
    mass_index, decay_index = {value: i for i, value in enumerate(mass_points)}, {value: i for i, value in enumerate(decays)}
    for row in rows:
        matrix[mass_index[(row["nominal_suu_mass"], row["nominal_chi_mass"])], decay_index[row["decay"]]] = row["physical_signal_retention"]
    fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(decays) + 2), max(6, 0.55 * len(mass_points) + 2)))
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(image, ax=ax, label="Best physical signal retention")
    ax.set_xticks(np.arange(len(decays))), ax.set_xticklabels(decays)
    ax.set_yticks(np.arange(len(mass_points)))
    ax.set_yticklabels([f"{suu:g}/{chi:g}" for suu, chi in mass_points])
    ax.set_xlabel("Decay channel"), ax.set_ylabel("Nominal MSuu/MChi [GeV]")
    ax.set_title("Best physical configuration in each signal regime")
    fig.tight_layout()
    fig.savefig(output_file, dpi=190)
    plt.close(fig)


def _plot_top_regrets(
    output_file: Path, top_indices: Sequence[int], retention: np.ndarray,
    best: np.ndarray, keys: Sequence[ConfigurationKey], samples: Sequence[str],
) -> None:
    if not top_indices:
        return
    values = best[None, :] - np.asarray(retention[top_indices], float)
    fig, ax = plt.subplots(figsize=(max(14, 0.18 * len(samples) + 4), max(5, 0.42 * len(top_indices) + 2)))
    image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", vmin=0, cmap="magma")
    fig.colorbar(image, ax=ax, label="Additive signal-retention regret")
    ax.set_xticks(np.arange(len(samples))), ax.set_xticklabels(samples, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(top_indices)))
    ax.set_yticklabels([configuration_slug(keys[index]) for index in top_indices], fontsize=6)
    ax.set_xlabel("Signal regime"), ax.set_ylabel("Global configuration")
    ax.set_title("Top physical configurations across regimes")
    fig.tight_layout()
    fig.savefig(output_file, dpi=190)
    plt.close(fig)


def run_evaluation(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    definitions = load_sample_definitions(
        args.sample_lists_dir,
        expected_count=None if (args.discovered_only or args.samples) else 114,
    )
    requested = args.samples
    metadata = discover_compact_files(
        input_dir=args.input_dir if args.input_file is None else None,
        paths=args.input_file,
        pattern=args.input_glob,
        samples=requested,
    )
    if not metadata:
        raise RuntimeError("no compact scan files were discovered")
    discovered_samples = sorted({item.sample_name for item in metadata})
    for sample in discovered_samples:
        if sample not in definitions:
            if not args.discovered_only:
                raise KeyError(f"discovered sample {sample!r} has no packaged regime definition")
            definitions[sample] = _fallback_definition(sample)
    if requested:
        expected_names = list(dict.fromkeys(requested))
        missing_definition = [sample for sample in expected_names if sample not in definitions]
        if missing_definition:
            raise KeyError("samples lack regime definitions: " + ", ".join(missing_definition))
    elif args.discovered_only:
        expected_names = discovered_samples
    else:
        expected_names = list(definitions)
    expected_samples = tuple(sorted(expected_names, key=lambda sample: _regime_sort_key(definitions[sample])))
    regime_index = {sample: index for index, sample in enumerate(expected_samples)}
    unexpected = sorted(set(discovered_samples) - set(expected_samples))
    if unexpected:
        raise ValueError("inputs fall outside expected regime set: " + ", ".join(unexpected))

    keys, key_to_index = build_configuration_universe(metadata)
    if not keys:
        raise RuntimeError("no configurations were found in metadata")
    _write_manifests(output_dir, metadata, definitions, expected_samples,
                     args.true_mass_overrides, args.physicality, args)
    print(f"Discovered {len(metadata)} files for {len(discovered_samples)} samples")
    print(f"Coverage denominator: {len(expected_samples)} regimes; configuration union: {len(keys)}")

    temp_parent = Path(args.temp_dir).resolve() if args.temp_dir else None
    with tempfile.TemporaryDirectory(prefix="compact_scan_eval_", dir=temp_parent) as temporary:
        retention_path = Path(temporary) / "retention.float32"
        physicality_path = Path(temporary) / "physicality.float32"
        shape = (len(keys), len(expected_samples))
        retention = np.memmap(retention_path, mode="w+", dtype=np.float32, shape=shape)
        physicality = np.memmap(physicality_path, mode="w+", dtype=np.float32, shape=shape)
        pareto_components = {
            name: np.memmap(
                Path(temporary) / f"{name}.float32",
                mode="w+", dtype=np.float32, shape=shape,
            )
            for name in PARETO_COMPONENT_FIELDS
        }
        retention[:] = np.nan
        physicality[:] = np.nan
        for values in pareto_components.values():
            values[:] = np.nan
        metric_sums = {
            name: np.zeros(len(keys), dtype=np.float64)
            for name in ONLINE_METRIC_FIELDS
        }
        metric_counts = {
            name: np.zeros(len(keys), dtype=np.int32)
            for name in ONLINE_METRIC_FIELDS
        }
        index_cache: Dict[Tuple[Any, int], np.ndarray] = {}
        records_by_sample = defaultdict(list)
        for item in metadata:
            records_by_sample[item.sample_name].append(item)
        for sample in discovered_samples:
            reference_ids, reference_weights = None, None
            true_mass = true_chi_mass_for_sample(sample, definitions, args.true_mass_overrides)
            records = sorted(records_by_sample[sample], key=lambda item: item.ak_radius)
            print(f"[{regime_index[sample] + 1}/{len(expected_samples)}] {sample}: {len(records)} AK files", flush=True)
            for item in records:
                payload = read_event_payload(item, args.max_events)
                if reference_ids is None:
                    reference_ids, reference_weights = payload.event_ids, payload.gen_weight
                elif not np.array_equal(reference_ids, payload.event_ids):
                    raise ValueError(f"{sample}: event IDs/order differ across AK files")
                elif not np.allclose(reference_weights, payload.gen_weight, rtol=1e-6, atol=1e-6):
                    raise ValueError(f"{sample}: generator weights differ across AK files")
                fingerprint = (
                    item.ak_radius, item.config_collection_pt_cut,
                    item.config_ca_radius, item.config_cos_thrust,
                    item.default_gate_jet_counts, item.default_gate_pt_cuts,
                )
                for batch_number, batch in enumerate(iter_configuration_metrics(
                    item, payload, true_mass, args.physicality, args.chunk_size
                )):
                    cache_key = (fingerprint, batch_number)
                    indices = index_cache.get(cache_key)
                    if indices is None:
                        indices = np.asarray([key_to_index[key] for key in batch.keys], dtype=np.int32)
                        index_cache[cache_key] = indices
                    column = regime_index[sample]
                    if np.any(~np.isnan(retention[indices, column])):
                        raise ValueError(f"{sample}: duplicate configuration values while processing {item.path}")
                    retention[indices, column] = np.asarray(batch.values["signal_retention"], dtype=np.float32)
                    physicality[indices, column] = np.asarray(batch.values["physicality_score"], dtype=np.float32)
                    for name, values in pareto_components.items():
                        values[indices, column] = np.asarray(batch.values[name], dtype=np.float32)
                    for name in ONLINE_METRIC_FIELDS:
                        values = np.asarray(batch.values[name], dtype=np.float64)
                        finite = np.isfinite(values)
                        metric_sums[name][indices[finite]] += values[finite]
                        metric_counts[name][indices[finite]] += 1
        retention.flush(), physicality.flush()
        for values in pareto_components.values():
            values.flush()

        mean_metrics = {
            name: np.divide(
                metric_sums[name], metric_counts[name],
                out=np.full(len(keys), np.nan), where=metric_counts[name] > 0,
            )
            for name in ONLINE_METRIC_FIELDS
        }

        aggregate = aggregate_scan_matrices(
            retention, physicality, keys, args.mean_weight,
            args.minimum_regime_coverage, args.physicality_statistic,
            mean_metrics, pareto_components,
        )
        aggregate_rows = (aggregate_row(i, key, aggregate) for i, key in enumerate(keys))
        write_csv(output_dir / "aggregate_configurations.csv.gz", aggregate_rows,
                  AGGREGATE_FIELDS, compressed=True)

        pareto_indices = np.flatnonzero(aggregate["pareto"])
        pareto_sort_values = [
            aggregate[f"{args.physicality_statistic}_{name}"][pareto_indices]
            for name in PARETO_COMPONENT_FIELDS
        ] + [
            aggregate["worst_regret"][pareto_indices],
            aggregate["mean_regret"][pareto_indices],
        ]
        pareto_indices = pareto_indices[np.lexsort(tuple(reversed(pareto_sort_values)))]
        write_csv(output_dir / "pareto_front.csv",
                  (aggregate_row(i, keys[i], aggregate) for i in pareto_indices),
                  AGGREGATE_FIELDS)

        qualified = np.flatnonzero(aggregate["physicality_pass_all"])
        qualified = qualified[np.argsort(aggregate["global_objective"][qualified], kind="stable")]
        ranking_fields = ("rank",) + AGGREGATE_FIELDS
        write_csv(
            output_dir / "physicality_qualified_ranking.csv",
            ({"rank": rank, **aggregate_row(i, keys[i], aggregate)}
             for rank, i in enumerate(qualified, 1)), ranking_fields,
        )

        best_rows = _best_by_regime_rows(retention, physicality, keys, expected_samples, definitions)
        best_fields = (
            "regime_index", "sample", "decay", "nominal_suu_mass", "nominal_chi_mass",
            "generated_suu_mass", "generated_chi_mass", "unconstrained_configuration",
            "unconstrained_signal_retention", "unconstrained_physicality_score",
            "physical_configuration", "physical_signal_retention",
            "physical_physicality_score", "retention_gain_if_unconstrained",
        )
        write_csv(output_dir / "per_regime_best.csv", best_rows, best_fields)

        top_indices = list(qualified[: args.top_configurations])
        if not top_indices:
            top_indices = list(pareto_indices[np.argsort(aggregate["global_objective"][pareto_indices])][: args.top_configurations])
        detail_fields = (
            "rank", "configuration", "n_gate_jets", "gate_pt_cut",
            "collection_pt_cut", "ak_radius", "ca_radius", "cos_thrust_cut",
            "regime_index", "sample", "decay", "nominal_suu_mass",
            "nominal_chi_mass", "generated_suu_mass", "generated_chi_mass",
            "available", "signal_retention", "physicality_score",
            "physicality_pass", "retention_regret", "signal_retention_rank",
        )
        detail_rows = []
        best = aggregate["best_retention_by_regime"]
        # Rank each regime once, then look up every displayed configuration.
        # This avoids top-config x regime repeated scans over the full memmap.
        top_rank = np.full((len(top_indices), len(expected_samples)), np.nan)
        for column in range(len(expected_samples)):
            regime_values = np.asarray(retention[:, column], dtype=float)
            available_values = np.sort(regime_values[np.isfinite(regime_values)])
            if not len(available_values):
                continue
            selected_values = np.asarray(retention[top_indices, column], dtype=float)
            selected_available = np.isfinite(selected_values)
            top_rank[selected_available, column] = 1 + len(available_values) - np.searchsorted(
                available_values, selected_values[selected_available], side="right"
            )
        for rank, config_index in enumerate(top_indices, 1):
            key = keys[config_index]
            for column, sample in enumerate(expected_samples):
                value = float(retention[config_index, column])
                score = float(physicality[config_index, column])
                available = not math.isnan(value)
                detail_rows.append({
                    "rank": rank, "configuration": configuration_slug(key), **key.as_dict(),
                    "regime_index": column, **definitions[sample].as_dict(),
                    "available": available, "signal_retention": value,
                    "physicality_score": score,
                    "physicality_pass": available and math.isfinite(score) and score <= 1.0,
                    "retention_regret": best[column] - value if available else np.nan,
                    "signal_retention_rank": top_rank[rank - 1, column],
                })
        write_csv(output_dir / "top_configuration_regime_details.csv.gz",
                  detail_rows, detail_fields, compressed=True)

        _plot_tradeoff(plot_dir / "physicality_vs_retention_regret.png", aggregate,
                       args.physicality_statistic)
        _plot_component_tradeoffs(
            plot_dir / "pareto_component_tradeoffs.png", aggregate,
            args.physicality_statistic,
        )
        _plot_pareto_hyperparameters(
            plot_dir / "pareto_hyperparameters.png", pareto_indices, keys,
            aggregate, args.physicality_statistic,
        )
        _plot_coverage(plot_dir / "coverage_vs_retention_regret.png", aggregate)
        _plot_regime_best(plot_dir / "best_physical_retention_by_regime.png", best_rows)
        _plot_top_regrets(plot_dir / "top_configuration_regret_heatmap.png",
                          top_indices, retention, best, keys, expected_samples)
        print(f"Pareto configurations: {len(pareto_indices)}")
        print(f"Physicality-qualified configurations: {len(qualified)}")
        print(
            "Qualified ranking requires coverage and physicality pass in all "
            f"{len(expected_samples)} expected regimes; the looser requested "
            "coverage threshold applies only to aggregate/Pareto eligibility."
        )
        if qualified.size:
            winner = keys[int(qualified[0])]
            print(f"Best qualified configuration: {configuration_slug(winner)}")
            print(f"  objective={aggregate['global_objective'][qualified[0]]:.6g}")
            print(f"  worst physicality={aggregate['worst_physicality'][qualified[0]]:.6g}")
        retention._mmap.close()
        physicality._mmap.close()
        for values in pareto_components.values():
            values._mmap.close()
    print(f"Outputs written under {output_dir.resolve()}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_evaluation(parse_args(argv))


if __name__ == "__main__":
    main()
