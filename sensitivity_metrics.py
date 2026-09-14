#!/usr/bin/env python3
"""Yield, physicality, significance, and ranking primitives for sensitivity scans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from compact_scan_metrics import (
    PhysicalityDefinition, fixed_response_histogram, fixed_histogram_estimates,
    physicality_from_arrays,
)


INFEASIBLE_OBJECTIVE = -1.0


def validated_edges(values: Sequence[float], name: str) -> np.ndarray:
    edges = np.asarray(values, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} must contain at least two finite, increasing edges")
    return edges


def normalization_factor(cross_section_pb: float, luminosity_pb: float,
                         sum_gen_weights: float, filter_efficiency: float = 1.0,
                         k_factor: float = 1.0) -> float:
    """Scale signed generator weights using an independently audited denominator."""
    values = (cross_section_pb, luminosity_pb, sum_gen_weights, filter_efficiency, k_factor)
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in values):
        raise ValueError("cross section, luminosity, generator denominator, efficiency, and k factor must be finite and positive")
    if filter_efficiency > 1:
        raise ValueError("filter_efficiency must not exceed one")
    return cross_section_pb * luminosity_pb * filter_efficiency * k_factor / sum_gen_weights


@dataclass
class WeightedHistogram:
    sumw: np.ndarray
    sumw2: np.ndarray
    entries: np.ndarray
    flow_entries: int = 0
    flow_sumw: float = 0.0
    flow_sumw2: float = 0.0

    @classmethod
    def empty(cls, edges: Sequence[np.ndarray]) -> "WeightedHistogram":
        shape = tuple(len(axis) - 1 for axis in edges)
        return cls(np.zeros(shape), np.zeros(shape), np.zeros(shape, dtype=np.int64))

    def add(self, other: "WeightedHistogram") -> None:
        if self.sumw.shape != other.sumw.shape:
            raise ValueError("histogram shapes differ")
        self.sumw += other.sumw
        self.sumw2 += other.sumw2
        self.entries += other.entries
        self.flow_entries += other.flow_entries
        self.flow_sumw += other.flow_sumw
        self.flow_sumw2 += other.flow_sumw2

    def as_dict(self) -> dict[str, Any]:
        return {"sumw": self.sumw.tolist(), "sumw2": self.sumw2.tolist(),
                "entries": self.entries.tolist(), "flow_entries": self.flow_entries,
                "flow_sumw": self.flow_sumw, "flow_sumw2": self.flow_sumw2}


def weighted_histogram(coordinates: np.ndarray, weights: np.ndarray,
                       edges: Sequence[np.ndarray], flow_policy: str = "fold") -> WeightedHistogram:
    """Histogram signed weights and squares using the campaign's flow policy.

    ``exclude`` leaves masses outside the fixed analysis rectangle out of its
    bins; ``fold`` preserves the legacy edge-folding convention. Both record
    signed flow yields and variances separately. Exact upper edges are included.
    """
    if flow_policy not in ("fold", "exclude"):
        raise ValueError("histogram flow must be fold or exclude")
    weights = np.asarray(weights, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim == 1:
        coordinates = coordinates[:, None]
    if coordinates.shape != (len(weights), len(edges)) or weights.ndim != 1:
        raise ValueError("histogram coordinates/weights have inconsistent dimensions")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(coordinates)):
        raise ValueError("selected histogram coordinates and weights must be finite")
    clipped = coordinates.copy()
    flow = np.zeros(len(weights), dtype=bool)
    for axis, bins in enumerate(edges):
        bins = validated_edges(bins, "histogram edges")
        flow |= (coordinates[:, axis] < bins[0]) | (coordinates[:, axis] > bins[-1])
        clipped[:, axis] = np.clip(clipped[:, axis], bins[0], bins[-1])
    keep = ~flow if flow_policy == "exclude" else np.ones(len(weights), dtype=bool)
    return WeightedHistogram(
        np.histogramdd(clipped[keep], bins=edges, weights=weights[keep])[0],
        np.histogramdd(clipped[keep], bins=edges, weights=weights[keep] * weights[keep])[0],
        np.histogramdd(clipped[keep], bins=edges)[0].astype(np.int64), int(np.count_nonzero(flow)),
        float(np.sum(weights[flow])), float(np.sum(np.square(weights[flow]))),
    )


def asimov_significance(signal: np.ndarray, background: np.ndarray,
                        background_sumw2: np.ndarray | None = None,
                        minimum_background_effective_events: float = 10.0) -> dict[str, Any]:
    """Independent-bin Asimov discovery Z with auxiliary MC constraints.

    The auxiliary measurement has variance ``sumw2`` and effective count
    b**2/sumw2. This is a planning approximation to finite background MC, not
    the analysis's complete correlated nuisance-parameter likelihood.
    Empty signal/background bins contribute zero. A signal in an unsupported
    background bin or negative expected yield makes the objective infeasible.
    A positive minimum effective count adds an explicit statistics gate; zero
    disables that gate while retaining the auxiliary MC uncertainty. There are
    no epsilon backgrounds or adaptive bin merging.
    """
    s, b = np.asarray(signal, dtype=float), np.asarray(background, dtype=float)
    v = np.zeros_like(b) if background_sumw2 is None else np.asarray(background_sumw2, dtype=float)
    if s.shape != b.shape or s.shape != v.shape:
        raise ValueError("signal, background, and background variance shapes differ")
    if not math.isfinite(minimum_background_effective_events) or minimum_background_effective_events < 0:
        raise ValueError("minimum_background_effective_events must be finite and nonnegative")
    reasons = []
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(v)):
        reasons.append("nonfinite_yield_or_variance")
    if np.any(s < 0):
        reasons.append("negative_signal_bin")
    if np.any(b < 0) or np.any((b == 0) & (v > 0)):
        reasons.append("nonpositive_background_after_signed_cancellation")
    if np.any(v < 0):
        reasons.append("negative_background_variance")
    active = s > 0
    if np.any(active & (b <= 0)):
        reasons.append("signal_bin_without_positive_background")
    neff = np.divide(b * b, v, out=np.full_like(b, np.inf), where=v > 0)
    neff[(b == 0) & (v == 0)] = 0
    if np.any(active & (neff < minimum_background_effective_events)):
        reasons.append("insufficient_background_effective_events")
    # Retain valid cell values for plots even when another cell invalidates the
    # full benchmark. No per-trial masking is used to create a scalar objective.
    valid = (np.isfinite(s) & np.isfinite(b) & np.isfinite(v) & (s >= 0) & (b >= 0)
             & (v >= 0) & ~((b == 0) & (v > 0)) & ~(active & (b <= 0)))
    supported = valid & (~active | (neff >= minimum_background_effective_events))
    q0 = np.where(valid, 0., np.nan)
    calculate = active & valid
    # Long-double arithmetic reduces cancellation in small-signal bins.
    ss, bb, vv = (np.asarray(x[calculate], dtype=np.longdouble) for x in (s, b, v))
    exact = vv == 0
    terms = np.zeros_like(ss)
    terms[exact] = (ss[exact] + bb[exact]) * np.log1p(ss[exact] / bb[exact]) - ss[exact]
    uncertain = ~exact
    n, bs, vs = ss[uncertain] + bb[uncertain], bb[uncertain], vv[uncertain]
    terms[uncertain] = (n * np.log1p(ss[uncertain] * bs / (bs * bs + n * vs))
                        - bs * bs / vs * np.log1p(vs * ss[uncertain] / (bs * (bs + vs))))
    numerical_bad = ~np.isfinite(terms) | (terms < -1e-9)
    if np.any(numerical_bad):
        reasons.append("invalid_likelihood_value")
        indices = np.flatnonzero(calculate)
        valid.flat[indices[numerical_bad]] = False
        supported.flat[indices[numerical_bad]] = False
    q0[calculate] = np.where(numerical_bad, np.nan, np.maximum(2.0 * terms, 0))
    total = float(np.sum(q0))
    if np.all(valid) and not math.isfinite(total):
        reasons.append("nonfinite_total_likelihood")
    feasible = not reasons
    return {"feasible": feasible, "significance": float(np.sqrt(total)) if feasible else None,
            "q0": total if feasible else None, "per_bin_q0": q0.tolist(),
            "per_bin_valid": valid.tolist(), "per_bin_supported": supported.tolist(),
            "diagnostic_significance": float(np.sqrt(total)) if math.isfinite(total) else None,
            "background_effective_events": neff.tolist(), "failure_reasons": reasons}


@dataclass
class PhysicalityAccumulator:
    """Existing unweighted physicality score, evaluated before SR mass cuts."""
    definition: PhysicalityDefinition
    histogram: np.ndarray
    n_gate: int = 0
    n_valid: int = 0
    n_tail: int = 0

    @classmethod
    def empty(cls, definition: PhysicalityDefinition) -> "PhysicalityAccumulator":
        return cls(definition, np.zeros(definition.response_hist_bins + 2, dtype=np.int64))

    def fill(self, mass1: np.ndarray, mass2: np.ndarray, valid: np.ndarray,
             baseline_and_gate: np.ndarray, chi_mass: float) -> None:
        if not math.isfinite(chi_mass) or chi_mass <= 0:
            raise ValueError("generated_chi_mass_gev must be finite and positive")
        selected = np.asarray(baseline_and_gate, bool)
        valid = (np.asarray(valid, bool) & np.isfinite(mass1) & np.isfinite(mass2)
                 & (mass1 >= 0) & (mass2 >= 0) & selected)
        r1, r2 = mass1[valid] / chi_mass, mass2[valid] / chi_mass
        self.n_gate += int(np.count_nonzero(selected))
        self.n_valid += int(np.count_nonzero(valid))
        self.histogram += fixed_response_histogram(np.concatenate((r1, r2)), self.definition)
        self.n_tail += int(np.count_nonzero(
            (r1 < self.definition.tail_response_min) | (r1 > self.definition.tail_response_max)
            | (r2 < self.definition.tail_response_min) | (r2 > self.definition.tail_response_max)))

    def result(self) -> dict[str, Any]:
        median, fwhm, peak, resolution = fixed_histogram_estimates(self.histogram, self.definition)
        invalid = 1 - self.n_valid / self.n_gate if self.n_gate else math.inf
        tails = self.n_tail / self.n_valid if self.n_valid else math.inf
        score = physicality_from_arrays(abs(median - 1), resolution, invalid, tails,
                                        self.n_valid, self.definition)
        result = {name: bool(value) if name == "physicality_pass" else float(value)
                  for name, value in score.items()}
        result.update(n_gate_events=self.n_gate, n_valid_events=self.n_valid,
                      n_tail_events=self.n_tail, mass_bias=float(abs(median - 1)),
                      fwhm_resolution=float(resolution), invalid_fraction=invalid,
                      tail_fraction=tails, median_mass_response=float(median),
                      fwhm_mass_response=float(fwhm), fwhm_peak_response=float(peak))
        return result


def sensitivity_rankings(results: Sequence[Mapping[str, Any]],
                         required_signals: Sequence[str]) -> list[dict[str, Any]]:
    """Compatibility entry point for worst-plus-mean relative Asimov regret.

    The recorded optimizer objective is retained; the retrospective rank uses
    each benchmark's best feasible score in the supplied comparison cohort.
    """
    from sensitivity_objective import relative_regret_rankings
    return relative_regret_rankings(results, required_signals, mean_weight=0.25)
