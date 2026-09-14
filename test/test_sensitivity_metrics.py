#!/usr/bin/env python3
"""Analytic tests for the sensitivity objective and feasibility constraints."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact_scan_metrics import PhysicalityDefinition
from sensitivity_metrics import (
    PhysicalityAccumulator, asimov_significance, normalization_factor,
    sensitivity_rankings, weighted_histogram,
)


class SensitivityMetricsTest(unittest.TestCase):
    def test_known_background_analytic_limit_and_bin_additivity(self):
        s, b = np.array([5., 10.]), np.array([10., 100.])
        expected_q = 2 * ((s + b) * np.log1p(s / b) - s).sum()
        result = asimov_significance(s, b)
        self.assertTrue(result["feasible"])
        self.assertAlmostEqual(result["q0"], expected_q, places=12)
        self.assertAlmostEqual(result["significance"], math.sqrt(expected_q), places=12)
        small = asimov_significance(np.array([0.01]), np.array([1000.]))
        self.assertAlmostEqual(small["significance"], 0.01 / math.sqrt(1000), places=8)

    def test_mc_uncertainty_matches_independent_poisson_control_likelihood(self):
        s, b, variance = 5., 10., 4.
        tau = b / variance
        on, off = s + b, tau * b
        null_b = (on + off) / (1 + tau)
        q = 2 * (on * math.log(on / null_b) + off * math.log(off / (tau * null_b)))
        result = asimov_significance(np.array([s]), np.array([b]), np.array([variance]))
        self.assertAlmostEqual(result["q0"], q, places=12)
        self.assertLess(result["significance"], asimov_significance(np.array([s]), np.array([b]))["significance"])
        self.assertAlmostEqual(result["background_effective_events"][0], 25)

    def test_missing_negative_and_low_statistics_background_fail_closed(self):
        for signal, background, variance, reason in (
            ([1.], [0.], [0.], "signal_bin_without_positive_background"),
            ([1.], [-1.], [1.], "nonpositive_background_after_signed_cancellation"),
            ([0.], [0.], [2.], "nonpositive_background_after_signed_cancellation"),
            ([-1.], [10.], [1.], "negative_signal_bin"),
            ([1.], [10.], [20.], "insufficient_background_effective_events"),
        ):
            with self.subTest(reason=reason):
                result = asimov_significance(np.array(signal), np.array(background), np.array(variance))
                self.assertFalse(result["feasible"])
                self.assertIsNone(result["significance"])
                self.assertIn(reason, result["failure_reasons"])
        empty = asimov_significance(np.zeros(2), np.zeros(2), np.zeros(2))
        self.assertTrue(empty["feasible"])
        self.assertEqual(empty["significance"], 0)

    def test_signed_weights_sumw2_normalization_and_fixed_flow_folding(self):
        scale = normalization_factor(2, 1000, 100, 0.5, 1.2)
        self.assertEqual(scale, 12)
        values = np.array([-1., 0.5, 1.5, 2., 9.])
        weights = scale * np.array([1., -0.5, 2., 1., -1.])
        result = weighted_histogram(values, weights, [np.array([0., 1., 2.])])
        np.testing.assert_allclose(result.sumw, [6., 24.])
        np.testing.assert_allclose(result.sumw2, [180., 864.])
        self.assertEqual(result.flow_entries, 2)
        np.testing.assert_array_equal(result.entries, [2, 3])
        with self.assertRaises(ValueError):
            normalization_factor(1, 1000, -10)
        with self.assertRaises(ValueError):
            weighted_histogram(np.array([float("nan")]), np.array([1.]), [np.array([0., 1.])])

    def test_fixed_two_dimensional_template(self):
        coords = np.array([[1000., 500.], [3000., 500.], [3000., 1500.], [9000., 4000.]])
        hist = weighted_histogram(coords, np.ones(4),
                                  [np.array([0., 2000., 4000.]), np.array([0., 1000., 2000.])])
        np.testing.assert_array_equal(hist.sumw, [[1., 0.], [1., 2.]])
        self.assertEqual(hist.flow_entries, 1)

    def test_sparse_two_dimensional_bins_cannot_create_infinite_sensitivity(self):
        background = np.array([[100., 0.], [20., 0.]])
        variance = np.array([[100., 0.], [20., 0.]])
        signal = np.array([[2., 0.], [1., 0.]])
        result = asimov_significance(signal, background, variance)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["per_bin_q0"][0][1], 0)
        signal[1, 1] = 0.001
        unsupported = asimov_significance(signal, background, variance)
        self.assertFalse(unsupported["feasible"])
        self.assertIn("signal_bin_without_positive_background", unsupported["failure_reasons"])
        self.assertIsNone(unsupported["significance"])

    def test_physicality_uses_pre_sr_population_and_widened_mass_window(self):
        definition = PhysicalityDefinition(min_valid_events=1, tail_response_min=0.5, tail_response_max=1.5)
        gate = PhysicalityAccumulator.empty(definition)
        gate.fill(np.array([500., 1500., 1000., 1000.]), np.array([1000., 1000., 1000., 1000.]),
                  np.array([True, True, False, True]), np.array([True, True, True, False]), 1000.)
        result = gate.result()
        self.assertEqual(result["n_gate_events"], 3)
        self.assertEqual(result["n_valid_events"], 2)
        self.assertEqual(result["n_tail_events"], 0)
        self.assertAlmostEqual(result["invalid_fraction"], 1 / 3)
        self.assertFalse(result["physicality_pass"])

    def test_rank_and_regret_use_sensitivity_and_globally_feasible_trials(self):
        def trial(name, zs, feasible=True):
            return {"configuration": name, "feasible": feasible,
                    "objective": -999, "per_signal": {
                        signal: {"significance": z, "feasible": True}
                        for signal, z in zip(("a", "b"), zs)}}
        rows = sensitivity_rankings([trial("one", [2, 4]), trial("two", [4, 1]),
                                     trial("unphysical", [100, 100], False),
                                     trial("missing", [100])], ["a", "b"])
        self.assertEqual([r["configuration"] for r in rows], ["one", "two"])
        self.assertEqual(rows[0]["objective"], -999)
        self.assertEqual(rows[0]["regret_objective"], .5625)
        self.assertEqual(rows[0]["per_signal_best_significance"], {"a": 4, "b": 4})
        self.assertEqual(rows[0]["mean_regret"], 0.25)
        self.assertEqual(rows[1]["worst_regret"], 0.75)


if __name__ == "__main__":
    unittest.main()
