#!/usr/bin/env python3
"""End-to-end tests for the compact global and diagnostic evaluators."""

import contextlib
import csv
import gzip
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import ROOT
except ImportError:  # Tests are intended to run inside CMSSW.
    ROOT = None

TEST_DIR = Path(__file__).resolve().parent
REPOSITORY = Path(os.environ.get(
    "EXISTING_OPTIMIZATION_REPO",
    str(TEST_DIR.parent),
)).resolve()
sys.path.insert(0, str(REPOSITORY))

import compact_scan_metrics as metrics  # noqa: E402
import evaluate_compact_scan as global_evaluator  # noqa: E402
import evaluate_compact_scan_diagnostics as diagnostics  # noqa: E402
from test_compact_scan_metrics import make_compact_fixture  # noqa: E402


def _read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as input_file:
        return list(csv.DictReader(input_file))


class GlobalAggregationTest(unittest.TestCase):
    def test_regret_physicality_coverage_and_pareto(self):
        retention = np.asarray([
            [0.9, 0.7],
            [0.75, 0.8],
            [0.5, np.nan],
        ])
        physicality = np.asarray([
            [0.5, 0.8],
            [0.6, 1.2],
            [0.2, np.nan],
        ])
        aggregate = global_evaluator.aggregate_scan_matrices(
            retention,
            physicality,
            mean_weight=0.25,
            minimum_regime_coverage=1.0,
        )
        self.assertEqual(aggregate["n_regimes"].tolist(), [2, 2, 1])
        np.testing.assert_allclose(
            aggregate["regime_coverage"], [1.0, 1.0, 0.5])
        self.assertAlmostEqual(aggregate["worst_regret"][0], 0.1)
        self.assertAlmostEqual(aggregate["mean_regret"][0], 0.05)
        self.assertAlmostEqual(aggregate["global_objective"][0], 0.1125)
        self.assertEqual(
            aggregate["physicality_failure_count"].tolist(), [0, 1, 0])
        self.assertAlmostEqual(
            aggregate["physicality_failure_fraction"][1], 0.5)
        self.assertEqual(
            aggregate["eligible_coverage"].tolist(), [True, True, False])
        self.assertEqual(aggregate["pareto"].tolist(), [True, False, False])

    def test_infinite_q90_and_equal_objective_pareto_ordering(self):
        aggregate = global_evaluator.aggregate_scan_matrices(
            np.asarray([
                [0.8, 0.8],
                [0.8, 0.8],
            ]),
            np.asarray([
                [0.5, np.inf],
                [0.6, 0.6],
            ]),
            physicality_statistic="q90",
        )
        self.assertTrue(np.isposinf(aggregate["q90_physicality"][0]))
        self.assertEqual(
            global_evaluator.pareto_mask(
                np.asarray([0.5, 0.6]),
                np.asarray([0.1, 0.1]),
            ).tolist(),
            [True, False],
        )

    def test_seven_objective_pareto_keeps_independent_tradeoffs(self):
        objectives = np.asarray([
            [0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1],
            [0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.2],
            [0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1],
        ])
        self.assertEqual(
            global_evaluator.pareto_mask(objectives).tolist(),
            [True, True, False, True],
        )


@unittest.skipIf(ROOT is None, "PyROOT is not available")
class CompactEvaluatorCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ROOT.gROOT.SetBatch(True)
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.root_file = Path(cls.tempdir.name) / "compact.root"
        make_compact_fixture(cls.root_file)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_global_evaluator_writes_ranked_and_pareto_outputs(self):
        output_dir = Path(self.tempdir.name) / "global"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = global_evaluator.main([
                "--input-file", str(self.root_file),
                "--output-dir", str(output_dir),
                "--sample-lists-dir",
                str(REPOSITORY / "test" / "signalMCFiles"),
                "--discovered-only",
                "--bias-limit", "0.5",
                "--resolution-limit", "1.0",
                "--invalid-limit", "0.6",
                "--tail-limit", "1.0",
                "--min-valid-events", "1",
                "--response-hist-bins", "20",
                "--top-configurations", "2",
                "--chunk-size", "2",
            ])
        self.assertIsNone(result)
        self.assertIn("Coverage denominator: 1 regimes", stdout.getvalue())

        required = (
            "input_manifest.csv",
            "regime_manifest.csv",
            "evaluation_parameters.csv",
            "aggregate_configurations.csv.gz",
            "pareto_front.csv",
            "physicality_qualified_ranking.csv",
            "per_regime_best.csv",
            "top_configuration_regime_details.csv.gz",
            "plots/physicality_vs_retention_regret.png",
            "plots/pareto_component_tradeoffs.png",
            "plots/best_physical_retention_by_regime.png",
            "plots/top_configuration_regret_heatmap.png",
        )
        for relative in required:
            path = output_dir / relative
            self.assertTrue(path.is_file(), msg=f"missing {path}")
            self.assertGreater(path.stat().st_size, 0, msg=f"empty {path}")

        aggregate_rows = _read_csv(
            output_dir / "aggregate_configurations.csv.gz")
        self.assertEqual(len(aggregate_rows), 20)
        slugs = [row["configuration"] for row in aggregate_rows]
        self.assertEqual(len(set(slugs)), len(slugs))
        ungated = [
            row for row in aggregate_rows if row["n_gate_jets"] == "0"]
        self.assertEqual(len(ungated), 4)
        self.assertTrue(all(row["gate_pt_cut"] == "" for row in ungated))

        ranking = _read_csv(
            output_dir / "physicality_qualified_ranking.csv")
        self.assertGreater(len(ranking), 0)
        self.assertEqual(ranking[0]["rank"], "1")
        regime_best = _read_csv(output_dir / "per_regime_best.csv")
        self.assertEqual(len(regime_best), 1)
        self.assertEqual(regime_best[0]["sample"], "WbWb_4000_1000")

    def test_diagnostic_evaluator_writes_selected_configuration_outputs(self):
        output_dir = Path(self.tempdir.name) / "diagnostics"
        key = metrics.configuration_key(
            1, 200.0, 100.0, 0.8, 0.8, 0.0)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = diagnostics.main([
                "--input", str(self.root_file),
                "--sample", "WbWb_4000_1000",
                "--configuration", "1:200:100:0.8:0.8:0",
                "--output-dir", str(output_dir),
                "--true-chi-mass", "1000",
                "--bias-limit", "0.5",
                "--resolution-limit", "1.0",
                "--invalid-fraction-limit", "0.25",
                "--tail-fraction-limit", "1.0",
                "--minimum-valid-events", "1",
                "--response-hist-bins", "20",
                "--max-problem-events", "10",
            ])
        self.assertEqual(result, 0)
        self.assertIn("evaluating 1 unique configuration", stdout.getvalue())

        selected_dir = (
            output_dir
            / "WbWb_4000_1000"
            / metrics.configuration_slug(key)
        )
        required = (
            "configuration_metrics.csv",
            "status_breakdown.csv",
            "problem_events.csv",
            "summary.txt",
            "pooled_mass_response.png",
            "paired_mass_response.png",
            "mass_asymmetry.png",
            "reco_status.png",
            "n_ca_jets.png",
            "n_ambiguous_ca_jets.png",
            "gate_ak_multiplicity.png",
        )
        for filename in required:
            path = selected_dir / filename
            self.assertTrue(path.is_file(), msg=f"missing {path}")
            self.assertGreater(path.stat().st_size, 0, msg=f"empty {path}")
        self.assertFalse((selected_dir / "event_table.csv.gz").exists())

        metric_rows = _read_csv(
            selected_dir / "configuration_metrics.csv")
        self.assertEqual(len(metric_rows), 1)
        row = metric_rows[0]
        self.assertEqual(row["n_events"], "4")
        self.assertEqual(row["n_gate_events"], "2")
        self.assertEqual(row["n_valid_events"], "1")
        self.assertAlmostEqual(float(row["signal_retention"]), 0.25)
        self.assertAlmostEqual(
            float(row["gate_efficiency"])
            * float(row["reco_given_gate_efficiency"]),
            float(row["signal_retention"]),
        )
        self.assertEqual(row["physicality_pass"], "False")

        status_rows = {
            row["status_name"]: row
            for row in _read_csv(selected_dir / "status_breakdown.csv")
        }
        self.assertEqual(
            status_rows["complexity_guard"]["count_gate"], "1")
        problem_rows = _read_csv(selected_dir / "problem_events.csv")
        self.assertGreaterEqual(len(problem_rows), 1)
        self.assertEqual(problem_rows[0]["event"], "2")
        self.assertEqual(problem_rows[0]["status_name"], "complexity_guard")
        summary = (selected_dir / "summary.txt").read_text()
        self.assertIn("End-to-end signal retention: 0.25", summary)
        self.assertIn("Complexity-guard fraction given gate: 0.5", summary)


if __name__ == "__main__":
    unittest.main()
