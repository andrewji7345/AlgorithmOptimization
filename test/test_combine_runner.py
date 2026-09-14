#!/usr/bin/env python3
"""Reject corrupt fits and keep expected inference blind and reproducible."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from combine_runner import (CombineRunError, _run_command, diagnostic_range_policy,
                            parse_expected_limits, parse_expected_significance,
                            parse_fit_diagnostics, run_expected, significance_fit_policy,
                            validate_fit_diagnostics_interval)


class CombineRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_tree(self, name, branches):
        path = self.directory / "result.root"
        with uproot.recreate(path) as output:
            output[name] = {key: np.asarray(values) for key, values in branches.items()}
        return path

    def test_expected_quantiles_accept_float32_and_arbitrary_row_order(self):
        path = self.write_tree("limit", {"limit": [3., 1., 5., 2., 4.],
                                         "quantileExpected": np.asarray([.5, .025, .975, .16, .84], dtype="f4")})
        result = parse_expected_limits(path)
        self.assertEqual(result["median"], 3.)
        self.assertFalse(result["observed_computed"])

    def test_reject_observed_duplicates_nonfinite_negative_and_unordered(self):
        for values, quantiles in (
            ([1, 2, 3, 4, 5, 6], [.025, .16, .5, .84, .975, -1]),
            ([1, 2, 3, 4, 5], [.025, .16, .5, .5, .975]),
            ([1, 2, math.inf, 4, 5], [.025, .16, .5, .84, .975]),
            ([1, 2, 3, 4, -5], [.025, .16, .5, .84, .975]),
            ([1, 2, 4, 3, 5], [.025, .16, .5, .84, .975]),
            ([1, 2, 3, 4, 5], [.025, .16, .5, .84, -1]),
        ):
            with self.subTest(values=values, quantiles=quantiles):
                path = self.write_tree("limit", {"limit": values, "quantileExpected": quantiles})
                with self.assertRaises(CombineRunError):
                    parse_expected_limits(path)

    def test_reject_missing_or_corrupt_root(self):
        path = self.write_tree("wrong_tree", {"limit": [1.]})
        with self.assertRaises(CombineRunError):
            parse_expected_significance(path)
        path.write_text("not root")
        with self.assertRaises(CombineRunError):
            parse_expected_limits(path)

    def test_significance_and_asimov_fit_validation(self):
        path = self.write_tree("limit", {"limit": [2.3]})
        self.assertEqual(parse_expected_significance(path), 2.3)
        for val in ([math.nan], [-1.], [1., 2.]):
            with self.assertRaises(CombineRunError):
                parse_expected_significance(self.write_tree("limit", {"limit": val}))
        good = {"fit_status": [0], "r": [1.00001], "rErr": [.2], "numbadnll": [0]}
        self.assertEqual(parse_fit_diagnostics(self.write_tree("tree_fit_sb", good))["fit_status"], 0)
        for change in ({"fit_status": [1]}, {"r": [2.]}, {"rErr": [0.]}, {"r": [math.nan]}):
            with self.assertRaises(CombineRunError):
                parse_fit_diagnostics(self.write_tree("tree_fit_sb", dict(good, **change)))

    def test_diagnostic_range_uses_limit_band_and_preserves_discovery_floor(self):
        weak = diagnostic_range_policy(400.)
        self.assertEqual(weak["r_max"], 800.)
        self.assertEqual(weak["discovery_r_max"], 20.)
        self.assertEqual(weak["expected_limit_quantile"], "0.975")
        self.assertEqual(weak["expected_limit_quantile_value"], 400.)
        self.assertEqual(weak["upper_interval_fraction_max"], .99)
        self.assertEqual(diagnostic_range_policy(4.)["r_max"], 20.)
        self.assertEqual(diagnostic_range_policy(4., signal_strength=10.)["r_max"], 50.)
        for invalid in (0., -1., math.nan, math.inf):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                diagnostic_range_policy(invalid)
            with self.subTest(injection=invalid), self.assertRaises(ValueError):
                diagnostic_range_policy(4., signal_strength=invalid)
        with self.assertRaises(ValueError):
            diagnostic_range_policy(1e308)

    def test_diagnostic_upper_minos_validation_and_legacy_parsing(self):
        good = {"fit_status": [0], "r": [1.], "rErr": [100.], "numbadnll": [0],
                "rHiErr": [12.]}
        fit = parse_fit_diagnostics(self.write_tree("tree_fit_sb", good), r_max=20.)
        self.assertEqual(fit["interval_validation"]["upper_endpoint"], 13.)
        # A range-dependent symmetric HESSE error must not substitute for MINOS.
        self.assertEqual(fit["rErr"], 100.)
        for upper_error in (0., -1., math.nan, math.inf, 18.8, 19., 20.):
            with self.subTest(upper_error=upper_error), self.assertRaises(CombineRunError):
                parse_fit_diagnostics(self.write_tree("tree_fit_sb", dict(good, rHiErr=[upper_error])),
                                      r_max=20.)
        legacy = {key: values for key, values in good.items() if key != "rHiErr"}
        path = self.write_tree("tree_fit_sb", legacy)
        self.assertEqual(parse_fit_diagnostics(path)["fit_status"], 0)
        with self.assertRaises(CombineRunError):
            parse_fit_diagnostics(path, r_max=20.)
        for invalid in (0., -1., math.nan, math.inf):
            with self.subTest(r_max=invalid), self.assertRaises(ValueError):
                validate_fit_diagnostics_interval({"r": 1., "rHiErr": 1.}, invalid)

    def test_subprocess_timeout_and_nonzero_preserve_logs(self):
        for code, timeout in (("import time; print('started',flush=True); time.sleep(3)", .1),
                              ("print('failure'); raise SystemExit(7)", 5)):
            records = []
            with self.assertRaises(CombineRunError):
                _run_command([sys.executable, "-c", code], self.directory,
                             self.directory / "command.log", timeout, records)
            self.assertEqual(len(records), 1)
            self.assertGreater(records[0]["wall_seconds"], 0)
            self.assertNotEqual(records[0]["returncode"], 0)
            self.assertTrue((self.directory / "command.log").exists())

    def test_no_shell_interpretation(self):
        text = "$(touch unwanted); `touch unwanted2`"
        output = _run_command([sys.executable, "-c", "import sys; print(sys.argv[1])", text],
                              self.directory, self.directory / "literal.log", 5, [])
        self.assertEqual(output.strip(), text)
        self.assertFalse((self.directory / "unwanted").exists())

    def test_failed_runtime_creates_failure_receipt(self):
        card = self.directory / "card.txt"
        card.write_text("invalid card")
        with self.assertRaises(CombineRunError) as caught:
            run_expected(card, self.directory / "out", command_prefix=["/does/not/exist"])
        self.assertEqual(caught.exception.result["status"], "failed")
        result = json.loads((self.directory / "out" / "combine_result.json").read_text())
        self.assertTrue(result["expected_only"])
        self.assertEqual(result["status"], "failed")

    def test_stale_outputs_and_invalid_configuration_are_rejected(self):
        card = self.directory / "card.txt"
        card.write_text("card")
        with self.assertRaises(ValueError):
            run_expected(card, self.directory)
        with self.assertRaises(ValueError):
            run_expected(card, self.directory / "unused", signal_strength=1000)
        self.assertFalse((self.directory / "unused").exists())

    def test_command_policy_is_blind_and_expected(self):
        card = self.directory / "card.txt"
        card.write_text("card")
        out = self.directory / "out"

        def fake_run(argv, cwd, log_path, timeout_seconds, commands):
            commands.append({"argv": argv})
            if "text2workspace.py" in argv:
                (out / "workspace.root").write_bytes(b"workspace")
            if "Significance" in argv:
                (out / "higgsCombine.expected.Significance.mH125.123456.root").write_bytes(b"root")
            return " <<< Combine >>>\n <<< v11.0.0 >>>\nFit S+B, status = 0, numBadNLL = 0, covariance quality = 3\n"

        limits = {"median": 200., "quantiles": {"0.975": 400.}}
        with mock.patch("combine_runner._run_command", side_effect=fake_run), \
             mock.patch("combine_runner.parse_expected_limits", return_value=limits), \
             mock.patch("combine_runner.parse_expected_significance", return_value=1.), \
             mock.patch("combine_runner.parse_fit_diagnostics", return_value={"fit_status": 0}) as parser:
            result = run_expected(card, out)
        commands = [record["argv"] for record in result["commands"]]
        limit = next(cmd for cmd in commands if "AsymptoticLimits" in cmd)
        self.assertEqual(limit[limit.index("--run") + 1], "blind")
        for method in ("Significance", "FitDiagnostics"):
            cmd = next(cmd for cmd in commands if method in cmd)
            self.assertEqual(cmd[cmd.index("-t") + 1], "-1")
            self.assertEqual(cmd[cmd.index("--expectSignal") + 1], "1")
            self.assertNotIn("--toysFrequentist", cmd)
        ranges = {method: float(next(cmd for cmd in commands if method in cmd)[
                  next(cmd for cmd in commands if method in cmd).index("--rMax") + 1])
                  for method in ("AsymptoticLimits", "Significance", "FitDiagnostics")}
        self.assertEqual(ranges, {"AsymptoticLimits": 1000., "Significance": 20.,
                                  "FitDiagnostics": 800.})
        self.assertEqual(result["fit_diagnostics_range_policy"], diagnostic_range_policy(400.))
        self.assertEqual(result["significance_fit_policy"], significance_fit_policy())
        significance = next(cmd for cmd in commands if "Significance" in cmd)
        self.assertEqual(significance[significance.index("--cminDefaultMinimizerStrategy") + 1], "2")
        self.assertEqual(float(significance[significance.index("--cminDefaultMinimizerTolerance") + 1]), 1e-5)
        diagnostics = next(cmd for cmd in commands if "FitDiagnostics" in cmd)
        self.assertNotIn("--cminDefaultMinimizerStrategy", diagnostics)
        self.assertNotIn("--cminDefaultMinimizerTolerance", diagnostics)
        parser.assert_called_once_with(out / "fitDiagnostics.expected.root", 1., r_max=800.)
        self.assertEqual(result["status"], "complete")

    def test_clipped_minos_or_bad_covariance_keeps_result_failed(self):
        card = self.directory / "card.txt"
        card.write_text("card")
        for upper_error, covariance, error in ((799., 3, "MINOS upper interval"),
                                               (10., 2, "covariance quality is 2")):
            out = self.directory / f"out_{covariance}"

            def fake_run(argv, cwd, log_path, timeout_seconds, commands):
                commands.append({"argv": argv})
                if "text2workspace.py" in argv:
                    (out / "workspace.root").write_bytes(b"workspace")
                elif "Significance" in argv:
                    (out / "higgsCombine.expected.Significance.mH125.123456.root").write_bytes(b"root")
                elif "FitDiagnostics" in argv:
                    with uproot.recreate(out / "fitDiagnostics.expected.root") as output:
                        output["tree_fit_sb"] = {key: np.asarray([value]) for key, value in {
                            "fit_status": 0, "r": 1., "rErr": 10., "numbadnll": 0,
                            "rHiErr": upper_error}.items()}
                return (" <<< v11.0.0 >>>\nFit S+B, status = 0, numBadNLL = 0, "
                        f"covariance quality = {covariance}\n")

            with self.subTest(covariance=covariance), \
                 mock.patch("combine_runner._run_command", side_effect=fake_run), \
                 mock.patch("combine_runner.parse_expected_limits",
                            return_value={"median": 200., "quantiles": {"0.975": 400.}}), \
                 mock.patch("combine_runner.parse_expected_significance", return_value=.1), \
                 self.assertRaisesRegex(CombineRunError, error):
                run_expected(card, out)
            receipt = json.loads((out / "combine_result.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["fit_diagnostics_r_max"], 800.)
            self.assertEqual(receipt["significance_r_max"], 20.)


if __name__ == "__main__":
    unittest.main()
