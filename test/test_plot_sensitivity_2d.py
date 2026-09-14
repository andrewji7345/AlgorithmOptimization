import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plot_sensitivity_2d import build_plot_data, plot_2d_outputs


def example_result():
    """Asymmetric axes, signed cancellations, empty bins, and two QCD inputs."""
    def histogram(values, variance, entries):
        return {"sumw": values, "sumw2": variance, "entries": entries,
                "flow_entries": 2, "flow_sumw": -3., "flow_sumw2": 5.}
    first = histogram([[5., 0., -3.], [6., 10., 16.]], [[1., 0., 5.], [30., .4, .3]],
                      [[5, 0, 3], [6, 10, 16]])
    second = histogram([[4., 0., 1.], [10., 15., 20.]], [[0., 0., 3.], [34., .6, .7]],
                       [[4, 0, 1], [10, 15, 20]])
    background = {field: (np.asarray(first[field]) + np.asarray(second[field])).tolist()
                  for field in ("sumw", "sumw2", "entries")}
    background.update(flow_entries=4, flow_sumw=-6., flow_sumw2=10.)
    signal = histogram([[5., 0., 1.], [2., 1., 0.]], [[.25, 0., .1], [.2, .1, 0.]],
                       [[10, 0, 10], [10, 10, 0]])
    return {
        "configuration_slug": "ak4_ng2_tg100_tk120_ak0p4_ca0p6_c0p85", "purpose": "pilot",
        "parameters": {"mass_bin_edges_gev": [2500., 3000., 3500.],
                       "chi_mass_bin_edges_gev": [750., 1000., 1250., 1500.],
                       "luminosity_pb": 41480., "minimum_background_effective_events": 10.,
                       "histogram_flow": "exclude"},
        "background": background, "required_signals": ["WbWb_4000_1000"],
        "samples": {"QCD/a": {"kind": "background", "category": "QCD", "histogram": first},
                    "QCD:a": {"kind": "background", "category": "QCD", "histogram": second},
                    "WbWb_4000_1000": {"kind": "signal", "histogram": signal}},
        "per_signal": {"WbWb_4000_1000": {
            "significance": None, "feasible": False,
            "failure_reasons": ["negative_background_bin", "insufficient_background_effective_events"],
            "per_bin_q0": [[4., 0., None], [1., .25, 0.]],
            "per_bin_valid": [[True, True, False], [True, True, True]]}}}


class PlotSensitivity2DTest(unittest.TestCase):
    def test_categories_are_weighted_sums_and_identifiers_cannot_collide(self):
        data = build_plot_data(example_result())
        category = next(record for record in data["plots"] if record["kind"] == "background_category")
        self.assertEqual(category["members"], ["QCD/a", "QCD:a"])
        self.assertEqual(category["histogram"]["sumw"][0, 2], -2.)
        self.assertEqual(category["histogram"]["sumw2"][0, 2], 8.)
        self.assertEqual(category["histogram"]["entries"][0, 2], 4)
        self.assertEqual(category["histogram"]["flow_entries"], 4)
        self.assertEqual(category["histogram"]["flow_sumw"], -6.)
        ids = [record["id"] for record in data["plots"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(any("/" in value or ":" in value for value in ids))

    def test_partial_failure_keeps_other_scores_and_background_quality_visible(self):
        data = build_plot_data(example_result())
        score = next(record for record in data["plots"] if record["kind"] == "asimov")
        self.assertFalse(score["feasible"])
        self.assertEqual(score["values"].shape, (2, 3))
        self.assertEqual(score["values"][0, 0], 2.)
        self.assertEqual(score["values"][1, 1], .5)
        self.assertEqual(score["values"][0, 1], 0.)
        self.assertTrue(np.isnan(score["values"][0, 2]))
        self.assertTrue(score["low_background_statistics"][1, 0])
        self.assertFalse(score["low_background_statistics"][0, 0])
        self.assertEqual(score["background_effective_events"][1, 0], 4.)

    def test_zero_mc_gate_still_has_diagnostic_hatching_and_actual_mass_metadata(self):
        result = example_result()
        result["parameters"]["minimum_background_effective_events"] = 0
        result["comparison_samples"] = {"WbWb_4000_1000": {
            "generated_suu_mass_gev": 6200., "generated_chi_mass_gev": 1950.,
            "decay_channel": "HtZt", "dataset": "/actual_generated_sample/processing/MINIAODSIM"}}
        data = build_plot_data(result)
        self.assertEqual(data["minimum_background_effective_events"], 0.)
        self.assertEqual(data["plot_minimum_background_effective_events"], 10.)
        score = next(record for record in data["plots"] if record["kind"] == "asimov")
        self.assertTrue(score["low_background_statistics"][1, 0])
        self.assertEqual(score["physics_metadata"]["generated_suu_mass_gev"], 6200.)
        self.assertEqual(score["physics_metadata"]["generated_chi_mass_gev"], 1950.)

    def test_mismatched_total_axis_and_score_contracts_are_rejected(self):
        for mutation, expected in [
            (lambda result: result["background"]["sumw"][0].__setitem__(0, 10.), "total background"),
            (lambda result: result["parameters"].__setitem__("chi_mass_bin_edges_gev", None), "increasing bin edges"),
            (lambda result: result["samples"]["QCD/a"]["histogram"].__setitem__("sumw", [[1, 2], [3, 4], [5, 6]]), "x,y order"),
            (lambda result: result["per_signal"]["WbWb_4000_1000"]["per_bin_valid"][0].__setitem__(2, True), "finite nonnegative q0"),
        ]:
            result = copy.deepcopy(example_result())
            mutation(result)
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                build_plot_data(result)

    def test_real_render_and_root_preserve_orientation_signed_yields_and_errors(self):
        import uproot
        from matplotlib.image import imread
        with tempfile.TemporaryDirectory() as directory:
            manifest = plot_2d_outputs(example_result(), directory)
            output = Path(directory)
            self.assertEqual(len(manifest["plots"]), 6)
            for record in manifest["plots"]:
                png, pdf = [output / name for name in record["files"]]
                self.assertGreater(png.stat().st_size, 10000)
                self.assertGreater(pdf.stat().st_size, 3000)
                self.assertEqual(pdf.read_bytes()[:5], b"%PDF-")
                rendered = imread(png)
                self.assertGreater(rendered.shape[0], 1000)
                self.assertGreater(rendered.shape[1], rendered.shape[0])
            for name, digest in manifest["artifacts_sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            raw = (output / manifest["histogram_json"]).read_text()
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)
            exported = json.loads(raw)
            score_json = next(item for item in exported["plots"] if item["kind"] == "asimov")
            self.assertIsNone(score_json["values"][0][2])
            total_id = next(item["id"] for item in manifest["plots"] if item["kind"] == "background_total")
            score_id = score_json["id"]
            with uproot.open(output / manifest["histogram_root"],
                             handler=uproot.source.file.MemmapSource) as root:
                hist = root[total_id + "/yield"]
                values, x, y = hist.to_numpy()
                np.testing.assert_array_equal(x, [2500., 3000., 3500.])
                np.testing.assert_array_equal(y, [750., 1000., 1250., 1500.])
                self.assertEqual(values.shape, (2, 3))
                self.assertEqual(values[0, 2], -2.)
                self.assertEqual(values[1, 2], 36.)
                self.assertEqual(hist.variances()[0, 2], 8.)
                self.assertEqual(hist.variances()[1, 0], 64.)
                self.assertEqual(root[score_id + "/z"].values()[1, 1], .5)
                self.assertEqual(root[score_id + "/z"].variances()[1, 1], 0.)
                self.assertEqual(root[score_id + "/valid"].values()[0, 2], 0.)


if __name__ == "__main__":
    unittest.main()
