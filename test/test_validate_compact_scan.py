#!/usr/bin/env python3
"""Integration tests for validate_compact_scan.py using tiny ROOT fixtures."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from array import array

try:
    import ROOT
except ImportError:  # Tests are intended to run inside CMSSW.
    ROOT = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_compact_scan  # noqa: E402


def _fill_vector(vector, values):
    vector.clear()
    for value in values:
        vector.push_back(value)


def _make_fixture(path, defect=None):
    output = ROOT.TFile(path, "RECREATE")
    directory = output.mkdir("compactScan")
    directory.cd()

    metadata = ROOT.TTree("Metadata", "compact scan metadata")
    schema_version = array("I", [1 if defect == "schema1" else 2])
    sample_name = ROOT.std.string("unit_test_sample")
    ak_radius = array("f", [0.8])
    max_ambiguous = array("I", [12])
    puppi_weighted = array("b", [1])
    use_jec = array("b", [0])
    ca_algorithm = ROOT.std.string("cambridge_y_phi")
    mass_objective = ROOT.std.string("abs(m1-m2)/(m1+m2)")
    gate_constraint = ROOT.std.string(
        "collectionPtCut<=gatePtCut; nGateJets=0 is ungated")
    legacy_radius_constraint = array("b", [0])
    processed_events = array("Q", [2])
    sum_weights = array("d", [0.5])
    sum_weights2 = array("d", [1.25])

    metadata.Branch("schemaVersion", schema_version, "schemaVersion/i")
    metadata.Branch("sampleName", sample_name)
    metadata.Branch("akRadius", ak_radius, "akRadius/F")
    metadata.Branch("maxAmbiguousCAJets", max_ambiguous, "maxAmbiguousCAJets/i")
    metadata.Branch("puppiWeighted", puppi_weighted, "puppiWeighted/O")
    metadata.Branch("useJEC", use_jec, "useJEC/O")
    metadata.Branch("caAlgorithm", ca_algorithm)
    metadata.Branch("massObjective", mass_objective)
    metadata.Branch("gateRecoConstraint", gate_constraint)
    metadata.Branch("enforceLegacyRadiusConstraint", legacy_radius_constraint,
                    "enforceLegacyRadiusConstraint/O")
    metadata.Branch("processedEvents", processed_events, "processedEvents/l")
    metadata.Branch("sumWeights", sum_weights, "sumWeights/D")
    metadata.Branch("sumWeights2", sum_weights2, "sumWeights2/D")

    vu32 = ROOT.std.vector("unsigned int")
    vu16 = ROOT.std.vector("unsigned short")
    vu8 = ROOT.std.vector("unsigned char")
    vf = ROOT.std.vector("float")
    vs = ROOT.std.vector("string")
    default_gate_counts = vu16()
    collection_pt_cuts = vf()
    ca_radii = vf()
    cos_thrust_cuts = vf()
    default_gate_pt = vf()
    status_codes = vu8()
    status_names = vs()
    base_pt = vf()
    base_ca = vf()
    config_id = vu32()
    config_base = vu32()
    config_pt = vf()
    config_ca = vf()
    config_cos = vf()

    _fill_vector(default_gate_counts, [0, 4])
    _fill_vector(collection_pt_cuts, [100.0, 200.0])
    _fill_vector(ca_radii, [0.8])
    _fill_vector(cos_thrust_cuts, [0.0, 0.5])
    _fill_vector(default_gate_pt, [300.0, 500.0])
    _fill_vector(status_codes, list(range(9)))
    _fill_vector(status_names, list(validate_compact_scan.STATUS_NAMES))
    _fill_vector(base_pt, [100.0, 200.0])
    _fill_vector(base_ca, [0.8, 0.8])
    _fill_vector(config_id, [0, 1, 2, 3])
    _fill_vector(config_base, [0, 0, 1, 1])
    _fill_vector(config_pt, [100.0, 100.0, 200.0, 200.0])
    _fill_vector(config_ca, [0.8, 0.8, 0.8, 0.8])
    _fill_vector(config_cos, [0.0, 0.5, 0.0, 0.5])

    for name, value in (
        ("collectionPtCuts", collection_pt_cuts),
        ("caRadii", ca_radii),
        ("cosThrustCuts", cos_thrust_cuts),
        ("defaultGateJetCounts", default_gate_counts),
        ("defaultGatePtCuts", default_gate_pt),
        ("statusCodes", status_codes),
        ("statusNames", status_names),
        ("baseCollectionPtCut", base_pt),
        ("baseCaRadius", base_ca),
        ("configId", config_id),
        ("configBaseIndex", config_base),
        ("configCollectionPtCut", config_pt),
        ("configCaRadius", config_ca),
        ("configCosThrust", config_cos),
    ):
        metadata.Branch(name, value)
    metadata.Fill()

    events = ROOT.TTree("Events", "compact scan events")
    run = array("I", [1])
    lumi = array("I", [1])
    event = array("Q", [0])
    gen_weight = array("f", [0.0])
    ak_jet_pt = vf()
    n_ca_jets = vu16()
    reco_status = vu8()
    n_ambiguous = vu16()
    sj1_mass = vf()
    sj2_mass = vf()
    suu_mass = vf()
    events.Branch("run", run, "run/i")
    events.Branch("lumi", lumi, "lumi/i")
    events.Branch("event", event, "event/l")
    events.Branch("genWeight", gen_weight, "genWeight/F")
    for name, value in (
        ("akJetPt", ak_jet_pt), ("nCAJets", n_ca_jets),
        ("recoStatus", reco_status), ("nAmbiguous", n_ambiguous),
        ("sj1Mass", sj1_mass), ("sj2Mass", sj2_mass),
    ):
        events.Branch(name, value)

    if defect not in ("schema1", "missing_suu"):
        events.Branch("suuMass", suu_mass)

    nan = float("nan")
    entries = (
        {
            "event": 1, "weight": 1.0, "ak": [1000.0, 800.0, 300.0],
            "nca": [3, 0], "status": [0, 0, 5, 5], "namb": [0, 1, 0, 0],
            "m1": [100.0, 110.0, nan, nan],
            "m2": [120.0, 115.0, nan, nan],
        },
        {
            "event": 2, "weight": -0.5, "ak": [900.0, 500.0],
            "nca": [13, 2], "status": [0, 6, 0, 7], "namb": [12, 13, 1, 1],
            "m1": [90.0, nan, 80.0, nan],
            "m2": [95.0, nan, 85.0, nan],
        },
    )
    for index, values in enumerate(entries):
        event[0] = values["event"]
        gen_weight[0] = values["weight"]
        status_values = list(values["status"])
        mass1_values = list(values["m1"])
        mass2_values = list(values["m2"])
        ak_values = list(values["ak"])
        if defect == "bad_status" and index == 0:
            status_values[0] = 99
            mass1_values[0] = nan
            mass2_values[0] = nan
        if defect == "finite_invalid_mass" and index == 0:
            mass1_values[2] = 1.0
        if defect == "bad_ak_order" and index == 0:
            ak_values = [800.0, 1000.0, 300.0]
        if defect == "short_result" and index == 0:
            mass2_values.pop()
        if defect == "guard_at_cap" and index == 1:
            values = dict(values)
            values["namb"] = [12, 12, 1, 1]
        _fill_vector(ak_jet_pt, ak_values)
        _fill_vector(n_ca_jets, values["nca"])
        _fill_vector(reco_status, status_values)
        _fill_vector(n_ambiguous, values["namb"])
        _fill_vector(sj1_mass, mass1_values)
        _fill_vector(sj2_mass, mass2_values)
        pair_mass = [1000.0 if code == 0 else nan for code in status_values]
        if defect == "short_suu":
            pair_mass.pop()
        if index == 0:
            if defect == "nonfinite_suu":
                pair_mass[0] = nan
            elif defect == "negative_suu":
                pair_mass[0] = -1.0
            elif defect == "unphysical_suu":
                pair_mass[0] = 100.0
            elif defect == "finite_invalid_suu":
                pair_mass[2] = 1000.0
        _fill_vector(suu_mass, pair_mass)
        events.Fill()

    output.Write()
    output.Close()


@unittest.skipIf(ROOT is None, "PyROOT is not available")
class CompactScanValidatorTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def _run(self, defect=None):
        path = os.path.join(self.tempdir.name, "fixture.root")
        _make_fixture(path, defect)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = validate_compact_scan.main(["--strict-branches", path])
        return result, stdout.getvalue(), stderr.getvalue()

    def test_valid_fixture_passes(self):
        result, stdout, stderr = self._run()
        self.assertEqual(result, 0, msg=stderr)
        self.assertIn("PASS:", stdout)
        self.assertIn("projected at 10,000 events", stdout)

    def test_schema_one_remains_supported(self):
        result, _, stderr = self._run("schema1")
        self.assertEqual(result, 0, msg=stderr)

    def test_invalid_suu_mass_fails(self):
        for defect, message in (
            ("missing_suu", "suuMass"),
            ("short_suu", "suuMass length 3 does not equal Nconfig 4"),
            ("nonfinite_suu", "non-finite/negative suuMass"),
            ("negative_suu", "non-finite/negative suuMass"),
            ("unphysical_suu", "suuMass below sj1Mass + sj2Mass"),
            ("finite_invalid_suu", "must have NaN suuMass"),
        ):
            with self.subTest(defect=defect):
                result, _, stderr = self._run(defect)
                self.assertEqual(result, 1)
                self.assertIn(message, stderr)

    def test_out_of_range_status_fails(self):
        result, _, stderr = self._run("bad_status")
        self.assertEqual(result, 1)
        self.assertIn("out-of-range status 99", stderr)

    def test_invalid_mass_sentinel_fails(self):
        result, _, stderr = self._run("finite_invalid_mass")
        self.assertEqual(result, 1)
        self.assertIn("two NaN mass sentinels", stderr)

    def test_result_length_mismatch_fails(self):
        result, _, stderr = self._run("short_result")
        self.assertEqual(result, 1)
        self.assertIn("sj2Mass length 3 does not equal Nconfig 4", stderr)

    def test_unsorted_ak_jets_fail(self):
        result, _, stderr = self._run("bad_ak_order")
        self.assertEqual(result, 1)
        self.assertIn("akJetPt is not sorted descending", stderr)

    def test_cap_is_inclusive_but_guard_requires_more(self):
        # The valid fixture already contains a valid nAmbiguous==12 result.
        result, _, stderr = self._run("guard_at_cap")
        self.assertEqual(result, 1)
        self.assertIn("nAmbiguous=12 <= cap 12", stderr)


if __name__ == "__main__":
    unittest.main()
