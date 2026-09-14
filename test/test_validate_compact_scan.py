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


def _make_fixture(path, defect=None, analysis=False, radius=0.8, correction=None, observables=False):
    output = ROOT.TFile(path, "RECREATE")
    directory = output.mkdir("compactScan")
    directory.cd()

    metadata = ROOT.TTree("Metadata", "compact scan metadata")
    schema_version = array("I", [3 if analysis else 1 if defect == "schema1" else 2])
    sample_name = ROOT.std.string("unit_test_sample")
    ak_radius = array("f", [radius])
    max_ambiguous = array("I", [12])
    puppi_weighted = array("b", [1])
    use_jec = array("b", [int(analysis)])
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
    if analysis:
        analysis_profile = ROOT.std.string(
            "unknown" if defect == "bad_profile" else "AN-23-067-UL2017-cutbased-v1")
        correction_profile = ROOT.std.string(correction or "UL2017-AK4PFchs-AK8PFPuppi-JEC-JER-nominal-v1")
        sample_kind = ROOT.std.string("background")
        if defect != "missing_profile":
            metadata.Branch("analysisSelection", analysis_profile)
        metadata.Branch("correctionPrescription", correction_profile)
        metadata.Branch("sampleKind", sample_kind)

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
    if observables:
        observable_version = array("I", [2 if defect == "unknown_observable_version" else 1])
        systematic = ROOT.std.string("unknown" if defect == "bad_systematic" else "nominal")
        reconstruction = ROOT.std.string("AN23-067-PATAK8-CA8-Thrust-source-port-v1")
        variation_names = vs()
        names = list(validate_compact_scan.WEIGHT_VARIATION_NAMES)
        if defect == "bad_weight_names":
            names.reverse()
        _fill_vector(variation_names, names)
        metadata.Branch("analysisObservableVersion", observable_version, "analysisObservableVersion/i")
        metadata.Branch("analysisSystematic", systematic)
        if defect != "missing_observable_metadata":
            metadata.Branch("referenceReconstruction", reconstruction)
        metadata.Branch("weightVariationNames", variation_names)
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

    if analysis:
        analysis_scalars = {
            "analysisWeight": (array("f", [1.25]), "F"),
            "analysisHT": (array("f", [2500.]), "F"),
            "passesBaseline": (array("b", [1]), "O"),
            "passesTrigger": (array("b", [1]), "O"),
            "passesFilters": (array("b", [1]), "O"),
            "passesLeptonVeto": (array("b", [1]), "O"),
            "passesJetVeto": (array("b", [1]), "O"),
            "analysisNAK4": (array("H", [4]), "s"),
            "analysisNAK8": (array("H", [3]), "s"),
            "analysisNHeavyAK8": (array("H", [2]), "s"),
            "analysisNBTags": (array("H", [1]), "s"),
        }
        if correction and correction.endswith("btagGuard-v3"):
            analysis_scalars["analysisBTagWeight"] = (array("f", [2. if defect == "bad_btag_guard" else 1.]), "F")
            analysis_scalars["analysisBTagWeightFallback"] = (array("b", [1]), "O")
        for name, (buffer, leaf_type) in analysis_scalars.items():
            events.Branch(name, buffer, name + "/" + leaf_type)
        sr = vu8()
        reco_veto = vu8()
        nca4_first = vu16()
        nca4_second = vu16()
        events.Branch("passesSignalRegion", sr)
        events.Branch("passesRecoJetVeto", reco_veto)
        events.Branch("sj1NCA4E300", nca4_first)
        events.Branch("sj2NCA4E300", nca4_second)

    if observables:
        observable_vectors = {}
        for name in (validate_compact_scan.OBSERVABLE_VECTOR_COUNTS |
                     validate_compact_scan.OBSERVABLE_VECTOR_MASSES |
                     validate_compact_scan.OBSERVABLE_REGION_BRANCHES |
                     validate_compact_scan.OBSERVABLE_FLOAT_VECTORS | {"btagWeightVariationFallback"}):
            if name in validate_compact_scan.OBSERVABLE_VECTOR_COUNTS:
                buf = vu16()
            elif name in validate_compact_scan.OBSERVABLE_REGION_BRANCHES | {"btagWeightVariationFallback"}:
                buf = vu8()
            elif name == "analysisWeightVariations" and defect == "wrong_observable_type":
                buf = ROOT.std.vector("double")()
            else:
                buf = vf()
            observable_vectors[name] = buf
            if name != "sj2MassE100" or defect != "missing_observable_event":
                events.Branch(name, buf)
        observable_scalars = {}
        for name in validate_compact_scan.REFERENCE_MASSES | {"referenceWeight"}:
            observable_scalars[name] = (array("f", [0]), "F")
        for name in validate_compact_scan.REFERENCE_COUNTS:
            observable_scalars[name] = (array("H", [0]), "s")
        for name in ("referenceRecoStatus", "referenceRegion"):
            observable_scalars[name] = (array("B", [0]), "b")
        for name, (buffer, leaf_type) in observable_scalars.items():
            events.Branch(name, buffer, name + "/" + leaf_type)
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
        if analysis:
            mass1_values = [value + 1000. for value in mass1_values]
            mass2_values = [value + 1000. for value in mass2_values]
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
        pair_mass = [(5000.0 if analysis else 1000.0) if code == 0 else nan for code in status_values]
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
        if analysis:
            # A vetoed retained AK8 suppresses only configuration 1, while a
            # negative generator-weight event remains valid schema-v3 input.
            analysis_scalars["analysisNBTags"][0][0] = 1 if index == 0 else 0
            region_values = [1, 0, 0, 0] if index == 0 else [0, 0, 0, 0]
            veto_values = [1, 0, 1, 1]
            tag_counts = [2 if code == 0 else 0 for code in status_values]
            if defect == "bad_sr" and index == 0:
                region_values[1] = 1
            if defect == "bad_baseline" and index == 0:
                analysis_scalars["passesTrigger"][0][0] = 0
            if defect == "negative_analysis_weight" and index == 0:
                analysis_scalars["analysisWeight"][0][0] = -1.
            if defect == "short_reco_veto" and index == 0:
                veto_values.pop()
            if defect == "nonbinary_reco_veto" and index == 0:
                veto_values[0] = 2
            if defect == "bad_ca4_energy" and index == 0:
                tag_counts[0] = 5
            if defect == "invalid_reco_ca4" and index == 0:
                tag_counts[2] = 1
            _fill_vector(sr, region_values)
            _fill_vector(reco_veto, veto_values)
            _fill_vector(nca4_first, tag_counts)
            _fill_vector(nca4_second, tag_counts)
        if observables:
            obs_values = {
                "analysisWeightVariations": [1.25] * 14,
                "referenceWeightVariations": [1.25] * 14,
                "btagWeightVariationFallback": [0] * 8,
                "analysisBTagJetPt": [80., 90., 100., 110.],
                "analysisBTagJetEta": [0., .5, -.5, 1.],
                "analysisBTagJetDiscriminator": [.5 if index == 0 else .1, .1, .1, .1],
                "sj1NCA4E50": [3 if code == 0 else 0 for code in status_values],
                "sj2NCA4E50": [3 if code == 0 else 0 for code in status_values],
                "sj1MassE100": [500. if code == 0 else nan for code in status_values],
                "sj2MassE100": [500. if code == 0 else nan for code in status_values],
                "passesControlRegion": [0] * 4 if index == 0 else [1, 0, 1, 0],
                "passesAT1b": [0] * 4, "passesAT0b": [0] * 4,
            }
            ref_values = {"referenceWeight": 1.25, "referenceRecoStatus": 0 if index == 0 else 8,
                          "referenceRegion": 1 if index == 0 else 0,
                          "referenceSJ1Mass": 1100. if index == 0 else nan,
                          "referenceSJ2Mass": 1200. if index == 0 else nan,
                          "referenceSuuMass": 5000. if index == 0 else nan,
                          "referenceSJ1MassE100": 500. if index == 0 else nan,
                          "referenceSJ2MassE100": 500. if index == 0 else nan,
                          "referenceSJ1NCA4E50": 3 if index == 0 else 0,
                          "referenceSJ2NCA4E50": 3 if index == 0 else 0,
                          "referenceSJ1NCA4E300": 2 if index == 0 else 0,
                          "referenceSJ2NCA4E300": 2 if index == 0 else 0}
            if index == 0:
                if defect == "short_weight_variation": obs_values["analysisWeightVariations"].pop()
                if defect == "nan_weight_variation": obs_values["referenceWeightVariations"][0] = nan
                if defect == "invalid_btag_variation_flag": obs_values["btagWeightVariationFallback"][0] = 2
                if defect == "short_btag_jets": obs_values["analysisBTagJetPt"].pop()
                if defect == "wrong_btag_count": obs_values["analysisBTagJetDiscriminator"][1] = .9
                if defect == "nan_btag_eta": obs_values["analysisBTagJetEta"][0] = nan
                if defect == "region_overlap": obs_values["passesAT1b"][0] = 1
                if defect == "wrong_control_region": obs_values["passesControlRegion"][0] = 1
                if defect == "short_ca4_e50": obs_values["sj1NCA4E50"].pop()
                if defect == "inconsistent_ca4_thresholds": obs_values["sj1NCA4E50"][0] = 1
                if defect == "restricted_mass_too_large": obs_values["sj1MassE100"][0] = 2000.
                if defect == "finite_invalid_restricted_mass": obs_values["sj1MassE100"][2] = 0.
                if defect == "negative_reference_weight": ref_values["referenceWeight"] = -1.
                if defect == "invalid_reference_status": ref_values["referenceRecoStatus"] = 99
                if defect == "wrong_reference_region": ref_values["referenceRegion"] = 4
                if defect == "small_reference_pair_mass": ref_values["referenceSuuMass"] = 1000.
                if defect == "zero_reference_mass": ref_values["referenceSJ1Mass"] = 0.
            if index == 1 and defect == "finite_invalid_reference_mass":
                ref_values["referenceSJ1Mass"] = 1.
            for name, values in obs_values.items():
                _fill_vector(observable_vectors[name], values)
            for name, value in ref_values.items():
                observable_scalars[name][0][0] = value
        events.Fill()

    output.Write()
    output.Close()


@unittest.skipIf(ROOT is None, "PyROOT is not available")
class CompactScanValidatorTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def _run(self, defect=None, analysis=False, **kwargs):
        path = os.path.join(self.tempdir.name, "fixture.root")
        _make_fixture(path, defect, analysis, **kwargs)
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

    def test_schema_three_with_analysis_masks_passes(self):
        result, stdout, stderr = self._run(analysis=True)
        self.assertEqual(result, 0, msg=stderr)
        self.assertIn("PASS:", stdout)

    def test_radius_and_correction_profile_must_agree(self):
        profile = "UL2017-AK4CHS-baseline-AK4AK8Puppi-reco-JEC-JER-nominal-v2"
        for radius, correction, expected in ((0.4, profile, 0), (0.8, profile, 0),
                                               (0.6, profile, 1), (0.4, None, 1)):
            with self.subTest(radius=radius, correction=correction):
                code, _, err = self._run(analysis=True, radius=radius, correction=correction)
                self.assertEqual(code, expected, msg=err)

    def test_v3_btag_guard_is_checked(self):
        profile = "UL2017-AK4CHS-baseline-AK4AK8Puppi-reco-JEC-JER-btagGuard-v3"
        code, _, err = self._run(analysis=True, correction=profile)
        self.assertEqual(code, 0, msg=err)
        code, _, err = self._run("bad_btag_guard", analysis=True, correction=profile)
        self.assertEqual(code, 1)
        self.assertIn("invalid b-tag weight/fallback", err)

    def test_schema_three_rejects_invalid_analysis_inputs(self):
        for defect, message in (
            ("bad_sr", "signal-region bit contradicts"),
            ("bad_baseline", "baseline contradicts cutflow flags"),
            ("negative_analysis_weight", "invalid analysisWeight"),
            ("missing_profile", "analysisSelection"),
            ("bad_profile", "unsupported analysisSelection"),
            ("short_reco_veto", "passesRecoJetVeto width mismatch"),
            ("nonbinary_reco_veto", "passesRecoJetVeto"),
            ("bad_ca4_energy", "violates superjet rest-energy bound"),
            ("invalid_reco_ca4", "invalid reconstruction must have zero CA4 tag counts"),
        ):
            with self.subTest(defect=defect):
                result, _, stderr = self._run(defect, analysis=True)
                self.assertEqual(result, 1)
                self.assertIn(message, stderr)

    def test_additive_analysis_observables_pass(self):
        code, _, err = self._run(analysis=True, observables=True)
        self.assertEqual(code, 0, msg=err)

    def test_additive_contract_rejects_incomplete_unknown_and_wrong_types(self):
        for defect, expected in (
            ("missing_observable_metadata", "referenceReconstruction"),
            ("missing_observable_event", "sj2MassE100"),
            ("unknown_observable_version", "analysisObservableVersion"),
            ("bad_systematic", "unsupported analysisSystematic"),
            ("bad_weight_names", "weightVariationNames"),
            ("wrong_observable_type", "has type"),
        ):
            with self.subTest(defect=defect):
                code, _, err = self._run(defect, analysis=True, observables=True)
                self.assertEqual(code, 1, msg=err)
                self.assertIn(expected, err)

    def test_additive_observable_event_invariants(self):
        for defect, expected in (
            ("short_weight_variation", "14 finite nonnegative"),
            ("nan_weight_variation", "14 finite nonnegative"),
            ("invalid_btag_variation_flag", "eight boolean"),
            ("short_btag_jets", "lengths disagree"),
            ("wrong_btag_count", "contradict analysisNBTags"),
            ("nan_btag_eta", "invalid b-tag jet"),
            ("region_overlap", "exclusive tag/anti-tag"),
            ("wrong_control_region", "exclusive tag/anti-tag"),
            ("short_ca4_e50", "width mismatch"),
            ("inconsistent_ca4_thresholds", "inconsistent CA4 E50/E300"),
            ("restricted_mass_too_large", "exceeds full superjet mass"),
            ("finite_invalid_restricted_mass", "NaN MassE100"),
            ("negative_reference_weight", "invalid referenceWeight"),
            ("invalid_reference_status", "invalid referenceRecoStatus"),
            ("wrong_reference_region", "referenceRegion contradicts"),
            ("small_reference_pair_mass", "pair-mass bound"),
            ("zero_reference_mass", "positive finite superjet"),
            ("finite_invalid_reference_mass", "needs NaN masses"),
        ):
            with self.subTest(defect=defect):
                code, _, err = self._run(defect, analysis=True, observables=True)
                self.assertEqual(code, 1, msg=err)
                self.assertIn(expected, err)

    def test_region_boundaries_and_exclusivity(self):
        region = validate_compact_scan._expected_region
        for btags in (0, 1, 3):
            self.assertEqual(region(True, True, True, btags, 2, 2, 3, 3, 500., 500.), 1 if btags else 2)
            self.assertEqual(region(True, True, True, btags, 2, 0, 3, 0, 500., 0.), 3 if btags else 4)
            self.assertEqual(region(True, True, True, btags, 0, 2, 0, 3, 149.999, 500.), 3 if btags else 4)
            self.assertEqual(region(True, True, True, btags, 0, 2, 0, 3, 150., 500.), 0)
            self.assertEqual(region(True, True, True, btags, 0, 2, 1, 3, 0., 500.), 0)
            for mask in ((False, True, True), (True, False, True), (True, True, False)):
                self.assertEqual(region(*mask, btags, 2, 2, 3, 3, 500., 500.), 0)

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
