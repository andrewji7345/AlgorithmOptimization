#!/usr/bin/env python3
"""Deterministic integration tests for compact-scan metric factorization."""

import os
import math
import sys
import tempfile
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace

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


def _fill_vector(vector, values):
    vector.clear()
    for value in values:
        vector.push_back(value)


def make_compact_fixture(path, sample_name="WbWb_4000_1000", ak_radius=0.8,
                         schema_version_number=2, suu_defect=None):
    """Write four events with exact gate boundaries and one guard failure."""

    output = ROOT.TFile(str(path), "RECREATE")
    directory = output.mkdir("compactScan")
    directory.cd()

    metadata = ROOT.TTree("Metadata", "compact scan metadata")
    schema_version = array("I", [schema_version_number])
    sample = ROOT.std.string(sample_name)
    stored_ak_radius = array("f", [ak_radius])
    max_ambiguous = array("I", [12])
    puppi_weighted = array("b", [1])
    use_jec = array("b", [0])
    ca_algorithm = ROOT.std.string("cambridge_y_phi")
    mass_objective = ROOT.std.string("abs(m1-m2)/(m1+m2)")
    gate_constraint = ROOT.std.string(
        "collectionPtCut<=gatePtCut; nGateJets=0 is ungated")
    legacy_radius_constraint = array("b", [0])
    processed_events = array("Q", [4])
    sum_weights = array("d", [2.5])
    sum_weights2 = array("d", [3.25])

    metadata.Branch("schemaVersion", schema_version, "schemaVersion/i")
    metadata.Branch("sampleName", sample)
    metadata.Branch("akRadius", stored_ak_radius, "akRadius/F")
    metadata.Branch("maxAmbiguousCAJets", max_ambiguous,
                    "maxAmbiguousCAJets/i")
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
    collection_pt_cuts = vf()
    ca_radii = vf()
    cos_thrust_cuts = vf()
    default_gate_counts = vu16()
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

    _fill_vector(collection_pt_cuts, [100.0, 200.0])
    _fill_vector(ca_radii, [0.8])
    _fill_vector(cos_thrust_cuts, [0.0, 0.5])
    _fill_vector(default_gate_counts, [0, 1, 2])
    _fill_vector(default_gate_pt, [200.0, 300.0])
    _fill_vector(status_codes, range(len(metrics.STATUS_NAMES)))
    _fill_vector(status_names, metrics.STATUS_NAMES)
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
    lumi = array("I", [7])
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
        ("akJetPt", ak_jet_pt),
        ("nCAJets", n_ca_jets),
        ("recoStatus", reco_status),
        ("nAmbiguous", n_ambiguous),
        ("sj1Mass", sj1_mass),
        ("sj2Mass", sj2_mass),
    ):
        events.Branch(name, value)

    if schema_version_number >= 2 and suu_defect != "missing":
        events.Branch("suuMass", suu_mass)

    nan = float("nan")
    entries = (
        ([500.0, 300.0, 200.0], [2, 2], [0, 0, 0, 0],
         [0, 1, 0, 1], [1000.0] * 4, [1000.0] * 4, 1.0),
        ([300.0, 250.0], [13, 13], [6, 6, 6, 6],
         [13, 13, 13, 13], [nan] * 4, [nan] * 4, 1.0),
        ([200.0], [2, 0], [0, 0, 1, 1],
         [0, 0, 0, 0], [800.0, 900.0, nan, nan],
         [1200.0, 1100.0, nan, nan], -0.5),
        ([], [0, 0], [1, 1, 1, 1],
         [0, 0, 0, 0], [nan] * 4, [nan] * 4, 1.0),
    )
    for entry_number, values in enumerate(entries, start=1):
        ak_values, nca, status, ambiguous, mass1, mass2, weight = values
        event[0] = entry_number
        gen_weight[0] = weight
        _fill_vector(ak_jet_pt, ak_values)
        _fill_vector(n_ca_jets, nca)
        _fill_vector(reco_status, status)
        _fill_vector(n_ambiguous, ambiguous)
        _fill_vector(sj1_mass, mass1)
        _fill_vector(sj2_mass, mass2)
        # Pair mass includes relative SJ momentum; it differs from m1 + m2.
        pair_mass = [4000.0 - 250.0 * (entry_number - 1)
                     if code == 0 else nan for code in status]
        if suu_defect == "short":
            pair_mass.pop()
        elif suu_defect == "nonfinite" and entry_number == 1:
            pair_mass[0] = nan
        elif suu_defect == "finite_invalid" and entry_number == 2:
            pair_mass[0] = 4000.0
        _fill_vector(suu_mass, pair_mass)
        events.Fill()

    output.Write()
    output.Close()


@unittest.skipIf(ROOT is None, "PyROOT is not available")
class SuuMassPayloadTest(unittest.TestCase):
    def test_versioned_pair_mass_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.root"
            for version in (1, 2):
                with self.subTest(version=version):
                    make_compact_fixture(path, schema_version_number=version)
                    payload = metrics.read_event_payload(path)
                    if version == 1:
                        self.assertIsNone(payload.suu_mass)
                    else:
                        self.assertEqual(payload.suu_mass.shape, (4, 4))
                        self.assertEqual(payload.suu_mass[0, 0], 4000.0)
                        self.assertNotEqual(payload.suu_mass[0, 0],
                                            payload.sj1_mass[0, 0] + payload.sj2_mass[0, 0])
                        self.assertTrue(math.isnan(payload.suu_mass[1, 0]))

    def test_malformed_version_two_pair_mass_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.root"
            for defect in ("missing", "short", "nonfinite", "finite_invalid"):
                with self.subTest(defect=defect):
                    make_compact_fixture(path, suu_defect=defect)
                    with self.assertRaisesRegex(ValueError, "suuMass"):
                        metrics.read_event_payload(path)


class SampleDefinitionTest(unittest.TestCase):
    def test_full_regime_grid_and_generated_mass_alias(self):
        definitions = metrics.load_sample_definitions(
            REPOSITORY / "test" / "signalMCFiles")
        self.assertEqual(len(definitions), 114)
        definition = definitions["HtZt_6000_2000"]
        self.assertEqual(definition.nominal_suu_mass, 6000.0)
        self.assertEqual(definition.nominal_chi_mass, 2000.0)
        self.assertEqual(definition.generated_suu_mass, 6200.0)
        self.assertEqual(definition.generated_chi_mass, 1950.0)
        self.assertEqual(
            metrics.true_chi_mass_for_sample(definition.sample, definitions),
            1950.0,
        )

    def test_production_grid_has_311168_configurations_per_regime(self):
        ak_radii = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        ca_radii = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        pt_cuts = list(range(100, 401, 20))
        cos_cuts = [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 0.95,
        ]
        gate_counts = tuple(range(7))
        per_ak = []
        for ak_radius in ak_radii:
            allowed_ca = [
                ca_radius for ca_radius in ca_radii
                if ca_radius + 1.0e-9 >= max(0.4, ak_radius - 0.2)
            ]
            reconstruction = [
                (pt_cut, ca_radius, cos_cut)
                for pt_cut in pt_cuts
                for ca_radius in allowed_ca
                for cos_cut in cos_cuts
            ]
            metadata = SimpleNamespace(
                ak_radius=ak_radius,
                default_gate_jet_counts=gate_counts,
                default_gate_pt_cuts=tuple(pt_cuts),
                config_collection_pt_cut=tuple(
                    item[0] for item in reconstruction),
                config_ca_radius=tuple(item[1] for item in reconstruction),
                config_cos_thrust=tuple(item[2] for item in reconstruction),
            )
            per_ak.append(
                len(metrics.enumerate_configuration_keys(metadata)))
        self.assertEqual(
            per_ak,
            [64064, 64064, 54912, 45760, 36608, 27456, 18304],
        )
        self.assertEqual(sum(per_ak), 311168)


@unittest.skipIf(ROOT is None, "PyROOT is not available")
class CompactScanMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ROOT.gROOT.SetBatch(True)
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tempdir.name) / "compact.root"
        make_compact_fixture(cls.path)
        cls.metadata = metrics.load_metadata(cls.path)
        cls.payload = metrics.read_event_payload(cls.metadata)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_gate_is_strictly_greater_than_threshold(self):
        self.assertEqual(
            metrics.gate_jet_multiplicity(self.payload, 200.0).tolist(),
            [2, 2, 0, 0],
        )
        self.assertEqual(
            metrics.gate_jet_multiplicity(self.payload, 300.0).tolist(),
            [1, 0, 0, 0],
        )
        self.assertEqual(
            metrics.gate_mask(self.payload, 2, 200.0).tolist(),
            [True, True, False, False],
        )

    def test_ungated_configuration_is_canonical_and_unique(self):
        key = metrics.configuration_key(0, 300.0, 100.0, 0.8, 0.8, 0.5)
        canonical = metrics.configuration_key(
            0, None, 100.0, 0.8, 0.8, 0.5)
        self.assertEqual(key, canonical)
        self.assertIsNone(key.gate_pt_cut)
        self.assertEqual(metrics.parse_configuration_slug(
            metrics.configuration_slug(key)), key)

        keys = metrics.enumerate_configuration_keys(self.metadata)
        ungated = [candidate for candidate in keys
                   if candidate.n_gate_jets == 0]
        self.assertEqual(len(keys), 20)
        self.assertEqual(len(ungated), self.metadata.n_configurations)
        self.assertTrue(all(candidate.gate_pt_cut is None
                            for candidate in ungated))

    def test_collection_threshold_cannot_exceed_gate_threshold(self):
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            metrics.configuration_key(
                1, 200.0, 300.0, 0.8, 0.8, 0.0)

    def test_efficiency_identity_and_complexity_guard_physicality(self):
        definition = metrics.PhysicalityDefinition(
            bias_limit=0.5,
            resolution_limit=0.5,
            invalid_limit=0.25,
            tail_limit=0.5,
            min_valid_events=1,
        )
        key = metrics.configuration_key(
            1, 200.0, 100.0, 0.8, 0.8, 0.0)
        result = metrics.calculate_configuration_metrics(
            self.metadata, self.payload, key, 1000.0, definition)

        self.assertEqual(result["n_events"], 4)
        self.assertEqual(result["n_gate_events"], 2)
        self.assertEqual(result["n_valid_events"], 1)
        self.assertEqual(result["n_valid_all_events"], 2)
        self.assertAlmostEqual(result["gate_efficiency"], 0.5)
        self.assertAlmostEqual(result["reco_given_gate_efficiency"], 0.5)
        self.assertAlmostEqual(result["signal_retention"], 0.25)
        self.assertAlmostEqual(
            result["signal_retention"],
            result["gate_efficiency"]
            * result["reco_given_gate_efficiency"],
        )
        self.assertEqual(result["status_complexity_guard_gated"], 1)
        self.assertAlmostEqual(result["invalid_fraction"], 0.5)
        self.assertAlmostEqual(result["ungated_reco_efficiency"], 0.5)
        self.assertAlmostEqual(result["peak_signal_retention"], 0.25)
        self.assertEqual(result["n_complexity_guard_events"], 1)
        self.assertAlmostEqual(
            result["complexity_guard_fraction_given_gate"], 0.5)
        self.assertEqual(result["n_other_invalid_events"], 0)
        self.assertAlmostEqual(
            result["other_invalid_fraction_given_gate"], 0.0)
        self.assertAlmostEqual(result["physicality_invalid_term"], 2.0)
        self.assertFalse(result["physicality_pass"])

        stricter = metrics.configuration_key(
            1, 300.0, 100.0, 0.8, 0.8, 0.0)
        stricter_result = metrics.calculate_configuration_metrics(
            self.metadata, self.payload, stricter, 1000.0, definition)
        self.assertEqual(stricter_result["n_gate_events"], 1)
        self.assertEqual(stricter_result["n_valid_events"], 1)
        self.assertAlmostEqual(stricter_result["gate_efficiency"], 0.25)
        self.assertAlmostEqual(
            stricter_result["reco_given_gate_efficiency"], 1.0)
        self.assertAlmostEqual(stricter_result["signal_retention"], 0.25)
        self.assertTrue(stricter_result["physicality_pass"])

    def test_vectorized_and_single_configuration_metrics_agree(self):
        definition = metrics.PhysicalityDefinition(
            bias_limit=0.5, resolution_limit=0.5, invalid_limit=0.25,
            tail_limit=0.5, min_valid_events=1)
        selected = metrics.configuration_key(
            1, 200.0, 100.0, 0.8, 0.8, 0.0)
        single = metrics.calculate_configuration_metrics(
            self.metadata, self.payload, selected, 1000.0, definition)
        found = None
        for batch in metrics.iter_configuration_metrics(
                self.metadata, self.payload, 1000.0,
                definition, chunk_size=2):
            if selected in batch.keys:
                found = (batch.values, batch.keys.index(selected))
                break
        self.assertIsNotNone(found)
        selected_values, selected_index = found
        for name in (
                "n_gate_events", "n_valid_events", "gate_efficiency",
                "reco_given_gate_efficiency", "signal_retention",
                "ungated_reco_efficiency", "peak_signal_retention",
                "n_complexity_guard_events",
                "complexity_guard_fraction_given_gate",
                "n_other_invalid_events",
                "other_invalid_fraction_given_gate",
                "invalid_fraction", "physicality_score"):
            self.assertAlmostEqual(
                float(selected_values[name][selected_index]),
                float(single[name]))

    def test_duplicate_sample_and_ak_input_is_rejected(self):
        duplicate = Path(self.tempdir.name) / "duplicate.root"
        make_compact_fixture(duplicate)
        with self.assertRaisesRegex(
                ValueError, "duplicate compact sample/AK"):
            metrics.discover_compact_files(paths=[self.path, duplicate])


if __name__ == "__main__":
    unittest.main()
