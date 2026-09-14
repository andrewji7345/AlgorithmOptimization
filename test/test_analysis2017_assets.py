"""Integrity checks for the exact analysis calibration and sample inputs."""

import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Analysis2017AssetsTest(unittest.TestCase):
    def test_calibration_assets_match_provenance(self):
        directory = ROOT / "data" / "analysis_2017"
        manifest = json.loads((directory / "provenance.json").read_text())
        names = [entry["file"] for entry in manifest["files"]]
        assert len(names) == len(set(names))
        assert set(names) == {path.name for path in directory.iterdir() if path.name != "provenance.json"}
        for entry in manifest["files"]:
            content = (directory / entry["file"]).read_bytes()
            assert hashlib.sha256(content).hexdigest() == entry["sha256"]
            if "git_blob_sha" in entry:
                git_object = b"blob " + str(len(content)).encode() + b"\0" + content
                assert hashlib.sha1(git_object).hexdigest() == entry["git_blob_sha"]


    def test_globaltag_jer_labels_and_reduced_btag_inputs(self):
        directory = ROOT / "data" / "analysis_2017"
        mapping = (directory / "globaltag_mapping.txt").read_text()
        for name in ("AK4PFchs", "AK8PF", "AK4PFPuppi"):
            for kind in ("PtResolution", "SF"):
                assert "JR_Summer19UL17_JRV3_MC_" + kind + "_" + name in mapping
        btag = json.loads((directory / "btagging.json").read_text())
        assert {entry["name"] for entry in btag["corrections"]} == {"deepJet_mujets", "deepJet_incl"}


    def test_background_catalog_is_disjoint_and_concrete(self):
        catalog = json.loads((ROOT / "config" / "backgrounds_2017.json").read_text())
        samples = catalog["samples"]
        assert len(samples) == 23
        assert len({sample["dataset"] for sample in samples}) == len(samples)
        assert catalog["luminosity_pb"] == 41480
        assert sum(sample["category"] == "QCDMC" for sample in samples) == 10
        assert sum(sample["category"] == "TTbarMC" for sample in samples) == 3
        for sample in samples:
            assert sample["dataset"].startswith("/") and sample["dataset"].endswith("/MINIAODSIM")
            assert "RunIISummer20UL17MiniAODv2" in sample["dataset"]
            assert sample["cross_section_pb"] > 0
            assert "/QCD_HT" not in sample["dataset"]
            assert "/TTTo" not in sample["dataset"]


    def test_signal_cross_sections_match_analysis_table_three(self):
        catalog = json.loads((ROOT / "config" / "signal_benchmarks_2017.json").read_text())
        assert catalog["benchmark"]["y_uu"] == catalog["benchmark"]["y_chi"] == 2
        # Published table values in fb are rounded to two decimals.
        for sample, expected_fb in zip(catalog["samples"], (156.72, 5.71, 0.11)):
            assert math.isclose(sample["cross_section_pb"] * 1000, expected_fb, abs_tol=0.005)


if __name__ == "__main__":
    unittest.main()
