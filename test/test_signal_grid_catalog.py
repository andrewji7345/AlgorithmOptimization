#!/usr/bin/env python3
"""Signal inventory integrity, source normalization, and the explicit mass substitute."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
import build_signal_grid_catalog as builder
from resolve_samples import load_catalog, validate_file_response


class SignalGridCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog, cls.audit = builder.build(REPOSITORY)
        cls.samples = {s["name"]: s for s in cls.catalog["samples"]}

    def temporary_repo(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        repo = Path(temp.name)
        shutil.copytree(REPOSITORY / "test/signalMCFiles", repo / "test/signalMCFiles")
        (repo / "config").mkdir()
        shutil.copy2(REPOSITORY / "config/signal_benchmarks_2017.json", repo / "config/signal_benchmarks_2017.json")
        return repo

    def test_complete_grid_and_unique_actual_dataset_identifiers(self):
        self.assertEqual(len(self.samples), 114)
        self.assertEqual(Counter(s["decay_mode"] for s in self.samples.values()), {m: 19 for m in builder.MODES})
        nominal = {(s["nominal_suu_mass_gev"], s["nominal_chi_mass_gev"]) for s in self.samples.values()}
        self.assertEqual(nominal, {(s, c) for s, cs in builder.GRID.items() for c in cs})
        self.assertEqual(len({s["dataset"] for s in self.samples.values()}), 114)
        self.assertEqual(self.audit["actual_generated_mass_pair_count"], 20)
        self.assertEqual(self.audit["unresolved_samples"], [])
        self.assertTrue(self.catalog["all_samples_normalized"])

    def test_every_lfn_matches_its_explicit_dataset_and_is_unique(self):
        seen = set()
        for sample in self.samples.values():
            path = REPOSITORY / "config" / sample["input_file_list"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), sample["input_file_list_sha256"])
            lfns = [line[line.index("/store/mc/"):].strip() for line in path.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
            self.assertEqual(len(lfns), sample["local_list_file_count"])
            self.assertEqual(validate_file_response("\n".join(lfns), sample["dataset"]), sorted(lfns))
            self.assertFalse(seen.intersection(lfns))
            seen.update(lfns)
            self.assertFalse(sample["full_dataset_file_coverage_verified"])
            self.assertIsNone(sample["full_dataset_sum_gen_weights"])
        self.assertEqual(len(seen), 1237)

    def test_all_three_existing_cross_sections_preserved(self):
        expected = {"WbWb_4000_1000": .1567165012696888,
                    "WbWb_6000_2000": .005711471665638484,
                    "WbWb_8000_3000": .00011091388424720394}
        for name, value in expected.items():
            self.assertTrue(math.isclose(self.samples[name]["cross_section_pb"], value, rel_tol=2e-14))

    def test_interpolation_uses_production_and_actual_mass_branching(self):
        sample = self.samples["HtZt_6000_2000"]
        self.assertEqual((sample["generated_suu_mass_gev"], sample["generated_chi_mass_gev"]), (6200, 1950))
        self.assertEqual((sample["nominal_suu_mass_gev"], sample["nominal_chi_mass_gev"]), (6000, 2000))
        self.assertEqual((sample["suu_mass_gev"], sample["chi_mass_gev"]), (6200, 1950))
        production = 137.0 ** .8 * 23.1 ** .2
        rho = 1950. / 6200.
        phase = (1 - 2 * rho * rho) * math.sqrt(1 - 4 * rho * rho)
        branching = phase / (1 + phase)  # equal y_uu=y_chi cancels the common width factor
        mixed_hadronic = 2 * (.25 * .58 * .6741) * (.25 * .6991 * .6741)
        expected_pb = production * branching * mixed_hadronic / 1000.
        self.assertTrue(math.isclose(sample["production_cross_section_fb"], production, rel_tol=2e-14))
        self.assertTrue(math.isclose(sample["suu_to_chichi_branching_fraction"], branching, rel_tol=2e-14))
        self.assertTrue(math.isclose(sample["cross_section_pb"], expected_pb, rel_tol=2e-14))
        self.assertFalse(math.isclose(sample["cross_section_pb"], sample["nominal_grid_hypothesis_cross_section_pb"], rel_tol=.01))
        self.assertEqual([s["name"] for s in self.samples.values() if s["production_interpolation"]], [sample["name"]])

    def test_interpolation_preserves_anchors_and_rejects_extrapolation(self):
        for mass, value in builder.PRODUCTION_FB.items():
            self.assertEqual(builder.production_cross_section_fb(mass), value)
        for mass in (3999, 8001, float("nan"), float("inf")):
            with self.subTest(mass=mass), self.assertRaises(ValueError):
                builder.production_cross_section_fb(mass)
        for anchors in ({6000: 0, 7000: 23.1}, {6000: -1, 7000: 23.1}, {6000: float("nan"), 7000: 23.1}):
            with self.assertRaises(ValueError):
                builder.production_cross_section_fb(6200, anchors)

    def test_vendored_source_is_hash_locked(self):
        provenance = json.loads((REPOSITORY / "data/analysis_2017/signal_normalization_provenance.json").read_text())
        self.assertEqual(provenance["sha256"], builder.SOURCE_SHA256)
        self.assertEqual(hashlib.sha256((REPOSITORY / builder.SOURCE_RELATIVE).read_bytes()).hexdigest(), builder.SOURCE_SHA256)
        with tempfile.TemporaryDirectory() as temp:
            modified = Path(temp) / "source.py"
            modified.write_bytes((REPOSITORY / builder.SOURCE_RELATIVE).read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "differs from audited commit"):
                builder.build(REPOSITORY, modified)

    def test_duplicate_and_incorrect_mass_files_fail_closed(self):
        repo = self.temporary_repo()
        target = repo / "test/signalMCFiles/WbWb_4000_1000.txt"
        original = target.read_text()
        line = next(s for s in original.splitlines() if s.strip() and not s.startswith("#"))
        target.write_text(original + "\n" + line + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate LFN"):
            builder.build(repo, REPOSITORY / builder.SOURCE_RELATIVE)
        target.write_text(original.replace("MSuu-4000_MChi-1000", "MSuu-4100_MChi-1000"))
        with self.assertRaisesRegex(ValueError, "Undocumented generated-mass mismatch"):
            builder.build(repo, REPOSITORY / builder.SOURCE_RELATIVE)

    def test_missing_mass_channel_fails_closed(self):
        repo = self.temporary_repo()
        (repo / "test/signalMCFiles/WbWb_4000_1000.txt").unlink()
        with self.assertRaisesRegex(ValueError, "complete six-mode"):
            builder.build(repo, REPOSITORY / builder.SOURCE_RELATIVE)

    def test_committed_catalog_and_audit_reproduce_without_das(self):
        self.assertEqual(json.loads((REPOSITORY / "config/signal_grid_2017.json").read_text()), self.catalog)
        self.assertEqual(json.loads((REPOSITORY / "data/analysis_2017/signal_grid_inventory_audit.json").read_text()), self.audit)
        _, samples = load_catalog(REPOSITORY / "config/signal_grid_2017.json")
        self.assertEqual(len(samples), 114)
        result = subprocess.run([sys.executable, str(REPOSITORY / "build_signal_grid_catalog.py"), "--check"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.audit["DAS_queries"], 0)
        self.assertEqual(self.audit["event_files_opened"], 0)


if __name__ == "__main__":
    unittest.main()
