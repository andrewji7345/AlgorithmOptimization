#!/usr/bin/env python3
"""DAS resolver retries, provenance, atomic failure handling, and pilot scope."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
import resolve_samples as resolver
from evaluate_sensitivity import validate_campaign


DATASET = "/QCD_Pt_170to300_TuneCP5_13TeV_pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/MINIAODSIM"
FILE_BASE = "/store/mc/RunIISummer20UL17MiniAODv2/QCD_Pt_170to300_TuneCP5_13TeV_pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/240000/"


class SampleResolverTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sample = {"name": "QCDMC_Pt_170to300", "kind": "background",
                       "dataset": DATASET, "dbs_instance": "prod/global"}
        self.catalog = self.root / "catalog.json"
        self.catalog.write_text(json.dumps({"samples": [self.sample], "source_commit": "fixture"}))
        self.campaign = self.root / "campaign.json"
        self.campaign.write_text(json.dumps({"purpose": "pilot", "samples": [
            {**self.sample, "normalization_scope": "representative_subset", "input_files": ["old.root"]}]}))
        self.output = self.root / "lists"

    def test_validated_sorted_unique_files_and_exact_dataset_match(self):
        self.assertEqual(resolver.validate_file_response(FILE_BASE + "b.root\n" + FILE_BASE + "a.root\n", DATASET),
                         [FILE_BASE + "a.root", FILE_BASE + "b.root"])
        for raw in ("", "ERROR unavailable", FILE_BASE + "a.root\n" + FILE_BASE + "a.root\n",
                    FILE_BASE.replace("170to300", "300to470") + "a.root",
                    FILE_BASE + "../a.root"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                resolver.validate_file_response(raw, DATASET)

    def test_transient_failures_retry_with_bounded_timeout_and_literal_query(self):
        good = SimpleNamespace(returncode=0, stdout=FILE_BASE + "a.root\n", stderr="")
        with mock.patch.object(resolver.subprocess, "run", side_effect=[
            subprocess.TimeoutExpired("dasgoclient", 2), good]) as run, mock.patch.object(resolver.time, "sleep") as sleep:
            files, argv = resolver.query_files(self.sample, "/bin/dasgoclient", 2, 2)
        self.assertEqual(files, [FILE_BASE + "a.root"])
        self.assertEqual(argv, ["/bin/dasgoclient", "-query", "file dataset=" + DATASET + " instance=prod/global", "-limit", "0"])
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.kwargs["timeout"], 2)
        sleep.assert_called_once()

    def test_pilot_cap_audits_full_available_count_and_campaign_path(self):
        files = [FILE_BASE + "a.root", FILE_BASE + "b.root"]
        with mock.patch.object(resolver, "query_files", return_value=(files, ["dasgoclient", "-query", "fixture"])):
            result = resolver.resolve_catalog(self.catalog, self.output, campaign_path=self.campaign, max_files=1)
        self.assertEqual(result["file_count"], 1)
        path = self.output / "QCDMC_Pt_170to300.txt"
        content = path.read_text()
        self.assertEqual(content, "root://cmsxrootd.fnal.gov/" + FILE_BASE + "a.root\n")
        provenance = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual(provenance["available_files"], 2)
        self.assertEqual(provenance["selected_files"], 1)
        self.assertFalse(provenance["complete_file_coverage"])
        self.assertEqual(provenance["input_list_sha256"], hashlib.sha256(content.encode()).hexdigest())
        updated = json.loads(self.campaign.read_text())["samples"][0]
        self.assertEqual(updated["input_file_list"], "lists/QCDMC_Pt_170to300.txt")
        self.assertNotIn("input_files", updated)

    def test_query_failure_preserves_existing_lists_and_campaign(self):
        self.output.mkdir()
        existing = self.output / "QCDMC_Pt_170to300.txt"
        existing.write_text("previous verified input\n")
        previous = self.campaign.read_bytes()
        with mock.patch.object(resolver, "query_files", side_effect=RuntimeError("No valid DAS response")):
            with self.assertRaisesRegex(RuntimeError, "No lists or campaign were updated"):
                resolver.resolve_catalog(self.catalog, self.output, campaign_path=self.campaign)
        self.assertEqual(existing.read_text(), "previous verified input\n")
        self.assertEqual(self.campaign.read_bytes(), previous)

    def test_production_cannot_receive_a_truncated_file_list(self):
        data = json.loads(self.campaign.read_text())
        data["purpose"] = "production"
        data["samples"][0]["normalization_scope"] = "full_dataset"
        self.campaign.write_text(json.dumps(data))
        with mock.patch.object(resolver, "query_files") as query:
            with self.assertRaisesRegex(ValueError, "pilot/representative_subset"):
                resolver.resolve_catalog(self.catalog, self.output, campaign_path=self.campaign, max_files=1)
        query.assert_not_called()

    def test_catalog_rejects_wildcards_and_unknown_sample_requests(self):
        with self.assertRaisesRegex(ValueError, "unknown catalog samples"):
            resolver.resolve_catalog(self.catalog, self.output, names=["missing"])
        self.sample["dataset"] = DATASET.replace("-v1", "-v*")
        self.catalog.write_text(json.dumps({"samples": [self.sample]}))
        with self.assertRaisesRegex(ValueError, "wildcards"):
            resolver.load_catalog(self.catalog)

    def test_committed_pilot_contains_exact_23_backgrounds_and_3_signals(self):
        campaign = json.loads((REPOSITORY / "config/sensitivity_2017.json").read_text())
        backgrounds = json.loads((REPOSITORY / "config/backgrounds_2017.json").read_text())["samples"]
        signals = [sample for sample in campaign["samples"] if sample["kind"] == "signal"]
        actual_backgrounds = [sample for sample in campaign["samples"] if sample["kind"] == "background"]
        self.assertEqual(len(backgrounds), 23)
        self.assertEqual({sample["name"] for sample in actual_backgrounds}, {sample["name"] for sample in backgrounds})
        self.assertEqual(campaign["required_signals"], ["WbWb_4000_1000", "WbWb_6000_2000", "WbWb_8000_3000"])
        self.assertEqual(len(signals), 3)
        self.assertEqual(campaign["purpose"], "pilot")
        self.assertEqual(campaign["minimum_background_effective_events"], 10)
        for sample in campaign["samples"]:
            self.assertEqual(sample["normalization_scope"], "representative_subset")
            self.assertEqual(sample["sum_gen_weights"], "metadata")
            self.assertEqual(sample["generated_events"], "metadata")
            sample["files"] = [sample["name"] + ".root"]
        validate_campaign(campaign)


if __name__ == "__main__":
    unittest.main()
