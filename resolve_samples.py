#!/usr/bin/env python3
"""Resolve an explicit CMS dataset catalog into audited, nonempty MiniAOD lists."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parent
DATASET_RE = re.compile(r"^/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/(?:MINIAODSIM|USER)$")
FILE_RE = re.compile(r"^/store/(?:mc|user|group)/[A-Za-z0-9_./-]+\.root$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_catalog(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    catalog = json.loads(path.read_text())
    samples = catalog.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError("catalog must contain a nonempty samples list")
    names, datasets = set(), set()
    for sample in samples:
        name, dataset = sample.get("name", ""), sample.get("dataset", "")
        if not isinstance(name, str) or NAME_RE.fullmatch(name) is None or name in names:
            raise ValueError("catalog sample names must be safe, nonempty and unique")
        if not isinstance(dataset, str) or DATASET_RE.fullmatch(dataset) is None or dataset in datasets:
            raise ValueError(f"{name}: explicit, unique MiniAOD dataset required; wildcards are not allowed")
        if sample.get("dbs_instance", "prod/global") not in ("prod/global", "prod/phys03"):
            raise ValueError(f"{name}: dbs_instance must be prod/global or prod/phys03")
        names.add(name)
        datasets.add(dataset)
    return catalog, samples


def validate_file_response(raw: str, dataset: str) -> list[str]:
    files = [line.strip() for line in raw.splitlines() if line.strip()]
    if not files:
        raise ValueError(f"DAS returned no files for {dataset}")
    primary, processing, tier = dataset.strip("/").split("/")
    for name in files:
        if FILE_RE.fullmatch(name) is None or ".." in Path(name).parts:
            raise ValueError(f"DAS returned a malformed file entry: {name[:200]!r}")
        if tier == "MINIAODSIM" and name.startswith("/store/mc/"):
            parts = Path(name).parts
            if len(parts) < 9 or parts[4] != primary or parts[5] != tier or f"{parts[3]}-{parts[6]}" != processing:
                raise ValueError(f"DAS file does not belong to requested dataset: {name}")
    if len(set(files)) != len(files):
        raise ValueError(f"DAS returned duplicate file entries for {dataset}")
    return sorted(files)


def query_files(sample: Mapping[str, Any], executable: str, timeout_seconds: float,
                attempts: int, retry_delay_seconds: float = 2.0) -> tuple[list[str], list[str]]:
    query = f"file dataset={sample['dataset']} instance={sample.get('dbs_instance', 'prod/global')}"
    command = [executable, "-query", query, "-limit", "0"]
    failure = ""
    for attempt in range(attempts):
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_seconds)
            if result.returncode:
                raise RuntimeError(f"dasgoclient exited {result.returncode}: {result.stderr[-1000:]}")
            return validate_file_response(result.stdout, sample["dataset"]), command
        except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError) as error:
            failure = str(error)
            if attempt + 1 < attempts:
                time.sleep(min(retry_delay_seconds * (attempt + 1), 30))
    raise RuntimeError(f"{sample['name']}: DAS failed after {attempts} attempts: {failure}")


def resolve_catalog(catalog_path: Path, output_dir: Path, *, campaign_path: Path | None = None,
                    names: Sequence[str] | None = None, max_files: int | None = None,
                    redirector: str = "root://cmsxrootd.fnal.gov/", executable: str = "dasgoclient",
                    timeout_seconds: float = 90, attempts: int = 3, workers: int = 4) -> dict[str, Any]:
    catalog_path, output_dir = catalog_path.resolve(), output_dir.resolve()
    catalog, all_samples = load_catalog(catalog_path)
    if attempts < 1 or workers < 1 or workers > 8 or timeout_seconds <= 0 or (max_files is not None and max_files <= 0):
        raise ValueError("attempts, timeout and max_files must be positive; workers must be in [1,8]")
    if re.fullmatch(r"root://[A-Za-z0-9.-]+(?::[0-9]+)?/?", redirector) is None:
        raise ValueError("redirector must be a root://host[:port]/ URL")
    selected_names = set(names) if names else {sample["name"] for sample in all_samples}
    missing = selected_names - {sample["name"] for sample in all_samples}
    if missing:
        raise ValueError("unknown catalog samples: " + ", ".join(sorted(missing)))
    samples = [sample for sample in all_samples if sample["name"] in selected_names]
    campaign = None
    if campaign_path is not None:
        campaign_path = campaign_path.resolve()
        campaign = json.loads(campaign_path.read_text())
        matching = {sample["name"]: sample for sample in campaign["samples"]}
        if not selected_names <= set(matching):
            raise ValueError("every resolved catalog sample must be present in the campaign")
        for sample in samples:
            target = matching[sample["name"]]
            if target.get("dataset") != sample["dataset"]:
                raise ValueError(f"{sample['name']}: campaign dataset differs from catalog")
            if max_files is not None and (campaign.get("purpose") != "pilot"
                                         or target.get("normalization_scope") != "representative_subset"):
                raise ValueError("--max-files requires pilot/representative_subset normalization in the campaign")
    # No output changes occur until every requested query succeeds and validates.
    resolved = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(query_files, sample, executable, timeout_seconds, attempts): sample for sample in samples}
        errors = []
        for future in as_completed(pending):
            sample = pending[future]
            try:
                resolved[sample["name"]] = future.result()
                print(f"Resolved {sample['name']}: {len(resolved[sample['name']][0])} files", file=sys.stderr, flush=True)
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError("No lists or campaign were updated:\n" + "\n".join(errors))
    timestamp = datetime.now(timezone.utc).isoformat()
    source_hash = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    entries = []
    for sample in samples:
        all_files, command = resolved[sample["name"]]
        chosen = all_files if max_files is None else all_files[:max_files]
        uris = [redirector.rstrip("/") + "/" + name for name in chosen]
        content = "\n".join(uris) + "\n"
        list_path = output_dir / (sample["name"] + ".txt")
        provenance = {
            "schema_version": 1, "resolved_at_utc": timestamp, "sample": sample["name"],
            "dataset": sample["dataset"], "dbs_instance": sample.get("dbs_instance", "prod/global"),
            "catalog": str(catalog_path), "catalog_sha256": source_hash,
            "source_repository": catalog.get("source_repository"), "source_commit": catalog.get("source_commit"),
            "query_argv": command, "redirector": redirector.rstrip("/") + "/",
            "available_files": len(all_files), "selected_files": len(chosen),
            "max_files": max_files, "complete_file_coverage": len(chosen) == len(all_files),
            "selection": "all_files" if max_files is None else "lexicographic_first_n_pilot_subset",
            "input_list_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "normalization_note": "File coverage is not a signed generator-weight denominator; use all preselection genWeights for the declared scope.",
        }
        atomic_text(list_path, content)
        provenance_path = list_path.with_suffix(".json")
        atomic_text(provenance_path, json.dumps(provenance, indent=2, allow_nan=False) + "\n")
        if campaign is not None:
            target = next(item for item in campaign["samples"] if item["name"] == sample["name"])
            target["input_file_list"] = os.path.relpath(list_path, campaign_path.parent)
            target["input_provenance"] = os.path.relpath(provenance_path, campaign_path.parent)
            target.pop("input_files", None)
        entries.append({"name": sample["name"], "file_list": str(list_path),
                        "available_files": len(all_files), "selected_files": len(chosen)})
    if campaign is not None:
        atomic_text(campaign_path, json.dumps(campaign, indent=2, allow_nan=False) + "\n")
    return {"catalog": str(catalog_path), "samples": entries, "sample_count": len(entries),
            "file_count": sum(entry["selected_files"] for entry in entries),
            "campaign": str(campaign_path) if campaign_path is not None else None}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=REPOSITORY / "config/backgrounds_2017.json")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "test/backgroundMCFiles")
    parser.add_argument("--campaign", type=Path, default=REPOSITORY / "config/sensitivity_2017.json")
    parser.add_argument("--no-update-campaign", action="store_true", help="Resolve lists without changing a campaign")
    parser.add_argument("--sample", action="append", default=[], help="Resolve only named catalog samples; repeat")
    parser.add_argument("--max-files", type=int, help="Explicit deterministic pilot cap; default resolves every file")
    parser.add_argument("--redirector", default="root://cmsxrootd.fnal.gov/")
    parser.add_argument("--dasgoclient", default=shutil.which("dasgoclient") or "/cvmfs/cms.cern.ch/common/dasgoclient")
    parser.add_argument("--timeout-seconds", type=float, default=90)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        result = resolve_catalog(args.catalog, args.output_dir,
                                 campaign_path=None if args.no_update_campaign else args.campaign,
                                 names=args.sample, max_files=args.max_files, redirector=args.redirector,
                                 executable=args.dasgoclient, timeout_seconds=args.timeout_seconds,
                                 attempts=args.attempts, workers=args.workers)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        parser.exit(2, f"sample resolution failed: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
