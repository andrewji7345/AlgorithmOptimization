#!/usr/bin/env python3
"""Build a uniquely named, relocatable CMSSW bundle for sensitivity workers."""
import argparse
import json
import os
from pathlib import Path
import sys
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmssw-base", type=Path, default=os.environ.get("CMSSW_BASE"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.cmssw_base is None:
        parser.error("Set CMSSW_BASE or provide --cmssw-base")
    base = args.cmssw_base.resolve()
    package = base / "src/SuuAnalysis/ExistingOptimization"
    for needed in (base / "lib", package / "test/runSensitivityScan_cfg.py",
                   package / "data/analysis_2017"):
        if not needed.exists():
            parser.error(f"Build/install sensitivity code and corrections before packaging: {needed}")
    target = args.output.resolve()
    if target.exists():
        parser.error(f"Refusing to overwrite bundle: {target}")
    if target == base or base in target.parents:
        parser.error("Bundle output must be outside the CMSSW release")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    excluded_names = {".git", ".pytest_cache", "__pycache__", "legacy", "tmp", ".SCRAM"}
    # .SCRAM is necessary for scram ProjectRename/runtime. Exclude only its
    # transient compiler caches, leaving architecture and project metadata.
    excluded_names.remove(".SCRAM")

    def select(member):
        parts = Path(member.name).parts
        if any(part in excluded_names for part in parts):
            return None
        if member.name.endswith(".root") and "data" not in parts:
            return None
        if member.name.endswith((".pdf", ".pyc", ".log", ".tgz", ".tar.gz")):
            return None
        if any(part.startswith("logs_") or part.startswith("trial_") for part in parts):
            return None
        return member

    try:
        with tarfile.open(temporary, "w:gz", dereference=False) as archive:
            archive.add(base, arcname=base.name, filter=select)
        # Hard-link publishes atomically with no replacement if another
        # packager happened to choose the same destination.
        os.link(temporary, target)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps({"bundle": str(target), "bytes": target.stat().st_size,
                      "cmssw_release": base.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
