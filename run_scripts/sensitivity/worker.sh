#!/bin/bash
set -euo pipefail

scratch="${_CONDOR_SCRATCH_DIR:-${PWD}}"
cd "${scratch}"
trap 'status=$?; if (( status != 0 )); then cd "${scratch}"; python3 worker.py failure || true; fi; exit "${status}"' EXIT

source /cvmfs/cms.cern.ch/cmsset_default.sh
python3 worker.py prepare
release="$(cat release_name.txt)"
cd "${scratch}/${release}/src"
scramv1 b ProjectRename
eval "$(scramv1 runtime -sh)"
cd "${scratch}"
python3 worker.py run
