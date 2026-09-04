#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: $0 SAMPLE AK_RADIUS AK_TAG MAX_EVENTS OUTPUT_NAME TARBALL_NAME" >&2
}

if (( $# != 6 )); then
    usage
    exit 1
fi

sample="$1"
ak_radius="$2"
ak_tag="$3"
max_events="$4"
output_name="$5"
tarball_name="$6"

if [[ ! "${sample}" =~ ^[A-Za-z0-9]+_[0-9]+_[0-9]+$ ]]; then
    echo "ERROR: invalid sample name: ${sample}" >&2
    exit 2
fi

if [[ ! "${ak_radius}" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]] ||
   [[ "${ak_radius}" =~ ^0(\.0+)?$ ]] ||
   (( ${#ak_radius} > 16 )); then
    echo "ERROR: AK_RADIUS must be positive: ${ak_radius}" >&2
    exit 3
fi

expected_ak_tag="${ak_radius//./p}"
if [[ "${ak_tag}" != "${expected_ak_tag}" ]]; then
    echo "ERROR: AK_TAG ${ak_tag} does not match AK_RADIUS ${ak_radius}" >&2
    exit 4
fi

if [[ ! "${max_events}" =~ ^[1-9][0-9]*$ ]] ||
   (( ${#max_events} > 10 )) ||
   (( 10#${max_events} > 2147483647 )); then
    echo "ERROR: MAX_EVENTS must be a positive 32-bit integer: ${max_events}" >&2
    exit 5
fi

if [[ ! "${output_name}" =~ ^[A-Za-z0-9_.-]+\.root$ ]] ||
   [[ "${output_name}" == */* ]] ||
   (( ${#output_name} > 255 )); then
    echo "ERROR: invalid output basename: ${output_name}" >&2
    exit 6
fi

if [[ ! "${tarball_name}" =~ ^[A-Za-z0-9_.-]+\.tgz$ ]] ||
   [[ "${tarball_name}" == */* ]] ||
   (( ${#tarball_name} > 255 )); then
    echo "ERROR: invalid tarball basename: ${tarball_name}" >&2
    exit 7
fi

if [[ -z "${_CONDOR_SCRATCH_DIR:-}" ]] ||
   [[ ! -d "${_CONDOR_SCRATCH_DIR}" ]]; then
    echo "ERROR: _CONDOR_SCRATCH_DIR is not a valid directory." >&2
    exit 8
fi

cmssw_release="CMSSW_15_0_19"
tarball_url="root://cmseos.fnal.gov//store/user/aji/condor_inputs/${tarball_name}"
eos_output_dir="root://cmseos.fnal.gov//store/user/aji/rootfiles_existingOptimization_compact"

cfg_relative="SuuAnalysis/ExistingOptimization/test/runCompactOptimizationScan_cfg.py"
input_list_relative="SuuAnalysis/ExistingOptimization/test/signalMCFiles/${sample}.txt"

echo "Starting compact grouped job at $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Scratch directory: ${_CONDOR_SCRATCH_DIR}"
echo "Parameters: sample=${sample}, akRadius=${ak_radius}, maxEvents=${max_events}"
echo "Output: ${output_name}"

cd "${_CONDOR_SCRATCH_DIR}"

echo "Downloading ${tarball_url}"
xrdcp --nopbar "${tarball_url}" "${tarball_name}"

tar -xzf "${tarball_name}"
rm -f -- "${tarball_name}"

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "${cmssw_release}/src"

# Repair paths embedded in the precompiled CMSSW area after relocation.
scramv1 b ProjectRename
eval "$(scramv1 runtime -sh)"

cfg_path="${CMSSW_BASE}/src/${cfg_relative}"
input_list="${CMSSW_BASE}/src/${input_list_relative}"

if [[ ! -f "${cfg_path}" ]]; then
    echo "ERROR: configuration not found: ${cfg_path}" >&2
    exit 9
fi

if [[ ! -s "${input_list}" ]]; then
    echo "ERROR: input list is missing or empty: ${input_list}" >&2
    exit 10
fi

cd "${_CONDOR_SCRATCH_DIR}"

cmsRun "${cfg_path}" \
    inputRootFiles="${input_list}" \
    outputRootFile="${output_name}" \
    maxEvents="${max_events}" \
    akRadius="${ak_radius}"

if [[ ! -s "${output_name}" ]]; then
    echo "ERROR: cmsRun did not produce a nonempty ${output_name}" >&2
    exit 11
fi

validator_path="${CMSSW_BASE}/src/SuuAnalysis/ExistingOptimization/test/validate_compact_scan.py"
if [[ ! -f "${validator_path}" ]]; then
    echo "ERROR: compact scan validator not found: ${validator_path}" >&2
    exit 12
fi

echo "Validating compact ROOT schema and invariants"
python3 "${validator_path}" --strict-branches "${output_name}"

destination="${eos_output_dir}/${output_name}"
echo "Copying output to ${destination}"
# Output names contain a per-submission tag. Deliberately omit --force so a
# reused tag cannot silently overwrite an earlier result.
xrdcp --nopbar "${output_name}" "${destination}"

# Condor should return only the log streams; EOS owns the successful ROOT file.
rm -f -- "${output_name}"

echo "Job completed successfully at $(date --iso-8601=seconds)"
