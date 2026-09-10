#!/bin/bash

set -euo pipefail

if (( $# != 10 )); then
    echo "Usage: $0 SAMPLE LOWER_PT EVENT_PT MIN_EVENT_JETS AK CA TH AK_INT CA_INT TH_INT" >&2
    exit 1
fi

sample="$1"
lower_pt="$2"
event_pt="$3"
min_event_jets="$4"
ak="$5"
ca="$6"
th="$7"
ak_int="$8"
ca_int="$9"
th_int="${10}"

cmssw_release="CMSSW_15_0_19"
tarball_name="CMSSW_15_0_19_existingOptimization_two_threshold.tgz"
tarball_url="root://cmseos.fnal.gov//store/user/aji/condor_inputs/${tarball_name}"
eos_output_dir="root://cmseos.fnal.gov//store/user/aji/rootfiles_existingOptimization"

cfg_relative="SuuAnalysis/ExistingOptimization/test/runExistingOptimizationNtuplizer_cfg.py"
input_list_relative="SuuAnalysis/ExistingOptimization/test/signalMCFiles/${sample}.txt"

# Keep the complete legacy stem and add a self-describing, parseable suffix.
output_name="${sample}_pt${lower_pt}_ak${ak_int}_ca${ca_int}_th${th_int}_two_threshold_${event_pt}_${lower_pt}.root"

echo "Starting job at $(date)"
echo "Host: $(hostname)"
echo "OS: $(cat /etc/redhat-release)"
echo "Scratch directory: ${_CONDOR_SCRATCH_DIR}"
echo "Parameters: sample=${sample}, lowerPt=${lower_pt}, eventPt=${event_pt}, minEventJets=${min_event_jets}, ak=${ak}, ca=${ca}, th=${th}"
echo "Output: ${output_name}"

cd "${_CONDOR_SCRATCH_DIR}"

echo "Downloading ${tarball_url}"
xrdcp --nopbar --force "${tarball_url}" "${tarball_name}"

tar -xzf "${tarball_name}"
rm -f "${tarball_name}"

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "${cmssw_release}/src"

# Repair paths embedded in the precompiled CMSSW area after relocation.
scramv1 b ProjectRename
eval "$(scramv1 runtime -sh)"

cfg_path="${CMSSW_BASE}/src/${cfg_relative}"
input_list="${CMSSW_BASE}/src/${input_list_relative}"

if [[ ! -f "${cfg_path}" ]]; then
    echo "ERROR: configuration not found: ${cfg_path}" >&2
    exit 2
fi

if [[ ! -f "${input_list}" ]]; then
    echo "ERROR: input list not found: ${input_list}" >&2
    exit 3
fi

cd "${_CONDOR_SCRATCH_DIR}"

cmsRun "${cfg_path}" \
    inputRootFiles="${input_list}" \
    jetPtCut="${lower_pt}" \
    eventJetPtCut="${event_pt}" \
    minEventJets="${min_event_jets}" \
    akRadius="${ak}" \
    caRadius="${ca}" \
    cosThrust="${th}" \
    outputRootFile="${output_name}"

if [[ ! -s "${output_name}" ]]; then
    echo "ERROR: cmsRun did not produce a nonempty ${output_name}" >&2
    exit 4
fi

destination="${eos_output_dir}/${output_name}"
echo "Copying output to ${destination}"
xrdcp --nopbar --force "${output_name}" "${destination}"

# Prevent Condor from trying to return a second copy through NFS.
rm -f "${output_name}"

echo "Job completed successfully at $(date)"
