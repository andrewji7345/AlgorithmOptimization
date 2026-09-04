#!/bin/bash

set -euo pipefail

mode="${1:---test}"
if (( $# > 1 )) || [[ "${mode}" != "--test" && "${mode}" != "--full" ]]; then
    echo "Usage: $0 [--test|--full]" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

cmssw_src="$(readlink -f "${HOME}/nobackup/research/CMSSW_15_0_19/src")"
cmssw_dir="$(dirname "${cmssw_src}")"
cmssw_parent="$(dirname "${cmssw_dir}")"
cmssw_release="$(basename "${cmssw_dir}")"

cfg_relative="SuuAnalysis/ExistingOptimization/test/runExistingOptimizationNtuplizer_cfg.py"
input_dir_relative="SuuAnalysis/ExistingOptimization/test/signalMCFiles"

manifest_name="scan_parameters_two_threshold.tsv"
log_dir="logs_two_threshold"
submit_file="submit_ntuplizer_scan_two_threshold.jdl"
tarball_name="CMSSW_15_0_19_existingOptimization_two_threshold.tgz"
# Build outside the CMSSW tree so this scan and the ParticleTransformer target
# production can package concurrently without reading each other's live
# archive.
local_tarball="$(mktemp "/tmp/${tarball_name%.tgz}.XXXXXX.tgz")"
trap 'rm -f "${local_tarball}"' EXIT
eos_tarball_dir="/store/user/aji/condor_inputs"
eos_output_dir="/store/user/aji/rootfiles_existingOptimization"

if [[ "${cmssw_release}" != "CMSSW_15_0_19" ]]; then
    echo "ERROR: expected CMSSW_15_0_19, found ${cmssw_release}" >&2
    exit 2
fi

if [[ ! -f "${cmssw_src}/${cfg_relative}" ]]; then
    echo "ERROR: configuration not found: ${cmssw_src}/${cfg_relative}" >&2
    exit 3
fi

if [[ ! -d "${cmssw_src}/${input_dir_relative}" ]]; then
    echo "ERROR: input-list directory not found: ${cmssw_src}/${input_dir_relative}" >&2
    exit 4
fi

for command_name in voms-proxy-info tar xrdfs xrdcp condor_submit; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 5
    fi
done

if ! voms-proxy-info --exists --valid 00:30 >/dev/null 2>&1; then
    echo "ERROR: no CMS proxy valid for at least 30 minutes." >&2
    echo "Run: voms-proxy-init --valid 192:00 -voms cms" >&2
    exit 6
fi

channels=("WbWb" "WbZt" "WbHt" "ZtZt" "HtHt" "HtZt")
masses=("4000_1000" "6000_2000" "8000_3000")

event_jet_pt_cut=300
min_event_jets=4
thrust_int=85
thrust="0.85"

if [[ "${mode}" == "--test" ]]; then
    channels=("WbWb")
    masses=("4000_1000")
fi

for channel in "${channels[@]}"; do
    for mass in "${masses[@]}"; do
        sample="${channel}_${mass}"
        input_list="${cmssw_src}/${input_dir_relative}/${sample}.txt"
        if [[ ! -s "${input_list}" ]]; then
            echo "ERROR: missing or empty input list: ${input_list}" >&2
            exit 7
        fi
    done
done

mkdir -p "${log_dir}"
: > "${manifest_name}"

for channel in "${channels[@]}"; do
    for mass in "${masses[@]}"; do
        sample="${channel}_${mass}"

        if [[ "${mode}" == "--test" ]]; then
            lower_pt_values=(100)
            ak_int_values=(4)
        else
            lower_pt_values=($(seq 100 20 300))
            ak_int_values=($(seq 4 2 16))
        fi

        for lower_pt in "${lower_pt_values[@]}"; do
            for ak_int in "${ak_int_values[@]}"; do
                # Enforce CA >= AK - 0.2, with both radii bounded below by 0.4.
                ca_start_int=$((ak_int - 2))
                if (( ca_start_int < 4 )); then
                    ca_start_int=4
                fi

                if [[ "${mode}" == "--test" ]]; then
                    ca_int_values=("${ca_start_int}")
                else
                    ca_int_values=($(seq "${ca_start_int}" 2 16))
                fi

                ak="$(printf '%d.%d' "$((ak_int / 10))" "$((ak_int % 10))")"

                for ca_int in "${ca_int_values[@]}"; do
                    ca="$(printf '%d.%d' "$((ca_int / 10))" "$((ca_int % 10))")"
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "${sample}" "${lower_pt}" "${event_jet_pt_cut}" \
                        "${min_event_jets}" "${ak}" "${ca}" "${thrust}" \
                        "${ak_int}" "${ca_int}" "${thrust_int}" \
                        >> "${manifest_name}"
                done
            done
        done
    done
done

job_count="$(wc -l < "${manifest_name}")"
if [[ "${mode}" == "--test" ]]; then
    expected_job_count=1
else
    # 18 samples x 11 lower thresholds x 34 allowed AK/CA pairs.
    expected_job_count=6732
fi

if (( job_count != expected_job_count )); then
    echo "ERROR: generated ${job_count} jobs; expected ${expected_job_count}" >&2
    exit 8
fi

echo "Prepared ${job_count} two-threshold parameter points (${mode})."
echo "Manifest: ${script_dir}/${manifest_name}"

echo "Building compact CMSSW tarball..."
tar \
    --exclude-vcs \
    --exclude-caches-all \
    --exclude='*.root' \
    --exclude='*.tgz' \
    --exclude='*/tmp/*' \
    --exclude='*/rootfiles_existingOptimization/*' \
    --exclude='*/SuuAnalysis/ParticleTransformer/*' \
    --exclude='*/SuuAnalysis/ExistingOptimization/results/*' \
    --exclude='*/SuuAnalysis/ExistingOptimization/run_scripts/*' \
    -czf "${local_tarball}" \
    -C "${cmssw_parent}" \
    "${cmssw_release}"

echo "Tarball size: $(du -h "${local_tarball}" | cut -f1)"

xrdfs root://cmseos.fnal.gov mkdir -p "${eos_tarball_dir}"
xrdfs root://cmseos.fnal.gov mkdir -p "${eos_output_dir}"

echo "Uploading the two-threshold CMSSW tarball to EOS..."
xrdcp --nopbar --force \
    "${local_tarball}" \
    "root://cmseos.fnal.gov/${eos_tarball_dir}/${tarball_name}"

rm -f "${local_tarball}"

echo "Submitting ${job_count} jobs..."
condor_submit "${submit_file}"
