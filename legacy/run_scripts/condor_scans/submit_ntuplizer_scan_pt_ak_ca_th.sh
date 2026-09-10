#!/bin/bash

set -euo pipefail

mode="${1:---test}"
if [[ "${mode}" != "--test" && "${mode}" != "--full" ]]; then
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

tarball_name="CMSSW_15_0_19_existingOptimization.tgz"
local_tarball="${script_dir}/${tarball_name}"
eos_tarball_dir="/store/user/aji/condor_inputs"
eos_output_dir="/store/user/aji/rootfiles_existingOptimization"

if [[ "${cmssw_release}" != "CMSSW_15_0_19" ]]; then
    echo "ERROR: expected CMSSW_15_0_19, found ${cmssw_release}" >&2
    exit 2
fi

if [[ ! -f "${cmssw_src}/${cfg_relative}" ]]; then
    echo "ERROR: configuration not found:" >&2
    echo "  ${cmssw_src}/${cfg_relative}" >&2
    echo "Adjust cfg_relative in submit_scan.sh if the path differs." >&2
    exit 3
fi

if [[ ! -d "${cmssw_src}/${input_dir_relative}" ]]; then
    echo "ERROR: input-list directory not found:" >&2
    echo "  ${cmssw_src}/${input_dir_relative}" >&2
    echo "Adjust input_dir_relative in both shell scripts if its actual location differs." >&2
    exit 4
fi

if ! voms-proxy-info --exists --valid 00:30 >/dev/null 2>&1; then
    echo "ERROR: no CMS proxy valid for at least 30 minutes." >&2
    echo "Run: voms-proxy-init --valid 192:00 -voms cms" >&2
    exit 5
fi

channels=("WbWb" "WbZt" "WbHt" "ZtZt" "HtHt" "HtZt")
masses=("4000_1000" "6000_2000" "8000_3000")
thrust_ints=(85)

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
            exit 6
        fi
    done
done

mkdir -p logs
: > scan_parameters.tsv

for channel in "${channels[@]}"; do
    for mass in "${masses[@]}"; do
        sample="${channel}_${mass}"

        if [[ "${mode}" == "--test" ]]; then
            pt_values=(100)
            ak_int_values=(4)
        else
            pt_values=($(seq 100 20 400))
            ak_int_values=($(seq 4 2 16))
        fi

        for pt in "${pt_values[@]}"; do
            for ak_int in "${ak_int_values[@]}"; do
                # CA starts at max(AK - 0.2, 0.4).
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

                    for th_int in "${thrust_ints[@]}"; do
                        th="$(printf '0.%02d' "${th_int}")"
                        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                            "${sample}" "${pt}" "${ak}" "${ca}" "${th}" \
                            "${ak_int}" "${ca_int}" "${th_int}" \
                            >> scan_parameters.tsv
                    done
                done
            done
        done
    done
done

job_count="$(wc -l < scan_parameters.tsv)"
echo "Prepared ${job_count} parameter points (${mode})."

echo "Building compact CMSSW tarball..."
tar \
    --exclude-vcs \
    --exclude='*/SuuAnalysis/ExistingOptimization/legacy/*' \
    --exclude-caches-all \
    --exclude='*.root' \
    --exclude='*/tmp/*' \
    --exclude='*/rootfiles_existingOptimization/*' \
    --exclude="${tarball_name}" \
    -czf "${local_tarball}" \
    -C "${cmssw_parent}" \
    "${cmssw_release}"

echo "Tarball size: $(du -h "${local_tarball}" | cut -f1)"

xrdfs root://cmseos.fnal.gov mkdir -p "${eos_tarball_dir}"
xrdfs root://cmseos.fnal.gov mkdir -p "${eos_output_dir}"

echo "Uploading CMSSW tarball to EOS..."
xrdcp --nopbar --force \
    "${local_tarball}" \
    "root://cmseos.fnal.gov/${eos_tarball_dir}/${tarball_name}"

rm -f "${local_tarball}"

echo "Submitting ${job_count} jobs..."
condor_submit submit_ntuplizer_scan_pt_ak_ca_th.jdl
