#!/bin/bash

set -euo pipefail
export LC_ALL=C

usage() {
    cat <<'EOF'
Usage: submit_ntuplizer_scan_compact_grouped.sh [OPTIONS]

Build a compact grouped scan manifest and, unless --dry-run is supplied,
package CMSSW, upload it to EOS, and submit the jobs.

Options:
  --test                 One sample at AK R=0.8 (default)
  --full                 All discovered samples at AK R=0.4,...,1.6
  --dry-run              Validate and write the manifest only
  --max-events N         Positive event limit per sample (default: 1000)
  --ak-radii CSV         Override the mode's comma-separated AK radii
  --tag TAG              Unique output tag (default: generated UTC tag)
  -h, --help             Show this help
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

mode="test"
mode_was_set=false
dry_run=false
max_events=1000
ak_radii_csv=""
scan_tag=""

while (( $# > 0 )); do
    case "$1" in
        --test|--full)
            requested_mode="${1#--}"
            if [[ "${mode_was_set}" == true && "${requested_mode}" != "${mode}" ]]; then
                die "--test and --full are mutually exclusive"
            fi
            mode="${requested_mode}"
            mode_was_set=true
            shift
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --max-events)
            (( $# >= 2 )) || die "--max-events requires a value"
            max_events="$2"
            shift 2
            ;;
        --max-events=*)
            max_events="${1#*=}"
            shift
            ;;
        --ak-radii)
            (( $# >= 2 )) || die "--ak-radii requires a value"
            ak_radii_csv="$2"
            shift 2
            ;;
        --ak-radii=*)
            ak_radii_csv="${1#*=}"
            shift
            ;;
        --tag)
            (( $# >= 2 )) || die "--tag requires a value"
            scan_tag="$2"
            shift 2
            ;;
        --tag=*)
            scan_tag="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            die "unknown option: $1"
            ;;
    esac
done

if [[ ! "${max_events}" =~ ^[1-9][0-9]*$ ]] ||
   (( ${#max_events} > 10 )) ||
   (( 10#${max_events} > 2147483647 )); then
    die "--max-events must be a positive 32-bit integer"
fi

if [[ -z "${scan_tag}" ]]; then
    scan_tag="${mode}_nev${max_events}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
elif [[ ! "${scan_tag}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
     (( ${#scan_tag} > 64 )); then
    die "--tag must be at most 64 characters using letters, digits, '.', '_' or '-'"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

# Resolve from this script rather than HOME so both /uscms/home and
# /uscms/homes/a mounts work without special cases.
cmssw_src="$(readlink -f "${script_dir}/../../../..")"
cmssw_dir="$(dirname "${cmssw_src}")"
cmssw_parent="$(dirname "${cmssw_dir}")"
cmssw_release="$(basename "${cmssw_dir}")"

cfg_relative="SuuAnalysis/ExistingOptimization/test/runCompactOptimizationScan_cfg.py"
input_dir_relative="SuuAnalysis/ExistingOptimization/test/signalMCFiles"
input_dir="${cmssw_src}/${input_dir_relative}"

manifest_name="scan_parameters_compact_grouped.tsv"
manifest_path="${script_dir}/${manifest_name}"
log_dir="${script_dir}/logs_compact_grouped"
submit_file="submit_ntuplizer_scan_compact_grouped.jdl"

eos_host="root://cmseos.fnal.gov"
eos_tarball_dir="/store/user/aji/condor_inputs"
eos_output_dir="/store/user/aji/rootfiles_existingOptimization_compact"
tarball_name="${cmssw_release}_existingOptimization_compact_${scan_tag}.tgz"

[[ "${cmssw_release}" == "CMSSW_15_0_19" ]] ||
    die "expected CMSSW_15_0_19, found ${cmssw_release}"
[[ -f "${cmssw_src}/${cfg_relative}" ]] ||
    die "configuration not found: ${cmssw_src}/${cfg_relative}"
[[ -d "${input_dir}" ]] ||
    die "input-list directory not found: ${input_dir}"
[[ -f "${script_dir}/${submit_file}" ]] ||
    die "submit description not found: ${script_dir}/${submit_file}"
[[ -f "${script_dir}/run_ntuplizer_scan_compact_grouped.sh" ]] ||
    die "worker not found: ${script_dir}/run_ntuplizer_scan_compact_grouped.sh"

shopt -s nullglob
all_input_lists=("${input_dir}"/*.txt)
shopt -u nullglob

(( ${#all_input_lists[@]} == 114 )) ||
    die "discovered ${#all_input_lists[@]} sample lists in ${input_dir}; expected 114"

declare -A seen_samples=()
for input_list in "${all_input_lists[@]}"; do
    [[ -s "${input_list}" ]] || die "input list is empty: ${input_list}"

    sample="$(basename "${input_list}" .txt)"
    [[ "${sample}" =~ ^[A-Za-z0-9]+_[0-9]+_[0-9]+$ ]] ||
        die "unexpected sample-list name: ${input_list}"
    [[ -z "${seen_samples[${sample}]:-}" ]] ||
        die "duplicate sample name: ${sample}"
    seen_samples["${sample}"]=1

    has_input=false
    while IFS= read -r line || [[ -n "${line}" ]]; do
        trimmed="${line#"${line%%[![:space:]]*}"}"
        if [[ -n "${trimmed}" && "${trimmed:0:1}" != "#" ]]; then
            has_input=true
            break
        fi
    done < "${input_list}"
    [[ "${has_input}" == true ]] ||
        die "input list has no data-file entries: ${input_list}"
done

if [[ "${mode}" == "test" ]]; then
    test_input_list="${input_dir}/WbWb_4000_1000.txt"
    [[ -s "${test_input_list}" ]] || die "test sample is unavailable: ${test_input_list}"
    selected_input_lists=("${test_input_list}")
    default_ak_radii=(0.8)
else
    selected_input_lists=("${all_input_lists[@]}")
    default_ak_radii=(0.4 0.6 0.8 1.0 1.2 1.4 1.6)
fi

if [[ -n "${ak_radii_csv}" ]]; then
    [[ "${ak_radii_csv}" != ,* && "${ak_radii_csv}" != *, &&
       "${ak_radii_csv}" != *,,* ]] ||
        die "--ak-radii must be a nonempty comma-separated list"
    IFS=',' read -r -a ak_radii <<< "${ak_radii_csv}"
else
    ak_radii=("${default_ak_radii[@]}")
fi

(( ${#ak_radii[@]} > 0 )) || die "no AK radii were selected"

declare -A seen_ak_radii=()
for ak_radius in "${ak_radii[@]}"; do
    if [[ ! "${ak_radius}" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]] ||
       [[ "${ak_radius}" =~ ^0(\.0+)?$ ]] ||
       (( ${#ak_radius} > 16 )); then
        die "invalid AK radius: ${ak_radius}"
    fi
    canonical_ak_radius="${ak_radius}"
    if [[ "${canonical_ak_radius}" == *.* ]]; then
        while [[ "${canonical_ak_radius}" == *0 ]]; do
            canonical_ak_radius="${canonical_ak_radius%0}"
        done
        canonical_ak_radius="${canonical_ak_radius%.}"
    fi
    [[ -z "${seen_ak_radii[${canonical_ak_radius}]:-}" ]] ||
        die "duplicate AK radius: ${ak_radius}"
    seen_ak_radii["${canonical_ak_radius}"]=1
done

manifest_tmp="$(mktemp "${script_dir}/.${manifest_name}.XXXXXX")"
local_tarball=""
cleanup() {
    if [[ -n "${manifest_tmp}" && -e "${manifest_tmp}" ]]; then
        rm -f -- "${manifest_tmp}"
    fi
    if [[ -n "${local_tarball}" && -e "${local_tarball}" ]]; then
        rm -f -- "${local_tarball}"
    fi
}
trap cleanup EXIT

for input_list in "${selected_input_lists[@]}"; do
    sample="$(basename "${input_list}" .txt)"
    for ak_radius in "${ak_radii[@]}"; do
        ak_tag="${ak_radius//./p}"
        output_name="compactOptimization_${scan_tag}_${sample}_akR${ak_tag}_nev${max_events}.root"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${sample}" "${ak_radius}" "${ak_tag}" "${max_events}" \
            "${output_name}" "${tarball_name}" >> "${manifest_tmp}"
    done
done

expected_jobs=$(( ${#selected_input_lists[@]} * ${#ak_radii[@]} ))
job_count="$(wc -l < "${manifest_tmp}")"
(( job_count == expected_jobs )) ||
    die "generated ${job_count} jobs; expected ${expected_jobs}"

mv -- "${manifest_tmp}" "${manifest_path}"
manifest_tmp=""
mkdir -p "${log_dir}"

echo "Prepared ${job_count} compact grouped jobs."
echo "Mode: ${mode}"
echo "Discovered sample lists: ${#all_input_lists[@]}"
echo "Selected sample lists: ${#selected_input_lists[@]}"
echo "AK radii: ${ak_radii[*]}"
echo "Maximum events per sample: ${max_events}"
echo "Scan tag: ${scan_tag}"
echo "Manifest: ${manifest_path}"
echo "EOS output directory: ${eos_host}/${eos_output_dir}"
echo "First manifest rows:"
sed -n '1,5p' "${manifest_path}"

if [[ "${dry_run}" == true ]]; then
    echo "Dry run complete: no tarball was built or uploaded and no jobs were submitted."
    exit 0
fi

for command_name in voms-proxy-info tar xrdfs xrdcp condor_submit; do
    command -v "${command_name}" >/dev/null 2>&1 ||
        die "required command is unavailable: ${command_name}"
done

if ! voms-proxy-info --exists --valid 01:00 >/dev/null 2>&1; then
    die "no CMS proxy is valid for at least one hour; initialize one before submitting"
fi

local_tarball="$(mktemp "/tmp/${tarball_name%.tgz}.XXXXXX.tgz")"

echo "Building compact CMSSW tarball..."
tar \
    --exclude-vcs \
    --exclude-caches-all \
    --exclude='*.root' \
    --exclude='*.tgz' \
    --exclude='*/tmp/*' \
    --exclude='*/rootfiles_existingOptimization*/*' \
    --exclude='*/SuuAnalysis/ParticleTransformer/*' \
    --exclude='*/SuuAnalysis/ExistingOptimization/results/*' \
    --exclude='*/SuuAnalysis/ExistingOptimization/run_scripts/*' \
    -czf "${local_tarball}" \
    -C "${cmssw_parent}" \
    "${cmssw_release}"

echo "Tarball size: $(du -h "${local_tarball}" | cut -f1)"

xrdfs "${eos_host}" mkdir -p "${eos_tarball_dir}"
xrdfs "${eos_host}" mkdir -p "${eos_output_dir}"

remote_tarball="${eos_host}/${eos_tarball_dir}/${tarball_name}"
echo "Uploading the run-specific CMSSW tarball to ${remote_tarball}"
# The run tag makes this object unique. Omit --force to protect an existing
# explicitly named run from accidental replacement.
xrdcp --nopbar "${local_tarball}" "${remote_tarball}"

rm -f -- "${local_tarball}"
local_tarball=""

echo "Submitting ${job_count} jobs..."
condor_submit "${submit_file}"
