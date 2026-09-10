#!/bin/bash

legacy_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for th in $(seq 1.0 0.05 1.0); do
    echo "========================================"
    echo "Evaluating cosine thrust cutoff = ${th}"
    echo "========================================"

    # Calculate 100 * th as integer
    th_int=$(printf "%.0f" $(echo "$th * 100" | bc))

    python3 "${legacy_dir}/evaluate_ntuplizer.py" \
        --input /eos/uscms/store/user/aji/rootfiles_existingOptimization/WbWb_4000_1000_jetPtCut100_ak6_ca4_th${th_int}.root \
        --output "${legacy_dir}/results/evaluate_ntuplizer_jetPtCut100_ak6_ca4_th${th_int}/" \
        --new

    if [ $? -ne 0 ]; then
        echo "ERROR: evaluation failed for th = ${th}"
        exit 1
    fi
done
