#!/bin/bash

for th in $(seq 0.50 0.05 0.60); do
    echo "========================================"
    echo "Running cosine thrust cutoff = ${th}"
    echo "========================================"

    # Calculate 100 * th as integer
    th_int=$(printf "%.0f" $(echo "$th * 100" | bc))

    cmsRun ./SuuAnalysis/ExistingOptimization/test/runExistingOptimizationNtuplizer_cfg.py \
                jetPtCut=100 \
                akRadius=0.6 \
                caRadius=0.4 \
                cosThrust=${th} \
                outputRootFile=rootfiles_existingOptimization/WbWb_4000_1000_jetPtCut100_ak6_ca4_th${th_int}.root

    if [ $? -ne 0 ]; then
        echo "ERROR: cmsRun failed for th = ${th}"
        exit 1
    fi
done