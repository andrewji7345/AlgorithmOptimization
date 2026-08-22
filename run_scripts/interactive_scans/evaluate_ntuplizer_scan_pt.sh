#!/bin/bash

for pt in $(seq 20 20 400); do
    echo "========================================"
    echo "Evaluating jet pT cutoff = ${pt} GeV"
    echo "========================================"

    mv /eos/uscms/store/user/aji/rootfiles_existingOptimization/WbWb_4000_1000_jetCutPt${pt}.root /eos/uscms/store/user/aji/rootfiles_existingOptimization/WbWb_4000_1000_jetPtCut${pt}.root

    python3 ./evaluate_ntuplizer.py \
        --input /eos/uscms/store/user/aji/rootfiles_existingOptimization/WbWb_4000_1000_jetPtCut${pt}.root \
        --output evaluate_ntuplizer_jetPtCut${pt}/

    if [ $? -ne 0 ]; then
        echo "ERROR: evaluation failed for pT = ${pt}"
        exit 1
    fi
done