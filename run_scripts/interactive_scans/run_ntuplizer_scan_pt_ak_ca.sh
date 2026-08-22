for pt in $(seq 100 20 160); do
    for ak in $(seq 0.4 0.2 1.6); do
        # 1. Compute dynamic lower and upper bounds using bc
        ca_min=$(echo "$ak + 0.4" | bc)
        ca_max=$(echo "1.6" | bc)

        for ca in $(seq $ca_min 0.2 $ca_max); do
            # Standardize ca format to 1 decimal place to prevent floating point mismatch (e.g., 0.20 vs 0.2)
            ca_formatted=$(printf "%.1f" $ca)

            # 2. Skip condition using standard comparison on formatted string or bc
            if [ "$ca_formatted" = "0.2" ] || [ "$ca_formatted" = "1.8" ]; then
                continue
            fi
            
            # Calculate 10 * ak and 10 * ca as integers
            ak_int=$(printf "%.0f" $(echo "$ak * 10" | bc))
            ca_int=$(printf "%.0f" $(echo "$ca * 10" | bc))

            echo "========================================"
            echo "Running jet pT cutoff = ${pt} GeV, ak radius = ${ak}, ca radius = ${ca}"
            echo "========================================"

            cmsRun ./SuuAnalysis/ExistingOptimization/test/runExistingOptimizationNtuplizer_cfg.py \
                jetPtCut=${pt} \
                akRadius=${ak} \
                caRadius=${ca} \
                outputRootFile=rootfiles_existingOptimization/WbWb_4000_1000_jetPtCut${pt}_ak${ak_int}_ca${ca_int}.root

            if [ $? -ne 0 ]; then
                echo "ERROR: cmsRun failed for pT = ${pt}, akRadius = ${ak}, caRadius = ${ca}"
                exit 1
            fi
        done
    done
done