#!/bin/bash

run_script="/work/xg24i002/x10041/lora_vqa/script/miyabi/qsub/run_vip-llava.sh"

init_lora_weights_list=("eva" "corda" "lora_ga" "gaussian" "True" "olora" "pissa" "orthogonal" )
fit_init_lora_weights_list=("gaussian" "true" "olora" "orthogonal" )
seed_list=(11 )

for seed in "${seed_list[@]}"; do  
    for init_lora_weights in "${init_lora_weights_list[@]}"; do
        qsub_output="$(qsub -v DATASET="${DATASET}",SEED="${seed}",init_lora_weights="${init_lora_weights}" \
        "${run_script}")"
        qsub_outputs+=("${qsub_output}")
    done

    for init_lora_weights in "${fit_init_lora_weights_list[@]}"; do
        qsub_output="$(qsub -v DATASET="${DATASET}",SEED="${seed}",init_lora_weights="${init_lora_weights}",use_cleaned_svd_ref_trainer=True \
        "${run_script}")"
        qsub_outputs+=("${qsub_output}")
    done
done

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="qsub_history_${timestamp}.log"

{
    echo "===== TRAIN_CONFIG CONTENT ====="
    cat "${run_script}"
    echo "===== QSUB OUTPUTS ====="
    printf '%s\n' "${qsub_outputs[@]}"
} > "${log_file}"
