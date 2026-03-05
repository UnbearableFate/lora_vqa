#!/bin/bash

run_script="/work/xg24i002/x10041/lora_vqa/script/miyabi/qsub/run_llava-v1.6-mistral-7b-hf.sh"
DATASET="HuggingFaceM4/ChartQA"

init_lora_weights_list=("corda" )
fit_init_lora_weights_list=("gaussian" "True" "olora" "orthogonal" )
seed_list=(11 23 37)
output_dir="output_chartqa_$(date +%Y%m%d-%H)"

for seed in "${seed_list[@]}"; do  
    for init_lora_weights in "${init_lora_weights_list[@]}"; do
        qsub_output="$(qsub -v DATASET="${DATASET}",SEED="${seed}",init_lora_weights="${init_lora_weights}",use_cleaned_svd_ref_trainer=False,output_dir="${output_dir}" \
        "${run_script}")"
        qsub_outputs+=("${qsub_output} ${init_lora_weights} ${seed}")
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
