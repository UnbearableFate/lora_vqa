#!/bin/bash

eval_scripts="/work/xg24i002/x10041/lora_vqa/script/miyabi/eval/eval_by_llm_transformers.sh"

adapter_root_path="/work/xg24i002/x10041/lora_vqa/output_kvqa_more-more-alpha_20260304-17/Kvasir-VQA-x1/LFM2.5-VL-1.6B/r16"
csv_path_dir="experiments_kqv_more-more-alpha_$(date +%Y%m%d-%H)"

for adapter_dir in "${adapter_root_path}"/*; do
    if [[ -d "${adapter_dir}" ]]; then
        if [[ ! -f "${adapter_dir}/adapter_config.json" ]]; then
            echo "Skipping ${adapter_dir}, no adapter_config.json found."
            continue
        fi
        adapter_path="${adapter_dir}"
        echo "Submitting evaluation job for adapter at ${adapter_path}"
        qsub -v adapter_path="${adapter_path}",csv_path_dir="${csv_path_dir}" "${eval_scripts}"
    fi
done
