#!/bin/bash

eval_scripts="/work/xg24i002/x10041/lora_vqa/script/miyabi/eval/eval.sh"

adapter_root_path="/work/xg24i002/x10041/lora_vqa/output_test"

for adapter_dir in "${adapter_root_path}"/*; do
    if [[ -d "${adapter_dir}" ]]; then
        if [[ ! -f "${adapter_dir}/adapter_config.json" ]]; then
            echo "Skipping ${adapter_dir}, no adapter_config.json found."
            continue
        fi
        adapter_path="${adapter_dir}"
        echo "Submitting evaluation job for adapter at ${adapter_path}"
        qsub -v adapter_path="${adapter_path}" "${eval_scripts}"
    fi
done
