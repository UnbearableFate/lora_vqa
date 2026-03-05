#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -m abe

source "/work/xg24i002/x10041/lora_vqa/script/miyabi/common.sh"

adapter_path="${adapter_path:-/work/xg24i002/x10041/lora_vqa/output_hosoku/Kvasir-VQA-x1/llava-v1.6-mistral-7b-hf/r16/lora_Kvasir-VQA-x1_r16_a1_eva_s23_20260224-072316}"
csv_path_dir="${csv_path_dir:-experiments_buchong_$(date +%Y%m%d)}"
judge_model_name_or_path="Qwen/Qwen3-4B-Instruct-2507"

echo "Evaluating adapter at path: ${adapter_path}"
echo "Using judge model: ${judge_model_name_or_path}"

"$PYTHON_EXEC" -m src.cli evaluate_by_llm_transformers \
    --adapter_path "${adapter_path}" \
    --judge_model_name_or_path "${judge_model_name_or_path}" \
    --eval_batch_size 32 \
    --judge_batch_size 16 \
    --csv_path_dir "${csv_path_dir}" \
    --attn_implementation "flash_attention_2"
