#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -m abe

source "/work/xg24i002/x10041/lora_vqa/script/miyabi/common.sh"

adapter_path="${adapter_path:-/work/xg24i002/x10041/lora_vqa/output_llava_hosoku/ChartQA/Ministral-3-3B-Instruct-2512-BF16/r16/lora_ChartQA_r16_a1_gaussian_s23_20260224-085209}"
echo "Evaluating adapter at path: ${adapter_path}"

"$PYTHON_EXEC" -m src.cli evaluate \
    --adapter_path "${adapter_path}" \
    --eval_batch_size 32 \
    --csv_path_dir "experiments_buchong_$(date +%Y%m%d-%H)" \
    --attn_implementation "flash_attention_2" \