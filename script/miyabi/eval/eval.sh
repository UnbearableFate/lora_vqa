#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -m abe

source "/work/xg24i002/x10041/lora_vqa/script/miyabi/common.sh"

adapter_path="${adapter_path:-/work/xg24i002/x10041/lora_vqa/output_0210/ChartQA/Ministral-3-3B-Instruct-2512-BF16/r16/lora_ChartQA_r16_a1_kaiming_s11_20260210-232007}"
echo "Evaluating adapter at path: ${adapter_path}"

"$PYTHON_EXEC" -m src.cli evaluate \
    --adapter_path "${adapter_path}" \
    --eval_batch_size 32 \
    --csv_path_dir "experiments_test_minis" \
    --attn_implementation "flash_attention_2" \