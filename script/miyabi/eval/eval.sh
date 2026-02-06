#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=01:20:00
#PBS -j oe
#PBS -m abe

source "/work/xg24i002/x10041/lora_vqa/script/miyabi/common.sh"

DATASET="HuggingFaceM4/ChartQA"
MODEL_NAME="LiquidAI/LFM2.5-VL-1.6B"

timestamp=$(date +"%Y%m%d-%H%M%S")

"$PYTHON_EXEC" -m src.cli evaluate \
    --adapter_path "/work/xg24i002/x10041/lora_vqa/output/ChartQA/LFM2.5-VL-1.6B/r16/lora_ChartQA_r16_a1_kaiming_s11_20260203-182317"\
    --dataset_name "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --eval_batch_size 32 \
    --seed 11 \