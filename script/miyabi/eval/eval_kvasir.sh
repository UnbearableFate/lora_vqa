#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=01:20:00
#PBS -j oe
#PBS -m abe

source "/work/xg24i002/x10041/lora_vqa/script/miyabi/common.sh"

DATASET="Kvasir-VQA-x1"
MODEL_NAME="llava-hf/vip-llava-7b-hf"

timestamp=$(date +"%Y%m%d-%H%M%S")

"$PYTHON_EXEC" -m src.cli evaluate \
    --adapter_path "output/Kvasir-VQA-x1/vip-llava-7b-hf/r16/lora_Kvasir-VQA-x1_r16_a1_kaiming_s11_20260205-002628" \
    --dataset_name "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --eval_batch_size 32 \
    --seed 11 \