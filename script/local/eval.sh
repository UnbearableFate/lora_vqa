#!/bin/bash

DATASET="Kvasir-VQA-x1"
MODEL_NAME="llava-hf/vip-llava-7b-hf"

export CUDA_VISIBLE_DEVICES="0"

timestamp=$(date +"%Y%m%d-%H%M%S")

python -m src.cli evaluate \
    --adapter_path "output/Kvasir-VQA-x1/vip-llava-7b-hf/r16/lora_Kvasir-VQA-x1_r16_a1_kaiming_s11_20260204-014648" \
    --dataset_name "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --eval_batch_size 16 \
    --seed 11 \