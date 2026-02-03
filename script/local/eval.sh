#!/bin/bash

DATASET="HuggingFaceM4/ChartQA"
MODEL_NAME="LiquidAI/LFM2.5-VL-1.6B"

export CUDA_VISIBLE_DEVICES="0"

timestamp=$(date +"%Y%m%d-%H%M%S")

python -m src.cli evaluate \
    --adapter_path "output/ChartQA/LFM2.5-VL-1.6B/r16/lora_ChartQA_r16_a1_kaiming_s11_20260203-024248" \
    --dataset_name "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --eval_batch_size 16 \
    --seed 11 \