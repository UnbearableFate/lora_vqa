#!/bin/bash

DATASET="HuggingFaceM4/ChartQA"
MODEL_NAME="LiquidAI/LFM2.5-VL-1.6B"

export CUDA_VISIBLE_DEVICES="0"

timestamp=$(date +"%Y%m%d-%H%M%S")

python -m src.cli train \
    --timestamp "${timestamp}" \
    --output_dir "output" \
    --dataset_name "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --seed 11 \
    --global_batch_size 16 \
    --per_device_batch_size 16 \
    --num_train_epochs 4 \
    --learning_rate 4e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --target_modules "q_proj,k_proj,v_proj,out_proj,w1,w2,w3" \
    --lora_r 16 \
    --lora_alpha 1 \
    --lora_dropout 0.01 \
    --lora_bias none \
    --peft_variant lora \
    --init_lora_weights true \
    --init_num_samples 128 \
    --init_batch_size 1 \
    --eval_steps 100 \
    --eval_batch_size 16 \
    --logging_steps 50 \