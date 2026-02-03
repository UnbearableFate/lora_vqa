#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=8:mpiprocs=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -m abe

module load cuda/12.8

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"
echo "Current working directory: $(pwd)"

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-accelerate_config/accelerate_config.yaml}
MASTER_PORT=${MASTER_PORT:-29500}
MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE")
NNODES=$(sort -u "$PBS_NODEFILE" | wc -l)
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

export MASTER_ADDR MASTER_PORT
export ACCELERATE_CONFIG_FILE="$ACCELERATE_CONFIG"

ENV_VARS=("MASTER_ADDR=${MASTER_ADDR}" "MASTER_PORT=${MASTER_PORT}" "ACCELERATE_CONFIG_FILE=${ACCELERATE_CONFIG}")
ENV_LIST=$(IFS=,; echo "${ENV_VARS[*]}")
if [[ -n "${OMPI_MCA_mca_base_env_list:-}" ]]; then
    export OMPI_MCA_mca_base_env_list="${OMPI_MCA_mca_base_env_list},${ENV_LIST}"
else
    export OMPI_MCA_mca_base_env_list="${ENV_LIST}"
fi

PYTHON_PATH="/work/xg24i002/x10041/my_peft/torch291/bin/python"

HF_HOME="/work/xg24i002/x10041/hf_home"
HF_DATASETS_CACHE="/work/xg24i002/x10041/data"
export HF_HOME HF_DATASETS_CACHE

DATASET="HuggingFaceM4/ChartQA"
MODEL_NAME="LiquidAI/LFM2.5-VL-1.6B"
target_modules="q_proj,k_proj,v_proj,out_proj,w1,w2,w3"


timestamp=$(date +%Y%m%d-%H%M%S)
mpirun --mca mpi_abort_print_stack 1 \
       --report-bindings \
       --bind-to core \
       -np "${WORLD_SIZE}" \
       /usr/bin/env \
           MASTER_ADDR="${MASTER_ADDR}" \
           MASTER_PORT="${MASTER_PORT}" \
           ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG}" \
       bash -c 'set -euo pipefail; \
                : "${MASTER_ADDR:?MASTER_ADDR not set}"; \
                : "${MASTER_PORT:?MASTER_PORT not set}"; \
                : "${ACCELERATE_CONFIG_FILE:?ACCELERATE_CONFIG_FILE not set}"; \
                export RANK=$OMPI_COMM_WORLD_RANK; \
                export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE; \
                export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK; \
                export LOCAL_WORLD_SIZE=$OMPI_COMM_WORLD_LOCAL_SIZE; \
                export HF_HOME='"${HF_HOME}"'; \
                export HF_DATASETS_CACHE='"${HF_DATASETS_CACHE}"'; \
                echo "Running on rank $RANK out of $WORLD_SIZE"; \
                '"${PYTHON_PATH}"' -m src.cli train \
                    --timestamp '"${timestamp}"' \
                    --output_dir "output" \
                    --dataset_name '"${DATASET}"' \
                    --model_name '"${MODEL_NAME}"' \
                    --seed 11 \
                    --global_batch_size 32 \
                    --per_device_batch_size 4 \
                    --num_train_epochs 8 \
                    --learning_rate 4e-4 \
                    --weight_decay 0.0 \
                    --warmup_ratio 0.03 \
                    --target_modules '"${target_modules}"' \
                    --modules_to_save "lm_head" \
                    --lora_r 16 \
                    --lora_alpha 1 \
                    --lora_dropout 0.0 \
                    --lora_bias none \
                    --peft_variant lora \
                    --init_lora_weights true \
                    --init_num_samples 128 \
                    --init_batch_size 1 \
                    --eval_steps 100 \
                    --eval_batch_size 4 \
                    --logging_steps 50 \
                    --use_wandb true \
                    --wandb_online true \
                    '