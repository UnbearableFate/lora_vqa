#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=16:mpiprocs=1
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

DATASET=${DATASET:-"SimulaMet/Kvasir-VQA-x1"}
SEED=${SEED:-11}
use_cleaned_svd_ref_trainer=${use_cleaned_svd_ref_trainer:-False}
init_lora_weights=${init_lora_weights:-True}

MODEL_NAME="llava-hf/llava-v1.6-mistral-7b-hf"
#target_modules="k_proj,v_proj,q_proj,out_proj,fc1,fc2,linear_1,linear_2,o_proj,gate_proj,up_proj,down_proj"
target_modules="k_proj,v_proj,q_proj,out_proj,o_proj,gate_proj,up_proj,down_proj"

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
                    --output_dir "output_test" \
                    --dataset_name '"${DATASET}"' \
                    --model_name '"${MODEL_NAME}"' \
                    --seed '"${SEED}"' \
                    --global_batch_size 32 \
                    --per_device_batch_size 2 \
                    --num_train_epochs 2 \
                    --learning_rate 4e-4 \
                    --weight_decay 0.01 \
                    --warmup_ratio 0.03 \
                    --target_modules '"${target_modules}"' \
                    --lora_r 16 \
                    --lora_alpha 1 \
                    --lora_dropout 0.0 \
                    --lora_bias none \
                    --peft_variant lora \
                    --init_lora_weights '"${init_lora_weights}"' \
                    --init_num_samples 2048 \
                    --init_batch_size 1 \
                    --eval_steps 100 \
                    --eval_batch_size 2 \
                    --logging_steps 50 \
                    --use_cleaned_svd_ref_trainer '"${use_cleaned_svd_ref_trainer}"' \
                    --repeat_n 1 \
                    --repeat_warmup_ratio 0.03 \
                    --repeat_decay_ratio 0.03 \
                    --repeat_end_lr_rate 0.97 \
                    --final_warmup_ratio 0.03 \
                    --use_wandb True \
                    --wandb_online True \
                    '
