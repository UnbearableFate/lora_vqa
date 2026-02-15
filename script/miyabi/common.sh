#!/bin/bash

module load cuda/12.8

set -euo pipefail

cd "/work/xg24i002/x10041/lora_vqa"
echo "Current working directory: $(pwd)"

module load cuda/12.8
export HF_DATASETS_CACHE="/work/xg24i002/x10041/data"
export HF_HOME="/work/xg24i002/x10041/hf_home"
export PYTHON_EXEC="/work/xg24i002/x10041/my_peft/torch291/bin/python"
export CUDA_VISIBLE_DEVICES="0"