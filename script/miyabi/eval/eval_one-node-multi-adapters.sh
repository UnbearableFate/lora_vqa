#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -m abe

source "/work/xg24i002/x10041/lora_vqa/script/miyabi/common.sh"

adapter_root="${adapter_root:-/work/xg24i002/x10041/lora_vqa/output_go/Kvasir-VQA-x1/LFM2.5-VL-1.6B/r16}"
done_root="$(dirname "${adapter_root}")/done"

echo "Using adapter_root: ${adapter_root}"
echo "Using done_root: ${done_root}"

if [[ ! -d "${adapter_root}" ]]; then
    echo "Error: adapter_root does not exist: ${adapter_root}"
    echo "Hint: run with a valid path, e.g. adapter_root=/path/to/r16 bash ./script/miyabi/eval/eval_one-node-multi-adapters.sh"
    exit 1
fi

mkdir -p "${done_root}"

shopt -s nullglob
adapter_paths=("${adapter_root}"/*)

if [[ ${#adapter_paths[@]} -eq 0 ]]; then
    echo "No entries found under adapter_root: ${adapter_root}"
    exit 0
fi

for adapter_path in "${adapter_paths[@]}"; do
    if [[ -d "${adapter_path}" ]]; then
        if [[ ! -f "${adapter_path}/adapter_config.json" ]]; then
            echo "Skipping ${adapter_path}, no adapter_config.json found."
            continue
        fi
        echo "Evaluating adapter at path: ${adapter_path}"

        if "$PYTHON_EXEC" -m src.cli evaluate \
            --adapter_path "${adapter_path}" \
            --eval_batch_size 32 \
            --csv_path_dir "experiments_0214" \
            --attn_implementation "flash_attention_2"; then
            target_path="${done_root}/$(basename "${adapter_path}")"
            if [[ -e "${target_path}" ]]; then
                echo "Target already exists, skip moving: ${target_path}"
            else
                mv "${adapter_path}" "${done_root}/"
                echo "Moved ${adapter_path} to ${done_root}"
            fi
        else
            echo "Evaluation failed for ${adapter_path}, keeping it in place."
        fi
    fi
done
