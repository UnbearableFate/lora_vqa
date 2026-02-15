#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -m abe

source "/work/xg24i002/x10041/lora_vqa/script/miyabi/common.sh"

"${PYTHON_EXEC}" src/data_process.py