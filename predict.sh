#!/bin/sh
set -o noglob
exec docker run --gpus=all -eTORCHINDUCTOR_FX_GRAPH_CACHE=1 -eTORCHINDUCTOR_CACHE_DIR=/scratch/sidtableflip/inductorcache --rm -v /scratch:/scratch -ti anarkiwi/sidtableflip /sidtableflip/predict.py --model_state $(find /scratch/sidtableflip/tb_logs/ -name \*ckpt|sort|tail -1) $*
