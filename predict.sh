#!/bin/bash
set -o noglob

FLAGS=""
GPUS=$(nvidia-smi -L 2>/dev/null)
if [[ $? -eq 0 && ! -z "$GPUS" ]] ; then
    FLAGS=--gpus=all
fi
exec docker run $FLAGS -eTORCHINDUCTOR_FX_GRAPH_CACHE=1 -eTORCHINDUCTOR_CACHE_DIR=/scratch/sidtableflip/inductorcache --rm --name sidtableflip-predict -v /scratch:/scratch -ti anarkiwi/sidtableflip /sidtableflip/predict.py $*
