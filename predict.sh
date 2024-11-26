#!/bin/sh
set -o noglob
exec docker run --gpus=all -eTORCHINDUCTOR_FX_GRAPH_CACHE=1 -eTORCHINDUCTOR_CACHE_DIR=/scratch/sidtableflip/inductorcache --rm --name sidtableflip-predict -v /scratch:/scratch -ti anarkiwi/sidtableflip /sidtableflip/predict.py $*
