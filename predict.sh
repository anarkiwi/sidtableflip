#!/bin/sh
set -o noglob
exec docker run --gpus=all --rm -v /scratch:/scratch -ti anarkiwi/sidtableflip /sidtableflip/predict.py $*
