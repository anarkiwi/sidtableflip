#!/bin/sh
set -o noglob
exec docker run --gpus=all --rm -v /scratch:/scratch -ti sidtableflip /sidtableflip/predict.py $*
