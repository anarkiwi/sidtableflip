#!/bin/sh
set -o noglob
exec docker run --rm -v /scratch:/scratch -ti anarkiwi/sidtableflip /sidtableflip/predict.py $*
