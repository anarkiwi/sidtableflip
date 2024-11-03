#!/bin/sh
set -o noglob
exec docker run --gpus=all --rm -v /scratch:/scratch -ti anarkiwi/sidtableflip /sidtableflip/predict.py --reglog /scratch/hvsc/C64Music/MUSICIANS/G/Goto80/Automatas/1/Automatas-1.dump.zst $*
