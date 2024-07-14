#!/bin/sh
docker run --gpus=all --rm -v /scratch:/scratch -ti sidtableflip /sidtableflip/train.py $*
