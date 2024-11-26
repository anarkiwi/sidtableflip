#!/bin/sh
PIP_OPTS="--index-url http://192.168.5.1:5001/index/ --trusted-host 192.168.5.1"

docker build --build-arg PIP_OPTS="$PIP_OPTS" -f Dockerfile.tensorboard . -t anarkiwi/tensorboard
docker build --build-arg PIP_OPTS="$PIP_OPTS" -f Dockerfile . -t anarkiwi/sidtableflip
