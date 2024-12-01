#!/bin/sh
docker rm -f sidtableflip-train
sudo rm -rf /scratch/sidtableflip/tb_logs/*
./build.sh && ./train.sh && docker logs -f sidtableflip-train
