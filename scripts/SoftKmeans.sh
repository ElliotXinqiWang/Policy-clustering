#!/bin/bash
env_name="halfcheetah"
# env_name="hopper"
# env_name="ant"
# env_name="walker2d"

# dataset="expert"
dataset="medium-expert"
# dataset="medium-replay"
# dataset="full-replay"
# dataset="medium"
# dataset="random"

Kvalues=(3 4 5 6 7 8 10 12 15)
# Kvalues=(2) # for debugging

for k in "${Kvalues[@]}"; do
    echo "Running Soft Kmeans with k=$k"
    CUDA_VISIBLE_DEVICES=0 python algos/Kmeans_original_soft.py \
        --K_value "$k" \
        --env "${env_name}-${dataset}-v2" \
        --temperature 15.0 \
        --project "1205SoftKmeans" \
        --seed 2 
done
