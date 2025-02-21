#!/bin/bash
# env_name="halfcheetah"
env_name="hopper"
# env_name="ant"
# env_name="walker2d"

# dataset="expert"
dataset="medium-expert"
# dataset="medium-replay"
# dataset="full-r
# dataset="medium"
# dataset="random"

# Kvalues=(2 3 4 6 8 10 12 15)
Kvalues=(3 4 6)
seeds=(3 4 5)
# Kvalues=(6) # for debugging
for seed in "${seeds[@]}"; do
    for k in "${Kvalues[@]}"; do
        echo "Running Kmeans with k=$k"
        CUDA_VISIBLE_DEVICES=0 python algos/Kmeans_original.py \
            --K_value "$k" \
            --env "${env_name}-${dataset}-v2" \
            --seed "$seed" \
            --project "0129Kmeans_D4RL" \
            --max_updates 10
    done
done