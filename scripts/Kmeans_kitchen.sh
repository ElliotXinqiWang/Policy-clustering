#!/bin/bash
# env_name="halfcheetah"
# env_name="hopper"
# env_name="ant"
# env_name="walker2d"
env_name="kitchen"

# dataset="expert"
# dataset="medium-expert"
# dataset="medium-replay"
# dataset="full-replay"
# dataset="medium"
# dataset="random"
# dataset="mixed"
dataset="partial"
# dataset="complete"

Kvalues=(3 4 5 6 7 8 10 12 15)
# Kvalues=(2) # for debugging

for k in "${Kvalues[@]}"; do
    echo "Running Kmeans with k=$k"
    CUDA_VISIBLE_DEVICES=3 python algos/Kmeans_original.py \
        --K_value "$k" \
        --env "${env_name}-${dataset}-v0" \
        --seed 2 \
        --project "1205Kmeans" \
        --max_traj_len 280 # 280 for kitchen
done
