#!/bin/bash
# env_name="halfcheetah"
# env_name="hopper"
env_name="ant"
# env_name="walker2d"

# dataset="expert"
dataset="medium-expert"
# dataset="medium-replay"
# dataset="full-r
# dataset="medium"
# dataset="random"

# Kvalues=(2 3 4 6 8 10 12 15)
Kvalues=(3)
seeds=(0 1 2 3 4 5 6 7 8 9)
SELECTED_GPU=$(python scripts/select_gpu.py)
echo "Selected GPU: $SELECTED_GPU"
# Kvalues=(6) # for debugging
for seed in "${seeds[@]}"; do
    for k in "${Kvalues[@]}"; do
        echo "Running Kmeans with k=$k"
        CUDA_VISIBLE_DEVICES=$SELECTED_GPU python algos/Kmeans_original.py \
            --K_value "$k" \
            --env "${env_name}-${dataset}-v2" \
            --seed "$seed" \
            --project "test_PGkmeans" \
            --max_updates 15
    done
done