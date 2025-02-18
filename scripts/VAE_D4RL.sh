#!/bin/bash"
# env_name="halfcheetah"
# env_name="hopper"
# env_name="ant"
env_name="walker2d"

# dataset="expert"
dataset="medium-expert"
# dataset="medium-replay"
# dataset="full-replay"
# dataset="medium"
# dataset="random"
seeds=(0 1 2 3 4 5 6 7 8 9)
# seeds=(0)
for seed in "${seeds[@]}"; do
    echo "Running VAE + Kmeans"
    CUDA_VISIBLE_DEVICES=3 python algos/VAE_kmeans_D4RL.py \
        --env "${env_name}-${dataset}-v2" \
        --seed "$seed" \
        --project "0129VAE_D4RL" \
        --max_updates 200 
done