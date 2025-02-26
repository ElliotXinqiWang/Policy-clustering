#!/bin/bash"
# env_name="halfcheetah"
env_name="hopper"
# env_name="ant"
# env_name="walker2d"

# dataset="expert"
dataset="medium-expert"
# dataset="medium-replay"
# dataset="full-replay"
# dataset="medium"
# dataset="random"
# seeds=(0 1 2 3 4 5 6 7 8 9)
codebook=16
seeds=(0)
SELECTED_GPU=$(python scripts/select_gpu.py)
echo "Selected GPU: $SELECTED_GPU"
for seed in "${seeds[@]}"; do
    echo "Running VAE + Kmeans"
    CUDA_VISIBLE_DEVICES=$SELECTED_GPU python algos/VAE_kmeans_all.py \
        --env "${env_name}-${dataset}-v2" \
        --seed "$seed" \
        --project "0129VAE_D4RL" \
        --max_updates 200 \
        --vqvae_codebook "$codebook"\
        # --algo "vqvae_gumble_softmax" --batch_size 256\
        --algo "vqvae"\

done