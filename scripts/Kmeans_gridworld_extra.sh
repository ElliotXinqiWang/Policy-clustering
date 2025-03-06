#!/bin/bash"
# env_name="MiniGrid-Reacher"
env_name="MiniGrid-Reacher-extra-good"
# env_name="MiniGrid-Reacher-noisy"
# env_name="MiniGrid-Reacher-MDP"
# env_name="hopper"
# env_name="ant"
# env_name="walker2d"

# dataset="expert"
# dataset="medium-expert"
# dataset="medium-replay"
# dataset="full-replay"
# dataset="medium"
# dataset="random"
seeds=(1 2 3 4 5)
# Kvalues=(5 6 7 8 10 12 15)
# seeds=(0)
Kvalues=(5) # for debugging
rule_based_dataset_files=(
    "datasets/rule_based/MiniGrid-Reacher-extra-good/batch_8000.pkl"
    "datasets/rule_based/MiniGrid-Reacher-extra-bad/batch_20000.pkl"
    "datasets/rule_based/MiniGrid-Reacher-extra-med/batch_20000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-extra-good/batch_2000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-extra-bad/batch_2000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-extra-med/batch_2000.pkl"
)
SELECTED_GPU=$(python scripts/select_gpu.py)
echo "Selected GPU: $SELECTED_GPU"
for seed in "${seeds[@]}"; do
    for k in "${Kvalues[@]}"; do
        echo "Running Kmeans with k=$k"
        CUDA_VISIBLE_DEVICES=$SELECTED_GPU python algos/Kmeans_original_gridworld.py \
            --env "$env_name" \
            --K_value "$k" \
            --seed "$seed" \
            --project "0124gridworldKmeansextra_rule_based_large" \
            --max_updates 50 \
            --load_from_rule_based_dataset true \
            --rule_based_dataset_files "${rule_based_dataset_files[@]}"
    done
done