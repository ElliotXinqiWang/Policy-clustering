#!/bin/bash"
# env_name="MiniGrid-Reacher"
# env_name="MiniGrid-Reacher-extra-good"
# env_name="MiniGrid-Reacher-noisy"
env_name="MiniGrid-Reacher-MDP"
# env_name="hopper"
# env_name="ant"
# env_name="walker2d"

# dataset="expert"
# dataset="medium-expert"
# dataset="medium-replay"
# dataset="full-replay"
# dataset="medium"
# dataset="random"
seeds=(0 1 2 3 4 5 6 7 8 9)
Kvalues=(5 6 7 8 10 12 15)
# Kvalues=(5)
# seeds=(0)
# Kvalues=(8) # for debugging
rule_based_dataset_files=(
    "datasets/rule_based/MiniGrid-Reacher-MDP/balanced_20000.pkl"
    "datasets/rule_based/MiniGrid-Reacher-MDP/rightfirst_20000.pkl"
    "datasets/rule_based/MiniGrid-Reacher-MDP/downfirst_20000.pkl"
    "datasets/rule_based/MiniGrid-Reacher-MDP/zigzag1_20000.pkl"
    "datasets/rule_based/MiniGrid-Reacher-MDP/zigzag2_20000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-MDP/balanced_8000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-MDP/rightfirst_8000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-MDP/downfirst_8000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-MDP/zigzag1_8000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-MDP/zigzag2_8000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-MDP/balanced_2000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-MDP/rightfirst_2000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-MDP/downfirst_2000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-MDP/zigzag1_2000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-MDP/zigzag2_2000.pkl"
)
SELECTED_GPU=$(python scripts/select_gpu.py)
echo "Selected GPU: $SELECTED_GPU"
for seed in "${seeds[@]}"; do
    for k in "${Kvalues[@]}"; do
        echo "Running Kmeans with k=$k"
        CUDA_VISIBLE_DEVICES=$SELECTED_GPU python algos/Kmeans_original.py \
            --env "$env_name" \
            --K_value "$k" \
            --seed "$seed" \
            --project "0126gridworldKmeansMDP_rule_based_large" \
            --max_updates 15 \
            --load_from_rule_based_dataset true \
            --rule_based_dataset_files "${rule_based_dataset_files[@]}"
    done
done