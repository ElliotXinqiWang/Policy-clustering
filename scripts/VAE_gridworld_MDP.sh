#!/bin/bash"
# env_name="MiniGrid-Reacher"
# env_name="MiniGrid-Reacher-extra-good"
# env_name="MiniGrid-Reacher-noisy"
env_name="MDPtakeball"
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
seeds=(0 1 2 3 4 5 6 7 8 9)
# Kvalues=(5 6 7 8 10 12 15)
# seeds=(0)
# Kvalues=(5) # for debugging
rule_based_dataset_files=(
    # "datasets/rule_based/MiniGrid-Reacher-MDP/balanced_20000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-MDP/rightfirst_20000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-MDP/downfirst_20000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-MDP/zigzag1_20000.pkl"
    # "datasets/rule_based/MiniGrid-Reacher-MDP/zigzag2_20000.pkl"
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
    "datasets/rule_based/MDPtakeball/fixed_0_20000.pkl"
    "datasets/rule_based/MDPtakeball/fixed_1_20000.pkl"
    "datasets/rule_based/MDPtakeball/fixed_2_20000.pkl"
    "datasets/rule_based/MDPtakeball/fixed_3_20000.pkl"
)
for seed in "${seeds[@]}"; do
    echo "Running VAE + Kmeans"
    CUDA_VISIBLE_DEVICES=2 python algos/VAE_kmeans_gridworld.py \
        --env "$env_name" \
        --seed "$seed" \
        --project "0128VAE_${env_name}" \
        --max_updates 200 \
        --rule_based_dataset_files "${rule_based_dataset_files[@]}"
done