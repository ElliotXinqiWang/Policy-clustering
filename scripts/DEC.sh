#!/bin/bash"
env_name="halfcheetah-medium-expert-v2"
# env_name="hopper-medium-expert-v2"
# env_name="ant-medium-expert-v2"
# env_name="walker2d-medium-expert-v2"
# env_name="MiniGrid-Reacher-MDP"
# env_name="MDPtakeball"

# dataset="expert"
# dataset="medium-expert"
# dataset="medium-replay"
# dataset="full-replay"
# dataset="medium"
# dataset="random"
seeds=(0 1 2)
# seeds=(1)
# k_values=(10 12 15)
k_values=(5)
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
    # "datasets/rule_based/MDPtakeball/fixed_0_20000.pkl"
    # "datasets/rule_based/MDPtakeball/fixed_1_20000.pkl"
    # "datasets/rule_based/MDPtakeball/fixed_2_20000.pkl"
    # "datasets/rule_based/MDPtakeball/fixed_3_20000.pkl"
)
SELECTED_GPU=$(python scripts/select_gpu.py)
echo "Selected GPU: $SELECTED_GPU"
for seed in "${seeds[@]}"; do
    for k in "${k_values[@]}"; do
        echo "Running DEC"
        CUDA_VISIBLE_DEVICES=$SELECTED_GPU python algos/DEC.py \
            --env "${env_name}" \
            --seed "$seed" \
            --project "0129DEC_different K" \
            --max_updates 400 \
            --K_value "$k" \
            --rule_based_dataset_files "${rule_based_dataset_files[@]}"
    done
done