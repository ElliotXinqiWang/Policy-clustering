#!/bin/bash"
# env_name="MiniGrid-Reacher"
# env_name="MiniGrid-Reacher-extra-good"
# env_name="MiniGrid-Reacher-noisy"
env_name="Gridworld-reacher-continous"
# env_name="hopper"
# env_name="ant"
# env_name="walker2d"

# dataset="expert"
# dataset="medium-expert"
# dataset="medium-replay"
# dataset="full-replay"
# dataset="medium"
# dataset="random"
seeds=(0 1 2 3 4)
# Kvalues=(5 6 7 8 10 12 15)
# seeds=(0)
Kvalues=(6)
# Kvalues=(8) # for debugging
rule_based_dataset_files=(
    "datasets/Gridworld-reacher-continous/continous/data_20000_0.pkl"
    "datasets/Gridworld-reacher-continous-dr/dr/data_20000_0.pkl"
    "datasets/Gridworld-reacher-continous-lu/lu/data_20000_0.pkl"
)
SELECTED_GPU=$(python scripts/select_gpu.py)
echo "Selected GPU: $SELECTED_GPU"
for seed in "${seeds[@]}"; do
    for k in "${Kvalues[@]}"; do
        echo "Running Kmeans with k=$k"
        CUDA_VISIBLE_DEVICES=$SELECTED_GPU python algos/Kmeans_original_gridworld_continous.py \
            --env "$env_name" \
            --K_value "$k" \
            --seed "$seed" \
            --project "Kmeans_girdworld" \
            --max_updates 15 \
            --rule_based_dataset_files "${rule_based_dataset_files[@]}" \
            --use_rnn true \
            # --learning_rate 0.01 
    done
done