# envs=("MiniGrid-Reacher-extra-bad" "MiniGrid-Reacher-extra-good" "MiniGrid-Reacher-extra-med")
env_name="MiniGrid-Reacher-extra-med"
model_load_paths=("behavior_models/PPO_MiniGrid-Reacher-extra-good/best_params.pkl" "behavior_models/PPO_MiniGrid-Reacher-extra-bad/best_params.pkl" "behavior_models/PPO_MiniGrid-Reacher-extra-med/best_params.pkl")
for model_load_path in "${model_load_paths[@]}"; do
    echo "Collecting expert data in env $env_name"
    CUDA_VISIBLE_DEVICES=7 python Xland/Collect.py \
        --env "$env_name" \
        --seed 2 \
        --n_episodes 10 \
        --model_load_path "$model_load_path"
done