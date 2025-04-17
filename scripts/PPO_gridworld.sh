# env_name="MiniGrid-Reacher-extra-good"
# env_name="MiniGrid-Reacher-extra-bad"
# env_name="MiniGrid-Reacher-extra-med"
# env_name="Gridworld-reacher-continous"
env_name="Gridworld-reacher-continous-lu"
# env_name="Gridworld-reacher-continous-dr"

SELECTED_GPU=$(python scripts/select_gpu.py)
echo "Running PPO in env $env_name"
echo "Selected GPU: $SELECTED_GPU"
CUDA_VISIBLE_DEVICES=$SELECTED_GPU python algos/PPO/PPO_train.py \
    --env "$env_name" \
    --seed 2 \
    --max_updates 20000 \
    # --gamma 0.9 \
