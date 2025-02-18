# env_name="MiniGrid-Reacher-extra-good"
# env_name="MiniGrid-Reacher-extra-bad"
env_name="MiniGrid-Reacher-extra-med"

echo "Running PPO in env $env_name"
CUDA_VISIBLE_DEVICES=5 python Xland/PPO_train.py \
    --env "$env_name" \
    --seed 2 \
    --max_updates 20000