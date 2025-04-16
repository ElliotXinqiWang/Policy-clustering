
# env_name="MiniGrid-Reacher-extra-med"
# env_name="MiniGrid-Reacher-extra-good"
# env_name="MiniGrid-Reacher-extra-bad"
env_name="MiniGrid-Reacher-MDP"
# env_name="MDPtakeball"
# env_name="MDPtakeball-hard"
# agent_modes=("balanced" "zigzag2" "zigzag1" "downfirst" "rightfirst")
agent_modes=("random1" "random2")
takeball_targets=(0 1 2 3)
# takeball_targets=(0)
# agent_modes=("zigzag2")
size=20000

SELECTED_GPU=$(python scripts/select_gpu.py)
echo "Selected GPU: $SELECTED_GPU"

if [ "$env_name" == "MiniGrid-Reacher-MDP" ]; then
    for agent_mode in "${agent_modes[@]}"; do
        filename="${agent_mode}_${size}.pkl"
        echo "Collecting expert data in env $env_name, agent_mode $agent_mode"
        CUDA_VISIBLE_DEVICES=$SELECTED_GPU python gridworld/rule_based_collect.py \
            --env "$env_name" \
            --seed 66 \
            --n_episodes $size \
            --filename "$filename" \
            --merge_interval 128 \
            --MDP_reacher_agent_mode "$agent_mode"
    done
fi

# for ball_target in "${takeball_targets[@]}"; do
#     filename="fixed_${ball_target}_${size}.pkl"
#     echo "Collecting expert data in env $env_name"
#     CUDA_VISIBLE_DEVICES=0 python gridworld/rule_based_collect.py \
#         --env "$env_name" \
#         --seed 67 \
#         --n_episodes $size \
#         --filename "$filename" \
#         --merge_interval 128 \
#         --take_ball_target $ball_target
# done


# echo "Collecting expert data in env $env_name"
# CUDA_VISIBLE_DEVICES=1 python gridworld/rule_based_collect.py \
#     --env "$env_name" \
#     --seed 5 \
#     --n_episodes 8000 \
#     --filename "zigzag2_8000.pkl" \
#     --merge_interval 128 \
#     --MDP_reacher_agent_mode "$agent_mode"