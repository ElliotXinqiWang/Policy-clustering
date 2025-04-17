
from gridworld.env import SingleAgentGridworld, FixedGridworld, ExtraRewardGridworld
from gridworld.continuous_gridworld import SingleAgentEnv, TwoBarriorEnv, SpecifyPathEnv

def set_env(config):
    if config.env == "MiniGrid-Reacher":
        env = SingleAgentGridworld(grid_size=7, max_steps=20, distance_penalty=-0.5, goal_reward=10.0)
    elif config.env == "MiniGrid-Binary-Reacher":
        env = FixedGridworld(K=3, max_steps=20, distance_penalty=-0.5, goal_reward=10.0)
    elif config.env == "MiniGrid-Reacher-noisy":
        env = SingleAgentGridworld(grid_size=7, max_steps=20, distance_penalty=-0.3, goal_reward=10.0, epsilon=0.5)
    elif config.env == "MiniGrid-Reacher-extra-good":
        env = ExtraRewardGridworld(grid_size=7, max_steps=40, distance_penalty=-0.3, goal_reward=10.0, epsilon=config.epsilon, extra_reward=config.extra_reward)
    elif config.env == "MiniGrid-Reacher-extra-bad":
        env = ExtraRewardGridworld(grid_size=7, max_steps=40, distance_penalty=-0.3, goal_reward=10.0, epsilon=config.epsilon, extra_reward=-config.extra_reward)
    elif config.env == "MiniGrid-Reacher-extra-med":
        env = ExtraRewardGridworld(grid_size=7, max_steps=40, distance_penalty=-0.3, goal_reward=10.0, epsilon=config.epsilon, extra_reward=0)
    elif config.env == "Gridworld-reacher-continous":
        env = SpecifyPathEnv(max_steps=20, path=0)
    elif config.env == "Gridworld-reacher-continous-lu":
        env = SpecifyPathEnv(max_steps=20, path=1)
    elif config.env == "Gridworld-reacher-continous-dr":
        env = SpecifyPathEnv(max_steps=20, path=2)
    else:
        raise ValueError("Environment: ", config.env, " not supported")
    return env