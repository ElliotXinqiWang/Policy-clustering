from gridworld.env import ExtraRewardGridworld, MDPGridworld, MDPtakeball
from gridworld.rule_based_agent import grid_Reacher_agent, grid_Reacher_agent_good, grid_Reacher_agent_bad, MDP_reacher_agent, MDP_takeball_agent
import jax.numpy as jnp
import jax
import os
import argparse
import numpy as np
from typing import NamedTuple, Tuple, Dict
from dataclasses import dataclass, field
import tqdm
import pickle


@dataclass
class CollectConfig:
    # General
    extra_attributes: dict = field(default_factory=dict, init=False)
    # Collect
    env: str = "MDPtakeball"  # Minigrid environment name
    seed: int = 1  # Sets Gym, Jax and Numpy seeds
    n_episodes: int = 2048  # How many episodes to collect
    filename: str = "extra_good.pkl"  # Where to save the collected data
    merge_interval: int = 128  # How many episodes to merge into a single batch
    # env_kwargs
    epsilon: float = 0.3
    extra_reward: float = 10.0
    MDP_reacher_agent_mode: str = "balanced"
    take_ball_target: int = 0
    

    def __post_init__(self):
        return
            
    def __setattr__(self, key, value):
        # Add to extra_attributes if not in dataclass fields
        if key not in self.__dataclass_fields__:
            self.extra_attributes[key] = value
        else:
            super().__setattr__(key, value)
    
    def __getattr__(self, key):
        # Fetch from extra_attributes if not in dataclass fields
        if key in self.extra_attributes:
            return self.extra_attributes[key]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

def parse_args_and_update_config(config_class):
    parser = argparse.ArgumentParser()

    # 遍历 dataclass 的字段，根据其类型动态添加参数
    for field_name, field_info in config_class.__dataclass_fields__.items():
        default = field_info.default
        if isinstance(default, (int, float, str, bool)):
            parser.add_argument(f"--{field_name}", type=type(default), default=default, help=f"Default: {default}")
        elif default is None:
            parser.add_argument(f"--{field_name}", type=str, default=None, help="Default: None")

    # 解析命令行参数
    args = parser.parse_args()

    # 使用命令行参数覆盖默认配置
    kwargs = vars(args)
    config = config_class(**kwargs)
    return config


class Transitions(NamedTuple):
    obs: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    action: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    reward: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    done: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))

def get_one_trajectory(env, agent, rng):
    obss = []
    actions = []
    rewards = []
    dones = []
    
    
    reset_key, rng = jax.random.split(rng)
    obs, state = env.reset(key=reset_key)
    agent.reset()
    done = False
    while not done:
        dis, action = agent.get_action(state)
        if dis == np.inf:
            break
        step_key, rng = jax.random.split(rng)
        last_obs = obs
        obs, state, reward, done= env.step(step_key, state, action)
        
        obss.append(last_obs.flatten())
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
    
    if len(obss) == 0: # no path
        return rng, None
    else:
        return rng, (jnp.array(obss), jnp.array(actions), jnp.array(rewards), jnp.array(dones))

def MDP_get_one_trajectory(env, agent, rng, num_trajectories=2000, save_dir=None, filename=None, num_environments=128, iteration_length=128):
    num_traj_collected = 0
    traj_batchs = []
    def _collect_a_batch(collect_state):
        def _collect_step(collect_state, collect_timestep):
            obss, states, rng = collect_state
            actions = jax.vmap(agent.get_action, in_axes=0)(states)
            step_key, rng = jax.random.split(rng)
            step_keys = jax.random.split(step_key, num_environments)
            next_obss, next_states, rewards, dones = jax.vmap(env.step, in_axes=(0, 0, 0))(step_keys, states, actions)
            traj = Transitions(
                obs=obss.reshape(num_environments, -1),
                action=actions,
                reward=rewards,
                done=dones,
            )
            return (next_obss, next_states, rng), traj
        collect_state, trajs = jax.lax.scan(_collect_step, collect_state, jnp.arange(iteration_length))
        return collect_state, trajs
    jitted_collect_a_batch = jax.jit(_collect_a_batch)
    while num_traj_collected < num_trajectories:
        _reset_key, rng = jax.random.split(rng)
        _reset_keys = jax.random.split(_reset_key, num_environments)
        init_obs, init_state = jax.vmap(env.reset, in_axes=0)(_reset_keys)
        collect_state = (init_obs, init_state, rng)
        collect_state, trajs = jitted_collect_a_batch(collect_state)
        rng = collect_state[-1]
        sliced_transitions = continuous_dataset_to_trajectories(trajs, max_traj_len=40)
        
        traj_batchs.append(sliced_transitions)
        num_traj_collected += sliced_transitions.obs.shape[0]
        print("average returns of latest batch", sliced_transitions.reward.sum(axis=1).mean())
        print("average length of latest batch", jnp.argmax(sliced_transitions.done, axis=1).mean())
        print("num_traj_collected", num_traj_collected)
        print("collected batch obs shape", sliced_transitions.obs.shape)
    max_len = max([x.obs.shape[1] for x in traj_batchs])
    obs = jnp.concatenate([jnp.pad(x.obs, ((0, 0), (0, max_len - x.obs.shape[1]), (0, 0))) for x in traj_batchs], axis=0)
    action = jnp.concatenate([jnp.pad(x.action, ((0, 0), (0, max_len - x.action.shape[1]))) for x in traj_batchs], axis=0)
    reward = jnp.concatenate([jnp.pad(x.reward, ((0, 0), (0, max_len - x.reward.shape[1]))) for x in traj_batchs], axis=0)
    done = jnp.concatenate([jnp.pad(x.done, ((0, 0), (0, max_len - x.done.shape[1])), constant_values=False) for x in traj_batchs], axis=0)
    dataset = Transitions(obs=obs, action=action, reward=reward, done=done)
    print("average data length", jnp.argmax(dataset.done, axis=1).mean())
    print("average return", dataset.reward.sum(axis=1).mean())
    if save_dir is not None and filename is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(f"{save_dir}/{filename}", "wb") as f:
            pickle.dump(dataset, f)
            print(f"saved batch to {save_dir}/{filename}, batch_size: {dataset.obs.shape[0]}")
    return rng, dataset
    
def continuous_dataset_to_trajectories(dataset, max_traj_len=40):
    """
    continous_dataset is a transition with obs shape: (traj_length, num_envs, obs_dim), where trajectories are concatenated
    Convert the dataset to a Transition 
    where the obs shape is (num_traj, max_traj_len, obs_dim), i.e. split the dataset into trajs
    Args:
        dataset (_type_): the dataset to convert
        max_traj_len (_type_): the maximum length of the trajectories. set tobe 1000 by defaulf for mujoco envs.
    Returns:
        trajectories (_type_): the Transition
    """
    obs = []
    action = []
    reward = []
    done = []
    for env_idx in range(dataset.obs.shape[1]):
        start_idx = 0
        start_is_done = True
        # print the keys of the dataset
        # print("Dataset keys", dataset.keys())
        # see if all the terminals are False
        # print("All terminals are False?", not any(dataset.done[:, env_idx]))
        dones = dataset.done[:, env_idx]
        for i in range(len(dones)):
            if i - start_idx == max_traj_len or dones[i]:
                if not start_is_done:
                    start_idx = i + 1
                    start_is_done = True
                    continue
                else:
                    obs.append(dataset.obs[start_idx:i+1, env_idx])
                    action.append(dataset.action[start_idx:i+1, env_idx])
                    reward.append(dataset.reward[start_idx:i+1, env_idx])
                    done.append(dones[start_idx:i+1])
                    # make sure the last done is True for each traj
                    done[-1].at[-1].set(True)
                    
                    start_idx = i + 1
                    start_is_done = dones[i]
    # pad all the trajs to the same length. pad dones with 1
    max_len = max([len(obs[i]) for i in range(len(obs))])
    obs = [jnp.pad(obs[i], ((0, max_len - len(obs[i])), (0, 0))) for i in range(len(obs))]
    action = [jnp.pad(action[i], ((0, max_len - len(action[i])))) for i in range(len(action))]
    reward = [jnp.pad(reward[i], ((0, max_len - len(reward[i])))) for i in range(len(reward))]
    done = [jnp.pad(done[i], ((0, max_len - len(done[i]))), constant_values=False) for i in range(len(done))]
            
    return Transitions(jnp.array(obs), jnp.array(action), jnp.array(reward), jnp.array(done))

    

def get_a_batch_of_trajs(env, agent, rng, num_trajectories=32, save_dir=None, filename=None, merge_interval=8):
    trajectories = []
    num_collected = 0
    maximum_length = 0
    batch = None
    with tqdm.tqdm(total=num_trajectories) as pbar:
        for _ in range(num_trajectories * 2):
            rng, tr = get_one_trajectory(env, agent, rng)
            if tr is not None:
                num_collected += 1
                trajectories.append(tr)
                pbar.update(1)
            else:
                continue
            # assemble the trajs into a large Transition and clear the trajectories every 128 trajs
            if num_collected % merge_interval == 0 and num_collected > 0:
                new_maximum_length = max(max([len(x[0]) for x in trajectories]), maximum_length)
                # pad the trajs to the maximum length
                new_obss = jnp.array([jnp.pad(x[0], ((0, new_maximum_length - len(x[0])), (0, 0))) for x in trajectories])
                new_actions = jnp.array([jnp.pad(x[1], (0, new_maximum_length - len(x[1]))) for x in trajectories])
                new_rewards = jnp.array([jnp.pad(x[2], (0, new_maximum_length - len(x[2]))) for x in trajectories])
                new_dones = jnp.array([jnp.pad(x[3], (0, new_maximum_length - len(x[3]))) for x in trajectories])
                tr = Transitions(obs=new_obss, action=new_actions, reward=new_rewards, done=new_dones)
                if batch is None:
                    batch = tr
                else:
                    if new_maximum_length > maximum_length:
                        # pad the previous trajs to the new maximum length
                        batch = Transitions(
                            obs=jnp.pad(batch.obs, ((0, 0), (0, new_maximum_length - maximum_length), (0, 0))),
                            action=jnp.pad(batch.action, ((0, 0), (0, new_maximum_length - maximum_length))),
                            reward=jnp.pad(batch.reward, ((0, 0), (0, new_maximum_length - maximum_length))),
                            done=jnp.pad(batch.done, ((0, 0), (0, new_maximum_length - maximum_length))),
                        )
                    batch = Transitions(
                        obs=jnp.concatenate([batch.obs, tr.obs]),
                        action=jnp.concatenate([batch.action, tr.action]),
                        reward=jnp.concatenate([batch.reward, tr.reward]),
                        done=jnp.concatenate([batch.done, tr.done]),
                    )
                maximum_length = new_maximum_length
                tr = None # clear the trajs
                trajectories = []
            
            if num_collected >= num_trajectories:
                break
    if save_dir is not None and filename is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(f"{save_dir}/{filename}", "wb") as f:
            pickle.dump(batch, f)
            print(f"saved batch to {save_dir}/{filename}, batch_size: {batch.obs.shape[0]}")
    return rng, batch

if __name__ == "__main__":
    config = parse_args_and_update_config(CollectConfig)
    print("---------------------------------------")
    print(f"Collecting with rule-based agents, Env: {config.env}, Seed: {config.seed}")
    print("---------------------------------------")
    if config.env == "MiniGrid-Reacher-extra-good":
        env = ExtraRewardGridworld(grid_size=7, max_steps=40, distance_penalty=-0.3, goal_reward=10.0, epsilon=config.epsilon, extra_reward=config.extra_reward)
        agent = grid_Reacher_agent_good(env)
    elif config.env == "MiniGrid-Reacher-extra-bad":
        env = ExtraRewardGridworld(grid_size=7, max_steps=40, distance_penalty=-0.3, goal_reward=10.0, epsilon=config.epsilon, extra_reward=-config.extra_reward)
        agent = grid_Reacher_agent_bad(env)
    elif config.env == "MiniGrid-Reacher-extra-med":
        env = ExtraRewardGridworld(grid_size=7, max_steps=40, distance_penalty=-0.3, goal_reward=10.0, epsilon=config.epsilon, extra_reward=0)
        agent = grid_Reacher_agent(env)
    elif config.env == "MiniGrid-Reacher-MDP":
        env = MDPGridworld(max_steps=40, distance_penalty=-0.3, goal_reward=10.0, epsilon=config.epsilon)
        print("MDP_reacher_agent_mode", config.MDP_reacher_agent_mode)
        agent = MDP_reacher_agent(env, mode=config.MDP_reacher_agent_mode)
    elif config.env == "MDPtakeball":
        env = MDPtakeball(max_steps=40, distance_penalty=-0.0, goal_reward=10.0, epsilon=config.epsilon, target_ball=int(config.take_ball_target))
        agent = MDP_takeball_agent(env, mode=int(config.take_ball_target))
    else:
        raise ValueError("Environment: ", config.env, " not supported") 
    rng = jax.random.PRNGKey(config.seed)
    save_dir = "datasets/rule_based/" + config.env
    filename = config.filename
    if config.env == "MiniGrid-Reacher-MDP" or config.env == "MDPtakeball":
        # we can collect MDP data with jax.jit
        with jax.disable_jit(disable=False):
            rng, batch = MDP_get_one_trajectory(env, agent, rng, num_trajectories=config.n_episodes, save_dir=save_dir, filename=filename)
    else:
        rng, batch = get_a_batch_of_trajs(env, agent, rng, num_trajectories=config.n_episodes, save_dir=save_dir, filename=filename, merge_interval=config.merge_interval)
    print("saved batch obs shape", batch.obs.shape)
    print("saved batch action shape", batch.action.shape)
    print("saved batch reward shape", batch.reward.shape)
    print("saved batch done shape", batch.done.shape)
    
    # visualize first 10 trajs
    # for i in range(10):
    #     filename = f"traj_{i}.gif"
    #     env.visualize_obs(batch.obs[i].reshape([-1]+list(env.observation_shape)), filename=filename, rewards=batch.reward[i], actions=batch.action[i], interval=400)