import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from dataclasses import dataclass, field
from typing import NamedTuple, Optional
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym.spaces.box")
import pickle
import tqdm

import optax
import chex
from flax.training.train_state import TrainState
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import d4rl
import gym
import wandb
import matplotlib.pyplot as plt
import h5py
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from utils.networks import ScannedRNN, ContinuousActorRNN, DiscreteActorRNN, DEC
from gridworld.env import SingleAgentGridworld, FixedGridworld, ExtraRewardGridworld, MDPGridworld, MDPtakeball
from utils.plot_tools import plot_and_save_curves, plot_and_save_bar, plot_and_save_bars, plot_and_save_heatmap

# @dataclass
class Transitions(NamedTuple):
    obs: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    action: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    reward: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    done: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))

def continuous_dataset_to_trajectories(dataset, max_traj_len=20):
    """
    continous_dataset is a transition with obs shape: (traj_length, num_envs, obs_dim), where trajectories are concatenated
    Convert the dataset to a Transition 
    where obs shape is (num_traj, max_traj_len, obs_dim), i.e. split the dataset into trajs
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
                    start_idx = i
                    start_is_done = True
                    continue
                else:
                    obs.append(dataset.obs[start_idx:i+1, env_idx])
                    action.append(dataset.action[start_idx:i+1, env_idx])
                    reward.append(dataset.reward[start_idx:i+1, env_idx])
                    done.append(dones[start_idx:i+1])
                    # make sure the last done is True for each traj
                    done[-1].at[-1].set(True)
                    
                    start_idx = i
                    start_is_done = dones[i]
    # pad all the trajs to the same length. pad dones with 1
    max_len = max([len(obs[i]) for i in range(len(obs))])
    obs = [jnp.pad(obs[i], ((0, max_len - len(obs[i])), (0, 0))) for i in range(len(obs))]
    action = [jnp.pad(action[i], ((0, max_len - len(action[i])))) for i in range(len(action))]
    reward = [jnp.pad(reward[i], ((0, max_len - len(reward[i])))) for i in range(len(reward))]
    done = [jnp.pad(done[i], ((0, max_len - len(done[i]))), constant_values=False) for i in range(len(done))]
            
    return Transitions(jnp.array(obs), jnp.array(action), jnp.array(reward), jnp.array(done))

def load_rule_based_datasets(config):  
    rule_based_dataset_files = config.rule_based_dataset_files
    datasets = []
    for i, path in enumerate(rule_based_dataset_files):
        with open(path, "rb") as f:
            data = pickle.load(f)
            datasets.append(data)
            print("Dataset ", i, " loaded from ", path)
            print("Dataset shape: obs ", data.obs.shape, " action ", data.action.shape, " reward ", data.reward.shape, " done ", data.done.shape)
            print("average return: ", jnp.mean(data.reward.sum(axis=1)))
            print("average length: ", jnp.mean(jnp.argmax(data.done, axis=1) + 1))
    max_traj_len = max([datasets[i].obs.shape[1] for i in range(len(datasets))])
    for i in range(len(datasets)):
        if max_traj_len > datasets[i].obs.shape[1]:
            datasets[i] = datasets[i]._replace(
                obs=jnp.pad(datasets[i].obs, ((0, 0), (0, max_traj_len - datasets[i].obs.shape[1]), (0, 0))),
                action=jnp.pad(datasets[i].action, ((0, 0), (0, max_traj_len - datasets[i].action.shape[1]))),
                reward=jnp.pad(datasets[i].reward, ((0, 0), (0, max_traj_len - datasets[i].reward.shape[1]))),
                done=jnp.pad(datasets[i].done, ((0, 0), (0, max_traj_len - datasets[i].done.shape[1])), constant_values=True)
            )
    dataset = Transitions(
        obs=jnp.concatenate([dataset.obs for dataset in datasets], axis=0),
        action=jnp.concatenate([dataset.action for dataset in datasets], axis=0),
        reward=jnp.concatenate([dataset.reward for dataset in datasets], axis=0),
        done=jnp.concatenate([dataset.done for dataset in datasets], axis=0),
    )
    return dataset, jnp.concatenate([jnp.ones(len(datasets[i].obs)) * i for i in range(len(datasets))])
    

def load_datasets(config):
    datasets = []
    for i, path in enumerate(config.dataset_paths):
        dataset = None
        with tqdm.tqdm(range(config.dataset_sizes[i]), desc=f"Loading dataset {i}") as pbar:
            for j in pbar:
                with open(f"{path}/data_{j}.pkl", "rb") as f:
                    data = pickle.load(f)
                if dataset is None:
                    dataset = data
                else:
                    dataset = dataset._replace(
                        obs=jnp.concatenate([dataset.obs, data.obs], axis=0),
                        action=jnp.concatenate([dataset.action, data.action], axis=0),
                        reward=jnp.concatenate([dataset.reward, data.reward], axis=0),
                        done=jnp.concatenate([dataset.done, data.done], axis=0),
                    )
        sliced_dataset = continuous_dataset_to_trajectories(dataset, max_traj_len=config.max_traj_len)
        datasets.append(sliced_dataset)
        print("Dataset ", i, " loaded from ", path)
        print("Dataset shape: obs ", sliced_dataset.obs.shape, " action ", sliced_dataset.action.shape, " reward ", sliced_dataset.reward.shape, " done ", sliced_dataset.done.shape)
        
    done = jnp.concatenate([jnp.ones_like(done[:1]), done[:-1]])
    dataset = Transitions(
        obs=jnp.concatenate([dataset.obs for dataset in datasets], axis=0),
        action=jnp.concatenate([dataset.action for dataset in datasets], axis=0),
        reward=jnp.concatenate([dataset.reward for dataset in datasets], axis=0),
        done=jnp.concatenate([dataset.done for dataset in datasets], axis=0),
    )
    print("Total dataset shape: obs ", dataset.obs.shape, " action ", dataset.action.shape, " reward ", dataset.reward.shape, " done ", dataset.done.shape)
    data_idx = jnp.concatenate([jnp.ones(len(datasets[i].obs)) * i for i in range(len(datasets))]) # record the index of each trajectory for calculating the statistics
    if not os.path.exists("datasets/" + config.env):
        os.makedirs("datasets/" + config.env)
    with open("datasets/" + config.env + "/converted.pkl", "wb") as f:
        pickle.dump({"dataset": dataset, "data_idx": data_idx}, f)
        print("Sliced dataset saved to ", "datasets/" + config.env + "/converted.pkl")
    return dataset, data_idx

def load(config):
        
    # load dataset from local if exists
    dataset_filename = config.env + "/" + "|".join([str(size) for size in config.dataset_sizes]) + ".pkl"
    D4RL_envs = ["halfcheetah-medium-expert-v2","walker2d-medium-expert-v2","hopper-medium-expert-v2","ant-medium-expert-v2"]

    if config.env in D4RL_envs:
        print("Load dataset from local: ", f"datasets/{config.env}.hdf5")
        with h5py.File(f"datasets/{config.env}.hdf5", "r") as f:
            obs = jnp.array(f["obs"][:])
            action = jnp.array(f["action"][:])
            reward = jnp.array(f["reward"][:])
            done = jnp.array(f["done"][:])
        dataset = Transitions(obs, action, reward, done)
        returns = dataset.reward.sum(axis=1)
        cumsum_returns = jnp.cumsum(returns)
        medium_avg = cumsum_returns / jnp.arange(1, len(returns) + 1)
        total_sum = cumsum_returns[-1]
        expert_avg = (total_sum - cumsum_returns[:-1]) / jnp.arange(len(returns) - 1, 0, -1)
        diff = expert_avg - medium_avg[:-1]
        expert_start_idx = jnp.argmax(diff)
        print("This info only works for med-expert dataset. Expert start idx: ", expert_start_idx)
        print("medium average returns: ", jnp.mean(returns[:expert_start_idx]), "num of medium trajs: ", expert_start_idx)
        print("expert average returns: ", jnp.mean(returns[expert_start_idx:]), "num of expert trajs: ", len(dataset.obs) - expert_start_idx)
        data_idx = jnp.concatenate([jnp.zeros(expert_start_idx), jnp.ones(len(dataset.obs) - expert_start_idx)])
    elif config.env in ["MDPtakeball", "MiniGrid-Reacher-MDP"] or config.load_from_rule_based_dataset:
        dataset, data_idx = load_rule_based_datasets(config)

        # cnt=[0]*(int(jnp.max(data_idx)) + 1)
        # for i in range(len(data_idx)):
        #     idx=round(data_idx[i])
        #     if cnt[idx]<5:
        #         cnt[idx]+=1
        #         env.visualize_obs(dataset.obs[i].reshape(-1,7,7,4),f"logs/{config.env}_{idx}_{cnt[idx]}.gif",dataset.reward[i],dataset.action[i])
    elif not os.path.exists("datasets/" + dataset_filename):
        dataset, data_idx = load_datasets(config)
    else:
        with open("datasets/" + dataset_filename, "rb") as f:
            data = pickle.load(f)
            dataset = data["dataset"]
            data_idx = data["data_idx"]
    print("Dataset loaded, dataset size: ", dataset.obs.shape[0])
    # set the last done to be True if the episode is not done
    dataset = dataset._replace(done=jnp.concatenate([jnp.zeros_like(dataset.done[:, -1:]), dataset.done[:, :-1]], axis=1))
    dataset = dataset._replace(done=jnp.concatenate([dataset.done[:, :-1], jnp.ones_like(dataset.done[:, -1:])], axis=1))
    return dataset, data_idx

def load_env(config):
    if config.env == "MiniGrid-Reacher":
        env = SingleAgentGridworld(grid_size=7, max_steps=20, distance_penalty=-0.5, goal_reward=10.0)
    elif config.env == "MiniGrid-Binary-Reacher":
        env = FixedGridworld(K=3, max_steps=20, distance_penalty=-0.5, goal_reward=10.0)
    elif config.env == "MiniGrid-Reacher-noisy":
        env = SingleAgentGridworld(grid_size=7, max_steps=20, distance_penalty=-0.5, goal_reward=10.0, epsilon=0.5)
    elif config.env == "MiniGrid-Reacher-extra-good":
        env = ExtraRewardGridworld(grid_size=7, max_steps=40, distance_penalty=-0.3, goal_reward=10.0, epsilon=config.epsilon, extra_reward=config.extra_reward)
    elif config.env == "MiniGrid-Reacher-extra-bad":
        env = ExtraRewardGridworld(grid_size=7, max_steps=40, distance_penalty=-0.3, goal_reward=10.0, epsilon=config.epsilon, extra_reward=-config.extra_reward)
    elif config.env == "MiniGrid-Reacher-extra-med":
        env = ExtraRewardGridworld(grid_size=7, max_steps=40, distance_penalty=-0.3, goal_reward=10.0, epsilon=config.epsilon, extra_reward=0)
    elif config.env == "MiniGrid-Reacher-MDP":
        env = MDPGridworld(max_steps=40, distance_penalty=-0.3, goal_reward=10.0, epsilon=config.epsilon)
    elif config.env == "MDPtakeball" or config.env == "MDPtakeball-hard":
        env = MDPtakeball(max_steps=40, distance_penalty=-0.0, goal_reward=10.0, epsilon=config.epsilon, target_ball=int(config.take_ball_target))
    else:
        env = gym.make(config.env)
        # raise ValueError("Environment: ", config.env, " not supported") 
    return env