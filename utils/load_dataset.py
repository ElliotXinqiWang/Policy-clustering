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
    done1=jnp.concatenate([dataset.done for dataset in datasets], axis=0)
    done1 = jnp.concatenate([jnp.ones_like(done1[:1]), done1[:-1]])
    dataset = Transitions(
        obs=jnp.concatenate([dataset.obs for dataset in datasets], axis=0),
        action=jnp.concatenate([dataset.action for dataset in datasets], axis=0),
        reward=jnp.concatenate([dataset.reward for dataset in datasets], axis=0),
        done=done1,
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