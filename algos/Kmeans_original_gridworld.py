# In gridworld, we have a state and partial observation. Here the state is the observation.

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
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import optax
import chex
from flax.training.train_state import TrainState
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
# import d4rl
# import gym
import wandb
import matplotlib.pyplot as plt

from utils.networks import ScannedRNN, ContinuousActorRNN, DiscreteActorRNN
from gridworld.env import SingleAgentGridworld, FixedGridworld, ExtraRewardGridworld, MDPGridworld, MDPtakeball
from utils.plot_tools import plot_and_save_curves, plot_and_save_bar, plot_and_save_bars, plot_and_save_heatmap

@dataclass
class TrainConfig:
    # General
    device: str = "cuda"
    device_id: list = field(default_factory=lambda: [0])
    debug: bool = False
    disable_jit: bool = False
    extra_attributes: dict = field(default_factory=dict, init=False)
    make_plots: bool = True
    visualize_worst: bool = False
    evaluate: bool = True
    # Experiment
    alg: str = "Kmeans"  # Algorithm name
    env: str = "MiniGrid-Reacher-extra-good"  # Minigrid environment name
    seed: int = 1  # Sets Gym, Jax and Numpy seeds
    max_updates: int = 15  # Number of updates
    n_episodes: int = 128  # How many episodes run during evaluation
    checkpoints_path: Optional[str] = None  # Save path
    batch_size: int = 2048  # Batch size for all networks
    load_from_rule_based_dataset: bool = True
    true_k_available: bool = True
    use_rnn: bool = False
    dataset_paths: list[str] = field(default_factory=lambda: [
        "datasets/MiniGrid-Reacher-extra-med/good",
        "datasets/MiniGrid-Reacher-extra-med/bad",
        "datasets/MiniGrid-Reacher-extra-med/med",
    ])
    rule_based_dataset_files: list[str] = field(default_factory=lambda: [
        "datasets/rule_based/MiniGrid-Reacher-MDP/balanced_20000.pkl",
        "datasets/rule_based/MiniGrid-Reacher-MDP/rightfirst_20000.pkl",
        "datasets/rule_based/MiniGrid-Reacher-MDP/downfirst_20000.pkl",
        "datasets/rule_based/MiniGrid-Reacher-MDP/zigzag1_20000.pkl",
        "datasets/rule_based/MiniGrid-Reacher-MDP/zigzag2_20000.pkl",
    ])
    # dataset_sizes: list[int] = field(default_factory=lambda: [5, 0, 0, 5])
    dataset_sizes: list[int] = field(default_factory=lambda: [5, 5, 5])
    epsilon: float = 0.1
    extra_reward: float = 10.0
    take_ball_target: int = 0
    # Network
    hidden_dim: int = 128
    max_grad_norm: float = 0.5
    learning_rate: float = 1e-3
    adam_eps: float = 1e-8
    maximum_iterations: int = 50
    # Kmeans
    K_value: int = 4 # Number of clusters
    max_traj_len: int = 1000  # Max trajectory length
    normalize: bool = False  # Normalize states
    # Wandb logging
    project: str = "1217KmeansGBM"
    group: str = "PKmeans"
    name: str = ""

    def __post_init__(self):
        # self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        self.name = f"{self.name}-{self.env}-{self.K_value}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
            
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

# @dataclass
class Transitions(NamedTuple):
    obs: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    action: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    reward: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    done: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    
    
def parse_args_and_update_config(config_class):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rule_based_dataset_files",
        nargs='+',  # 
        help="List of rule-based dataset file paths"
    )
    #  dataclass ，
    for field_name, field_info in config_class.__dataclass_fields__.items():
        default = field_info.default
        if isinstance(default, (int, float, str, bool)):
            parser.add_argument(f"--{field_name}", type=type(default), default=default, help=f"Default: {default}")
        elif default is None:
            parser.add_argument(f"--{field_name}", type=str, default=None, help="Default: None")
    # 
    args = parser.parse_args()

    # 
    kwargs = vars(args)
    config = config_class(**kwargs)
    return config

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
    # raise ValueError("Not implemented")
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
        if config.dataset_sizes[i] == 0:
            continue
        dataset = None
        print("config.dataset_sizes[i]: ", config.dataset_sizes[i])
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
    filename = config.env + "/" + "|".join([str(size) for size in config.dataset_sizes]) + ".pkl"
    with open("datasets/" + filename, "wb") as f:
        pickle.dump({"dataset": dataset, "data_idx": data_idx}, f)
        print("Sliced dataset saved to ", "datasets/" + filename)
    return dataset, data_idx
    
def compute_mean_std(data, eps=1e-3):
    mean = jnp.mean(data, axis=(0, 1))
    std = jnp.std(data, axis=(0, 1)) + eps
    return mean, std

def train(config):
    # Set random seed
    rng = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)
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
    elif config.env == "MDPtakeball":
        env = MDPtakeball(max_steps=40, distance_penalty=-0.0, goal_reward=10.0, epsilon=config.epsilon, target_ball=int(config.take_ball_target))
    else:
        raise ValueError("Environment: ", config.env, " not supported") 
    
    # config.state_dim = env.observation_space.shape[0]
    # config.action_dim = env.action_space.shape[0]
    
    config.action_dim = 5
    obs_shape = env.observation_shape
    config.state_dim = obs_shape.prod()
    
    print(f"State dim: {config.state_dim}, Action dim: {config.action_dim}")
    
    # load dataset from local if exists
    dataset_filename = config.env + "/" + "|".join([str(size) for size in config.dataset_sizes]) + ".pkl"
    if config.load_from_rule_based_dataset:
        dataset, data_idx = load_rule_based_datasets(config)
    elif not os.path.exists("datasets/" + dataset_filename):
        dataset, data_idx = load_datasets(config)
    else:
        with open("datasets/" + dataset_filename, "rb") as f:
            data = pickle.load(f)
            dataset = data["dataset"]
            data_idx = data["data_idx"]
    print("Dataset loaded, dataset size: ", dataset.obs.shape[0])
    # # filter out dead episodes by removing the trajs with length 20 or 40
    # dataset_lengths = jnp.argmax(dataset.done, axis=1) + 1
    # needed_episode_idx = jnp.where((dataset_lengths != 20) & (dataset_lengths != 40))[0]
    # filter out episodes with short length, which can be easily learned by the network
    traj_lengths = jnp.argmax(dataset.done, axis=1) + 1
    # needed_episode_idx = jnp.where(total_return > 0.6)[0]
    needed_episode_idx = jnp.where(traj_lengths > 3)[0]
    dataset = Transitions(
        dataset.obs[needed_episode_idx],
        dataset.action[needed_episode_idx],
        dataset.reward[needed_episode_idx],
        dataset.done[needed_episode_idx]
    )
    data_idx = data_idx[needed_episode_idx]
    print("Dataset filtered, remaining dataset size: ", dataset.obs.shape[0])
    
    assert dataset.obs.shape[0] > 16 * config.K_value, "Dataset is too small, we need at least 16 trajs for each network"
    
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset.obs, eps=1e-3)
    else:
        state_mean, state_std = 0, 1
    # Shuffle the dataset
    dataset = dataset._replace(obs=(dataset.obs - state_mean) / state_std)
    print("config.debug:", config.debug)
    if config.debug:
        print("not shuffle the dataset for debug")
        idx = jnp.arange(len(dataset.obs))
    else:
        print("randomly shuffle the dataset")
        idx = jax.random.permutation(rng, len(dataset.obs))
    
    dataset = Transitions(dataset.obs[idx], dataset.action[idx], dataset.reward[idx], dataset.done[idx])
    
    
    data_idx = data_idx[idx]
    
    
    # Initialize K datasets by randomly seperating the dataset into K parts
    dataset_idxs = [] # record the index of each dataset
    for i in range(config.K_value):
        start_inx = i * len(dataset.obs) // config.K_value
        end_idx = (i + 1) * len(dataset.obs) // config.K_value
        dataset_idxs.append(jnp.arange(start_inx, end_idx))
    initial_nmi = normalized_mutual_info_score(data_idx, jnp.concatenate([jnp.ones(len(dataset_idxs[i])) * i for i in range(config.K_value)]))
    initial_ari = adjusted_rand_score(data_idx, jnp.concatenate([jnp.ones(len(dataset_idxs[i])) * i for i in range(config.K_value)]))
    print("Initial NMI: ", initial_nmi, " ARI: ", initial_ari)
    wandb.log({"NMI": initial_nmi, "ARI": initial_ari}, step=0)
    # if config.debug:
    #     dataset_idxs = []
    #     print("assigning datasets according to its index")
    #     for i in range(config.K_value):
    #         dataset_idxs.append(jnp.where(data_idx == i)[0])
    # Initialize K networks and create training states
    train_states = []
    model = DiscreteActorRNN(config.action_dim, config, not_rnn=not config.use_rnn)
    tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=config.learning_rate, eps=config.adam_eps),
        )
    # make a dummy observation with batch_size 1 and seq_len 2
    dummy_obs = jnp.zeros((2, 1, config.state_dim))
    dummy_dones = jnp.zeros((2, 1))
    dummy_h_state = ScannedRNN.initialize_carry(1, config.hidden_dim)
    for i in range(config.K_value):
        _rng_init, rng = jax.random.split(rng)

        params = model.init(_rng_init, dummy_h_state, (dummy_obs, dummy_dones))
        train_state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
        )
        train_states.append(train_state)
    
    
    def train_one_network(train_state, dataset_idx, rng, time_average_window=5, maximum_iterations=50, threshold=1e-6):
        """
        Train one network with the given dataset. Train until the convergence, i.e., the loss is not decreasing.
        """
        paded_size = math.ceil(len(dataset_idx) / config.batch_size) * config.batch_size
        def train_one_epoch(state_data_rng):
            train_state, obs, action, reward, done, rng = state_data_rng
            
            # shuffle pad the dataset to make sure the size is a multiple of batch_size
            # randomly choose data from the dataset to pad
            shuffle_rng, rng = jax.random.split(rng)
            shuffle_idx = jax.random.permutation(shuffle_rng, len(obs))
            
            pad_idx = jax.random.choice(rng, a=config.batch_size, shape=(paded_size - len(obs),), replace=False)
            shuffle_and_pad = lambda x: jnp.concatenate((x[shuffle_idx], x[pad_idx]), axis=0)
            # current shape: (num_traj, max_traj_len, data_dim)
            batchify = lambda x: jnp.reshape(x, [-1, config.batch_size] + list(x.shape[1:]))
            # current shape: (num_batch, batch_size, max_traj_len, data_dim)
            preprocess = lambda x: jnp.swapaxes(batchify(shuffle_and_pad(x)), 1, 2)
            # current shape: (num_batch, max_traj_len, batch_size, data_dim)
            # preprocess = lambda x: jnp.swapaxes(jnp.reshape(x, [-1, config.batch_size] + list(x.shape[1:])), 1, 2)
            trainset = [
                preprocess(obs),
                preprocess(action),
                preprocess(reward),
                preprocess(done)
            ]
            def train_one_batch(train_state, batch):
                def _loss_fn(params, h_state, batch):
                    # data shape: (max_traj_len, batch_size, data_dim)
                    obs, action, reward, done = batch
                    h_state, pi = model.apply(params, h_state, (obs, done))
                    # mask out the probs strictly after first done (exclusive)
                    done_mask = jnp.cumprod(1 - done.astype(jnp.int32), axis=0)
                    loss = -pi.log_prob(action)
                    loss = jnp.sum(done_mask * loss, axis=(0, 1)) / jnp.sum(done_mask, axis=(0, 1))
                    return loss
                # def print_input(batch):
                #     print("batch shape: ", jnp.mean(batch[0], axis=(0, 1)))
                #     raise ValueError("Stop")
                # jax.experimental.io_callback(print_input, None, batch)
                h_state = ScannedRNN.initialize_carry(config.batch_size, config.hidden_dim)
                grad_fn = jax.value_and_grad(_loss_fn)
                loss, grad = grad_fn(train_state.params, h_state, batch)
                train_state = train_state.apply_gradients(grads=grad)
                return train_state, loss
            train_state, losses = jax.lax.scan(train_one_batch, train_state, trainset)
            return train_state, losses.mean()
        
        jit_train_one_epoch = jax.jit(train_one_epoch)
        loss_history = []
        for i in range(maximum_iterations):
            rng, sub_rng = jax.random.split(rng)
            dataset_one_network = [
                dataset.obs[dataset_idx],
                dataset.action[dataset_idx],
                dataset.reward[dataset_idx],
                dataset.done[dataset_idx]
            ]
            train_state, loss = jit_train_one_epoch((train_state, *dataset_one_network, sub_rng))
            loss_history.append(loss)
            if len(loss_history) >= time_average_window + 1 and \
                jnp.abs(jnp.array(loss_history[-time_average_window:]) - jnp.array(loss_history[-time_average_window - 1:-1])).max() < threshold:
                print("Converged after ", i, "iterations")
                break
        print("Training loss: ", loss)
        return train_state, loss
        
    last_idxs = [jnp.zeros_like(dataset_idxs[i]) for i in range(config.K_value)]
    
    print("Start training")
    epoch = 0
    # log the size curve of each dataset in one plot
    plot_data = {"dataset_size":[],
                 "dataset_averages": [],
                 "dataset_stds": [],
                 "average_returns": [],
                 "NMI": [],
                 "ARI": [],}
    plot_data["dataset_size"].append([len(dataset_idxs[i]) for i in range(config.K_value)])
    
    while jnp.array([not jnp.array_equal(dataset_idxs[i], last_idxs[i]) and len(dataset_idxs[i])>10 for i in range(config.K_value)]).any():
        if epoch >= config.max_updates:
            break
        print("Epoch: ", epoch, "dataset sizes: ", [len(dataset_idxs[i]) for i in range(config.K_value)])
        # print("last_idx type", type(last_idxs[0]))
        last_idx_set = [set(np.array(last_idxs[i]).tolist()) for i in range(config.K_value)]
        idx_set = [set(np.array(dataset_idxs[i]).tolist()) for i in range(config.K_value)]
        l1_diff = [len(last_idx_set[i].symmetric_difference(idx_set[i])) for i in range(config.K_value)]
        print("l1 difference between idxs:", l1_diff)
        print("percentage of data changed: ", [f"{(l1_diff[i] / max(len(last_idx_set[i]), 1)):.2f}" for i in range(config.K_value)])
        epoch += 1
        # Train each network
        for i in range(config.K_value):
            # train the network if last_idxs[i] is different from dataset_idxs[i], and the dataset is not too small
            
            if not jnp.array_equal(dataset_idxs[i], last_idxs[i]) and len(dataset_idxs[i])>10:
                # print("average train action: ", jnp.mean(K_datasets[i].action, axis=(0, 1)))
                # print the statistics of the data
                train_state = train_states[i]
                _train_rng, rng = jax.random.split(rng)
                train_states[i], loss = train_one_network(train_state, dataset_idxs[i], _train_rng, maximum_iterations=config.maximum_iterations)
        
        # Predict the probability of each traj generated by each network
        dataset_to_be_inferenced = Transitions(
            dataset.obs.swapaxes(0, 1),
            dataset.action.swapaxes(0, 1),
            dataset.reward.swapaxes(0, 1),
            dataset.done.swapaxes(0, 1)
        )
    
        def _get_probability_of_traj(hidden, actions, obs, dones, params):
            # mask out the probs strictly after first done (exclusive)
            # clip the logprob to [-3, infty]
            out_hidden, pi = model.apply(params, hidden, (obs, dones))
            prob = pi.log_prob(actions)
            prob = jnp.clip(prob, -3, jnp.inf)
            # move it to 0 if log_prob > -0.1
            # prob = jnp.where(prob > -0.3, 0, -1) # shape: (max_traj_len, batch_size)
            
            done_mask = jnp.cumprod(1 - dones.astype(jnp.int32), axis=0)
            # traj_prob = jnp.sum(prob * done_mask, axis=0)
            # traj_prob = jnp.mean(prob * done_mask, axis=0)
            traj_prob = jnp.sum(prob * done_mask, axis=0) / jnp.sum(done_mask, axis=0)
            
            return out_hidden, traj_prob
        _jitted_get_probability_of_traj = jax.jit(_get_probability_of_traj)
        prob = []
        
        with jax.disable_jit(config.disable_jit):
            for i in range(config.K_value):
                _, probs = _jitted_get_probability_of_traj(
                    ScannedRNN.initialize_carry(len(dataset.obs), config.hidden_dim),
                    dataset_to_be_inferenced.action,
                    dataset_to_be_inferenced.obs,
                    dataset_to_be_inferenced.done,
                    train_states[i].params
                )
                prob.append(probs)
        prob = jnp.array(prob) # shape: (K_value, dataset_size)
        print("prob shape: ", prob.shape)
        print("average prob: ", jnp.mean(prob, axis=1))
        # Group the trajs according to the network with the highest probability
        highest_prob_idx = jnp.argmax(prob, axis=0)
        print("average in-dataset prob: ", [f"{jnp.mean(prob[i][dataset_idxs[i]]).item():.4f}" for i in range(config.K_value)])
        
        nmi = normalized_mutual_info_score(data_idx, highest_prob_idx)
        ari = adjusted_rand_score(data_idx, highest_prob_idx)
        wandb.log({"NMI": nmi, "ARI": ari}, step=epoch)
        print("NMI: ", nmi, " ARI: ", ari)
        plot_data["NMI"].append(nmi)
        plot_data["ARI"].append(ari)
        last_idxs = dataset_idxs.copy()
        dataset_idxs = [jnp.where(highest_prob_idx == i)[0] for i in range(config.K_value)]
        plot_data["dataset_size"].append([len(dataset_idxs[i]) for i in range(config.K_value)])
        
    wandb.log({"before merge NMI": nmi, "before merge ARI": ari})
    print("Epoch: ", epoch, "dataset sizes: ", [len(dataset_idxs[i]) for i in range(config.K_value)])
    # print("last_idx type", type(last_idxs[0]))
    last_idx_set = [set(np.array(last_idxs[i]).tolist()) for i in range(config.K_value)]
    idx_set = [set(np.array(dataset_idxs[i]).tolist()) for i in range(config.K_value)]
    l1_diff = [len(last_idx_set[i].symmetric_difference(idx_set[i])) for i in range(config.K_value)]
    print("l1 difference between idxs:", l1_diff)
    print("percentage of data changed: ", [f"{(l1_diff[i] / max(len(last_idx_set[i]), 1)):.2f}" for i in range(config.K_value)])
    
    # Merge the datasets together until the total number of datasets matches the ground truth
    print("objective function value(total probs):", sum([prob[i][dataset_idxs[i]].sum() for i in range(config.K_value)]))
    
    if config.true_k_available:
        true_k = max(data_idx) + 1
        remaining_k = sum([len(dataset_idxs[i]) > 0 for i in range(config.K_value)])
        if remaining_k < true_k:
            print("num of clusters smaller than the true num of clusters")
        else:
            # save the heatmap before merging
            num_catagories = int(jnp.max(data_idx)) + 1
            heatmap_matrix = np.zeros((config.K_value, num_catagories))
            for i in range(config.K_value):
                for j in range(num_catagories):
                    heatmap_matrix[i, j] = jnp.sum(data_idx[dataset_idxs[i]] == j)
            plot_data["heatmap_matrix_before_merge"] = heatmap_matrix
            
            print("merging for ", remaining_k - true_k, " times")
            dataset_idxs = [np.array(dataset_idxs[i]) for i in range(config.K_value)]
            prob_s = [np.array(prob[i]) for i in range(config.K_value)]
            for _ in range(int(remaining_k - true_k)):
                # merge the two datsets such that the total probability of the merged dataset is the highest
                merge_cost = np.ones((len(dataset_idxs), len(dataset_idxs))) * (- np.inf) # record the cost of merging i into j
                for i in range(len(dataset_idxs)):
                    if len(dataset_idxs[i]) == 0:
                        continue
                    for j in range(len(dataset_idxs)):
                        if len(dataset_idxs[j]) == 0:
                            continue
                        merge_cost[i][j] = - np.sum(prob_s[i][dataset_idxs[i]]) + np.sum(prob_s[j][dataset_idxs[i]])
                    # set the cost of merging i into itself to be infinity
                    merge_cost[i][i] = -np.inf
                # find the pair with the minimum cost
                min_idx = np.unravel_index(np.argmax(merge_cost), merge_cost.shape)
                print("Merging ", min_idx[0], " into ", min_idx[1], " with cost: ", merge_cost[min_idx], "cost per traj: ", merge_cost[min_idx] / len(dataset_idxs[min_idx[0]]))
                dataset_idxs[min_idx[1]] = np.concatenate([dataset_idxs[min_idx[1]], dataset_idxs[min_idx[0]]])
                dataset_idxs[min_idx[0]] = np.array([])
            dataset_idxs = [jnp.array(dataset_idxs[i], dtype=int) for i in range(config.K_value)]
            print("final dataset sizes after merging: ", [len(dataset_idxs[i]) for i in range(config.K_value)])
            print("objective function value(total probs) after merging:", sum([prob[i][dataset_idxs[i]].sum() for i in range(config.K_value)]))    
        plot_data["dataset_size"].append([len(dataset_idxs[i]) for i in range(config.K_value)])
    
    # get the statistics of the each dataset
    for i in range(config.K_value):
        if len(dataset_idxs[i]) < 10:
            # skip the degenerate datasets
            plot_data["dataset_averages"].append(0)
            plot_data["dataset_stds"].append(0)
        else:
            total_return = jnp.sum(dataset.reward[dataset_idxs[i]], axis=1)
            Mreturn = jnp.mean(total_return)
            NormMreturn = env.get_normalized_score(Mreturn) * 100
            Stdreturn = jnp.std(total_return)
            plot_data["dataset_averages"].append(NormMreturn)
            plot_data["dataset_stds"].append(Stdreturn)
    
    num_catagories = int(jnp.max(data_idx)) + 1
    heatmap_matrix = np.zeros((config.K_value, num_catagories))
    for i in range(config.K_value):
        for j in range(num_catagories):
            heatmap_matrix[i, j] = jnp.sum(data_idx[dataset_idxs[i]] == j)
    plot_data["heatmap_matrix"] = heatmap_matrix
    # make the predicted idx with latest dataset_idxs
    predicted_idx = np.zeros_like(data_idx)
    for i in range(config.K_value):
        predicted_idx[np.array(dataset_idxs[i], dtype=int)] = i
    nmi = normalized_mutual_info_score(data_idx, predicted_idx)
    ari = adjusted_rand_score(data_idx, predicted_idx)
    wandb.log({"NMI": nmi, "ARI": ari})
    print("NMI: ", nmi, " ARI: ", ari)
    
    if config.evaluate:
        # train the network for the last time until convergence for final evaluation
        print("Training for the last time")
        for i in range(config.K_value):
            # train the network if the dataset is not too small
            if len(dataset_idxs[i]) > 50:
                _train_rng, rng = jax.random.split(rng)
                train_states[i], loss = train_one_network(train_states[i], dataset_idxs[i], _train_rng, threshold=1e-4)
        
        
        print("Training finished. Start evaluation.")
        # Evaluate and save the models
        K_rewards = []
        K_reward_stds = []
        for i in range(config.K_value):
            # save the parameters
            if not os.path.exists(f"models/{config.K_value}_{config.env}"):
                os.makedirs(f"models/{config.K_value}_{config.env}", exist_ok=True)
            model_path = f"models/{config.K_value}_{config.env}/model_{i}.pkl"
            with open(model_path, "wb") as f:
                f.write(serialization.to_bytes(train_states[i].params))
            print("Model saved to ", model_path)
        
    if config.visualize_worst:
        # visualize the worst trajs in each dataset
        idx_prob_truth = []
        for i in range(config.K_value):
            if len(dataset_idxs[i]) == 0:
                continue
            idx_prob_truth.append(np.array([dataset_idxs[i], prob[i][dataset_idxs[i]], data_idx[dataset_idxs[i]]]))
        # sort the idx_prob_truth according to the probability
        print("idx_prob_truth shapes", idx_prob_truth[0][0].shape, idx_prob_truth[0][1].shape, idx_prob_truth[0][2].shape)
        print("idx_prob_truth samples", idx_prob_truth[0][0][:3], idx_prob_truth[0][1][:3], idx_prob_truth[0][2][:3])
        sorted_idxs = [np.argsort(idx_prob_truth[i][1, :]) for i in range(config.K_value)]
        print("sorted idxs[0]: ", sorted_idxs[0][:10])
        idx_prob_truth = [idx_prob_truth[i][:, sorted_idxs[i]] for i in range(config.K_value)]
        # idx_prob_truth = [sorted(idx_prob_truth[i], key=lambda x: x[1]) for i in range(config.K_value)]
        print("sorted idx_prob_truth shapes", idx_prob_truth[0][0].shape, idx_prob_truth[0][1].shape, idx_prob_truth[0][2].shape)
        print("sorted idx_prob_truth samples", idx_prob_truth[0][0][:3], idx_prob_truth[0][1][:3], idx_prob_truth[0][2][:3])
        # visualize 3 trajs with the lowest probability in each dataset
        
        for i in range(config.K_value):
            sorted_idx = idx_prob_truth[i][0].astype(int)
            sorted_prob = idx_prob_truth[i][1]
            sorted_truth = idx_prob_truth[i][2].astype(int)
            worst_lengths = jnp.argmax(dataset.done[sorted_idx], axis=1)
            print("Worst trajs in dataset ", i, " with probability: ", sorted_prob[:3], "and length: ", worst_lengths[:3], "and idx: ", sorted_idx[:3], "and truth: ", sorted_truth[:3])
            if not os.path.exists(f"visualization/{config.env}/Kmeans_{config.K_value}"):
                os.makedirs(f"visualization/{config.env}/Kmeans_{config.K_value}", exist_ok=True)
            for j in range(3):
                idx = sorted_idx[j]
                traj_len = worst_lengths[j]
                filename = f"visualization/{config.env}/Kmeans_{config.K_value}/worst_traj_{i}_{j}.gif"
                # env.visualize_obs(dataset.obs[idx], filename, rewards=dataset.reward[idx], actions=dataset.action[idx])
                obs = dataset.obs[idx][:traj_len + 1].reshape([-1] + obs_shape.tolist())
                env.visualize_obs(obs, filename, rewards=dataset.reward[idx][:traj_len + 1], actions=dataset.action[idx][:traj_len + 1], interval=500)
        
    
        
    def _eval_one_network_step(rng, eval_params):
        def _eval_step(eval_state, eval_timestep):
            last_obs, state, last_done, hstate, rng = eval_state
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            hstate, pi = model.apply(eval_params, hstate, ac_in)
            action = pi.sample(seed=_rng).squeeze()

            # STEP ENV
            rng, step_key = jax.random.split(rng)
            step_keys = jax.random.split(step_key, config.n_episodes)
            obs, state, reward, done = jax.vmap(env.step, in_axes=(0, 0, 0))(step_keys, state, action)
            # normalize the obs
            obs = (obs.reshape((obs.shape[0], -1)) - state_mean) / state_std
            eval_state = (obs, state, done, hstate, rng)
            return eval_state, (done, reward)
        
        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, config.n_episodes)
        obs, state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)
        obs = obs.reshape((config.n_episodes, -1))
        obs = (obs - state_mean) / state_std
        init_hstate = ScannedRNN.initialize_carry(config.n_episodes, 128)
        eval_state = (obs, state, jnp.zeros((config.n_episodes), dtype=bool), init_hstate, rng)
        
        eval_state, evaluated_reward = jax.lax.scan(_eval_step, eval_state, None, config.max_traj_len)
        # create a mask to mask out the rewards after the first done(exclusive)
        # done_mask = jnp.cumprod(1 - evaluated_reward[0].astype(jnp.int32), axis=0)
        dones, rewards = evaluated_reward
        done_mask = jnp.cumprod(1 - dones.astype(jnp.int32), axis=0)
        evaluated_returns = jnp.sum(rewards * done_mask, axis=0)
        return evaluated_returns
    
    evaluated_rewards = []
    for i in range(config.K_value):
        rng, eval_rng = jax.random.split(rng)
        evaluated_rewards.append(_eval_one_network_step(eval_rng, train_states[i].params))
    # jitted_eval = jax.jit(jax.vmap(_eval_one_network_step, in_axes=(0, 0)))
    # rng, eval_rng = jax.random.split(rng)
    # eval_rngs = jax.random.split(eval_rng, config.K_value)
    # evaluated_rewards = jitted_eval(eval_rngs, jnp.arange(config.K_value))
    # evaluated_rewards = jax.lax.scan(jitted_eval, rng, jnp.arange(config.K_value))
    # evaluated_rewards shape: (K_value, n_episodes)
    K_rewards = [env.get_normalized_score(evaluated_rewards[i].mean()) * 100 for i in range(config.K_value)]
    K_reward_stds = [evaluated_rewards[i].std() for i in range(config.K_value)]
    
    print("Normalized Rewards of each network: ", *K_rewards)
    plot_data["average_returns"] = K_rewards
    plot_data["return_stds"] = K_reward_stds
    
    # draw a distribution curve of the returns in each dataset
    dataset_return = dataset.reward.sum(axis=1)
    dataset_bins = np.linspace(dataset_return.min(), dataset_return.max(), 40)
    dataset_hist = []
    for i in range(config.K_value):
        hist, _ = np.histogram(dataset_return[dataset_idxs[i]], bins=dataset_bins, density=False)
        dataset_hist.append(hist)
    plot_data["dataset_hist"] = dataset_hist
    
    if config.make_plots:
        # save the data 
        plot_data["dataset_size"] = jnp.array(plot_data["dataset_size"])
        plot_data["dataset_averages"] = jnp.array(plot_data["dataset_averages"])
        plot_data["dataset_stds"] = jnp.array(plot_data["dataset_stds"])
        plot_data["average_returns"] = jnp.array(plot_data["average_returns"])
        plot_data["return_stds"] = jnp.array(plot_data["return_stds"])
        plot_data["dataset_hist"] = jnp.array(plot_data["dataset_hist"])
        
        data_save_path = f"results/{config.alg}/{config.env}/{config.K_value}/data"
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path, exist_ok=True)
        with open(os.path.join(data_save_path, "plot_data.pkl"), "wb") as f:
            pickle.dump(plot_data, f)
        print("Data saved to ", os.path.join(data_save_path, "plot_data.pkl")) 
        
        # log the dataset size curve
        plot_save_path = f"results/{config.alg}/{config.env}/{config.K_value}/plots"
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path, exist_ok=True)
        dataset_size_image = plot_and_save_curves(
            curves=plot_data["dataset_size"].T,
            title="Dataset Size",
            xlabel="Epoch",
            ylabel="# of Trajectories",
            labels=[f"Dataset_{i}" for i in range(config.K_value)],
            save_path=os.path.join(plot_save_path, "dataset_size_curve.png"),
            max_y=len(dataset.obs),
            min_y=0
        )
        wandb.log({"dataset_size_curve": wandb.Image(dataset_size_image)})
        
        dataset_hist_image = plot_and_save_curves(
            curves=plot_data["dataset_hist"],
            title="Returns Distribution",
            xlabel="Returns",
            ylabel="# of Trajectories",
            labels=[f"D_{i}" for i in range(config.K_value)],
            save_path=os.path.join(plot_save_path, "dataset_hist_curve.png"),
            min_y=0
        )
        # wandb.log({"dataset_hist_curve": wandb.Image(dataset_hist_image)})
            
        
        #log the final dataset_size, average returns and the statistics of the dataset
        final_dataset_size_image = plot_and_save_bar(
            data=plot_data["dataset_size"][-1],
            title="Final Dataset Size",
            xlabel="Dataset Index",
            ylabel="# of Trajectories",
            categories=[f"{i}" for i in range(config.K_value)],
            save_path=os.path.join(plot_save_path, "final_dataset_size.png"),
            ymin=0,
            ymax=len(dataset.obs)
        )
        wandb.log({"final_dataset_size": wandb.Image(final_dataset_size_image)})
        
        average_returns_image = plot_and_save_bar(
            data=plot_data["average_returns"],
            title="Average Returns",
            xlabel="Dataset Index",
            ylabel="Normalized Returns",
            categories=[f"{i}" for i in range(config.K_value)],
            save_path=os.path.join(plot_save_path, "average_returns.png"),
            stds=plot_data["return_stds"],
            ymin=-10,
            ymax=120
        )
        # wandb.log({"average_returns": wandb.Image(average_returns_image)})
        
        dataset_averages_image = plot_and_save_bar(
            data=plot_data["dataset_averages"],
            title="Average Returns of Datasets",
            xlabel="Dataset Index",
            ylabel="Normalized Returns",
            categories=[f"{i}" for i in range(config.K_value)],
            save_path=os.path.join(plot_save_path, "dataset_averages.png"),
            ymin=-10,
            ymax=120
        )
        # wandb.log({"normalized_dataset_averages": wandb.Image(dataset_averages_image)})
        
        dataset_stds_image = plot_and_save_bar(
            data=plot_data["dataset_stds"],
            title="Std Returns of Datasets",
            xlabel="Dataset Index",
            ylabel="Normalized Returns",
            categories=[f"{i}" for i in range(config.K_value)],
            save_path=os.path.join(plot_save_path, "dataset_stds.png")
        )
        # wandb.log({"dataset_stds": wandb.Image(dataset_stds_image)})
        
        dataset_categorical_image = plot_and_save_heatmap(
            matrix=plot_data["heatmap_matrix"],
            save_path=os.path.join(plot_save_path, "final_dataset_categorical.png")
        )
        wandb.log({"dataset_categorical": wandb.Image(dataset_categorical_image)})
        
        if plot_data.get("heatmap_matrix_before_merge") is not None:
            dataset_categorical_image = plot_and_save_heatmap(
                matrix=plot_data["heatmap_matrix_before_merge"],
                save_path=os.path.join(plot_save_path, "before_merge_dataset_categorical.png")
            )
            wandb.log({"before_merge_dataset_categorical": wandb.Image(dataset_categorical_image)})
    




if __name__ == "__main__":
    config = parse_args_and_update_config(TrainConfig)
    print(config.debug)
    if config.debug:
        print(config)
        config.disable_jit = True
    wandb.init(
        project=config.project,
        entity="policy-clustering",#!/bin/bash
        group=config.group,
        name=config.name,
        config=config,
        mode="online",
        # mode="disabled",
    )
    
    print("---------------------------------------")
    print(f"Training K_MEANS, Env: {config.env}, Seed: {config.seed}")
    print("---------------------------------------")
    train(config)
