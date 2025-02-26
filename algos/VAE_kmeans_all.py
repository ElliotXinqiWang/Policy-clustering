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
# limit the number of threads to 128
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans


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
import h5py
import matplotlib.pyplot as plt

from utils.networks import ScannedRNN, ContinuousActorRNN, DiscreteActorRNN, VAE, EncoderWrapper,VQVAE,VQVAE_gumble_softmax
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
    # Experiment
    alg: str = "VAE+Kmeans"  # Algorithm name
    env: str = "MiniGrid-Reacher"  # Minigrid environment name
    seed: int = 1  # Sets Gym, Jax and Numpy seeds
    max_updates: int = 200  # Number of updates
    n_episodes: int = 5  # How many episodes run during evaluation
    checkpoints_path: Optional[str] = None  # Save path
    batch_size: int = 64  # Batch size for all networks
    load_from_rule_based_dataset: bool = True
    true_k_available: bool = True
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
    dataset_sizes: list[int] = field(default_factory=lambda: [5, 5, 5])
    epsilon: float = 0.1
    # Network
    hidden_dim: int = 64
    max_grad_norm: float = 0.5
    learning_rate: float = 1e-3
    adam_eps: float = 1e-8
    # Kmeans
    k_value: int = 5 # Number of clusters
    max_traj_len: int = 20  # Max trajectory length
    normalize: bool = True  # Normalize states
    # vae
    vae_latent_dim: int = 32
    vae_kl_weight: float = 0.5
    # Wandb logging
    project: str = "1017VAEKmeans"
    group: str = "PKmeans"
    name: str = ""
    algo: str = "vae"
    vqvae_codebook: int = -1
    vqvae_alpha: float = 1
    vqvae_beta: float = 0.25
    vqvae_entropy_weight: float = 1

    take_ball_target: int = 0

    def __post_init__(self):
        # self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        self.name = f"{self.name}-{self.env}-{self.k_value}"
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
        nargs='+',  # 表示支持多个参数值
        help="List of rule-based dataset file paths"
    )

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
    with open("datasets/" + config.env + "/converted.pkl", "wb") as f:
        pickle.dump({"dataset": dataset, "data_idx": data_idx}, f)
        print("Sliced dataset saved to ", "datasets/" + config.env + "/converted.pkl")
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
        env = gym.make(config.env)
        # raise ValueError("Environment: ", config.env, " not supported") 
    D4RL_envs = ["halfcheetah-medium-expert-v2","walker2d-medium-expert-v2","hopper-medium-expert-v2","ant-medium-expert-v2"]
    
    if config.env in D4RL_envs:
        config.state_dim = env.observation_space.shape[0]
        config.action_dim = env.action_space.shape[0]
    else:
        config.action_dim = 5
        obs_shape = env.observation_shape
        config.state_dim = obs_shape.prod()
    
    print(f"State dim: {config.state_dim}, Action dim: {config.action_dim}")
    
    
    # load dataset from local if exists
    dataset_filename = config.env + "/" + "|".join([str(size) for size in config.dataset_sizes]) + ".pkl"
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
    elif config.env in ["MDPtakeball", "MiniGrid-Reacher-MDP"]:
        dataset, data_idx = load_rule_based_datasets(config)
    elif not os.path.exists("datasets/" + dataset_filename):
        dataset, data_idx = load_datasets(config)
    else:
        with open("datasets/" + dataset_filename, "rb") as f:
            data = pickle.load(f)
            dataset = data["dataset"]
            data_idx = data["data_idx"]
    # fdebug=open("logs/debug.txt","w")
    # for i in range(len(dataset.obs)):
    #     if data_idx[i]<0.5:
    #         print(i,file=fdebug)
    #         print(jnp.transpose(dataset.obs[i][:6].reshape(6,9,9,3),axes=(0,3,2,1))[:,1,:,:],file=fdebug)
    #         print(dataset.action[i][:6],file=fdebug)
    #         break
    # print(file=fdebug)
    # for i in range(len(dataset.obs)):
    #     if data_idx[i]>0.5:
    #         print(i,file=fdebug)
    #         print(jnp.transpose(dataset.obs[i][:6].reshape(6,9,9,3),axes=(0,3,2,1))[:,1,:,:],file=fdebug)
    #         print(dataset.action[i][:6],file=fdebug)
    #         break
    # exit(0)
    print("Dataset loaded, dataset size: ", dataset.obs.shape[0])
    # set the last done to be True if the episode is not done
    dataset = dataset._replace(done=jnp.concatenate([dataset.done[:, :-1], jnp.ones_like(dataset.done[:, -1:])], axis=1))
    
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
    
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset.obs, eps=1e-3)
    else:
        state_mean, state_std = 0, 1
    # Shuffle the dataset
    dataset = dataset._replace(obs=(dataset.obs - state_mean) / state_std)

    idx = jax.random.permutation(rng, len(dataset.obs))
    dataset = Transitions(dataset.obs[idx], dataset.action[idx], dataset.reward[idx], dataset.done[idx])
    # data_idx = jnp.concatenate([jnp.zeros(expert_start_idx), jnp.ones(len(dataset.obs) - expert_start_idx)])
    data_idx = data_idx[idx]
    print("Dataset shape: obs ", dataset.obs.shape, " action ", dataset.action.shape, " reward ", dataset.reward.shape, " done ", dataset.done.shape)
    
    DiscreteEnvNames = ["MiniGrid-Reacher", "MiniGrid-Binary-Reacher", "MiniGrid-Reacher-noisy", "MiniGrid-Reacher-extra-good", "MiniGrid-Reacher-extra-bad", "MiniGrid-Reacher-extra-med", "MiniGrid-Reacher-MDP", "MDPtakeball"]
    if config.algo == "vae":
        vae = VAE(latent_dim=config.vae_latent_dim, Encoder_hidden_dim=32, action_dim=config.action_dim, discrete_action=(config.env in DiscreteEnvNames))
    elif config.algo == "vqvae":
        vae = VQVAE(latent_dim=config.vae_latent_dim, Encoder_hidden_dim=32, action_dim=config.action_dim, alpha=config.vqvae_alpha, beta=config.vqvae_beta,
                    discrete_policy=(config.env in DiscreteEnvNames), k=config.k_value if config.vqvae_codebook == -1 else config.vqvae_codebook)
    elif config.algo == "vqvae_gumble_softmax":
        vae = VQVAE_gumble_softmax(latent_dim=config.vae_latent_dim, Encoder_hidden_dim=32, action_dim=config.action_dim,
                                   discrete_policy=(config.env in DiscreteEnvNames), alpha=config.vqvae_alpha, beta=config.vqvae_beta,
                                   k=config.k_value if config.vqvae_codebook == -1 else config.vqvae_codebook)                    
    # Initialize model and optimizer
    init_x = jnp.zeros((2, 1, config.state_dim))
    ac_init_in = (init_x, jnp.zeros((2, 1)))
    rng, init_rng = jax.random.split(rng)
    rng, reparam_rng = jax.random.split(rng)
    if config.algo == "vae":
        network_params = vae.init(init_rng, ac_init_in, reparam_rng)
    elif config.algo == "vqvae":
        network_params = vae.init(init_rng, ac_init_in)
    elif config.algo == "vqvae_gumble_softmax":
        network_params = vae.init(init_rng, ac_init_in, reparam_rng, 0)
    
    # Initialize optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate, eps=config.adam_eps),
    )
    train_state = TrainState.create(
        apply_fn=vae.apply,
        params=network_params,
        tx=tx,
    )

    
    paded_size = math.ceil(len(dataset.obs) / config.batch_size) * config.batch_size
    # Training step
    def epoch_step_func(runner_state, iter, method='vae'): 
        train_state, obs, action, reward, done, rng = runner_state
        rng, shuffle_rng = jax.random.split(rng) 
        shuffle_idx = jax.random.permutation(shuffle_rng, len(obs))
        
        pad_idx = jax.random.choice(rng, a=config.batch_size, shape=(paded_size - len(obs),), replace=False)
        shuffle_and_pad = lambda x: jnp.concatenate((x[shuffle_idx], x[pad_idx]), axis=0)
        batchify = lambda x: jnp.reshape(x, [-1, config.batch_size] + list(x.shape[1:]))
        preprocess = lambda x: jnp.swapaxes(batchify(shuffle_and_pad(x)), 1, 2)
        trainset = [
                preprocess(obs),
                preprocess(action),
                preprocess(reward),
                preprocess(done)
            ]
        def train_one_batch(train_state_rng, batch):
            train_state, rng = train_state_rng
            def loss_fn(params, batch, rng):
                obs, action, reward, done = batch
                done_mask = jnp.cumprod(1 - done.astype(jnp.int32), axis=0)
                done_mask = jnp.concatenate([jnp.ones_like(done_mask[:1]), done_mask[:-1]])
                if method == 'vae':
                    pi, mu, log_var = vae.apply(params, (obs, done), rng)
                    recon_loss = -pi.log_prob(action)
                    recon_loss = jnp.sum(done_mask * recon_loss, axis=(0, 1)) / jnp.sum(done_mask, axis=(0, 1))
                    kl_loss = mu ** 2 + jnp.exp(log_var) - log_var - 1
                    print(action.shape)
                    return recon_loss + config.vae_kl_weight * kl_loss.mean()
                elif method == 'vqvae':
                    pi, z, loss = vae.apply(params, (obs, done))
                    recon_loss = -pi.log_prob(action)
                    recon_loss = jnp.sum(done_mask * recon_loss, axis=(0, 1)) / jnp.sum(done_mask, axis=(0, 1))
                    return loss + recon_loss
                elif method == 'vqvae_gumble_softmax':
                    pi, zq, ze, loss = vae.apply(params, (obs, done), rng, iter)
                    recon_loss = -pi.log_prob(action)
                    recon_loss = jnp.sum(done_mask * recon_loss, axis=(0, 1)) / jnp.sum(done_mask, axis=(0, 1))
                    s=jnp.mean(ze,axis=(0))
                    # jax.debug.print("s.sum:{}",s.sum())
                    entropy= -jnp.sum(s*jnp.log(s+1e-8))
                    # entropy= jnp.sum(jnp.log(s+1e-8))
                    return loss + recon_loss - config.vqvae_entropy_weight * entropy
                else:
                    raise ValueError("Unknown method: ", method)
            grad_fn = jax.value_and_grad(loss_fn)
            rng, vae_rng = jax.random.split(rng)
            loss, grad = grad_fn(train_state.params, batch, vae_rng)
            train_state = train_state.apply_gradients(grads=grad)
            return (train_state, rng), loss
        train_state_rng, loss = jax.lax.scan(train_one_batch, (train_state, rng), trainset)
        train_state, rng = train_state_rng
        return train_state, loss.mean()
    epoch_step=jax.jit(epoch_step_func, static_argnames=('method'))
    
    loss_history = []
    for i in range(config.max_updates):
        rng, update_rng = jax.random.split(rng)
        train_state, loss = epoch_step((train_state, dataset.obs, dataset.action, dataset.reward, dataset.done, update_rng), i, method=config.algo)
        loss_history.append(loss)
        print(f"Update {i}, Loss: {loss}")
        wandb.log({"Loss": loss})
        if len(loss_history) > 10 and jnp.abs(loss - jnp.mean(jnp.array(loss_history[-10:]))) < 1e-4:
            break
    print("Training finished after ", i, " updates")
    
    # Encode data into latent space
    def encode_func(params, x, rng, method='vae'):
        if method == 'vae':
            pi, mu, log_var = vae.apply(params, x, rng)
            std = jnp.exp(0.5 * log_var)
            eps = jax.random.normal(rng, mu.shape)
            return mu + eps * std
        elif method == 'vqvae':
            pi, z, loss = vae.apply(params, x)
            return z
        elif method == 'vqvae_gumble_softmax':
            pi, zq, ze,  loss = vae.apply(params, x, rng, 0)
            return zq
        else:
            raise ValueError("Unknown method: ", method)
    encode = jax.jit(encode_func, static_argnames=('method'))
    pred_obs = jnp.swapaxes(dataset.obs, 0, 1)
    pred_done = jnp.swapaxes(dataset.done, 0, 1)
    latent_representations = encode(train_state.params, (pred_obs, pred_done), rng, method=config.algo)

    # Perform KMeans clustering
    if not config.true_k_available:
        kmeans = KMeans(n_clusters=config.k_value, random_state=42)
        labels = kmeans.fit_predict(latent_representations)
    else:
        true_k = int(jnp.max(data_idx)) + 1
        config.k_value = true_k
        kmeans = KMeans(n_clusters=true_k, random_state=42)
        labels = kmeans.fit_predict(latent_representations)
    predicted_labels = labels
    nmi = normalized_mutual_info_score(data_idx, predicted_labels)
    ari = adjusted_rand_score(data_idx, predicted_labels)
    print(f"NMI: {nmi}, ARI: {ari}")
    wandb.log({"NMI": nmi, "ARI": ari})
    
    dataset_idxs = [jnp.where(predicted_labels == i)[0] for i in range(config.k_value)]
    num_catagories = int(jnp.max(data_idx)) + 1
    heatmap_matrix = np.zeros((config.k_value, num_catagories))
    for i in range(config.k_value):
        for j in range(num_catagories):
            heatmap_matrix[i, j] = jnp.sum(data_idx[dataset_idxs[i]] == j)
    
    plot_save_path = f"results/{config.alg}/{config.env}/{config.k_value}/plots"
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    dataset_categorical_image = plot_and_save_heatmap(
        matrix=heatmap_matrix,
        save_path=os.path.join(plot_save_path, "final_dataset_categorical.png")
    )
    wandb.log({"dataset_categorical": wandb.Image(dataset_categorical_image)})
    # # Plot latent space with clustering results
    # plt.figure(figsize=(8, 6))
    # plt.scatter(latent_representations[:, 0], latent_representations[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    # plt.title("Latent Space Representation with KMeans Clustering")
    # plt.xlabel("Latent Dimension 1")
    # plt.ylabel("Latent Dimension 2")
    # plt.colorbar(label="Cluster")
    # plt.show()




if __name__ == "__main__":
    config = parse_args_and_update_config(TrainConfig)
    if config.debug:
        print(config)
        config.disable_jit = True
    wandb.init(
        project=config.project,
        entity="policy-clustering",#!/bin/bash
        group=config.group,
        name=config.name,
        config=config,
        mode="disabled" if config.debug else "online",
        # mode="disabled",
    )
    
    print("---------------------------------------")
    print(f"Training VAE + kmeans, Env: {config.env}, Seed: {config.seed}")
    print("---------------------------------------")
    train(config)
