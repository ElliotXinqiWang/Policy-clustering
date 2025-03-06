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

from utils.networks import ScannedRNN, ContinuousActorRNN, DiscreteActorRNN, VAE, EncoderWrapper,VQVAE,VQVAE_gumble_softmax,VQVAE_modify
from gridworld.env import SingleAgentGridworld, FixedGridworld, ExtraRewardGridworld, MDPGridworld, MDPtakeball
from utils.plot_tools import plot_and_save_curves, plot_and_save_bar, plot_and_save_bars, plot_and_save_heatmap
from utils.load_dataset import load_datasets, load_rule_based_datasets, Transitions

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
    extra_reward: float = 10.0
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
    vqvae_alpha: float = 1.0
    vqvae_beta: float = 0.25
    vqvae_entropy_weight: float = 1.0
    encoder_attention: bool = False
    encoder_hidden_dim: int = 32
    encoder_attention_features_dim: int = 4
    vqvae_modify_use_sigma: bool = False
    vqvae_modify_sum_method: str = "sum"

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
    dataset = dataset._replace(done=jnp.concatenate([dataset.done[:, :-1], jnp.ones_like(dataset.done[:, -1:])], axis=1))
    
    if config.true_k_available:
        true_k = int(jnp.max(data_idx)) + 1
        config.k_value = true_k
    print("Run on config: ", config)
    
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

    # idx = jax.random.permutation(rng, len(dataset.obs))
    # dataset = Transitions(dataset.obs[idx], dataset.action[idx], dataset.reward[idx], dataset.done[idx])
    # data_idx = jnp.concatenate([jnp.zeros(expert_start_idx), jnp.ones(len(dataset.obs) - expert_start_idx)])
    # data_idx = data_idx[idx]
    print("Dataset shape: obs ", dataset.obs.shape, " action ", dataset.action.shape, " reward ", dataset.reward.shape, " done ", dataset.done.shape)
    
    DiscreteEnvNames = ["MiniGrid-Reacher", "MiniGrid-Binary-Reacher", "MiniGrid-Reacher-noisy", "MiniGrid-Reacher-extra-good", "MiniGrid-Reacher-extra-bad", "MiniGrid-Reacher-extra-med", "MiniGrid-Reacher-MDP", "MDPtakeball"]
    if config.algo == "vae":
        vae = VAE(latent_dim=config.vae_latent_dim, Encoder_hidden_dim=config.encoder_hidden_dim, action_dim=config.action_dim, discrete_action=(config.env in DiscreteEnvNames))
    elif config.algo == "vqvae":
        vae = VQVAE(latent_dim=config.vae_latent_dim, Encoder_hidden_dim=config.encoder_hidden_dim, action_dim=config.action_dim, alpha=config.vqvae_alpha, beta=config.vqvae_beta,
                    discrete_policy=(config.env in DiscreteEnvNames), k=config.k_value if config.vqvae_codebook == -1 else config.vqvae_codebook, attention=config.encoder_attention, encoder_attention_features_dim=config.encoder_attention_features_dim)
    elif config.algo == "vqvae_gumble_softmax":
        vae = VQVAE_gumble_softmax(latent_dim=config.vae_latent_dim, Encoder_hidden_dim=config.encoder_hidden_dim, action_dim=config.action_dim,
                                   discrete_policy=(config.env in DiscreteEnvNames), alpha=config.vqvae_alpha, beta=config.vqvae_beta,
                                   k=config.k_value if config.vqvae_codebook == -1 else config.vqvae_codebook)    
    elif config.algo == "vqvae_modify":
        vae = VQVAE_modify(latent_dim=config.vae_latent_dim, Encoder_hidden_dim=config.encoder_hidden_dim, action_dim=config.action_dim, alpha=config.vqvae_alpha, beta=config.vqvae_beta,
                           discrete_policy=(config.env in DiscreteEnvNames), k=config.k_value if config.vqvae_codebook == -1 else config.vqvae_codebook,
                           attention=config.encoder_attention, encoder_attention_features_dim=config.encoder_attention_features_dim,
                           use_sigma=config.vqvae_modify_use_sigma, method=config.vqvae_modify_sum_method)
    else:
        raise ValueError("Unknown algo: ", config.algo)
    # Initialize model and optimizer
    init_x = jnp.zeros((2, 1, config.state_dim))
    ac_init_in = (init_x, jnp.zeros((2, 1)))
    init_act = jnp.zeros((2, 1)) if config.env in DiscreteEnvNames else jnp.zeros((2, 1, config.action_dim))
    rng, init_rng = jax.random.split(rng)
    rng, reparam_rng = jax.random.split(rng)
    if config.algo == "vae":
        network_params = vae.init(init_rng, ac_init_in, init_act, reparam_rng)
    elif config.algo == "vqvae" or config.algo == "vqvae_modify":
        network_params = vae.init(init_rng, ac_init_in, init_act)
    elif config.algo == "vqvae_gumble_softmax":
        network_params = vae.init(init_rng, ac_init_in, init_act, reparam_rng, 0)
    
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
                if method == 'vae':
                    pi, mu, log_var = vae.apply(params, (obs, done), action, rng)
                    recon_loss = -pi.log_prob(action)
                    recon_loss = jnp.sum(done_mask * recon_loss, axis=(0, 1)) / jnp.sum(done_mask, axis=(0, 1))
                    kl_loss = mu ** 2 + jnp.exp(log_var) - log_var - 1
                    print(action.shape)
                    return recon_loss + config.vae_kl_weight * kl_loss.mean()
                elif method == 'vqvae' or method == 'vqvae_modify':
                    pi, z, loss = vae.apply(params, (obs, done), action)
                    recon_loss = -pi.log_prob(action)
                    recon_loss = jnp.sum(done_mask * recon_loss, axis=(0, 1)) / jnp.sum(done_mask, axis=(0, 1))
                    return loss + recon_loss
                elif method == 'vqvae_gumble_softmax':
                    pi, zq, ze, loss = vae.apply(params, (obs, done), action, rng, iter)
                    recon_loss = -pi.log_prob(action)
                    recon_loss = jnp.sum(done_mask * recon_loss, axis=(0, 1)) / jnp.sum(done_mask, axis=(0, 1))
                    s=jnp.mean(ze,axis=(0))
                    # jax.debug.print("s.sum:{}",s.sum())
                    entropy= -jnp.sum(s*jnp.log(s+1e-8))
                    # entropy= jnp.sum(jnp.log(s+1e-8))
                    return loss + recon_loss - config.vqvae_entropy_weight * entropy
                    # return - config.vqvae_entropy_weight * entropy
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
    def encode_func(params, x, act, rng, method='vae'):
        if method == 'vae':
            pi, mu, log_var = vae.apply(params, x, act, rng)
            std = jnp.exp(0.5 * log_var)
            eps = jax.random.normal(rng, mu.shape)
            return mu + eps * std
        elif method == 'vqvae' or method == 'vqvae_modify':
            pi, z, loss = vae.apply(params, x, act)
            # (obs, done) = x
            # done_mask = jnp.cumprod(1 - done.astype(jnp.int32), axis=0)
            # recon_loss = -pi.log_prob(act)
            # recon_loss = jnp.sum(done_mask * recon_loss, axis=(0, 1)) / jnp.sum(done_mask, axis=(0, 1))
            # jax.debug.print("{}",recon_loss)
            return z
        elif method == 'vqvae_gumble_softmax':
            pi, zq, ze,  loss = vae.apply(params, x, act, rng, 0)
            return zq
        else:
            raise ValueError("Unknown method: ", method)
    encode = jax.jit(encode_func, static_argnames=('method'))
    pred_obs = jnp.swapaxes(dataset.obs, 0, 1)
    pred_done = jnp.swapaxes(dataset.done, 0, 1)
    latent_representations = encode(train_state.params, (pred_obs, pred_done), jnp.swapaxes(dataset.action,0,1), rng, method=config.algo)

    # Perform KMeans clustering
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
