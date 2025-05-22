import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import h5py
import uuid
from dataclasses import dataclass, field
from typing import NamedTuple, Optional
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym.spaces.box")
import pickle


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

from utils.networks import ScannedRNN, ContinuousActorRNN
from utils.plot_tools import plot_and_save_curves, plot_and_save_bar, plot_and_save_bars

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
    alg: str = "Soft_Kmeans"  # Algorithm name
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, Jax and Numpy seeds
    max_updates: int = 15 # Maximum number of updates
    n_episodes: int = 5  # How many episodes run during evaluation
    checkpoints_path: Optional[str] = None  # Save path
    batch_size: int = 64  # Batch size for all networks
    save_model: bool = False  # Save the model
    # Network
    hidden_dim: int = 128
    max_grad_norm: float = 0.5
    learning_rate: float = 2e-3
    adam_eps: float = 1e-8
    # Soft Kmeans
    K_value: int = 2 # Number of clusters
    max_traj_len: int = 1000  # Max trajectory length
    normalize: bool = True  # Normalize states
    temperature: float = 5.0  # Temperature for softmax
    uniform_init: bool = False  # Initialize the weights uniformly
    # Wandb logging
    project: str = "Soft_Kmeans"
    group: str = "PKmeans"
    name: str = "test"

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

def dataset_to_trajectories(dataset, max_traj_len=1000):
    """
    Dataset is a dictionary with keys: observations, actions, rewards, terminals, timeouts
    dataset['obs'] is of shape (num_traj * max_traj_len, obs_dim)
    Convert the dataset to a Transition 
    where obs shape is (num_traj, max_traj_len, obs_dim)
    Args:
        dataset (_type_): the dataset to convert
        max_traj_len (_type_): the maximum length of the trajectories. set tobe 1000 by defaulf for mujoco envs.
    Returns:
        trajectories (_type_): the Transition
    """
    start_idx = 0
    start_is_done = True
    obs = []
    action = []
    reward = []
    done = []
    # print the keys of the dataset
    # print("Dataset keys", dataset.keys())
    # see if all the terminals are False
    print("All terminals are False?", not any(dataset['terminals']))
    # a Traj is done when timeout or terminal is True
    if 'timeouts' not in dataset:
        dataset['timeouts'] = jnp.zeros_like(dataset['terminals'])
    dones = dataset['terminals'] | dataset['timeouts']
    
    for i in range(len(dataset['terminals'])):
        if i - start_idx == max_traj_len or dones[i]:
            if not start_is_done:
                start_idx = i
                start_is_done = True
                continue
            else:
                obs.append(dataset['observations'][start_idx:i])
                action.append(dataset['actions'][start_idx:i])
                reward.append(dataset['rewards'][start_idx:i])
                done.append(dones[start_idx:i])
                # make sure the last done is True for each traj
                done[-1].at[-1].set(True)
                
                start_idx = i
                start_is_done = dones[i]
    
    # pad all the trajs to the same length. pad dones with 1
    max_len = max([len(obs[i]) for i in range(len(obs))])
    obs = [jnp.pad(obs[i], ((0, max_len - len(obs[i])), (0, 0))) for i in range(len(obs))]
    action = [jnp.pad(action[i], ((0, max_len - len(action[i])), (0, 0))) for i in range(len(action))]
    reward = [jnp.pad(reward[i], ((0, max_len - len(reward[i])))) for i in range(len(reward))]
    done = [jnp.pad(done[i], ((0, max_len - len(done[i]))), constant_values=False) for i in range(len(done))]
            
    return Transitions(jnp.array(obs), jnp.array(action), jnp.array(reward), jnp.array(done))
    
def compute_mean_std(data, eps=1e-3):
    mean = jnp.mean(data, axis=(0, 1))
    std = jnp.std(data, axis=(0, 1)) + eps
    return mean, std

def train(config):
    # Set random seed
    rng = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)
    
    env = gym.make(config.env)
    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]
    print(f"State dim: {config.state_dim}, Action dim: {config.action_dim}")
    
    # load dataset from local if available
    if os.path.exists(f"datasets/{config.env}.hdf5"):
        print("Load dataset from local: ", f"datasets/{config.env}.hdf5")
        with h5py.File(f"datasets/{config.env}.hdf5", "r") as f:
            obs = jnp.array(f["obs"][:])
            action = jnp.array(f["action"][:])
            reward = jnp.array(f["reward"][:])
            done = jnp.array(f["done"][:])
        dataset = Transitions(obs, action, reward, done)
    else:
        print("Load dataset from d4rl")
        dataset = d4rl.qlearning_dataset(env)
        print("first 10 teriminals", dataset['terminals'][:10])
        dataset = dataset_to_trajectories(dataset, config.max_traj_len)
        # save dataset to local
        if not os.path.exists(f"datasets/{config.env}.hdf5"):
            os.makedirs("datasets", exist_ok=True)
        print("Save dataset to local: ", f"datasets/{config.env}.hdf5")
        with h5py.File(f"datasets/{config.env}.hdf5", "w") as f:
            f.create_dataset("obs", data=np.array(dataset.obs))
            f.create_dataset("action", data=np.array(dataset.action))
            f.create_dataset("reward", data=np.array(dataset.reward))
            f.create_dataset("done", data=np.array(dataset.done))
    print(f"Dataset shape: obs {dataset.obs.shape}, action {dataset.action.shape}, reward {dataset.reward.shape}, done {dataset.done.shape}")
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset.obs, eps=1e-3)
    else:
        state_mean, state_std = 0, 1
    # Shuffle the dataset
    dataset = dataset._replace(obs=(dataset.obs - state_mean) / state_std)
    
    idx = jax.random.permutation(rng, len(dataset.obs))
    dataset = Transitions(dataset.obs[idx], dataset.action[idx], dataset.reward[idx], dataset.done[idx])
    
    # Initialize the weight matrix. We assign equal weights to each network for the first iteration
    if config.uniform_init:
        dataset_weights = np.ones((config.K_value, len(dataset.obs))) / config.K_value
    else:
        # we randomly assign the weights to each dataset
        dataset_weights = np.random.dirichlet(np.ones(config.K_value), size=len(dataset.obs)).T
    print("Dataset weights shape: ", dataset_weights.shape)
    dataset_weights = jnp.array(dataset_weights)
    last_weights = jnp.zeros_like(dataset_weights)
    
    # Initialize K networks and create training states
    train_states = []
    model = ContinuousActorRNN(config.action_dim, config.hidden_dim, config)
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
    
    
    def train_one_network(train_state, dataset_weight, rng, time_average_window=100, maximum_iterations=200):
        """
        Train one network with the given dataset. Train until the convergence, i.e., the loss is not decreasing.
        """
        paded_size = math.ceil(len(dataset_weight) / config.batch_size) * config.batch_size
        def train_one_epoch(state_data_rng_weight):
            train_state, obs, action, reward, done, rng, weight = state_data_rng_weight
            # repeat weight to match the shape of done
            weight = jnp.repeat(weight[:, None], config.max_traj_len, axis=1)
            
            # shuffle pad the dataset to make sure the size is a multiple of batch_size
            # randomly choose data from the dataset to pad
            shuffle_idx = jax.random.permutation(rng, len(obs))
            
            pad_idx = jax.random.choice(rng, a=config.batch_size, shape=(paded_size - len(obs),), replace=False)
            shuffle_and_pad = lambda x: jnp.concatenate((x[shuffle_idx], x[pad_idx]), axis=0)
            # current shape: (num_traj, max_traj_len, data_dim)
            batchify = lambda x: jnp.reshape(x, [-1, config.batch_size] + list(x.shape[1:]))
            # current shape: (num_batch, batch_size, max_traj_len, data_dim)
            preprocess = lambda x: jnp.swapaxes(batchify(shuffle_and_pad(x)), 1, 2)
            # current shape: (num_batch, max_traj_len, batch_size, data_dim)
            trainset = [
                preprocess(obs),
                preprocess(action),
                preprocess(reward),
                preprocess(done),
                preprocess(weight)
            ]
            
            def train_one_batch(train_state, batch):
                def _loss_fn(params, h_state, batch):
                    # data shape: (max_traj_len, batch_size, data_dim)
                    obs, action, reward, done, weight = batch
                    h_state, pi = model.apply(params, h_state, (obs, done))
                    # mask out the probs strictly after first done (exclusive)
                    done_mask = jnp.cumprod(1 - done.astype(jnp.int32), axis=0)
                    done_mask = jnp.concatenate([jnp.ones_like(done_mask[:1]), done_mask[:-1]])
                    loss = -pi.log_prob(action) * weight
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
            train_state, loss = jit_train_one_epoch((train_state, dataset.obs, dataset.action, dataset.reward, dataset.done, sub_rng, dataset_weight))
            loss_history.append(loss)
            if len(loss_history) >= time_average_window and \
            jnp.abs(loss - jnp.mean(jnp.array(loss_history[-time_average_window:]))) < 1e-3:
                print("Converged after ", i, "iterations")
                break
        print("Training loss: ", loss)
        return train_state, loss
    
    print("Start training")
    epoch = 0
    # log the size curve of each dataset in one plot
    plot_data = {"dataset_size":[],
                 "dataset_averages": [],
                 "dataset_stds": [],
                 "average_returns": [],}
    plot_data["dataset_size"].append([dataset_weights[i].sum() for i in range(config.K_value)])
    
    def weight_equal(a, b, threshold=1e-2):
        return jnp.abs(a - b).max() < threshold
    
    dataset_L1_changes = []
    
    
    while jnp.array([not weight_equal(dataset_weights[i], last_weights[i]) and dataset_weights[i].sum() > 10 for i in range(config.K_value)]).any():
        if epoch >= config.max_updates:
            break
        print("Epoch: ", epoch, "dataset sizes: ", [f"{float(dataset_weights[i].sum()):.2f}" for i in range(config.K_value)])
        dataset_L1_changes.append(jnp.array([jnp.abs(dataset_weights[i] - last_weights[i]).sum() for i in range(config.K_value)]))
        print("Dataset L1 changes: ", dataset_L1_changes[-1].tolist())
        epoch += 1
        # Train each network
        for i in range(config.K_value):
            # train the network if last_idxs[i] is different from dataset_idxs[i], and the dataset is not too small
            if not weight_equal(dataset_weights[i], last_weights[i]) and dataset_weights[i].sum() >10:
                # print("average train action: ", jnp.mean(K_datasets[i].action, axis=(0, 1)))
                # print the statistics of the data
                train_state = train_states[i]
                _train_rng, rng = jax.random.split(rng)
                train_states[i], loss = train_one_network(train_state, dataset_weights[i], _train_rng)
        
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
            
            done_mask = jnp.cumprod(1 - dones.astype(jnp.int32), axis=0)
            done_mask = jnp.concatenate([jnp.ones_like(done_mask[:1]), done_mask[:-1]])
            # traj_prob = jnp.sum(prob * done_mask, axis=0)
            traj_prob = jnp.mean(prob * done_mask, axis=0)
            
            return out_hidden, traj_prob
        _jitted_get_probability_of_traj = jax.jit(_get_probability_of_traj)
        log_prob = []
        
        with jax.disable_jit(config.disable_jit):
            for i in range(config.K_value):
                _, probs = _jitted_get_probability_of_traj(
                    ScannedRNN.initialize_carry(len(dataset.obs), config.hidden_dim),
                    dataset_to_be_inferenced.action,
                    dataset_to_be_inferenced.obs,
                    dataset_to_be_inferenced.done,
                    train_states[i].params
                )
                log_prob.append(probs)
        log_prob = jnp.array(log_prob) # shape: (K_value, dataset_size)
        print("average prob: ", jnp.mean(log_prob, axis=1))
        # update the weights
        last_weights = dataset_weights
        dataset_weights = jax.nn.softmax(log_prob * config.temperature, axis=0)
        plot_data["dataset_size"].append([dataset_weights[i].sum() for i in range(config.K_value)])
    
    dataset_L1_changes.append(jnp.array([jnp.abs(dataset_weights[i] - last_weights[i]).sum() for i in range(config.K_value)]))
    print("Last Dataset L1 changes: ", dataset_L1_changes[-1].tolist())
    
    plot_data["dataset_L1_changes"] = dataset_L1_changes
    # get the statistics of the each dataset
    total_return = dataset.reward.sum(axis=1)  # shape: (dataset_size,)
    for i in range(config.K_value):
        if dataset_weights[i].sum() < 10:
            # skip the degenerate datasets
            plot_data["dataset_averages"].append(np.nan)
            plot_data["dataset_stds"].append(np.nan)
        else:
            # calculate the weighted mean and std of the return
            Mreturn = (total_return * dataset_weights[i]).sum()/dataset_weights[i].sum()
            Stdreturn = jnp.sqrt(((total_return - Mreturn) ** 2 * dataset_weights[i]).sum()/dataset_weights[i].sum())
            Normreturn = env.get_normalized_score(Mreturn) * 100
            plot_data["dataset_averages"].append(Normreturn)
            plot_data["dataset_stds"].append(Stdreturn)
    
    print("Training finished. Start evaluation.")
    # Evaluate and save the models
    K_rewards = []
    K_reward_stds = []
    def _get_one_action(params, h_state, obs, dones):
        h_state, pi = model.apply(params, h_state, (obs, dones))
        return h_state, pi.sample(seed=rng)[0, 0]
    _jitted_get_one_action = jax.jit(_get_one_action)
    for i in range(config.K_value):
        # save the parameters
        if config.save_model:
            if not os.path.exists(f"models/{config.K_value}_{config.env}"):
                os.makedirs(f"models/{config.K_value}_{config.env}", exist_ok=True)
            model_path = f"models/{config.K_value}_{config.env}/model_{i}.pkl"
            with open(model_path, "wb") as f:
                f.write(serialization.to_bytes(train_states[i].params))
            print("Model saved to ", model_path)
        
        # evaluate the network
        episode_rewards = []
        for t in range(config.n_episodes):
            state, done = env.reset(), False
            episode_reward = 0.0
            h_state = ScannedRNN.initialize_carry(1, config.hidden_dim)
            while not done:
                rng, sub_rng = jax.random.split(rng)
                state = jnp.array(state)[None, None, ...] # add batch_size dimension and seq_len dimension
                state = (state - state_mean) / state_std # normalize the state
                done = jnp.array(done)[None, None, ...]
                h_state, action = _jitted_get_one_action(train_states[i].params, h_state, state, done)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
            episode_rewards.append(env.get_normalized_score(episode_reward) * 100)
        episode_rewards = jnp.array(episode_rewards)
        K_rewards.append(episode_rewards.mean())
        K_reward_stds.append(episode_rewards.std())
    print("Normalized Rewards of each network: ", K_rewards)
    plot_data["average_returns"] = K_rewards
    plot_data["return_stds"] = K_reward_stds
    
    env.close()
    
    # draw a distribution curve of the returns in each dataset
    dataset_returns = dataset.reward.sum(axis=1) # shape: (dataset_size,)
    dataset_bins = np.linspace(dataset_returns.min(), dataset_returns.max(), 40)
    dataset_hist = []
    for i in range(config.K_value):
        hist, _ = np.histogram(dataset_returns, bins=dataset_bins, weights=dataset_weights[i], density=False)
        dataset_hist.append(hist)
    plot_data["dataset_hist"] = dataset_hist
    

    if config.make_plots:
        # save the data 
        plot_data["dataset_size"] = jnp.array(plot_data["dataset_size"])
        plot_data["dataset_averages"] = jnp.array(plot_data["dataset_averages"])
        plot_data["dataset_stds"] = jnp.array(plot_data["dataset_stds"])
        plot_data["average_returns"] = jnp.array(plot_data["average_returns"])
        plot_data["dataset_L1_changes"] = jnp.array(plot_data["dataset_L1_changes"])
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
        
        dataset_L1_changes_image = plot_and_save_curves(
            curves=plot_data["dataset_L1_changes"].T,
            title="Dataset L1 Changes",
            xlabel="Epoch",
            ylabel="L1 Changes",
            labels=[f"Dataset_{i}" for i in range(config.K_value)],
            save_path=os.path.join(plot_save_path, "dataset_L1_changes_curve.png"),
            min_y=0
        )
        wandb.log({"dataset_L1_changes_curve": wandb.Image(dataset_L1_changes_image)})
        
        dataset_hist_image = plot_and_save_curves(
            curves=plot_data["dataset_hist"],
            title="Returns Distribution",
            xlabel="Returns",
            ylabel="# of Trajectories",
            labels=[f"D_{i}" for i in range(config.K_value)],
            save_path=os.path.join(plot_save_path, "dataset_hist_curve.png"),
            min_y=0
        )
        wandb.log({"dataset_hist_curve": wandb.Image(dataset_hist_image)})
        
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
        wandb.log({"average_returns": wandb.Image(average_returns_image)})
        
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
        wandb.log({"normalized_dataset_averages": wandb.Image(dataset_averages_image)})
        
        dataset_stds_image = plot_and_save_bar(
            data=plot_data["dataset_stds"],
            title="Std Returns of Datasets",
            xlabel="Dataset Index",
            ylabel="Normalized Returns",
            categories=[f"{i}" for i in range(config.K_value)],
            save_path=os.path.join(plot_save_path, "dataset_stds.png")
        )
        wandb.log({"dataset_stds": wandb.Image(dataset_stds_image)})
        
        
    




if __name__ == "__main__":
    config = parse_args_and_update_config(TrainConfig)
    if config.debug:
        print(config)
        config.disable_jit = True
    wandb.init(
        project=config.project,
        entity="elliotxinqiwang",#!/bin/bash
        group=config.group,
        name=config.name,
        config=config,
        mode="disabled" if config.debug else "online",
        # mode="disabled",
    )
    
    print("---------------------------------------")
    print(f"Training Soft K_MEANS, Env: {config.env}, Seed: {config.seed}")
    print("---------------------------------------")
    train(config)
