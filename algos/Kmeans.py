import argparse
import os
import h5py
import uuid
from dataclasses import dataclass, field
from typing import NamedTuple
from typing import Optional
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym.spaces.box")


import optax
import chex
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import d4rl
import gym
import wandb
import matplotlib.pyplot as plt

from utils.networks import ScannedRNN, ContinuousActorRNN
@dataclass
class TrainConfig:
    # General
    device: str = "cuda"
    device_id: list = field(default_factory=lambda: [0])
    debug: bool = False
    disable_jit: bool = False
    extra_attributes: dict = field(default_factory=dict, init=False)
    # Experiment
    env: str = "halfcheetah-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, Jax and Numpy seeds
    # eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 1  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    # load_model: str = ""  # Model load file name, "" doesn't load
    batch_size: int = 32  # Batch size for all networks
    # Network
    hidden_dim: int = 128
    max_grad_norm: float = 0.5
    learning_rate: float = 1e-3
    adam_eps: float = 1e-8
    # Kmeans
    K_value: int = 2 # Number of clusters
    max_traj_len: int = 1000  # Max trajectory length
    normalize: bool = True  # Normalize states
    # Wandb logging
    project: str = "Policy_Kmeans"
    group: str = "PKmeans"
    name: str = "test"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
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
                done[-1][-1].set(True)
                # print("Add traj in", start_idx, i)
                
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
    
    # Initialize K datasets by choosing first K*batch_size trajs
    K_datasets = []
    for i in range(config.K_value):
        # idx = jnp.random.choice(len(dataset.obs), size=config.batch_size)
        # K_datasets.append(Transitions(dataset.obs[idx], dataset.action[idx], dataset.reward[idx], dataset.done[idx]))
        K_datasets.append(Transitions(dataset.obs[i*config.batch_size:(i+1)*config.batch_size], dataset.action[i*config.batch_size:(i+1)*config.batch_size], dataset.reward[i*config.batch_size:(i+1)*config.batch_size], dataset.done[i*config.batch_size:(i+1)*config.batch_size]))
    
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
    
    
    def train_one_network(train_state, dataset, rng, time_average_window=100, maximum_iterations=500):
        """
        Train one network with the given dataset. Train until the convergence, i.e., the loss is not decreasing.
        """
        paded_size = math.ceil(len(dataset[0]) / config.batch_size) * config.batch_size
        def train_one_epoch(state_data_rng):
            train_state, dataset, rng = state_data_rng
            obs, action, reward, done = dataset
            # shuffle pad the dataset to make sure the size is a multiple of batch_size
            # randomly choose data from the dataset to pad
            shuffle_idx = jax.random.permutation(rng, len(obs))
            # print("padding size: ", paded_size - len(obs)) 0
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
                    loss = -pi.log_prob(action).mean()
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
            train_state, loss = jit_train_one_epoch((train_state, dataset, sub_rng))
            loss_history.append(loss)
            if len(loss_history) >= time_average_window and \
            jnp.abs(loss - jnp.mean(jnp.array(loss_history[-time_average_window:]))) < 1e-3:
                print("Converged after ", i, "iterations")
                break
        print("Training loss: ", loss)
        return train_state, loss
        
    
    Num_iterations = len(dataset.obs) // config.batch_size
    last_dataset_sizes = [0 for i in range(config.K_value)]
    current_dataset_sizes = [len(K_datasets[i].obs) for i in range(config.K_value)]
    print("Start training, will end in ", Num_iterations, "iterations")
    for epoch in range(Num_iterations):
        print("Epoch: ", epoch, "dataset sizes: ", current_dataset_sizes)
        # Train each network
        for i in range(config.K_value):
            # train the network if new trajs are added to the dataset
            if len(K_datasets[i].obs) > last_dataset_sizes[i]:
                # print("average train action: ", jnp.mean(K_datasets[i].action, axis=(0, 1)))
                # print the statistics of the data
                train_state = train_states[i]
                dataset_to_train = K_datasets[i]
                _train_rng, rng = jax.random.split(rng)
                train_states[i], loss = train_one_network(train_state, dataset_to_train, _train_rng)
        
        # Get a batch of data from the dataset, swap the axis to make it compatible with the model
        data_to_be_assigned = Transitions(
            dataset.obs[epoch*config.batch_size:(epoch+1)*config.batch_size].swapaxes(0, 1),
            dataset.action[epoch*config.batch_size:(epoch+1)*config.batch_size].swapaxes(0, 1),
            dataset.reward[epoch*config.batch_size:(epoch+1)*config.batch_size].swapaxes(0, 1),
            dataset.done[epoch*config.batch_size:(epoch+1)*config.batch_size].swapaxes(0, 1)
        )
        # test the loss on data_to_be_assigned
        def _loss_fn(params, h_state, batch):
            # data shape: (max_traj_len, batch_size, data_dim)
            obs, action, reward, done = batch
            h_state, pi = model.apply(params, h_state, (obs, done))
            loss = -pi.log_prob(action).mean()
            return loss
        
        
        # get the probability of each traj generated by each network
        def _get_probability_of_traj(hidden, actions, obs, dones, params):
            # mask out the probs strictly after first done (exclusive)
            # clip the logprob to [-3, infty]
            
            out_hidden, pi = model.apply(params, hidden, (obs, dones))
            prob = pi.log_prob(actions)
            # prob = jnp.clip(prob, -10, jnp.inf)
            
            done_mask = jnp.cumprod(1 - dones.astype(jnp.int32), axis=0)
            done_mask = jnp.concatenate([jnp.ones_like(done_mask[:1]), done_mask[:-1]])
            # traj_prob = jnp.sum(prob * done_mask, axis=0)
            traj_prob = jnp.mean(prob * done_mask, axis=0)
            
            return out_hidden, traj_prob
        _jitted_get_probability_of_traj = jax.jit(_get_probability_of_traj)
        prob = []
        with jax.disable_jit(config.disable_jit):
            for i in range(config.K_value):
                # print the shape of the data
                # print("Data shape: ", data_to_be_assigned.action.shape, data_to_be_assigned.obs.shape, data_to_be_assigned.done.shape)
                _, probs = _jitted_get_probability_of_traj(
                    ScannedRNN.initialize_carry(config.batch_size, config.hidden_dim),
                    data_to_be_assigned.action, 
                    data_to_be_assigned.obs,
                    data_to_be_assigned.done,
                    train_states[i].params
                )
                prob.append(probs)
        prob = jnp.array(prob)
        print("average prob: ", jnp.mean(prob, axis=1))
        # Add new trajs to the K datasets according to the probility of each traj generated by the K networks
        last_dataset_sizes = current_dataset_sizes.copy()
        idx = jnp.argmax(prob, axis=0)
        print("Assign trajs to dataset: ", idx)
        for i in range(config.batch_size):
            # add the traj to the dataset whose network has the highest probability
            add = lambda x, y: jnp.concatenate((x, jnp.expand_dims(y, axis=0)), axis=0)
            K_datasets[idx[i]] = K_datasets[idx[i]]._replace(obs=add(K_datasets[idx[i]].obs, data_to_be_assigned.obs[:, i]))
            K_datasets[idx[i]] = K_datasets[idx[i]]._replace(action=add(K_datasets[idx[i]].action, data_to_be_assigned.action[:, i]))
            K_datasets[idx[i]] = K_datasets[idx[i]]._replace(reward=add(K_datasets[idx[i]].reward, data_to_be_assigned.reward[:, i]))
            K_datasets[idx[i]] = K_datasets[idx[i]]._replace(done=add(K_datasets[idx[i]].done, data_to_be_assigned.done[:, i]))
            current_dataset_sizes[idx[i]] += 1
        # log the size curve of each dataset in one plot
        wandb.log(
            {f"Dataset size/dataset{i}": current_dataset_sizes[i] for i in range(config.K_value)},
            step=epoch,
        )

    # Train each network for the last time
    for i in range(config.K_value):
        if len(K_datasets[i].obs) > last_dataset_sizes[i]:
            print("last train for network ", i)
            _train_rng, rng = jax.random.split(rng)
            train_state = train_states[i]
            dataset_tuple = K_datasets[i]
            dataset_to_train = [dataset_tuple.obs, dataset_tuple.action, dataset_tuple.reward, dataset_tuple.done]
            train_states[i], loss = train_one_network(train_state, dataset_to_train, _train_rng) 
    
    print("Training finished. Start evaluation.")
    # Evaluate the model
    K_rewards = []
    def _get_one_action(params, h_state, obs, dones):
        _, pi = model.apply(params, h_state, (obs, dones))
        return pi.sample(seed=rng)[0, 0]
    _jitted_get_one_action = jax.jit(_get_one_action)
    for i in range(config.K_value):
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
                action = _jitted_get_one_action(train_states[i].params, h_state, state, done)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
            episode_rewards.append(episode_reward)
        K_rewards.append(jnp.mean(jnp.array(episode_rewards)))
    print("Rewards of each network: ", K_rewards)
    wandb.log(
        {f"Rewards/dataset{i}": K_rewards[i] for i in range(config.K_value)},
    )
        
    
    




if __name__ == "__main__":
    config = parse_args_and_update_config(TrainConfig)
    if config.debug:
        print(config)
        config.disable_jit = True
    wandb.init(
        project=config.project,
        entity="elliotxinqiwang",
        group=config.group,
        name=config.name,
        config=config,
        mode="disabled" if config.debug else "online",
        # mode="disabled",
    )
    
    print("---------------------------------------")
    print(f"Training K_MEANS, Env: {config.env}, Seed: {config.seed}")
    print("---------------------------------------")
    train(config)
