import os
import argparse
from dataclasses import dataclass, field
from typing import Optional
from typing import NamedTuple
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import optax 
from flax.training.train_state import TrainState
from flax import serialization
import wandb
from gridworld.env import SingleAgentGridworld, FixedGridworld, ExtraRewardGridworld


from utils.networks import ActorCriticRNN, ScannedRNN

class Transitions(NamedTuple):
    obs: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    action: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    reward: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    done: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))

@dataclass
class TrainConfig:
    # General
    device: str = "cuda"
    device_id: list = field(default_factory=lambda: [0])
    debug: bool = False
    disable_jit: bool = False
    extra_attributes: dict = field(default_factory=dict, init=False)
    make_plots: bool = True
    model_load_path: Optional[str] = "behavior_models/PPO_MiniGrid-Reacher-good/params_0.7.pkl"
    checkpoints_path: Optional[str] = "checkpoints"  # Path to save checkpoints
    # Experiment
    alg: str = "PPO"  # Algorithm used to trained the behavior model
    env: str = "MiniGrid-Reacher-noisy"  # Environment name
    seed: int = 5  # Sets Gym, Jax and Numpy seeds
    n_episodes: int = 10  # How many episodes we need to collect
    num_envs: int = 256  # Number of environments
    epsilon: float = 0.1
    # Network & training
    hidden_dim: int = 128
    num_steps: int = 128
    # Wandb logging
    project: str = "ppo"
    group: str = "PKmeans"
    name: str = "test"

    def __post_init__(self):
        # self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        self.name = f"{self.name}-{self.env}"
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

class PPO_Transitions(NamedTuple):
    obs: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    action: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    reward: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    done: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    value: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    log_prob: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
    
    
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


def get_rollout(config):
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
    else:
        raise ValueError("Environment: ", config.env, " not supported")    

    config.action_dim = 5
    obs_shape = env.observation_shape
    config.observation_dim = obs_shape.prod()
    
    def rollout_fn(rng):
        # Initialize networks
        rng, init_rng = jax.random.split(rng)
        actor_critic = ActorCriticRNN(action_dim=config.action_dim, config=config)
        init_x = jnp.zeros((2, 1, config.observation_dim))
        ac_init_in = (init_x, jnp.zeros((2, 1)))
        init_hidden = ScannedRNN.initialize_carry(2, 128)
        network_params = actor_critic.init(init_rng, init_hidden, ac_init_in)
        
        # load network
        def load_params(params):
            with open(config.model_load_path, "rb") as f:
                loaded_params = serialization.from_bytes(params, f.read())
            return loaded_params
        network_params = jax.experimental.io_callback(load_params, network_params, network_params)
            
        
        
        
        jitted_reset = jax.jit(env.reset)
        rng = jax.random.PRNGKey(0)
        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, config.num_envs)
        obs, state = jax.vmap(jitted_reset, in_axes=(0,))(reset_keys)
        obs = obs.reshape((config.num_envs, -1))
        init_hstate = ScannedRNN.initialize_carry(config.num_envs, 128)
        
        # COLLECT LOOP
        def _collect_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            def _env_step(runner_state, unused):
                behavior_params, last_obs, state, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = actor_critic.apply(behavior_params, hstate, ac_in)
                action = pi.sample(seed=_rng).squeeze()
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, step_key = jax.random.split(rng)
                step_keys = jax.random.split(step_key, config.num_envs)
                obs, state, reward, done = jax.vmap(env.step, in_axes=(0, 0, 0))(step_keys, state, action)
                transition = PPO_Transitions(
                    obs=last_obs,
                    action=action.squeeze(),
                    reward=reward,
                    done=done,
                    value=value.squeeze(),
                    log_prob=log_prob.squeeze()
                )
                runner_state = (behavior_params, obs.reshape((obs.shape[0], -1)), state, done, hstate, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )
            save_data = Transitions(
                obs=traj_batch.obs,
                action=traj_batch.action,
                reward=traj_batch.reward,
                done=traj_batch.done
            )

            def callback(avg_returns, update_steps, save_data):                
                wandb.log(
                    {
                        "returns": avg_returns,
                    }
                )
                model_name = config.model_load_path.split("/")[-2].split("-")[-1]
                save_path = f"datasets/{config.env}/{model_name}/data_{update_steps}.pkl"
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    pickle.dump(save_data, f)
                print("Step:", update_steps, "Returns: ", avg_returns)
            
            avg_returns = jnp.mean(jnp.sum(traj_batch.reward, axis=0)/ jnp.sum(traj_batch.done, axis=0))
            jax.experimental.io_callback(callback, None, avg_returns, update_steps, save_data)
            update_steps = update_steps + 1
            return (runner_state, update_steps), None

        rng, _rng = jax.random.split(rng)
        runner_state = (network_params, obs, state, jnp.zeros((config.num_envs), dtype=bool), init_hstate, _rng)
        (runner_state, _), _ = jax.lax.scan(
            _collect_step, (runner_state, 0), jnp.arange(config.n_episodes)
        )
        return runner_state
    return rollout_fn
    
    
    

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
        # mode="disabled" if config.debug else "online",
        mode="disabled",
    )
    
    print("---------------------------------------")
    print(f"Collecting, Env: {config.env}, Seed: {config.seed}")
    print("---------------------------------------")
    rollout_fn = get_rollout(config)
    jitted_rollout_fn = jax.jit(rollout_fn)
    rng = jax.random.PRNGKey(config.seed)
    with jax.disable_jit(config.disable_jit):
        runnerstate= jitted_rollout_fn(rng)
    

