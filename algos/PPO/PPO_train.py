import os
import argparse
from dataclasses import dataclass, field
from typing import Optional
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import optax 
from flax.training.train_state import TrainState
from flax import serialization
import wandb
from set_env import set_env

from utils.networks import ActorCriticRNN, ScannedRNN, ContinuousActorCriticRNN

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
    # Experiment
    alg: str = "PPO"  # Algorithm name
    env: str = "ContGrid-Diaganol"  # Environment name
    extra_reward: float = 10.0
    seed: int = 2  # Sets Gym, Jax and Numpy seeds
    max_updates: int = 200 # Maximum number of updates
    n_episodes: int = 5  # How many episodes run during evaluation
    checkpoints_path: Optional[str] = None  # Save path
    # batch_size: int = 32  # Batch size for all networks
    num_envs: int = 128  # Number of environments
    save_model: bool = True  # Save the model
    save_thresholds: list[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, float("inf")])
    epsilon: float = 0.1
    noise_constant: float = 0.01
    # Network & training
    hidden_dim: int = 128
    max_grad_norm: float = 0.5
    learning_rate: float = 1e-3
    adam_eps: float = 1e-8
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 2
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


def get_rollout(config):
    env = set_env(config)  

    # config.action_dim = 5
    config.action_dim = env.action_dim
    obs_shape = env.observation_shape
    config.observation_dim = obs_shape.prod()
    
    def rollout_fn(rng):
        # Initialize networks
        rng, init_rng = jax.random.split(rng)
        if env.action_type == "discrete":
            actor_critic = ActorCriticRNN(action_dim=config.action_dim, config=config)
        else:
            actor_critic = ContinuousActorCriticRNN(action_dim=config.action_dim, config=config)
        init_x = jnp.zeros((2, 1, config.observation_dim))
        ac_init_in = (init_x, jnp.zeros((2, 1)))
        init_hidden = ScannedRNN.initialize_carry(2, 128)
        
        network_params = actor_critic.init(init_rng, init_hidden, ac_init_in)
        best_network_params = network_params
        best_eval_return = -1e6
        
        # Initialize optimizer
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=config.learning_rate, eps=config.adam_eps),
        )
        train_state = TrainState.create(
            apply_fn=actor_critic.apply,
            params=network_params,
            tx=tx,
        )
        jitted_reset = jax.jit(env.reset)
        rng = jax.random.PRNGKey(0)
        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, config.num_envs)
        obs, state = jax.vmap(jitted_reset, in_axes=(0,))(reset_keys)
        obs = obs.reshape((config.num_envs, -1))
        init_hstate = ScannedRNN.initialize_carry(config.num_envs, 128)
        
        save_flags = 0
        
        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps, best_network_params, best_eval_return, save_flags= update_runner_state
            def _env_step(runner_state, unused):
                train_state, last_obs, state, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = actor_critic.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng).squeeze()
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, step_key = jax.random.split(rng)
                step_keys = jax.random.split(step_key, config.num_envs)
                obs, state, reward, done = jax.vmap(env.step, in_axes=(0, 0, 0))(step_keys, state, action)
                info = jnp.zeros((config.num_envs, 1))
                transition = PPO_Transitions(
                    obs=last_obs,
                    action=action.squeeze(),
                    reward=reward,
                    done=done,
                    value=value.squeeze(),
                    log_prob=log_prob.squeeze()
                )
                runner_state = (train_state, obs.reshape((obs.shape[0], -1)), state, done, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )
            eval_return = jnp.mean(jnp.sum(traj_batch.reward, axis=0) / jnp.sum(traj_batch.done, axis=0))
            best_network_params = jax.lax.cond(
                eval_return > best_eval_return,
                lambda x:runner_state[0].params,
                lambda x:x,
                best_network_params
            )
            best_eval_return = jnp.maximum(eval_return, best_eval_return)
            

            # CALCULATE ADVANTAGE
            train_state, obs, state, last_done, hstate, rng = runner_state
            last_obs_batch = obs
            
            ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = actor_critic.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config.gamma * next_value * (1 - done) - value
                    gae = (
                            delta
                            + config.gamma * config.gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = actor_critic.apply(params, init_hstate.transpose(),
                                                        (traj_batch.obs, traj_batch.done))
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config.clip_eps,
                                    1.0 + config.clip_eps,
                                )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                                loss_actor
                                + config.vf_coef * value_loss
                                - config.ent_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                # init_hstate = jnp.reshape(init_hstate, (config.num_steps, config.num_envs, -1))
                batch = (init_hstate, traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config.num_envs)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config.num_minibatches, -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            init_hstate = initial_hstate[None, :].squeeze().transpose() # (64, 128) -> (128, 64)
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            rng = update_state[-1]

            def callback(avg_returns, update_steps, params, save_flags):                
                wandb.log(
                    {
                        "returns": avg_returns,
                    }
                )
                print("Step:", update_steps, "Returns: ", avg_returns)
                if avg_returns > config.save_thresholds[save_flags]:
                    save_path = f"behavior_models/PPO_{config.env}/params_{config.save_thresholds[save_flags]}.pkl"
                    if not os.path.exists(f"behavior_models/PPO_{config.env}"):
                        os.makedirs(f"behavior_models/PPO_{config.env}", exist_ok=True)
                    with open(save_path, "wb") as f:
                        f.write(serialization.to_bytes(params))
                    print("Model saved to ", save_path)
                    save_flags = save_flags + 1
                return save_flags
            
            avg_returns = jnp.mean(jnp.sum(traj_batch.reward, axis=0) / jnp.sum(traj_batch.done, axis=0))
            save_flags = jax.experimental.io_callback(callback, save_flags, avg_returns, update_steps, train_state.params, save_flags)
            update_steps = update_steps + 1
            runner_state = (train_state, obs, state, last_done, hstate, rng)
            return (runner_state, update_steps, best_network_params, best_eval_return, save_flags), None

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, obs, state, jnp.zeros((config.num_envs), dtype=bool), init_hstate, _rng)
        (runner_state, _, best_network_params, best_eval_return, save_flags), _ = jax.lax.scan(
            _update_step, (runner_state, 0, best_network_params, best_eval_return, save_flags), jnp.arange(config.max_updates)
        )
        return runner_state, best_network_params, best_eval_return
    return rollout_fn
    
    
    

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
        # mode="disabled" if config.debug else "online",
        mode="disabled",
    )
    
    print("---------------------------------------")
    print(f"Training PPO, Env: {config.env}, Seed: {config.seed}")
    print("---------------------------------------")
    rollout_fn = get_rollout(config)
    jitted_rollout_fn = jax.jit(rollout_fn)
    with jax.disable_jit(config.disable_jit):
        runnerstate, best_params, best_reward = jitted_rollout_fn(jax.random.PRNGKey(config.seed))
    # network_params = runnerstate[0][0].params
    network_params = best_params
    print("best reward: ", best_reward)
    
    # if behavior_models/PPO_{config.env} not exists, create it
    if not os.path.exists(f"behavior_models/PPO_{config.env}"):
        os.makedirs(f"behavior_models/PPO_{config.env}", exist_ok=True)

    # save the best model
    with open(f"behavior_models/PPO_{config.env}/best_params.pkl", "wb") as f:
        f.write(serialization.to_bytes(network_params))
    print("Best model saved to ", f"behavior_models/PPO_{config.env}/best_params.pkl")
    
    # render a few episodes
    with jax.disable_jit(config.disable_jit):
        env= set_env(config)
        if env.action_type == "discrete":
            network = ActorCriticRNN(action_dim=config.action_dim, config=config)
        else:
            network = ContinuousActorCriticRNN(action_dim=config.action_dim, config=config)
        rng = jax.random.PRNGKey(config.seed)
        obs, state = env.reset(key=rng)
        obs = obs.reshape((1, -1))
        done = False
        def _step_fn(eval_state, timestep):
            hidden, rng, obs, state, last_done = eval_state
            ac_in = (obs[None, :], jnp.array([last_done]).reshape(1, 1))
            hidden, pi, _ = network.apply(network_params, hidden, ac_in)
            action = pi.sample(seed=rng).squeeze().squeeze()
            rng, _rng = jax.random.split(rng)
            obs, state, reward, done = env.step(_rng, state, action)
            return (hidden, rng, obs.reshape((1, -1)), state, done), (state, reward, action) 
        rng, _rng = jax.random.split(rng)
        jitted_step_fn = jax.jit(_step_fn)
        hidden = ScannedRNN.initialize_carry(1, 128)
        
        eval_state, (states, rewards, actions) = jax.lax.scan(jitted_step_fn, (hidden, _rng, obs, state, done), jnp.arange(120))
        
        env.visualize_states(states, filename=config.env + "_test.gif", rewards=rewards, actions=actions)
        