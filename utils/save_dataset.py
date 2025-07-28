# Import libraries
from utils.load_dataset import load, load_env
from types import SimpleNamespace
from typing import NamedTuple, Optional
import jax.numpy as jnp
from dataclasses import dataclass, field

# class Transitions(NamedTuple):
#     obs: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
#     action: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
#     reward: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
#     done: jnp.ndarray = field(default_factory=lambda: jnp.empty((0,)))
class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray


def convert_done_terminated_transitions(transitions):
    T, N = transitions.done.shape
    obs_dim = transitions.obs.shape[-1]
    action_dim = transitions.action.shape[-1]
    
    first_done_idx = jnp.argmax(transitions.done, axis=0)  # shape: (N,)
    
    steps = jnp.arange(T)[:, None]  # (T, 1)
    mask = steps <= first_done_idx  # (T, N)
    
    obs = transitions.obs.reshape(-1, obs_dim)
    actions = transitions.action.reshape(-1, action_dim)
    rewards = transitions.reward.reshape(-1)
    dones = transitions.done.reshape(-1)
    mask = mask.reshape(-1)
    
    next_obs = jnp.roll(obs, shift=-1, axis=0)
    next_mask = jnp.roll(mask, shift=-1, axis=0)
    
    invalid_next = (~mask) | dones | (~next_mask)
    next_obs = jnp.where(
        jnp.expand_dims(invalid_next, axis=-1),
        jnp.zeros_like(next_obs),
        next_obs
    )
    
    valid_idx = jnp.where(mask, size=mask.shape[0])[0]
    obs = obs[valid_idx]
    actions = actions[valid_idx]
    rewards = rewards[valid_idx]
    next_obs = next_obs[valid_idx]
    dones = dones[valid_idx]

    return Transition(
        observations=obs,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        dones=dones,
    )