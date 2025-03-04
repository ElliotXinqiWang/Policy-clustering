import jax
import jax.numpy as jnp
import numpy as onp
from gymnax.environments.spaces import Box, Discrete
import chex
from flax import struct
from typing import Tuple, Dict
import matplotlib.pyplot as plt

@struct.dataclass
class State:
    """Single-Agent Environment State"""
    p_pos: chex.Array  # [x, y] position of agent
    p_vel: chex.Array  # [x, y] velocity of agent
    done: bool  # Episode termination flag
    step: int  # Current step
    goal: chex.Array  # [x, y] target position

class SimpleEnv:
    def __init__(self):
        self.action_space = Discrete(4)  # Example: 4 discrete actions (up, down, left, right)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(4,))  # [x, y, v_x, v_y]
        self.max_steps = 100

    def reset(self) -> State:
        """Reset the environment state"""
        state = State(
            p_pos=jnp.array([0.0, 0.0]),
            p_vel=jnp.array([0.0, 0.0]),
            done=False,
            step=0,
            goal=jnp.array([1.0, 1.0]),
        )
        return state

    def step(self, state: State, action: int) -> Tuple[State, jnp.array, bool, Dict]:
        """Apply action and return new state, reward, done, and info"""
        step_size = 0.1
        movement = {
            0: jnp.array([0, step_size]),  # Up
            1: jnp.array([0, -step_size]), # Down
            2: jnp.array([-step_size, 0]), # Left
            3: jnp.array([step_size, 0]),  # Right
        }
        new_pos = state.p_pos + movement.get(action, jnp.array([0, 0]))
        done = jnp.linalg.norm(new_pos - state.goal) < 0.1 or state.step >= self.max_steps
        reward = -jnp.linalg.norm(new_pos - state.goal)  # Negative distance to goal
        new_state = State(p_pos=new_pos, p_vel=state.p_vel, done=done, step=state.step + 1, goal=state.goal)
        return new_state, reward, done, {}

    def render(self, state: State):
        """Render the environment"""
        plt.figure(figsize=(5, 5))
        plt.xlim(-1, 2)
        plt.ylim(-1, 2)
        plt.scatter(*state.p_pos, c='blue', label='Agent')
        plt.scatter(*state.goal, c='red', marker='x', label='Goal')
        plt.legend()
        plt.show()
