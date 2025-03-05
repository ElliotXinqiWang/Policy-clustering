import jax
import jax.numpy as jnp
import chex
from flax import struct
from gymnax.environments.spaces import Box
from typing import Tuple, Dict
import matplotlib.pyplot as plt

@struct.dataclass
class SingleState:
    """Single-Agent Environment State"""
    p_pos: chex.Array  # Agent position [x, y]
    p_vel: chex.Array  # Agent velocity [vx, vy]
    barriers: chex.Array  # Positions of barriers, shape=[n_barriers, 2]
    done: bool         # Whether the episode is done
    step: int          # Current step
    goal: chex.Array   # The target position [x, y]

class SingleAgentEnv:
    def __init__(self,
                 n_barriers: int = 1,
                 max_steps: int = 100,
                 agent_size: float = 0.05,
                 barrier_size: float = 0.05):
        """Initialize a single-agent environment with optional barriers."""
        self.n_barriers = n_barriers
        self.max_steps = max_steps
        self.agent_size = agent_size
        self.barrier_size = barrier_size

        # Force-control: action is 2D continuous for acceleration
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,))
        # Observation shape:
        # [p_pos(x,y), p_vel(x,y), goal(x,y), barrier0(x,y), barrier1(x,y), ...]
        obs_dim = 2 + 2 + 2 + 2 * n_barriers
        self.observation_space = Box(low=-jnp.inf, high=jnp.inf, shape=(obs_dim,))

        # Reward hyperparameters
        self.collision_penalty = -5.0  # collision penalty for each barrier we hit
        self.distance_coef = 1.0       # coefficient for distance to the goal

    def reset(self) -> SingleState:
        """Reset environment to default initial state."""
        # Default agent at origin, velocity=0
        p_pos = jnp.array([0.0, 0.0])
        p_vel = jnp.array([0.0, 0.0])

        # Place all barriers at (0,0) by default (or you can randomize them)
        barriers = jnp.zeros((self.n_barriers, 2))

        done = False
        step = 0
        goal = jnp.array([1.0, 1.0])
        return SingleState(p_pos=p_pos, p_vel=p_vel, barriers=barriers, done=done, step=step, goal=goal)

    def get_observation(self, state: SingleState) -> jnp.ndarray:
        """Combine agent pos, vel, goal pos, and barrier positions into a single vector."""
        # Flatten barrier positions
        barrier_flat = state.barriers.reshape(-1)
        return jnp.concatenate([state.p_pos, state.p_vel, state.goal, barrier_flat])

    def check_collision(self, pos1: jnp.ndarray, pos2: jnp.ndarray) -> bool:
        """Check if agent circle collides with barrier circle."""
        dist = jnp.linalg.norm(pos1 - pos2)
        return dist < (self.agent_size + self.barrier_size)

    def get_reward(self, state: SingleState, collisions: jnp.ndarray) -> float:
        """Compute reward = distance to goal + collision penalty (both negative)."""
        distance_reward = -self.distance_coef * jnp.linalg.norm(state.p_pos - state.goal)
        collision_cost = self.collision_penalty * collisions.sum()
        return distance_reward + collision_cost

    def step(self, state: SingleState, action: chex.Array) -> Tuple[SingleState, float, bool, Dict]:
        """Apply action as a force, update velocity and position, compute collisions, and optionally reset."""
        # Hyperparameters for acceleration and friction
        force_scale = 0.1
        friction = 0.99

        # Update velocity by applying action (force) then friction
        new_vel = state.p_vel + action * force_scale
        new_vel = new_vel * friction

        # Update position
        new_pos = state.p_pos + new_vel

        # Vectorized collision check with barriers using vmap
        collision_results = jax.vmap(lambda b: self.check_collision(new_pos, b))(state.barriers)

        # Compute if done (close to goal OR exceed max steps)
        dist_to_goal = jnp.linalg.norm(new_pos - state.goal)
        done_condition = (dist_to_goal < 0.1) | (state.step >= self.max_steps)

        # Tentative new state
        new_state = SingleState(
            p_pos=new_pos,
            p_vel=new_vel,
            barriers=state.barriers,
            done=done_condition,
            step=state.step + 1,
            goal=state.goal
        )

        # Compute reward
        reward_val = self.get_reward(new_state, collision_results)

        # If we are done, we reset immediately using a jax.lax.cond
        def do_reset(_):
            return self.reset()

        def do_keep(_):
            return new_state

        final_state = jax.lax.cond(done_condition, do_reset, do_keep, operand=None)

        return final_state, reward_val, bool(final_state.done), {}

    def render(self, state: SingleState):
        """Render the environment with matplotlib."""
        plt.figure(figsize=(5, 5))
        plt.xlim(-1, 2)
        plt.ylim(-1, 2)
        # Agent
        plt.scatter(state.p_pos[0], state.p_pos[1], label="Agent")
        # Goal
        plt.scatter(state.goal[0], state.goal[1], marker="x", label="Goal")
        # Barriers
        for i in range(self.n_barriers):
            plt.scatter(state.barriers[i, 0], state.barriers[i, 1], marker="s", label=f"Barrier {i}")
        plt.legend()
        plt.show()

    def render_from_state_list(self, state_list, filename="rollout.gif"):
        """Render all states in state_list as frames, then save as a GIF."""
        import imageio
        import numpy as np

        frames = []
        for s in state_list:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_xlim(-1, 2)
            ax.set_ylim(-1, 2)
            # Agent
            ax.scatter(s.p_pos[0], s.p_pos[1], label="Agent")
            # Goal
            ax.scatter(s.goal[0], s.goal[1], marker="x", label="Goal")
            # Barriers
            for i in range(self.n_barriers):
                ax.scatter(s.barriers[i, 0], s.barriers[i, 1], marker="s", label=f"Barrier {i}")
            ax.legend()

            # Convert figure to numpy image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            frames.append(image)
            plt.close(fig)

        imageio.mimsave(filename, frames, fps=5)
