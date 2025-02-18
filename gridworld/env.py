from collections import OrderedDict
from enum import IntEnum
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Dict
import chex
from flax import struct
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches

@struct.dataclass
class State:
    agent_pos: chex.Array
    goal_pos: chex.Array
    wall_map: chex.Array
    time: int
    terminal: bool
    info: Dict[str, chex.Array] = struct.field(pytree_node=True)


class Actions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


class SingleAgentGridworld:
    def __init__(
            self, 
            grid_size: int = 7, 
            max_steps: int = 30, 
            distance_penalty: float = -0.1, 
            goal_reward: float = 10.0, 
            epsilon: float = 0.00,
            num_obstacles: int = 5):
        """
        grid_size (int): The size of the grid (grid_size x grid_size).
        max_steps (int): Maximum steps before the environment terminates.
        distance_penalty (float): Penalty for each step based on Manhattan distance to the goal.
        goal_reward (float): Reward for reaching the goal.
        epsilon (float): Probability of taking a random action for exploration.
        """
        self.name = "SingleAgentGridworld"
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.distance_penalty = distance_penalty
        self.goal_reward = goal_reward
        self.epsilon = epsilon
        self.actions = jnp.array([Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT, Actions.STAY])
        self.observation_shape = jnp.array((grid_size, grid_size, 3))  # One-hot layers for agent, goal, and walls
        self.num_obstacles = num_obstacles
        self.maximum_reference_reward = goal_reward

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """Resets the environment to an initial state."""
        h, w = self.grid_size, self.grid_size
        
        # Initialize wall map without outer walls
        wall_map = jnp.zeros((h, w), dtype=jnp.bool_)
        
        # Generate random obstacle mask
        key, subkey = jax.random.split(key)
        available_positions = jnp.array([
            (x, y) for x in range(1, h - 1) for y in range(1, w - 1)
        ])
        shuffled_positions = jax.random.permutation(subkey, available_positions)
        obstacle_positions = shuffled_positions[:self.num_obstacles]

        # Use scan to update the wall map
        def update_wall_map(carry, pos):
            wall_map = carry
            wall_map = wall_map.at[pos[0], pos[1]].set(True)
            return wall_map, None

        wall_map, _ = lax.scan(update_wall_map, wall_map, obstacle_positions)

        # Randomly choose spawn and goal positions
        remaining_positions = shuffled_positions[self.num_obstacles:]
        spawn_pos = remaining_positions[0]
        goal_pos = remaining_positions[1]
        
        # delete obstacles near the spawn and goal
        def delete_obstacle(carry, pos):
            wall_map = carry
            wall_map = wall_map.at[pos[0], pos[1]].set(False)
            wall_map = wall_map.at[pos[0]+1, pos[1]].set(False)
            wall_map = wall_map.at[pos[0]-1, pos[1]].set(False)
            wall_map = wall_map.at[pos[0], pos[1]+1].set(False)
            wall_map = wall_map.at[pos[0], pos[1]-1].set(False)
            return wall_map, None
        wall_map, _ = lax.scan(delete_obstacle, wall_map, [spawn_pos, goal_pos])
        
        # add outer walls
        wall_map = wall_map.at[0, :].set(True)
        wall_map = wall_map.at[-1, :].set(True)
        wall_map = wall_map.at[:, 0].set(True)
        wall_map = wall_map.at[:, -1].set(True)

        state = State(
            agent_pos=spawn_pos,
            goal_pos=goal_pos,
            wall_map=wall_map,
            time=0,
            terminal=False,
            info={}
        )

        obs = self.get_obs(state)
        return obs, state

    def step(self, key: chex.PRNGKey, state: State, action: Actions) -> Tuple[chex.Array, State, float, bool]:
        # if state.terminal:
        #     raise ValueError("Cannot call step on a terminal state.")

        # Determine whether to take a random action
        key, subkey = jax.random.split(key)
        take_random_action = jax.random.uniform(subkey) < self.epsilon

        # Choose random action if needed
        key, subkey = jax.random.split(key)
        random_action = jax.random.choice(subkey, self.actions)
        action = lax.cond(take_random_action, lambda _: random_action, lambda _: action, None)
        # Calculate the new position
        move = jnp.array([0, 0])
        move = lax.cond(action == Actions.UP, lambda _: jnp.array([-1, 0]), lambda _: move, None)
        move = lax.cond(action == Actions.DOWN, lambda _: jnp.array([1, 0]), lambda _: move, None)
        move = lax.cond(action == Actions.LEFT, lambda _: jnp.array([0, -1]), lambda _: move, None)
        move = lax.cond(action == Actions.RIGHT, lambda _: jnp.array([0, 1]), lambda _: move, None)

        new_pos = state.agent_pos + move

        # Check for collisions with walls
        collides = state.wall_map[new_pos[0], new_pos[1]]
        new_pos = lax.cond(collides, lambda _: state.agent_pos, lambda _: new_pos, None)

        # Check for terminal condition (reaching the goal)
        reached_goal = jnp.all(new_pos == state.goal_pos)
        done = reached_goal | (state.time + 1 >= self.max_steps)

        # Compute reward
        distance = jnp.abs(new_pos - state.goal_pos).sum()
        reward = lax.cond(
            reached_goal,
            lambda _: self.goal_reward,
            # lambda _: self.distance_penalty * distance,
            lambda _: self.distance_penalty,
            None
        )
        # reward = self.goal_reward + self.distance_penalty * distance
        
        # Update state
        new_state = State(
            agent_pos=new_pos,
            goal_pos=state.goal_pos,
            wall_map=state.wall_map,
            time=state.time + 1,
            terminal=done,
            info=state.info
        )

        obs = self.get_obs(new_state)
        
        # calculate the reset obs and reset state
        key, subkey = jax.random.split(key)
        obs_re, state_re = self.reset(subkey)
        obs, new_state, reward, done = lax.cond(
            done,
            lambda _: (obs_re, state_re, reward, True),
            lambda _: (obs, new_state, reward, False),
            None
        )
        
        
        return obs, new_state, reward, done

    def get_obs(self, state: State) -> chex.Array:
        """Returns the observation as a one-hot encoded grid."""
        h, w = self.grid_size, self.grid_size
        obs = jnp.zeros((h, w, 3), dtype=jnp.uint8)

        # Add walls, agent, and goal to separate channels
        obs = obs.at[:, :, 0].set(state.wall_map)
        obs = obs.at[state.agent_pos[0], state.agent_pos[1], 1].set(1)
        obs = obs.at[state.goal_pos[0], state.goal_pos[1], 2].set(1)
        return obs

    def render(self, state: State) -> None:
        """Renders the current state of the environment."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = " "
        grid[state.wall_map] = "#"
        grid[state.goal_pos[0], state.goal_pos[1]] = "G"
        grid[state.agent_pos[0], state.agent_pos[1]] = "A"
        print("\n".join(["".join(row) for row in grid]))

    def get_normalized_score(self, reward: float) -> float:
        """Returns the normalized score for the environment."""
        return reward

    def visualize_states(self, states: list, filename: str = "gridworld.gif", rewards=None, actions=None):
        """Creates a GIF from a list of states."""
        fig, ax = plt.subplots(figsize=(self.grid_size / 2, self.grid_size / 2))

        def update(frame):
            ax.clear()
            state = states
            grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
            grid[:] = " "
            grid[state.wall_map[frame]] = "#"
            grid[state.goal_pos[frame][0], state.goal_pos[frame][1]] = "G"
            grid[state.agent_pos[frame][0], state.agent_pos[frame][1]] = "A"
            ax.imshow(state.wall_map[frame], cmap="gray", alpha=0.3)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            # Add rewards and actions if provided
            if rewards is not None:
                ax.text(
                    self.grid_size - 1.5, -0.8,
                    f"Reward: {rewards[frame]:.2f}" if frame < len(rewards) else "Reward: N/A",
                    ha="right", va="center", fontsize=10, color="blue", weight="bold"
                )

            if actions is not None:
                ax.text(
                    self.grid_size - 1.5, -1.2,
                    f"Action: {actions[frame]}" if frame < len(actions) else "Action: N/A",
                    ha="right", va="center", fontsize=10, color="green", weight="bold"
                )


        ani = animation.FuncAnimation(fig, update, frames=states.wall_map.shape[0], interval=200)
        ani.save(filename, writer="imagemagick")
        plt.close(fig)
        print("Saved GIF to", filename)

    def visualize_obs(self, obs_list, filename="gridworld_obs.gif", rewards=None, actions=None, interval=200):
        """
        Creates a GIF from a list of observations.
        Caution: This method assumes that the observations are in the format returned by get_obs.
        Do not normalize the observations before passing them to this method.
        """
        grid_size = obs_list[0].shape[0]
        fig, ax = plt.subplots(figsize=(grid_size / 2, grid_size / 2))

        def update(frame):
            ax.clear()
            obs = obs_list[frame]
            
            # Extract each layer from the observation
            wall_layer = obs[:, :, 0]
            agent_layer = obs[:, :, 1]
            goal_layer = obs[:, :, 2]

            # Create grid for text display
            grid = np.full((grid_size, grid_size), " ", dtype=str)
            grid[wall_layer == 1] = "#"
            grid[np.where(goal_layer == 1)] = "G"
            grid[np.where(agent_layer == 1)] = "A"

            # Render the grid with text
            ax.imshow(wall_layer, cmap="gray", alpha=0.3)
            for i in range(grid_size):
                for j in range(grid_size):
                    ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # Add rewards and actions if provided
            if rewards is not None:
                ax.text(
                    grid_size - 1.5, -0.8,
                    f"Reward: {rewards[frame]:.2f}" if frame < len(rewards) else "Reward: N/A",
                    ha="right", va="center", fontsize=10, color="blue", weight="bold"
                )

            if actions is not None:
                ax.text(
                    grid_size - 1.5, -1.2,
                    f"Action: {actions[frame]}" if frame < len(actions) else "Action: N/A",
                    ha="right", va="center", fontsize=10, color="green", weight="bold"
                )

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(obs_list), interval=interval)
        ani.save(filename, writer="imagemagick")
        plt.close(fig)
        print("Saved GIF to", filename)

class FixedGridworld(SingleAgentGridworld):
    def __init__(self, K: int = 3, max_steps: int = 30, distance_penalty: float = -0.1, goal_reward: float = 10.0):
        super().__init__(grid_size=2 * K + 1, max_steps=max_steps, distance_penalty=distance_penalty, goal_reward=goal_reward)
        # save K as a jax-traceable array
        self.K = jnp.array(K)
        self.name = "FixedGridworld"
        
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """Resets the environment to an initial state."""
        K = self.K
        h, w = self.grid_size, self.grid_size

        # Initialize wall map
        wall_map = jnp.zeros((h, w), dtype=jnp.bool_)

        # Outer walls
        wall_map = wall_map.at[0, :].set(True)
        wall_map = wall_map.at[-1, :].set(True)
        wall_map = wall_map.at[:, 0].set(True)
        wall_map = wall_map.at[:, -1].set(True)

        # Internal walls
        wall_map = wall_map.at[K, 3:-2].set(True)  # Horizontal wall
        wall_map = wall_map.at[K - 1, 1].set(True)  # Above spawn
        wall_map = wall_map.at[K + 1, 1].set(True)  # Below spawn

        # Spawn and goal positions
        spawn_pos = jnp.array([K, 1])
        goal_pos = jnp.array([K, 2 * K-1])

        # Randomize transition at [K, 2]
        key, subkey = jax.random.split(key)
        offset = jax.random.choice(subkey, jnp.array([-1, 1]))  # -1 or +1

        state = State(
            agent_pos=spawn_pos,
            goal_pos=goal_pos,
            wall_map=wall_map,
            time=0,
            terminal=False,
            info = {"transition_offset": offset}
        )

        obs = self.get_obs(state)
        return obs, state

    def step(self, key: chex.PRNGKey, state: State, action: Actions) -> Tuple[chex.Array, State, float, bool]:
        """Performs a step in the environment."""
        K = self.K

        # Calculate movement
        move = jnp.array([0, 0])
        move = lax.cond(action == Actions.UP, lambda _: jnp.array([-1, 0]), lambda _: move, None)
        move = lax.cond(action == Actions.DOWN, lambda _: jnp.array([1, 0]), lambda _: move, None)
        move = lax.cond(action == Actions.LEFT, lambda _: jnp.array([0, -1]), lambda _: move, None)
        move = lax.cond(action == Actions.RIGHT, lambda _: jnp.array([0, 1]), lambda _: move, None)

        new_pos = state.agent_pos + move

        # Handle wall collisions
        collides = state.wall_map[new_pos[0], new_pos[1]]
        new_pos = lax.cond(collides, lambda _: state.agent_pos, lambda _: new_pos, None)
        print("K type: ", type(K))
        print("transition_offset type: ", type(state.info["transition_offset"]))
        # Handle random transition at [K, 2]
        new_pos = lax.cond(
            jnp.all(new_pos == jnp.array([K, 2])),
            lambda _: jnp.array([K + state.info["transition_offset"], 2]),
            lambda _: new_pos,
            None
        )

        # Check for terminal condition
        reached_goal = jnp.all(new_pos == state.goal_pos)
        done = reached_goal | (state.time + 1 >= self.max_steps)

        # Compute reward
        distance = jnp.abs(new_pos - state.goal_pos).sum()
        reward = lax.cond(
            reached_goal,
            lambda _: self.goal_reward,
            # lambda _: self.distance_penalty * distance,
            lambda _: self.distance_penalty,
            None
        )

        # Update state
        new_state = State(
            agent_pos=new_pos,
            goal_pos=state.goal_pos,
            wall_map=state.wall_map,
            time=state.time + 1,
            terminal=done,
            info=state.info
        )

        obs = self.get_obs(new_state)
        
        # calculate the reset obs and reset state
        key, subkey = jax.random.split(key)
        obs_re, state_re = self.reset(subkey)
        obs, new_state, reward, done = lax.cond(
            done,
            lambda _: (obs_re, state_re, reward, True),
            lambda _: (obs, new_state, reward, False),
            None
        )

        return obs, new_state, reward, done


class ExtraRewardGridworld(SingleAgentGridworld):
    """
    A gridworld environment similar to SingleAgentGridworld with an additional reward term.
    This environment has two extra "special grids" generated randomly at the start.
    When reaching a special grid at the first time, the agent receives an additional reward.
    When initiating the environment, ExtraRewardGridworld takes a parameter `extra_reward` controlling the additional reward.
    extra_reward == 0: No additional reward. The same as SingleAgentGridworld.
    extra_reward > 0: The additional reward when reaching a special grid.
    extra_reward < 0: The additional penalty when reaching a special grid.
    """
    def __init__(
        self, 
        grid_size: int = 7, 
        max_steps: int = 30, 
        distance_penalty: float = -0.1, 
        goal_reward: float = 10, 
        epsilon: float = 0, 
        num_obstacles: int = 5,
        extra_reward: float = 1.0):
        super().__init__(grid_size, max_steps, distance_penalty, goal_reward, epsilon, num_obstacles)
        self.extra_reward = extra_reward
        self.maximum_reference_reward = jnp.max(jnp.array((goal_reward + extra_reward * 2, goal_reward)))
        self.observation_shape = jnp.array((grid_size, grid_size, 4))  # One-hot layers for agent, goal, special grids and walls
    
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        h, w = self.grid_size, self.grid_size
        
        # Initialize wall map without outer walls
        wall_map = jnp.zeros((h, w), dtype=jnp.bool_)
        
        # Generate random obstacle mask
        key, subkey = jax.random.split(key)
        available_positions = jnp.array([
            (x, y) for x in range(1, h - 1) for y in range(1, w - 1)
        ])
        shuffled_positions = jax.random.permutation(subkey, available_positions)
        obstacle_positions = shuffled_positions[:self.num_obstacles]

        # Use scan to update the wall map
        def update_wall_map(carry, pos):
            wall_map = carry
            wall_map = wall_map.at[pos[0], pos[1]].set(True)
            return wall_map, None
        wall_map, _ = lax.scan(update_wall_map, wall_map, obstacle_positions)
        # Randomly choose spawn and goal positions
        remaining_positions = shuffled_positions[self.num_obstacles:]
        spawn_pos = remaining_positions[0]
        goal_pos = remaining_positions[1]
        
        # Randomly choose two special grids
        special_pos = remaining_positions[2:4]
        # delete obstacles near the spawn, goal and special_pos
        def delete_obstacle(carry, pos):
            wall_map = carry
            wall_map = wall_map.at[pos[0], pos[1]].set(False)
            wall_map = wall_map.at[pos[0]+1, pos[1]].set(False)
            wall_map = wall_map.at[pos[0]-1, pos[1]].set(False)
            wall_map = wall_map.at[pos[0], pos[1]+1].set(False)
            wall_map = wall_map.at[pos[0], pos[1]-1].set(False)
            return wall_map, None
        wall_map, _ = lax.scan(delete_obstacle, wall_map, remaining_positions[:4])
        # add outer walls
        wall_map = wall_map.at[0, :].set(True)
        wall_map = wall_map.at[-1, :].set(True)
        wall_map = wall_map.at[:, 0].set(True)
        wall_map = wall_map.at[:, -1].set(True)
        state = State(
            agent_pos=spawn_pos,
            goal_pos=goal_pos,
            wall_map=wall_map,
            time=0,
            terminal=False,
            info={"special_pos": special_pos, "extra_reward": self.extra_reward, "special_reached": jnp.zeros(2, dtype=jnp.bool_)}
        )
        return self.get_obs(state), state
    
    def get_obs(self, state):
        h, w = self.grid_size, self.grid_size
        obs = jnp.zeros((h, w, 4), dtype=jnp.uint8)
        special_pos = state.info["special_pos"]

        # Add walls, agent, and goal to separate channels
        obs = obs.at[:, :, 0].set(state.wall_map)
        obs = obs.at[state.agent_pos[0], state.agent_pos[1], 1].set(1)
        obs = obs.at[state.goal_pos[0], state.goal_pos[1], 2].set(1)
        obs = obs.at[special_pos[0][0], special_pos[0][1], 3].set(1)
        obs = obs.at[special_pos[1][0], special_pos[1][1], 3].set(1)
        return obs

    def step(self, key: chex.PRNGKey, state: State, action: Actions) -> Tuple[chex.Array, State, float, bool]:
        # if state.terminal:
        #     raise ValueError("Cannot call step on a terminal state.")

        # Determine whether to take a random action
        key, subkey = jax.random.split(key)
        take_random_action = jax.random.uniform(subkey) < self.epsilon
        # Choose random action if needed
        key, subkey = jax.random.split(key)
        random_action = jax.random.choice(subkey, self.actions)
        action = lax.cond(take_random_action, lambda _: random_action, lambda _: action, None)
        # Calculate the new position
        move = jnp.array([0, 0])
        move = lax.cond(action == Actions.UP, lambda _: jnp.array([-1, 0]), lambda _: move, None)
        move = lax.cond(action == Actions.DOWN, lambda _: jnp.array([1, 0]), lambda _: move, None)
        move = lax.cond(action == Actions.LEFT, lambda _: jnp.array([0, -1]), lambda _: move, None)
        move = lax.cond(action == Actions.RIGHT, lambda _: jnp.array([0, 1]), lambda _: move, None)
        new_pos = state.agent_pos + move
        # Check for collisions with walls
        collides = state.wall_map[new_pos[0], new_pos[1]]
        new_pos = lax.cond(collides, lambda _: state.agent_pos, lambda _: new_pos, None)
        # Check for terminal condition (reaching the goal)
        reached_goal = jnp.all(new_pos == state.goal_pos)
        done = reached_goal | (state.time + 1 >= self.max_steps)
        # Compute reward
        # distance = jnp.abs(new_pos - state.goal_pos).sum()
        reward = lax.cond(
            reached_goal,
            lambda _: self.goal_reward,
            # lambda _: self.distance_penalty * distance,
            lambda _: self.distance_penalty,
            None
        )
        # reward = self.goal_reward + self.distance_penalty * distance
        # add special reward
        special_pos = state.info["special_pos"]
        special_reached = state.info["special_reached"]
        extra_reached = jnp.array([jnp.all(new_pos == special_pos[0]), jnp.all(new_pos == special_pos[1])])
        special_reward = extra_reached & ~special_reached 
        special_reached = special_reached | extra_reached
        reward = reward + special_reward.sum() * state.info["extra_reward"]
        # Update state
        new_state = State(
            agent_pos=new_pos,
            goal_pos=state.goal_pos,
            wall_map=state.wall_map,
            time=state.time + 1,
            terminal=done,
            info={"special_pos": special_pos, "extra_reward": state.info["extra_reward"], "special_reached": special_reached}
        )
        
        obs = self.get_obs(new_state)
        
        # calculate the reset obs and reset state
        key, subkey = jax.random.split(key)
        obs_re, state_re = self.reset(subkey)
        
        obs, new_state, reward, done = lax.cond(
            done,
            lambda _: (obs_re, state_re, reward, True),
            lambda _: (obs, new_state, reward, False),
            None
        )
        
        reward = reward / self.maximum_reference_reward
        
        return obs, new_state, reward, done
    
    def visualize_states(self, states: list, filename: str = "gridworld.gif", rewards=None, actions=None):
        """Creates a GIF from a list of states."""
        fig, ax = plt.subplots(figsize=(self.grid_size / 2, self.grid_size / 2))

        def update(frame):
            ax.clear()
            state = states
            grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
            grid[:] = " "
            grid[state.wall_map[frame]] = "#"
            grid[state.goal_pos[frame][0], state.goal_pos[frame][1]] = "G"
            grid[state.agent_pos[frame][0], state.agent_pos[frame][1]] = "A"
            grid[state.info["special_pos"][frame][0][0], state.info["special_pos"][frame][0][1]] = "S"
            grid[state.info["special_pos"][frame][1][0], state.info["special_pos"][frame][1][1]] = "S"

            # Color the special grid green if not reached, red if reached
            for i in range(2):
                special_pos = state.info["special_pos"][frame][i]
                if state.info["special_reached"][frame][i]:
                    rect_color = "red"
                else:
                    rect_color = "green"
                ax.add_patch(
                    patches.Rectangle(
                        (special_pos[1] - 0.5, special_pos[0] - 0.5), 1, 1, 
                        facecolor=rect_color, edgecolor="black", alpha=0.3
                    )
                )

            ax.imshow(state.wall_map[frame], cmap="gray", alpha=0.3)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # Add rewards and actions if provided
            if rewards is not None:
                ax.text(
                    self.grid_size - 1.5, -0.8,
                    f"Reward: {rewards[frame]:.2f}" if frame < len(rewards) else "Reward: N/A",
                    ha="right", va="center", fontsize=10, color="blue", weight="bold"
                )

            if actions is not None:
                ax.text(
                    self.grid_size - 1.5, -1.2,
                    f"Action: {actions[frame]}" if frame < len(actions) else "Action: N/A",
                    ha="right", va="center", fontsize=10, color="green", weight="bold"
                )

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=states.wall_map.shape[0], interval=200)
        ani.save(filename, writer="imagemagick")
        plt.close(fig)
        print("Saved GIF to", filename)

    def visualize_obs(self, obs_list, filename="gridworld_obs.gif", rewards=None, actions=None, interval=200):
        """
        Creates a GIF from a list of observations.
        Caution: This method assumes that the observations are in the format returned by get_obs.
        Do not normalize the observations before passing them to this method.
        """
        grid_size = obs_list[0].shape[0]
        fig, ax = plt.subplots(figsize=(grid_size / 2, grid_size / 2))

        def update(frame):
            ax.clear()
            obs = obs_list[frame]
            
            # Extract each layer from the observation
            wall_layer = obs[:, :, 0]
            agent_layer = obs[:, :, 1]
            goal_layer = obs[:, :, 2]
            special_layer = obs[:, :, 3]

            # Create grid for text display
            grid = np.full((grid_size, grid_size), " ", dtype=str)
            grid[wall_layer == 1] = "#"
            grid[np.where(goal_layer == 1)] = "G"
            grid[np.where(agent_layer == 1)] = "A"
            grid[np.where(special_layer == 1)] = "S"

            # Highlight special positions
            special_positions = np.argwhere(special_layer == 1)
            for pos in special_positions:
                # color = "red" if agent_layer[tuple(pos)] == 1 else "green"
                color = "yellow"
                ax.add_patch(
                    patches.Rectangle(
                        (pos[1] - 0.5, pos[0] - 0.5), 1, 1, 
                        facecolor=color, edgecolor="black", alpha=0.3
                    )
                )

            # Render the grid with text
            ax.imshow(wall_layer, cmap="gray", alpha=0.3)
            for i in range(grid_size):
                for j in range(grid_size):
                    ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # Add rewards and actions if provided
            if rewards is not None:
                ax.text(
                    grid_size - 1.5, -0.8,
                    f"Reward: {rewards[frame]:.2f}" if frame < len(rewards) else "Reward: N/A",
                    ha="right", va="center", fontsize=10, color="blue", weight="bold"
                )

            if actions is not None:
                ax.text(
                    grid_size - 1.5, -1.2,
                    f"Action: {actions[frame]}" if frame < len(actions) else "Action: N/A",
                    ha="right", va="center", fontsize=10, color="green", weight="bold"
                )

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(obs_list), interval=interval)
        ani.save(filename, writer="imagemagick")
        plt.close(fig)
        print("Saved GIF to", filename)


class MDPGridworld(SingleAgentGridworld):
    """
    A gridworld environment with a fixed map. The agent is spawned at a random location in [1:4, 1:4] and the goal is at [7, 7].
    """
    def __init__(self, distance_penalty: float = -0.3, max_steps: int = 40, goal_reward: float = 10.0, epsilon: float = 0.00):
        super().__init__(grid_size=9, max_steps=max_steps, distance_penalty=distance_penalty, goal_reward=goal_reward, epsilon=epsilon)
        self.name = "MDPGridworld"
        self.observation_shape = jnp.array((9, 9, 3))
        self.maximum_reference_reward = goal_reward
        self.wall_map = jnp.array([
            [True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True, True]
        ])
        self.goal_pos = jnp.array([7, 7])
    
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        h, w = self.grid_size, self.grid_size
        spawn_pos = jax.random.randint(key, (2,), minval=1, maxval=5)
        state = State(
            agent_pos=spawn_pos,
            goal_pos=self.goal_pos,
            wall_map=self.wall_map,
            time=0,
            terminal=False,
            info={}
        )
        obs = self.get_obs(state)
        return obs, state
    
class MDPtakeball(SingleAgentGridworld):
    """
    4 balls with index 0 to 3 are randomly placed at the 5 positions in the grid. All the balls will disappear when the agent get one of them.
    The agents goal is to take the ball and reach the goal position.
    """
    def __init__(self, distance_penalty: float = -0.3, max_steps: int = 40, goal_reward: float = 10.0, epsilon: float = 0.00, target_ball=0):
        super().__init__(grid_size=9, max_steps=max_steps, distance_penalty=distance_penalty, goal_reward=goal_reward, epsilon=epsilon)
        self.name = "MDPGridworld"
        self.observation_shape = jnp.array((9, 9, 7))
        self.maximum_reference_reward = goal_reward
        self.wall_map = jnp.array([
            [True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True, True]
        ])
        self.goal_pos = jnp.array([7, 7])
        self.ball_poss = jnp.array([[1, 1], [1, 3], [1, 5], [1, 7]])
        self.target_ball = target_ball
        
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        h, w = self.grid_size, self.grid_size
        spawn_pos_y = jax.random.randint(key, (1,), minval=1, maxval=4)[0]
        state = State(
            agent_pos=jnp.array([7, spawn_pos_y]),
            goal_pos=self.goal_pos,
            wall_map=self.wall_map,
            time=0,
            terminal=False,
            info={
                # "balls_idx": jax.random.permutation(key, jnp.arange(4)), # the index of the balls from left to right
                "balls_idx": jnp.array([0, 1, 2, 3]), # fixed index of the balls
                "ball_got": -1} # the index of the ball that the agent got
        )
        obs = self.get_obs(state)
        return obs, state
    
    def get_obs(self, state):
        h, w = self.grid_size, self.grid_size
        obs = jnp.zeros((h, w, 7), dtype=jnp.uint8)
        ball_poss = self.ball_poss
        ball_idx = state.info["balls_idx"]

        # Add walls, agent, and goal to separate channels
        obs = obs.at[:, :, 0].set(state.wall_map)
        obs = obs.at[state.agent_pos[0], state.agent_pos[1], 1].set(1)
        obs = obs.at[state.goal_pos[0], state.goal_pos[1], 2].set(1)
        obs = obs.at[ball_poss[0][0], ball_poss[0][1], ball_idx[0]+3].set(1)
        obs = obs.at[ball_poss[1][0], ball_poss[1][1], ball_idx[1]+3].set(1)
        obs = obs.at[ball_poss[2][0], ball_poss[2][1], ball_idx[2]+3].set(1)
        obs = obs.at[ball_poss[3][0], ball_poss[3][1], ball_idx[3]+3].set(1)
        
        obs_without_ball = obs.at[:, :, 3:7].set(0)
        
        obs = lax.cond(
            state.info["ball_got"] >= 0,
            lambda _: obs_without_ball,
            lambda _: obs,
            None
        )
        
        return obs
    
    def step(self, key: chex.PRNGKey, state: State, action: Actions) -> Tuple[chex.Array, State, float, bool]:
        # if state.terminal:
        #     raise ValueError("Cannot call step on a terminal state.")

        # Determine whether to take a random action
        key, subkey = jax.random.split(key)
        take_random_action = jax.random.uniform(subkey) < self.epsilon
        # Choose random action if needed
        key, subkey = jax.random.split(key)
        random_action = jax.random.choice(subkey, self.actions)
        action = lax.cond(take_random_action, lambda _: random_action, lambda _: action, None)
        # Calculate the new position
        move = jnp.array([0, 0])
        move = lax.cond(action == Actions.UP, lambda _: jnp.array([-1, 0]), lambda _: move, None)
        move = lax.cond(action == Actions.DOWN, lambda _: jnp.array([1, 0]), lambda _: move, None)
        move = lax.cond(action == Actions.LEFT, lambda _: jnp.array([0, -1]), lambda _: move, None)
        move = lax.cond(action == Actions.RIGHT, lambda _: jnp.array([0, 1]), lambda _: move, None)
        new_pos = state.agent_pos + move
        # Check for collisions with walls
        collides = state.wall_map[new_pos[0], new_pos[1]]
        new_pos = lax.cond(collides, lambda _: state.agent_pos, lambda _: new_pos, None)
        # Check for terminal condition (reaching the goal)
        reached_goal = jnp.all(new_pos == state.goal_pos)
        done = reached_goal | (state.time + 1 >= self.max_steps)
        # Compute reward. Give the reward if only the agent get the right ball
        # distance = jnp.abs(new_pos - state.goal_pos).sum()
        reward = lax.cond(
            reached_goal & (state.info["ball_got"] == self.target_ball),
            lambda _: self.goal_reward,
            # lambda _: self.distance_penalty * distance,
            lambda _: self.distance_penalty,
            None
        )
        # Update the ball_got. Change ball_got only if the agent is at the same position as the ball and ball_got is not set
        ball_idxs = state.info["balls_idx"]
        new_ball_got = lax.cond((new_pos == self.ball_poss[0]).all(), lambda _: ball_idxs[0], lambda _: state.info["ball_got"], None)
        new_ball_got = lax.cond((new_pos == self.ball_poss[1]).all(), lambda _: ball_idxs[1], lambda _: new_ball_got, None)
        new_ball_got = lax.cond((new_pos == self.ball_poss[2]).all(), lambda _: ball_idxs[2], lambda _: new_ball_got, None)
        new_ball_got = lax.cond((new_pos == self.ball_poss[3]).all(), lambda _: ball_idxs[3], lambda _: new_ball_got, None)
        new_ball_got = lax.cond(
            (state.info["ball_got"] == -1),
            lambda _: new_ball_got,
            lambda _: state.info["ball_got"],
            None
        )
        
        # Update state
        new_state = State(
            agent_pos=new_pos,
            goal_pos=state.goal_pos,
            wall_map=state.wall_map,
            time=state.time + 1,
            terminal=done,
            info={"balls_idx": ball_idxs, "ball_got": new_ball_got}
        )
        
        obs = self.get_obs(new_state)
        
        # calculate the reset obs and reset state
        key, subkey = jax.random.split(key)
        obs_re, state_re = self.reset(subkey)
        
        obs, new_state, reward, done = lax.cond(
            done,
            lambda _: (obs_re, state_re, reward, True),
            lambda _: (obs, new_state, reward, False),
            None
        )
        
        reward = reward / self.maximum_reference_reward
        
        return obs, new_state, reward, done
    
    
    def visualize_states(self, states: list, filename: str = "gridworld.gif", rewards=None, actions=None, interval=500):
        """Creates a GIF from a list of states."""
        grid_size = self.grid_size
        fig, ax = plt.subplots(figsize=(self.grid_size / 2, self.grid_size / 2))

        def update(frame):
            ax.clear()
            state = states
            grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
            grid[:] = " "
            grid[state.wall_map[frame]] = "#"
            grid[state.goal_pos[frame][0], state.goal_pos[frame][1]] = "G"
            grid[state.agent_pos[frame][0], state.agent_pos[frame][1]] = "A"
            grid[state.info["balls"][frame][0][0], state.info["balls"][frame][0][1]] = "0"
            grid[state.info["balls"][frame][1][0], state.info["balls"][frame][1][1]] = "1"
            grid[state.info["balls"][frame][2][0], state.info["balls"][frame][2][1]] = "2"
            grid[state.info["balls"][frame][3][0], state.info["balls"][frame][3][1]] = "3"

            # Color the balls 
            for i in range(4):
                ball_pos = state.info["balls"][frame][i]
                ax.add_patch(
                    patches.Rectangle(
                        (ball_pos[1] - 0.5, ball_pos[0] - 0.5), 1, 1, 
                        facecolor="green", edgecolor="black", alpha=0.3
                    )
                )

            ax.imshow(state.wall_map[frame], cmap="gray", alpha=0.3)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # Add rewards and actions if provided
            if rewards is not None:
                ax.text(
                    grid_size - 1.5, -0.8,
                    f"Reward: {rewards[frame]:.2f}" if frame < len(rewards) else "Reward: N/A",
                    ha="right", va="center", fontsize=10, color="blue", weight="bold"
                )

            if actions is not None:
                ax.text(
                    grid_size - 1.5, -1.2,
                    f"Action: {actions[frame]}" if frame < len(actions) else "Action: N/A",
                    ha="right", va="center", fontsize=10, color="green", weight="bold"
                )
            
            # Add information about the ball got
            ax.text(
                grid_size - 3.5, -0.8,
                f"Ball got: {state.info['ball_got'][frame]}" if state.info['ball_got'][frame] > -1 else "Ball got: N/A",
                ha="right", va="center", fontsize=10, color="red", weight="bold"
            )
         # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(states), interval=interval)
        ani.save(filename, writer="imagemagick")
        plt.close(fig)
        print("Saved GIF to", filename)
    
    
    def visualize_obs(self, obs_list, filename="gridworld_obs.gif", rewards=None, actions=None, interval=500):
        """
        Creates a GIF from a list of observations.
        Caution: This method assumes that the observations are in the format returned by get_obs.
        Do not normalize the observations before passing them to this method.
        """
        grid_size = obs_list[0].shape[0]
        fig, ax = plt.subplots(figsize=(grid_size / 2, grid_size / 2))

        def update(frame):
            ax.clear()
            obs = obs_list[frame]
            
            # Extract each layer from the observation
            wall_layer = obs[:, :, 0]
            agent_layer = obs[:, :, 1]
            goal_layer = obs[:, :, 2]

            # Create grid for text display
            grid = np.full((grid_size, grid_size), " ", dtype=str)
            grid[wall_layer == 1] = "#"
            grid[np.where(goal_layer == 1)] = "G"
            grid[np.where(agent_layer == 1)] = "A"
            
            # Add balls to the grid and highlight the balls
            for i in range(4):
                ball_layer = obs[:, :, 3 + i]
                ball_pos = np.argwhere(ball_layer == 1)
                if len(ball_pos) > 0:
                    grid[ball_pos[0][0], ball_pos[0][1]] = str(i)
                    ax.add_patch(
                        patches.Rectangle(
                            (ball_pos[0][1] - 0.5, ball_pos[0][0] - 0.5), 1, 1, 
                            facecolor="green", edgecolor="black", alpha=0.3
                        )
                    )

            # Render the grid with text
            ax.imshow(wall_layer, cmap="gray", alpha=0.3)
            for i in range(grid_size):
                for j in range(grid_size):
                    ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # Add rewards and actions if provided
            if rewards is not None:
                ax.text(
                    grid_size - 1.5, -0.8,
                    f"Reward: {rewards[frame]:.2f}" if frame < len(rewards) else "Reward: N/A",
                    ha="right", va="center", fontsize=10, color="blue", weight="bold"
                )

            if actions is not None:
                ax.text(
                    grid_size - 1.5, -1.2,
                    f"Action: {actions[frame]}" if frame < len(actions) else "Action: N/A",
                    ha="right", va="center", fontsize=10, color="green", weight="bold"
                )

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(obs_list), interval=interval)
        ani.save(filename, writer="imagemagick")
        plt.close(fig)
        print("Saved GIF to", filename)
    