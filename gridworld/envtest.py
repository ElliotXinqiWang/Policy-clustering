from gridworld.env import SingleAgentGridworld
from continuous_gridworld import SingleAgentEnv
import jax.numpy as jnp
import jax

# env = SingleAgentGridworld(grid_size=7, max_steps=30, distance_penalty=-0.1, goal_reward=10.0)

env = SingleAgentEnv(
    n_barriers=1,
    max_steps=100,
    agent_size=0.05,
    barrier_size=0.1,
    contact_force=0.1,
    noise_constant=0.01
)

action_to_move = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
    4: (0, 0)
}
# test with a random agnet
class random_agent:
    def __init__(self, num_actions=5):
        self.num_actions = num_actions
    
    def get_action(self, obs, key):
        return jax.random.choice(key, self.num_actions, shape=(1,))
    
class continous_random_agent:
    def __init__(self, action_dim=2):
        self.action_dim = action_dim
    
    def get_action(self, obs, key):
        return jax.random.uniform(key, shape=(self.action_dim,), minval=-1.0, maxval=1.0)
    
# test_agent = random_agent(num_actions=5)
test_agent = continous_random_agent(action_dim=2)


num_environments = 16
# scan through episodes
def env_step(runner_state, timestep):
    obs, state, rng = runner_state
    rng, act_key = jax.random.split(rng)
    # act_keys = jax.random.split(act_key, num_environments)
    # actions = jax.vmap(test_agent.get_action, in_axes=(0, 0))(obs, act_keys).squeeze()
    actions = test_agent.get_action(obs, act_key)
    
    rng, step_key = jax.random.split(rng)
    # step_keys = jax.random.split(step_key, num_environments)
    # obs, state, reward, done = jax.vmap(env.step, in_axes=(0, 0, 0))(step_keys, state, actions)
    obs, state, reward, done = env.step(step_key, state, actions)
    return (obs, state, rng), (obs, state, actions, reward, done)

rng = jax.random.PRNGKey(0)
rng, reset_key = jax.random.split(rng)
# reset_keys = jax.random.split(reset_key, num_environments)
# obs, state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)
obs, state = env.reset(reset_key)
runner_state = (obs, state, rng)
jitted_env_step = jax.jit(env_step)
runner_state, (obs, state, actions, reward, done) = jax.lax.scan(jitted_env_step, runner_state, jnp.arange(30))

vis = env.visualize_states(
    state,
    filename="continous.gif",
    rewards=reward,
    actions=actions,
    interval=200
)

print("obs shape: ", obs.shape)
# # print one of the episodes 
# print("goal position: ", state.goal_pos[0][0])
# print("agent position: ", state.agent_pos[0][0])
# for i in range(30):
#     print(f"Step {i}")
#     print(f"Agent position: {state.agent_pos[i][0]}, action: {action_to_move[actions[i][0].item()]}")
#     print(f"Reward: {reward[i][0]}, cumulative reward: {jnp.sum(reward[:i+1, 0])}")
#     print(f"Done: {done[i][0]}")
    