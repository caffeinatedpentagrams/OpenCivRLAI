import gym
from gym import spaces
import numpy as np

class MultiAgentTest(gym.Env):
    def __init__(self):
        self.num_agents = 2
        self.action_space = spaces.Discrete(4)  # Action space corresponding to 4 motion directions
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)  # Observation space
        self.agents_states = [np.zeros((4,), dtype=np.float32) for _ in range(self.num_agents)]
        self.step_count = 0
        self.max_steps = 100

    def reset(self):
        self.step_count = 0
        self.agents_states = [np.zeros((4,), dtype=np.float32) for _ in range(self.num_agents)]
        return self.agents_states

    def step(self, actions):
        self.step_count += 1
        rewards = [0] * self.num_agents
        dones = [False] * self.num_agents
        for i, action in enumerate(actions):
            self.agents_states[i] += np.random.rand(4) * 0.1  # State transition
            rewards[i] = np.sum(self.agents_states[i])
            if self.step_count >= self.max_steps:
                dones[i] = True
        return self.agents_states, rewards, dones, {}

# Environment
multi_env = MultiAgentTest()

# state_shape and num_acitons in the environment
state_shape = multi_env.observation_space.shape
num_actions = multi_env.action_space.n

# Initializing the Deep Q-Learner Agent for our custom multi-agent setting
agents = [DQNAgent(state_shape, num_actions) for _ in range(multi_env.num_agents)]

# Testing for 10 episodes (variable)
episodes = 10
for episode in range(episodes):
    states = multi_env.reset()
    total_rewards = [0] * multi_env.num_agents
    dones = [False] * multi_env.num_agents
    while not all(dones):
        for i, agent in enumerate(agents):
            if not dones[i]:
                action = agent.select_action(states[i])
                next_states, rewards, dones, _ = multi_env.step([action] * multi_env.num_agents)
                total_rewards[i] += rewards[i]
                agent.remember(states[i], action, rewards[i], next_states[i], dones[i])
                states[i] = next_states[i]
                agent.replay()
                agent.decay_epsilon()
    print(f"Episode {episode + 1}: Total Rewards = {total_rewards}")

# pass (Close)
