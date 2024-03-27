import gym

#Using some environments from the OpenAI gym for pilot tests before implementation in the FreeCiv Game Environment
env = gym.make('CartPole-v1')

state_shape = env.observation_space.shape
num_actions = env.action_space.n

# Initializing the Agent
agent = DQNAgent(state_shape, num_actions)

# Testing for 10 peisodes (while training/testing, we can add/modify certain hyperparameters as well)
episodes = 10
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        agent.replay()
        agent.decay_epsilon()
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close() 