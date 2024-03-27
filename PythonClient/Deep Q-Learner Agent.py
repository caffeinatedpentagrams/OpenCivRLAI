import numpy as np
import tensorflow as tf
from collections import deque
import random

#Model Architecture for the Deep Q-Learner

class QNetwork(tf.keras.Model):
    def __init__(self, state_shape, num_actions):
        super(QNetwork, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

class CustomReward:
    def __init__(self):
        #Noting that this is a custom reward formulation so I can modify it for many different cases
        pass

    def calculate_reward(self, state, action, next_state):
        # Init
        reward = 0
        # Multi-factor custom reward
        distance_reward = -0.1 * np.sqrt(np.sum(np.square(next_state['position'] - state['position'])))
        # Adjustments (based on actions performed and retracted from the client)
        if action == 0:  
            reward += distance_reward
        elif action == 1:  #Example scenario for attack
            # Health is directly affected; hence it can be incorporated in the reward
            health_diff = state['opponent_health'] - next_state['opponent_health']
            reward += 2 * health_diff
        # Formulation #1
        return reward

class DQNAgent:
    def __init__(self, state_shape, num_actions, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                 learning_rate=0.00025, batch_size=32, replay_buffer_size=10000):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.q_network = QNetwork(state_shape, num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.custom_reward = CustomReward()

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        else:
            q_values = self.q_network(np.expand_dims(state, axis=0))
            return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.q_network(np.expand_dims(next_state, axis=0))[0])
            target_f = self.q_network(np.expand_dims(state, axis=0)).numpy()
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f)
        states = np.array(states)
        targets = np.array(targets)
        self.q_network.train_on_batch(states, targets)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes):
        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                custom_reward = self.custom_reward.calculate_reward(state, action, next_state)
                self.remember(state, action, custom_reward, next_state, done)
                state = next_state
                self.replay()
                self.decay_epsilon()
