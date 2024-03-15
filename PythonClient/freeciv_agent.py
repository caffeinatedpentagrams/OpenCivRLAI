"""
Implementation of CNN-based AI
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Class :: Agent
class freeciv_agent(nn.Module):
    def __init__(self, input_tensor, output_tensor):
        super(freeciv_agent, self).__init__()
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        # Conv Layers
        self.conv1 = nn.Conv2d(in_channels=input_tensor[0], out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # FC layers
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, output_tensor)

    # NN :: Forward Pass
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        #Flattening
        x = x.view(-1, 64 * 3 * 3)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

    # Policy :: Sampling Actions
    def action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  
        action_probs = self.forward(state)  
        action = torch.multinomial(action_probs, num_samples=1).item()  
        return action

# Linear reward 
def reward(economy, time):
    a = 0.7
    b = 0.5
    c = 1
    reward = a * economy + b * time + c
    return reward
    # Gamma
    #discounted_future_reward = 0
    #for k in range(horizon):
    #    discounted_future_reward += gamma**k * reward(economy, time, 0, horizon - k)
    #return reward + gamma * discounted_future_reward

# Training loop
def main():
    input_tensor = (1, n, m)  
    output_tensor = k         

    agent = freeciv_agent(input_tensor, output_tensor)  
    optimizer = optim.Adam(agent.parameters(), lr=0.001)  #Adam optimizer

    episodes = 1000 #Tweakable

    for episode in range(episodes):
        # Initialization
        state = env.reset()

        while not done:
            # Action(state : s)
            action = agent.action(state)

            # Observe : s+1
            next_state, _, done = env.step(action)

            # Reward
            economy = calculate_economy(state)  # Client retrival needed
            time = calculate_time(state)        # Client retrival needed
            reward = reward(economy, time)

            # Trainining
            optimizer.zero_grad()  
            action_probs = agent(torch.tensor(state, dtype=torch.float32).unsqueeze(0))  # Action Prob
            loss = -torch.log(action_probs[0, action]) * reward  
            loss.backward()  # Backprop
            optimizer.step()  # Updating Parameters

            # Update state
            state = next_state

if __name__ == "__main__":
    main()
