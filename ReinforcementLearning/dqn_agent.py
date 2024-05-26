# dqn_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

import os

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64) # Q Change because of mismatch of shape
        self.fc2 = nn.Linear(64, 128) 
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_dim) # Q Change because of mismatch of shape

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.output(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(current_dir, 'Agent')
        os.makedirs(self.model_dir, exist_ok=True)
        self.state_size = state_size
        self.action_size = action_size
        #self.action_size = self.rows * self.cols
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99  # Discount factor for past rewards
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.model(next_state).detach()))
            current_q = self.model(state)[0][action]
            loss = (target - current_q) ** 2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_old(self, name):
        self.model.load_state_dict(torch.load(name))
        self.model.eval()  # Set the model to evaluation mode
        print("Model loaded from", name)

    def save_old(self, name):
        torch.save(self.model.state_dict(), name)
        print("Model saved to", name)

    def save(self, filename="model.pth"):
        full_path = os.path.join(self.model_dir, filename)
        torch.save(self.model.state_dict(), full_path)
        print("Model saved to", full_path)

    def load(self, filename="model.pth"):
        full_path = os.path.join(self.model_dir, filename)
        self.model.load_state_dict(torch.load(full_path))
        self.model.eval()  # Set the model to evaluation mode
        print("Model loaded from", full_path)