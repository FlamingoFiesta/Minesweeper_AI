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
        self.fc1 = nn.Linear(input_dim, 128) # Q Change because of mismatch of shape
        self.fc2 = nn.Linear(128, 256) 
        self.fc3 = nn.Linear(256, 128)
        self.output = nn.Linear(128, output_dim) # Q Change because of mismatch of shape

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.output(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(current_dir, 'Agent')
        os.makedirs(self.model_dir, exist_ok=True)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95  # Discount factor for past rewards
        self.epsilon = 1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.0025
        self.learning_rate_min = 0.00025
        self.learning_rate_decay = 0.9995
        self.batch_size = 64
        self.model = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size) # added target network 
        self.target_network.load_state_dict(self.model.state_dict())  # added target network
        self.update_target_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Initialize the update counter and target update frequency
        self.update_counter = 0
        self.target_update_frequency = 1000  # Update the target network every 1000 updates
        self.tau = 0.001  # Soft update parameter

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            return torch.argmax(q_values).item()
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 
        
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)  # Adding batch dimension
            next_state = torch.FloatTensor(next_state).unsqueeze(0)  # Adding batch dimension
            reward = torch.FloatTensor([reward])  # Convert reward to a tenso

            q_values = self.model(state)
            target_q_values = q_values.clone()

            if done:
                target_q_values[0, 0, action] = reward.item()
            else:
                q_next = self.target_network(next_state)

                #print(f"target_q_values shape: {target_q_values.shape}")
                #print(f"action: {action}")
                #print(f"target_q_values: {target_q_values}")
                #print(f"q_next: {q_next}")
                #print(f"reward: {reward}")
                ## PROBLEM: DIMENSIONS ARE OFF I NEED TO MAKE SURE THAT THE AGENT KNOWS DIFF BETWEEN CHOOSING A TILE W UNCOVERED TILES OR A GUESS
                target_q_values[0, 0, action] = reward.item() + self.gamma * torch.max(q_next).item()

            




            self.optimizer.zero_grad()
            loss = nn.functional.mse_loss(q_values, target_q_values)
            loss.backward()
            self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * np.exp(-self.epsilon_decay))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(self.learning_rate_min, param_group['lr'] * self.learning_rate_decay)

        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.soft_update(self.tau)

    def update_target_network(self):
        self.target_network.load_state_dict(self.model.state_dict())

    def soft_update(self, tau):
        for target_param, local_param in zip(self.target_network.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename="model.pth"):
        full_path = os.path.join(self.model_dir, filename)
        torch.save(self.model.state_dict(), full_path)
        print("Model saved to", full_path)

    def load(self, filename="model.pth"):
        full_path = os.path.join(self.model_dir, filename)
        self.model.load_state_dict(torch.load(full_path))
        self.model.eval()  # Set the model to evaluation mode
        print("Model loaded from", full_path)



def replayold_Q(self): # A LOT WRONG WITH THIS LOOKING TO IMPROVE !!!
        if len(self.memory) < self.batch_size:
            return 
        
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)  # Adding batch dimension
            next_state = torch.FloatTensor(next_state).unsqueeze(0)  # Adding batch dimension
            reward = torch.FloatTensor([reward])  # Convert reward to a tenso

            q_values = self.model(state)

            if action >= self.action_size:
               print(f"Action index {action} is out of bounds for action space of size {self.action_size}")
               continue


            if done:
                target_q_values[0][action] = reward.item()
            else:
                q_next = self.target_network(next_state)
                target_q_values[0][action] = reward.item() + self.gamma * torch.max(q_next).item()

            if done: 
                target = reward
            else:
                target = (reward + self.gamma * torch.max(self.target_network(next_state).detach()))

            target_q_values = q_values.clone().detach()
            target_q_values[0][action] = target.item()

            print(f"Target Q-values shape: {target_q_values.shape}, Target Q-values: {target_q_values}")
            print(f"State shape: {state.shape}, Next State shape: {next_state.shape}")
            print(f"Model output (q_values) shape: {q_values.shape}")
            print(f"Action: {action}, Reward: {reward}")


            loss = nn.functional.mse_loss(q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.soft_update(self.tau)

        self.epsilon = max(self.epsilon_min, self.epsilon * np.exp(-self.epsilon_decay))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(self.learning_rate_min, param_group['lr'] * self.learning_rate_decay)

def update_target_network(self):
    self.target_network.load_state_dict(self.model.state_dict())

def soft_update(self, tau):
    for target_param, local_param in zip(self.target_network.parameters(), self.model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def load_old(self, name):
        self.model.load_state_dict(torch.load(name))
        self.model.eval()  # Set the model to evaluation mode
        print("Model loaded from", name)

def save_old(self, name):
    torch.save(self.model.state_dict(), name)
    print("Model saved to", name)





























    def replayold(self):
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

        if self.learning_rate > self.learning_rate_min:
            self.learning_rate *= self.learning_rate_decay

    def replaynewold(self):
        if len(self.memory) < self.batch_size:
            return 
        
        for i in range(5):
            minibatch = random.sample(self.memory, self.batch_size)
            for state, action, reward, next_state, done in minibatch:
                state = torch.FloatTensor(state)
                next_state = torch.FloatTensor(next_state)
                target = reward
                if not done:
                    target = (reward + self.gamma * torch.max(self.target_network(next_state).detach()))
                current_q = self.model(state)[0][action]

                print(f"state: {state}")
                print(f"next_state: {next_state}")
                print(f"reward: {reward}")
                print(f"target: {target}")
                print(f"target_f before: {self.target_network}")

                loss = nn.functional.mse_loss(current_q, target)



                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.update_counter += 1
            if self.update_counter % self.target_update_frequency == 0:
                self.soft_update(self.tau)

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        if self.learning_rate > self.learning_rate_min:
            self.learning_rate = max(self.learning_rate * self.learning_rate_decay, self.learning_rate_min)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

    def soft_update(self, tau):
        for target_param, local_param in zip(self.target_network.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    
    def trainold(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.replay()

                # Update the target network periodically
                self.update_counter += 1
                if self.update_counter % self.target_update_frequency == 0:
                    self.update_target_network()

            print(f"Episode: {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")

        print("Training finished.")

    def trainNEWOLD(self, env, num_episodes):
        for episode in range(num_episodes):
            print(f"Starting episode {episode+1}/{num_episodes}")
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0  # Track the number of steps per episode

            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward 
                self.replay()
                steps += 1

                # Update the target network periodically
                self.update_counter += 1
                if self.update_counter % self.target_update_frequency == 0:
                    self.update_target_network()

            print(f"Episode {episode+1}/{num_episodes} finished. Total Reward: {total_reward}, Steps: {steps}, Epsilon: {self.epsilon:.4f}")

        print("Training finished.")


    