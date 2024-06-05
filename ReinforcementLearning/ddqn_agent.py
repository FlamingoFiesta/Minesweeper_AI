# ddqn_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from collections import namedtuple 
import pickle
import copy

import os

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed=0):
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed) 
        self.fc1 = nn.Linear(state_size, 64) # First hidden neuron layer.
        self.fc2 = nn.Linear(64, 32) # Second hidden neuron layer.
        self.fc3 = nn.Linear(32, action_size) # Output layer

    # How input states go through the network layers. It maps the state to action values
    def forward(self, state): 
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x # Ouput data
    
class DDQNAgent:

    def __init__(self, state_size, action_size, seed=0):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(current_dir, 'Agent')
        os.makedirs(self.model_dir, exist_ok=True)

        # Dimension Parameters
        self.state_size = state_size 
        self.action_size = action_size 
        self.seed = random.seed(seed)

        # Hyperparameters
        self.batch_size = 1024 
        self.gamma = 0.99  # Discount factor for past rewards
        self.epsilon = 0.99  # Exploration rate NEED TO CHANGE
        self.epsilon_decay = 0.99975 # Determines how much epsilon decays each time
        self.epsilon_min = 0.01 # Minimal amount of epsilon. By making it not 0, there is always the chance that the agent will explore.
        
        # PIECEWISE LEARNING RATE DECAY
        self.lr_piecewise = [0.001,0.0005,0.00025,0.00025/2,0.00025/4, 0.00025/10]
        self.step_milestones = [1000, 2000, 3500, 4250]
        self.decay_steps = [0.01, 0.005, 0.0025, 0.0001] 
        self.lrate = self.lr_piecewise[0]
        self.lrate_decay = []
        self.current_lr = 0

        # Implementation of Double DQN
        self.online_network = QNetwork(state_size, action_size, seed=0) # QNetwork which is actively updated during training, as well as selecting actions on the current values
        self.target_network = QNetwork(state_size, action_size, seed=0) # QNetwork which is only updated periodically, it computes the target Q-values. This stabilises the training process by updating more consistently
        
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.lrate, amsgrad=True)
        self.tau = 0.001  # Soft update parameter
        self.step_num = 0 # Initialize the update counter and target update frequency

    def update_learning_rate(self, num_steps):
        for i, milestone in enumerate(self.step_milestones):
            if num_steps == milestone:
                # Update to the new learning rate
                self.current_lr = self.learning_rates[i]
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr

        # Function to update the target network. However, the updates are small but frequent 
    def update_soft(self, tau): 
        for local_param, target_param in zip(self.target_model.parameters(), self.target_network_weights.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
            self.step_num += 1  
            
        # Function to establish an action, the action is used as an index on the 1D vectors of the fields. 
    def act(self, state):
        flattened_state = state.flatten()  # Obtain the vectors of the fields using flatten()
        valid_actions = np.where(flattened_state == 9)[0]  # The agent can only act on covered tiles

        # Generate a random number to decide between exploration and exploitation

        if random.random() < self.epsilon:
            action = np.random.choice(valid_actions)  # Then randomly an action is chosen
            
            return action
        
        else:
            state_tensor = torch.from_numpy(state).float()
            q_values = state_tensor.detach().numpy().flatten()
            valid_q_values = q_values[valid_actions]

            # Select the action with the highest Q-value among the valid actions
            action = valid_actions[np.argmax(valid_q_values)]

        return action  
        
    def optimise_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = ReplayMemory.sample(self.batch_size)
        batch = ReplayMemory.Transistion(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
       
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Calucates the Q-values of for current state-action pairs. 
        # Using the batch to go through the online network. The gathering() func is used to select the Q-values taken from action_batch.
        state_action_values = self.online_network(state_batch).gather(1, action_batch)

        # Predicties the Q-values.
        next_state_values = torch.zeros(self.batch_size) # First a tensor of 0 has to be created to store the new values.
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values

        # Calculates the target Q-value for current state-action pairs. 
        # The discounted next_value are added to the immediate rewards (reward_batches)
        # Target Q-values used to compute the loss at the 
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsquees(1))

        # Optimise the model
        self.optimizer.zero_grad()
        loss.backward()

        # The parameters are the weights and biases of the network layers
        # Gradients are clipped to prevent an "exploding gradient" !!!
        torch.nn.utils.clip_grad_value_(self.online_network.parameters(), 100)
        self.optimizer.step()

    def update_memory(self, action, reward, state, next_state):
        self.memory.append((action, reward, state, next_state))

    def save(self, filename="model.pth"):
        full_path = os.path.join(self.model_dir, filename)
        torch.save(self.model.state_dict(), full_path)
        print("Model saved to", full_path)
    
    # Save the memory buffer
    def save_memory(self, filename="memory.pkl"):
        full_path = os.path.join(self.model_dir, filename)
        with open(full_path, 'wb') as memory_file:
            pickle.dump(self.memory, memory_file)
        print("Memory saved to", full_path)

    def load(self, filename="model.pth"):
        full_path = os.path.join(self.model_dir, filename)
        self.model.load_state_dict(torch.load(full_path))
        self.model.eval()  # Set the model to evaluation mode
        print("Model loaded from", full_path)

    def load_memory(self, filename="memory.pkl"):
        full_path = os.path.join(self.model_dir, filename)
        with open(full_path, 'rb') as memory_file:
               self.memory = pickle.load(memory_file)
        print("Memory loaded from", full_path)


Transistion = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    #def push(self, *args):
    #    """Save a transition"""
    #    self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
