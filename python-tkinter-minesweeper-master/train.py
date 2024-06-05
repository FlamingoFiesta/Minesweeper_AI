import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ReinforcementLearning.minesweeper_env import MinesweeperEnv
from ReinforcementLearning.ddqn_agent import DDQNAgent
from ReinforcementLearning.ddqn_agent import QNetwork

import random
import numpy as np
import matplotlib.pyplot as plt
import torch 
from itertools import count


# Grid info.
SIZE_X = 5
SIZE_Y = 5
NUM_MINES = 4

# State Size
state_size = 5 * 5 * 10
# Action Size
action_size = 5 * 5

# Establishing the fiels:
minefield = [] 
playerfield = []

# Keeping track of stats
total_rewards = []
wins = 0
wins_after_500_episodes = 0
num_steps = 0
num_steps_list = []
win_rates = []
epsilon_tracker = []
lr_tracker = []

def main():

    state_size = 5 * 5 * 10
    action_size = 5 * 5

    env = MinesweeperEnv(SIZE_X, SIZE_Y, NUM_MINES)
    agent = DDQNAgent(state_size, action_size, seed=0)
    qnetwork = QNetwork(state_size, action_size, seed=0)
    
    num_steps = 0
    max_episodes = 5000
    for episode in range(max_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(100):
            print(state)
            action = agent.act(state)
            next_state, reward, terminated = env.step(action)
            
            agent.update_memory(action, reward, state, next_state)

            state = next_state
            total_reward += reward


        
        
    def calculate_win_rate(wins): # Calculate the win rate of the last 500 games. 
            return wins / 500 * 100 
    
    return training_total_rewards, training_win_rates, training_total_wins, training_epsilon_tracker, training_lr_tracker, num_steps_list


if __name__ == "__main__":
        
    training_total_rewards, training_win_rates, training_total_wins, training_epsilon_tracker, training_lr_tracker, num_steps_list = main()

    # Create a 1x3 grid of subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 2.5))

    # Plot for Total Rewards over Episodes
    axs[0].plot(range(1, len(training_total_rewards) + 1), training_total_rewards)
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Total Reward')
    axs[0].set_title('Total Rewards per Episode')

    # Plot for cummulative reward
    axs[1].plot(range(1, len(cummulative_reward_list) + 1), cummulative_reward_list)
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Total Reward')
    axs[1].set_title('Cummulative Reward (Reward Summed Up)')

    fig, axs = plt.subplots(1, 3, figsize=(10, 2.5))

    # Plot for Total Wins
    axs[0].plot(range(1, len(training_total_wins) + 1), training_total_wins)
    axs[0].set_xlabel('Training Number')
    axs[0].set_ylabel('Total Wins')
    axs[0].set_title('Total Wins per Training')

    # Plot for cummulative wins
    axs[1].plot(range(1, len(cummulative_wins_list) + 1), cummulative_wins_list)
    axs[1].set_xlabel('Training Number')
    axs[1].set_ylabel('Total Wins')
    axs[1].set_title('Cummulative Wins (Wins Summed Up)')

    # Plot for Win Rates
    axs[2].plot(range(1, len(training_win_rates) + 1), training_win_rates)
    axs[2].set_xlabel('Episodes (x 500)')
    axs[2].set_ylabel('Win Percentage after Every 500 Games')
    axs[2].set_title('Win Rates over Episodes')

    fig, axs1 = plt.subplots(1, 2, figsize=(15, 5))

    # Plot for Total Wins
    axs1[0].plot(range(1, len(training_epsilon_tracker) + 1), training_epsilon_tracker)
    axs1[0].set_xlabel('Time')
    axs1[0].set_ylabel('Epsilon')
    axs1[0].set_title('Epsilon Decay EXP')

    # Plot for cummulative wins
    axs1[1].plot(range(1, len(training_lr_tracker) + 1), training_lr_tracker, color='tab:red')
    axs1[1].set_xlabel('Time')
    axs1[1].set_ylabel('Learning Rate')
    axs1[1].set_title('Learning Rate over Time')

    axs2 = axs1[1].twinx()

    axs2.plot(range(1, len(num_steps_list) + 1), num_steps_list)
    axs2.set_xlabel('Time')
    axs2.set_ylabel('Number of Total Steps')

    plt.tight_layout()
    #plt.show()

