import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ReinforcementLearning.minesweeper_env import MinesweeperEnv
from ReinforcementLearning.dqn_agent import DQNAgent

import numpy as np
import matplotlib.pyplot as plt
import torch 

SIZE_X = 5
SIZE_Y = 5
NUM_MINES = 4

state_size = SIZE_X * SIZE_Y
action_size = SIZE_X * SIZE_Y

def train_dqn_agent(episodes, max_steps):
    env = MinesweeperEnv(SIZE_X, SIZE_Y, NUM_MINES)
    state_size = SIZE_X * SIZE_Y
    action_size = SIZE_X * SIZE_Y

    agent = DQNAgent(state_size, action_size)

    total_rewards = []
    win_rates = []

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                print(f'Episode: {episode}, Reward: {total_reward}, Done: {done}')
                break

        if len(agent.memory) > agent.batch_size:
            agent.replay()

        total_rewards.append(total_reward)

        if (episode + 1) % 500 == 0: #50 last people
            recent_rewards = total_rewards[-500:]
            
            win_rate = sum(recent_rewards) / 500
            win_rates.append(win_rate)

    
    return total_rewards, win_rates

if __name__ == "__main__":
        
    for i in range(30):
        
        agent = DQNAgent(state_size, action_size)
        
        try:
            agent.load()
            print("Model loaded succesfully.")
        except FileNotFoundError:
            print("No saved model found, starting with a new model")
        
            
        episodes = 4000
        total_rewards, total_wins = train_dqn_agent(episodes, max_steps=10)

        agent.save()

    plt.plot(total_rewards)
    plt.xlabel('episodes')
    plt.ylabel('total reward')
    plt.title('Total Awards over Episodes')

    plt.plot(total_wins)
    plt.xlabel('episodes')
    plt.ylabel('total wins')
    plt.title('Total Wins over Episodes')

