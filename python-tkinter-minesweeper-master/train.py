import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ReinforcementLearning.minesweeper_env import MinesweeperEnv
from ReinforcementLearning.dqn_agent import DQNAgent

import numpy as np
import matplotlib.pyplot as plt
import torch 
import asyncio

SIZE_X = 5
SIZE_Y = 5
NUM_MINES = 4

state_size = SIZE_X * SIZE_Y
action_size = SIZE_X * SIZE_Y

async def train_dqn_agent(agent, episodes, max_steps):
    env = MinesweeperEnv(SIZE_X, SIZE_Y, NUM_MINES)
    state_size = SIZE_X * SIZE_Y
    action_size = SIZE_X * SIZE_Y

    total_rewards = []
    wins = 0
    wins_after_500_episodes = 0
    win_rates = []

    def calculate_win_rate(wins): # Calculate the win rate of the last 
        return wins / 500 * 100

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, game_over = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                if not game_over:
                    wins += 1
                    wins_after_500_episodes += 1
                    print('! ! ! Agent has won ! ! !')

                print(f'Episode: {episode + 1}, Reward: {total_reward}, Done: {done}')
                break

        total_rewards.append(total_reward)

        if (episode + 1) % 500 == 0: 
            win_rate = calculate_win_rate(wins_after_500_episodes)
            win_rates.append(win_rate)
            wins_after_500_episodes = 0

        if len(agent.memory) > agent.batch_size:
            agent.replay()
            
        await asyncio.sleep(0)

    
    return total_rewards, win_rates, wins

if __name__ == "__main__":
        
    async def main():
        training_total_rewards = []
        training_win_rates = []
        training_total_wins = []

        training = 1

        for i in range(3):
            print('>>> TRAINING: ', training)
            agent = DQNAgent(state_size, action_size)
            
            try:
                agent.load()
                print("Model loaded succesfully.")
            except FileNotFoundError:
                print("No saved model found, starting with a new model")
              
            episodes = 1000
            total_rewards, win_rates, wins = await train_dqn_agent(agent, episodes, max_steps=10)
            
            training_total_rewards.extend(total_rewards)
            training_win_rates.extend(win_rates)
            training_total_wins.append(wins)

            agent.save()

            training += 1

        print(training_total_wins)
        return training_total_rewards, training_win_rates, training_total_wins

    training_total_rewards, training_win_rates, training_total_wins = asyncio.run(main())
    cummulative_reward_list = []
    cummulative_wins_list = []

    cummulative_reward = 0
    cummulative_wins = 0

    for reward in training_total_rewards:
        cummulative_reward += reward
        cummulative_reward_list.append(cummulative_reward)

    for win in training_total_wins:
        cummulative_wins += win
        cummulative_wins_list.append(cummulative_wins)

    # Create a 1x3 grid of subplots
    fig, axs = plt.subplots(1, 5, figsize=(10, 2.5))

    # Plot for Total Rewards over Episodes
    axs[0].plot(range(1, len(training_total_rewards) + 1), training_total_rewards)
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Total Reward')
    axs[0].set_title('Total Rewards of Each Episode')

    # Plot for cummulative reward
    axs[1].plot(range(1, len(cummulative_reward_list) + 1), cummulative_reward_list)
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Total Reward')
    axs[1].set_title('Cummulative Reward (Reward Summed Up)')

    # Plot for Total Wins
    axs[2].plot(range(1, len(training_total_wins) + 1), training_total_wins)
    axs[2].set_xlabel('Training Number')
    axs[2].set_ylabel('Total Wins')
    axs[2].set_title('Total Wins per Training')

    # Plot for cummulative wins
    axs[3].plot(range(1, len(cummulative_wins_list) + 1), cummulative_wins_list)
    axs[3].set_xlabel('Training Number')
    axs[3].set_ylabel('Total Wins')
    axs[3].set_title('Cummulative Wins (Wins Summed Up)')

    # Plot for Win Rates
    axs[4].plot(range(1, len(training_win_rates) + 1), training_win_rates)
    axs[4].set_xlabel('Episodes (x 500)')
    axs[4].set_ylabel('Win Percentage after Every 500 Games')
    axs[4].set_title('Win Rates over Episodes')

    # Adjust layout
    plt.tight_layout()

    # Display the plots
    plt.show()
