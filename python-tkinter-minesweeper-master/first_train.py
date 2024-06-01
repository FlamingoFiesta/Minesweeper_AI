import numpy as np
from tkinter import * 
from tkinter import messagebox as tkMessageBox
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import platform


from minesweeper import Minesweeper
from ReinforcementLearning.minesweeper_env import MinesweeperEnv
from ReinforcementLearning.dqn_agent import DQNAgent

import time

SIZE_X = 5
SIZE_Y = 5
NUM_MINES = 4

STATE_DEFAULT = 0
STATE_CLICKED = 1
STATE_FLAGGED = 2

BTN_CLICK = "<Button-1>"
BTN_FLAG = "<Button-2>" if platform.system() == 'Darwin' else "<Button-3>"

class Trains:
    def __init__(self, tk):
        self.tk = tk
        self.rows = SIZE_X
        self.cols = SIZE_Y
        self.mines = NUM_MINES
        self.game_over = False
        self.done = False

        self.total_episodes = 0
        self.wins = 0
        self.win_loss_rate = 0
        self.win_loss_rate_5000_games = []
        self.total_rewards = []
        self.steps_per_game = []
        self.current_step = 0
        self.cumulative_reward = 0
            

        action_size = SIZE_X * SIZE_Y
        self.env = MinesweeperEnv(SIZE_X, SIZE_Y, NUM_MINES)
        self.agent = DQNAgent(SIZE_X * SIZE_Y, action_size)
        self.Minesweeper = Minesweeper(tk)

        self.frame = Frame(self.tk)
        self.frame.pack()

        self.images = {
            "plain": PhotoImage(file = "images/tile_plain.gif"),
            "clicked": PhotoImage(file = "images/tile_clicked.gif"),
            "mine": PhotoImage(file = "images/tile_mine.gif"),
            "exploded_mine": PhotoImage(file = "images/tile_mine_exploded.gif"),
            "flag": PhotoImage(file = "images/tile_flag.gif"),
            "wrong": PhotoImage(file = "images/tile_wrong.gif"),
            "numbers": [PhotoImage(file=f"images/tile_{i}.gif") for i in range(1, 9)]
        }
        for i in range(1, 9):
            self.images["numbers"].append(PhotoImage(file = "images/tile_"+str(i)+".gif"))

        self.tiles = self.Minesweeper.tiles
        self.Minesweeper.setup()
        self.tiles = self.Minesweeper.tiles  # Link again after setup to ensure consistency


        #self.reset_train()


#     def trainnew(self):
#         print("Starting training!!!!!!!!!!!!!!!!!!")
#         IMAX = 100
#         total_rewards = []
#         steps_per_game = []
# 
#         while self.total_episodes < IMAX:
#             print("Episode:", self.total_episodes)
#             game_over = False
#             cumulative_reward = 0
#             current_step = 0
# 
#             self.reset_train()
# 
#             # Run the episode
#             self.run_episode(total_rewards, steps_per_game, cumulative_reward, current_step, game_over)
# 
#     def run_episode(self, total_rewards, steps_per_game, cumulative_reward, current_step, game_over):
#         if game_over:
#             print("Game Over!")
#             self.total_episodes += 1
# 
#             if self.total_episodes % 100 == 0:
#                 print(f'RIGHT HERE IM HERE IM HERE IM HERE {self.total_episodes}!!!!!!!!!!!!!!!!!!!!!!!')
# 
#             if game_over:
#                 self.minesweeper.reveal_all_mines()
#                 self.minesweeper.show_exploded_mine(action)  # Highlight the exploded mine
#                 result = 0  # a loss
#             else:
#                 self.wins += 1
#                 result = 1  # a win
# 
#             total_rewards.append(cumulative_reward)
#             steps_per_game.append(current_step)
#             self.minesweeper.update_stats()
# 
#             if self.total_episodes % 1000 == 0:  # Example: Save every 1000 episodes
#                 self.agent.save()
#                 print(f"Model saved after {self.total_episodes} episodes.")
# 
#             # Schedule the next episode
#             self.tk.after(1000, lambda: self.train())
#         else:
#             current_state = np.array([self.env.get_obs()]).flatten()
#             action = self.agent.act(current_state)
#             next_state, reward, done, game_over = self.env.step(action)
#             print('Action taken:', action, 'Reward:', reward, 'Done:', done)
# 
#             current_step += 1
#             cumulative_reward += reward
# 
#             self.minesweeper.update_gui_based_on_state()
# 
#             if not done:
#                 # Schedule the next step
#                 self.tk.after(2000, lambda: self.run_episode(total_rewards, steps_per_game, cumulative_reward, current_step, game_over))
#             else:
#                 # Process the end of the game
#                
#                 self.run_episode(total_rewards, steps_per_game, cumulative_reward, current_step, game_over) 
    
    
    def train(self):
        print("Starting training!!!!!!!!!!!!!!!!!!")
        IMAX = 100
        total_rewards = []
        steps_per_game = []

        total_episodes = 0      
        
        while total_episodes < IMAX:
            self.game_over = False
            

            while not self.game_over:
                current_state = np.array([self.env.get_obs()]).flatten()
                action = self.agent.act(current_state)
                next_state, reward, self.done, self.game_over = self.env.step(action)

                self.current_step += 1
                self.cumulative_reward += reward

                self.Minesweeper.update_gui_based_on_state()  # Refresh the entire board

                if not self.done:
                    # If the game is not done, schedule the next action
                    self.tk.after(1000) # SPEEED

                else:
                    # If the game is done, process the end of the game
                    print("Game Over!")
                    total_episodes += 1

                    if self.game_over:
                        self.Minesweeper.reveal_all_mines()
                        self.Minesweeper.show_exploded_mine(action)  # Highlight the exploded mine
                    # Ask if the user wants to restart

                        result = 0 # a loss

                    else:
                        wins += 1
                        result = 1 # a win

                    #Minesweeper.update_win_loss(result)
                    total_rewards.append(self.cumulative_reward)
                    steps_per_game.append(self.current_step)
                    #Minesweeper.log_performance()
                    self.Minesweeper.update_stats()


                    print(self.game_over)
                    self.reset_train()
                    print(self.game_over)

                    if total_episodes % 100 == 0:
                        self.agent.save()
                        print(f"Model saved after {self.total_episodes} episodes.")
            else:        
                self.reset_train() 
            
            if total_episodes % 1000 == 0:  # Example: Save every 1000 episodes
                self.agent.save()
                print(f"Model saved after {total_episodes} episodes.")

    def reset_trainold(self):
        self.env.reset()  # Reset the environment state
        for r in range(self.rows):
            for c in range(self.cols):
                tile = self.tiles[f"{r}_{c}"]
                tile["button"].config(image=self.images["plain"])  # Reset tiles to initial state
                #self.startTime = datetime.now()  # Q's add
        self.game_over = False

    def reset_trainoldold(self):
        # Assuming the reset_train method should initialize the tiles
        self.env.reset()
        for r in range(self.rows):
            for c in range(self.cols):
                key = f"{r}_{c}"
                if key  not in self.tiles:
                    print(f"Key {key} not found in tiles!")
                else:
                    tile = self.tiles[key]
                tile = self.tiles[f"{r}_{c}"]
                tile["button"].config(image=self.images["plain"])   # or some appropriate initial state

        self.game_over = False
        self.done = False
        self.current_step = 0
        self.cumulative_reward = 0

    def reset_train(self):
        for r in range(self.rows):
            for c in range(self.cols):
                key = f"{r}_{c}"
                if key in self.tiles:
                    tile = self.tiles[key]
                else:
                    raise KeyError(f"Key {key} not found in tiles!")  # Raise error to stop execution and debug
                
        self.game_over = False
        self.done = False
        self.current_step = 0
        self.cumulative_reward = 0

def main_train():
    # create Tk instance
    window = Tk()
    # set program title
    window.title("! Training ! Minesweeper ! Training !")
    # create game instance
    training = Trains(window)

    graph_window = Toplevel()
    graph_window.title("Win-Loss Graph")

    try:
        training.agent.load()  # Path is managed within the agent
        print("Model loaded successfully.")
        print('So this still works')
    except FileNotFoundError:
        print("No saved model found, starting with a new model.")

    fig = plt.figure()
    
    # Embed the figure in the Tkinter window using FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.get_tk_widget().pack(fill='both', expand=True)
    canvas.draw()

    

    # Create the FuncAnimation object that will update the plot
    ani = FuncAnimation(fig, training.Minesweeper.update_plot, interval=1000, blit=False, save_count=50)

    is_animation_running = True

    def improvement_graph():
        for win_loss in training.Minesweeper.win_loss_rate_5000_games:
            print(f"episode {win_loss * 500} had a win rate of {training.Minesweeper.win_loss_rate_5000_games[win_loss]}")       
        plt.plot(training.Minesweeper.win_loss_rate_5000_games)
        plt.show()

    def on_close():
        if not is_animation_running:
            ani.event_source.stop()
            plt.close(fig)
        if tkMessageBox.askyesno("Quit", "Do you want to save the model before quitting?"):
            training.agent.save()
        if graph_window.winfo_exists():
            graph_window.destroy()
        window.destroy()
        improvement_graph()

    def on_close_graph():
        ani.event_source.stop()
        plt.close(fig)
        graph_window.destroy()
        is_animation_running = False

    window.protocol("WM_DELETE_WINDOW", on_close)
    graph_window.protocol("WM_DELETE_WINDOW", on_close_graph)

    window.after(1000, training.train)

    window.mainloop()

if __name__ == "__main__":
    main_train()