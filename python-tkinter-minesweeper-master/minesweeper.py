# Python Version 2.7.3
# File: minesweeper.py

from tkinter import *
from tkinter import messagebox as tkMessageBox
from collections import deque
import random
import platform
import time
from datetime import time, date, datetime
from collections import deque # to calculate the win rate 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import random
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ReinforcementLearning.minesweeper_env import MinesweeperEnv
from ReinforcementLearning.dqn_agent import DQNAgent

SIZE_X = 3
SIZE_Y = 3
NUM_MINES = 1

STATE_DEFAULT = 0
STATE_CLICKED = 1
STATE_FLAGGED = 2

BTN_CLICK = "<Button-1>"
BTN_FLAG = "<Button-2>" if platform.system() == 'Darwin' else "<Button-3>"

window = None

class Minesweeper:

    def __init__(self, tk):

        self.tk = tk
        self.rows = SIZE_X
        self.cols = SIZE_Y
        self.mines = NUM_MINES
        self.game_over = False

        self.total_episodes = 0
        self.wins = 0
        self.total_rewards = []
        self.steps_per_game = []
        self.current_step = 0
        self.revealed_cells = 0
        #action_size = SIZE_X * SIZE_Y
        #self.env = MinesweeperEnv(SIZE_X, SIZE_Y, NUM_MINES)
        #self.agent = DQNAgent(SIZE_X * SIZE_Y, SIZE_X * SIZE_Y)
        self.after_ids = []

        action_size = 2 * SIZE_X * SIZE_Y  # Each cell can either be revealed or flagged

        self.env = MinesweeperEnv(SIZE_X, SIZE_Y, NUM_MINES)
        self.agent = DQNAgent(SIZE_X * SIZE_Y, action_size)

        # import images
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

        # set up frame
        self.frame = Frame(self.tk)
        self.frame.pack()

        # Win Loss Rate
        self.win_loss = deque(maxlen=50)

        # set up labels/UI
        self.labels = {
            "time": Label(self.frame, text = "00:00:00"),
            "mines": Label(self.frame, text = "Mines: 0"),
            "flags": Label(self.frame, text = "Flags: 0")
        }
        #self.labels["time"].grid(row = 0, column = 0, columnspan = SIZE_Y) # top full width
        #self.labels["mines"].grid(row = SIZE_X+1, column = 0, columnspan = int(SIZE_Y/2)) # bottom left
        #self.labels["flags"].grid(row = SIZE_X+1, column = int(SIZE_Y/2)-1, columnspan = int(SIZE_Y/2)) # bottom right

        self.stats_labels = {
            "stats": Label(self.frame, text="Statistics: "),
            "total_episodes": Label(self.frame, text="Total episodes: 0"),
            "total_wins": Label(self.frame, text="Total wins: 0"),
            "total_reward": Label(self.frame, text="Total reward: 0"),
            "win_rate": Label(self.frame, text="Win-loss rate (of last 50 games): 0%"),
            "percentage_cleared": Label(self.frame, text="Percentage cleared: 0%")
        }

        stats_column = self.cols + 1 
        self.stats_labels["stats"].grid(row=0, column=stats_column, sticky="nw")
        self.stats_labels["total_episodes"].grid(row=1, column=stats_column, sticky="nw")
        self.stats_labels["total_wins"].grid(row=2, column=stats_column, sticky="nw")
        self.stats_labels["total_reward"].grid(row=3, column=stats_column, sticky="nw") 
        self.stats_labels["win_rate"].grid(row=4, column=stats_column, sticky="nw")
        self.stats_labels["percentage_cleared"].grid(row=5, column=stats_column, sticky="nw") 

        self.labels["time"].grid(row=0, column=0, columnspan=self.cols) # top full width
        self.labels["mines"].grid(row=self.rows+1, column=0, columnspan=int(self.cols/2)) # bottom left
        self.labels["flags"].grid(row=self.rows+1, column=int(self.cols/2), columnspan=int(self.cols/2)) # bottom right

        self.startTime = None
        self.tiles = {}
        self.setup()
        #self.restart() # start game
        self.reset_game()
        self.updateTimer() # init timer

        
    def update_plot(self, frame):
        ax = plt.gca()
        ax.cla()

        win_loss_list = list(self.win_loss)
        win_loss_over_time = [float(sum(win_loss_list[:i+1])) / (i+1) * 100 for i in range(len(win_loss_list))]

        ax.plot(range(len(win_loss_over_time)), win_loss_over_time)
        ax.set_xlabel("Time")
        ax.set_ylabel("Win Rate (%)")
        ax.set_ylim(0, 100)

        if self.total_episodes >= 50:
            ax.set_xticks([0, len(win_loss_list) - 1])
            ax.set_xticklabels([f"{self.total_episodes - len(win_loss_list) + 1}", f"{self.total_episodes}"])

        id = self.tk.after(100, self.run_agent)
        self.after_ids.append(id)


    def run_agent_idk(self):
        if not self.game_over:
            current_state = np.array([self.env.get_obs()]).flatten()
            action = self.agent.act(current_state)
            next_state, reward, done, self.game_over = self.env.step(action)

            self.update_gui_based_on_state()  # Refresh the entire board
            #"""
            if not done:
                self.tk.after(1000, self.run_agent)  # schedule the next call, 
            else:
                print("Game Over!")  # Optionally handle game over
                if self.game_over:  # Only reveal all mines if the game is over due to a mine hit
                    self.reveal_all_mines()
                    self.show_exploded_mine(action)  # Highlight the exploded mine
            #"""
            """
            if done:
                print("Game Over!")
                if self.game_over:  # Specific handling if the game is over because a mine was hit
                    self.reveal_all_mines()
                    self.show_exploded_mine(action)
                    self.reset_game()
                self.tk.after(1000, self.run_agent)  # Reschedule only if the game is not over
        else:
                self.reset_game()  # Reset the game if it ended
            """

    def run_agent(self):
        if not self.game_over:
            current_state = np.array([self.env.get_obs()]).flatten()
            action = self.agent.act(current_state)
            next_state, reward, done, self.game_over = self.env.step(action)

            self.current_step += 1
            self.cumulative_reward += reward

            self.update_gui_based_on_state()  # Refresh the entire board
            if not done:
                # If the game is not done, schedule the next action
                self.tk.after(50, self.run_agent) #SPEED
            else:
                # If the game is done, process the end of the game
                print("Game Over!")
                self.total_episodes += 1

                if self.game_over:

                    self.reveal_all_mines()
                    self.show_exploded_mine(action)  # Highlight the exploded mine
                  # Ask if the user wants to restart

                    result = 0 # a loss

                else:
                    self.wins += 1

                    result = 1 # a win

                self.update_win_loss(result)
                
                self.total_rewards.append(self.cumulative_reward)
                self.steps_per_game.append(self.current_step)
                self.log_performance()
                self.update_stats()

                self.reset_game()

                if self.total_episodes % 10 == 0:  # Example: Save every 1 episodes
                    self.agent.save()
                    print(f"Model saved after {self.total_episodes} episodes.")


        else:
            # If the game is already marked as over, skip to asking for restart
            self.reset_game()
    
    def log_performance(self):
        # Print out the performance metrics
        print(f"Total Episodes: {self.total_episodes}, Wins: {self.wins}, Total Reward: {self.cumulative_reward}, Steps/Game: {np.mean(self.steps_per_game)}")    
    
    def reset_game_old(self):
        #self.state = self.env.reset()
        #self.setup()
        self.env.reset()
        for r in range(self.rows):
            for c in range(self.cols):
                tile = self.tiles[f"{r}_{c}"]["button"].config(image=self.images["plain"])
        self.run_agent()  # Restart the agent for a new game

    def reset_game(self):
        self.env.reset()  # Reset the environment state
        for r in range(self.rows):
            for c in range(self.cols):
                tile = self.tiles[f"{r}_{c}"]
                tile["button"].config(image=self.images["plain"])  # Reset tiles to initial state
        self.startTime = datetime.now() # Q's add
        self.game_over = False
        self.current_step = 0
        self.cumulative_reward = 0
        self.run_agent()  # Restart the agent for a new game

    def setup(self):
        self.tiles = {}
        self.flagCount = 0
        self.correctFlagCount = 0
        self.clickedCount = 0
        self.mines = 0
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                id = f"{x}_{y}"
                isMine = random.random() < 0.1
                self.mines += isMine
                tile = {
                    "id": id,
                    "isMine": isMine,
                    "state": STATE_DEFAULT,
                    "coords": {"x": x, "y": y},
                    "button": Button(self.frame, image=self.images["plain"]),
                    "mines": 0  # Updated later
                }
                tile["button"].bind(BTN_CLICK, self.onClickWrapper(x, y))
                tile["button"].bind(BTN_FLAG, self.onRightClickWrapper(x, y))
                tile["button"].grid(row=x+1, column=y)  # Offset for labels
                self.tiles[id] = tile

        # Calculate mines around each tile
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                tile_id = f"{x}_{y}"
                tile = self.tiles[tile_id]
                tile["mines"] = sum(1 for n in self.getNeighbors(x, y) if n["isMine"])

    def cell_revealed(self, x, y):
        self.revealed_cells += 1      
    
    def update_stats(self): # Q add: in progress
        self.percentage_cleared = self.revealed_cells / (self.cols * self.rows) * 100
        # Update the stats labels with the latest values
        self.stats_labels["total_episodes"].config(text=f"Total games: {self.total_episodes}")
        self.stats_labels["total_wins"].config(text=f"Total wins: {self.wins}")
        self.stats_labels["total_reward"].config(text=f"Total reward: {self.cumulative_reward}")
        self.stats_labels["win_rate"].config(text=f"Win-loss rate (of last 50 games): {self.win_loss_rate}%")
        self.stats_labels["percentage_cleared"].config(text=f'Percentage cleared: {self.percentage_cleared}%')

    def updateTimer(self):
        if not self.game_over:
            if self.startTime is not None:
                elapsed = datetime.now() - self.startTime
                # Manually format the elapsed time
                hours, remainder = divmod(elapsed.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                # Update the timer label with the elapsed time formatted as HH:MM:SS
                self.labels['time'].config(text=f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}")
            # Schedule this method to be called again after 1000 milliseconds (1 second)
            self.frame.after(1000, self.updateTimer)

    def update_win_loss(self, win):
        self.win_loss.append(win)

        self.win_loss_rate = float(sum(self.win_loss)) / len(self.win_loss) * 100

    """ old structure
    def restart(self):
        self.setup()
        self.refreshLabels()


    def refreshLabels(self):
        self.labels["flags"].config(text = "Flags: "+str(self.flagCount))
        self.labels["mines"].config(text = "Mines: "+str(self.mines))

    def gameOver(self, won):
        for x in range(0, SIZE_X):
            for y in range(0, SIZE_Y):
                if self.tiles[x][y]["isMine"] == False and self.tiles[x][y]["state"] == STATE_FLAGGED:
                    self.tiles[x][y]["button"].config(image = self.images["wrong"])
                if self.tiles[x][y]["isMine"] == True and self.tiles[x][y]["state"] != STATE_FLAGGED:
                    self.tiles[x][y]["button"].config(image = self.images["mine"])

        self.tk.update()

        msg = "You Win! Play again?" if won else "You Lose! Play again?"
        res = tkMessageBox.askyesno("Game Over", msg)
        if res:
            self.restart()
        else:
            self.tk.quit()

    def updateTimer(self):
        ts = "00:00:00"
        if self.startTime != None:
            delta = datetime.now() - self.startTime
            ts = str(delta).split('.')[0] # drop ms
            if delta.total_seconds() < 36000:
                ts = "0" + ts # zero-pad
        self.labels["time"].config(text = ts)
        self.frame.after(100, self.updateTimer)

    def getNeighbors(self, x, y):
        neighbors = []
        coords = [
            {"x": x-1,  "y": y-1},  #top right
            {"x": x-1,  "y": y},    #top middle
            {"x": x-1,  "y": y+1},  #top left
            {"x": x,    "y": y-1},  #left
            {"x": x,    "y": y+1},  #right
            {"x": x+1,  "y": y-1},  #bottom right
            {"x": x+1,  "y": y},    #bottom middle
            {"x": x+1,  "y": y+1},  #bottom left
        ]
        for n in coords:
            try:
                neighbors.append(self.tiles[n["x"]][n["y"]])
            except KeyError:
                pass
        return neighbors

    def onClickWrapper(self, x, y):
        return lambda Button: self.onClick(self.tiles[x][y])

    def onRightClickWrapper(self, x, y):
        return lambda Button: self.onRightClick(self.tiles[x][y])

    def onClick(self, tile):
        if self.startTime == None:
            self.startTime = datetime.now()

        if tile["isMine"] == True:
            # end game
            self.gameOver(False)
            return

        # change image
        if tile["mines"] == 0:
            tile["button"].config(image = self.images["clicked"])
            self.clearSurroundingTiles(tile["id"])
        else:
            tile["button"].config(image = self.images["numbers"][tile["mines"]-1])
        # if not already set as clicked, change state and count
        if tile["state"] != STATE_CLICKED:
            tile["state"] = STATE_CLICKED
            self.clickedCount += 1
        if self.clickedCount == (SIZE_X * SIZE_Y) - self.mines:
            self.gameOver(True)

    def onRightClick(self, tile):
        if self.startTime == None:
            self.startTime = datetime.now()

        # if not clicked
        if tile["state"] == STATE_DEFAULT:
            tile["button"].config(image = self.images["flag"])
            tile["state"] = STATE_FLAGGED
            tile["button"].unbind(BTN_CLICK)
            # if a mine
            if tile["isMine"] == True:
                self.correctFlagCount += 1
            self.flagCount += 1
            self.refreshLabels()
        # if flagged, unflag
        elif tile["state"] == 2:
            tile["button"].config(image = self.images["plain"])
            tile["state"] = 0
            tile["button"].bind(BTN_CLICK, self.onClickWrapper(tile["coords"]["x"], tile["coords"]["y"]))
            # if a mine
            if tile["isMine"] == True:
                self.correctFlagCount -= 1
            self.flagCount -= 1
            self.refreshLabels()

    def clearSurroundingTiles(self, id):
        queue = deque([id])

        while len(queue) != 0:
            key = queue.popleft()
            parts = key.split("_")
            x = int(parts[0])
            y = int(parts[1])

            for tile in self.getNeighbors(x, y):
                self.clearTile(tile, queue)

    def clearTile(self, tile, queue):
        if tile["state"] != STATE_DEFAULT:
            return

        if tile["mines"] == 0:
            tile["button"].config(image = self.images["clicked"])
            queue.append(tile["id"])
        else:
            tile["button"].config(image = self.images["numbers"][tile["mines"]-1])

        tile["state"] = STATE_CLICKED
        self.clickedCount += 1
    """

    def onClickWrapper(self, x, y):
        return lambda Button: self.onClick(self.tiles[f"{x}_{y}"])

    def onRightClickWrapper(self, x, y):
        return lambda Button: self.onRightClick(self.tiles[f"{x}_{y}"])

    def onClick(self, tile):
        if tile["state"] == STATE_DEFAULT:
            state, reward, done = self.env.step((tile["coords"]["x"], tile["coords"]["y"]))
            if done:
                self.reset_game()
            self.update_gui(tile)

    def onRightClick(self, tile):
        if tile["state"] == STATE_DEFAULT:
            state, reward, done = self.env.step((tile["coords"]["x"], tile["coords"]["y"]), flag=True)
            if done:
                self.reset_game()
            self.update_gui(tile)

    def update_gui(self, tile):
        # Update the tile based on its state after an action
        if tile["isMine"]:
            tile["button"].config(image=self.images["mine"])
        elif tile["mines"] > 0:
            tile["button"].config(image=self.images["numbers"][tile["mines"]-1])
        else:
            tile["button"].config(image=self.images["clicked"])

    def getNeighbors(self, x, y):
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < SIZE_X and 0 <= ny < SIZE_Y and (dx != 0 or dy != 0):
                    neighbors.append(self.tiles[f"{nx}_{ny}"])
        return neighbors
    
        """ old gui based on state
    def update_gui_based_on_state(self):
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                tile = self.tiles[f"{x}_{y}"]
                if self.env.explored[x][y]:
                    if self.env.grid[x][y] == -1:
                        tile["button"].config(image=self.images["mine"])
                    elif self.env.grid[x][y] == 0:
                        tile["button"].config(image=self.images["clicked"])
                    else:
                        tile["button"].config(image=self.images["numbers"][self.env.grid[x][y]-1])
                elif self.env.flagged[x][y]:
                    tile["button"].config(image=self.images["flag"])
                else:
                    tile["button"].config(image=self.images["plain"])
        """
    def update_gui_based_on_state(self):
        for x in range(self.rows):
            for y in range(self.cols):
                tile = self.tiles[f"{x}_{y}"]
                if self.env.explored[x][y]:
                    if self.env.grid[x][y] == -1:
                        tile["button"].config(image=self.images["mine"])
                    elif self.env.grid[x][y] == 0:
                        tile["button"].config(image=self.images["clicked"])
                    else:
                        # Ensure that the index is an integer
                        number_index = int(self.env.grid[x][y]) - 1
                        if number_index >= 0 and number_index < len(self.images["numbers"]):
                            tile["button"].config(image=self.images["numbers"][number_index])
                elif self.env.flagged[x][y]:
                    tile["button"].config(image=self.images["flag"])
                else:
                    tile["button"].config(image=self.images["plain"])

    def reveal_all_mines(self):
        for x in range(self.rows):
            for y in range(self.cols):
                if self.env.grid[x][y] == -1:  # Assuming -1 is the value for mines
                    tile = self.tiles[f"{x}_{y}"]
                    tile["button"].config(image=self.images["mine"])

    def show_exploded_mine(self, action):
        r = action // self.cols
        c = action % self.cols
        if self.env.grid[r][c] == -1:  # Ensure it's actually a mine cell
            tile = self.tiles[f"{r}_{c}"]
            tile["button"].config(image=self.images["exploded_mine"])


### END OF CLASSES ###

def main():
    # create Tk instance
    window = Tk()
    # set program title
    window.title("Minesweeper")
    # create game instance
    minesweeper = Minesweeper(window)

    graph_window = Toplevel()
    graph_window.title("Win-Loss Graph")

    try:
        minesweeper.agent.load()  # Path is managed within the agent
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No saved model found, starting with a new model.")

    fig = plt.figure()
    
    # Embed the figure in the Tkinter window using FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.get_tk_widget().pack(fill='both', expand=True)
    canvas.draw()

    # Create the FuncAnimation object that will update the plot
    ani = FuncAnimation(fig, minesweeper.update_plot, interval=1000, blit=False)   

    is_animation_running = True

    def on_close():
        if not is_animation_running:
            ani.event_source.stop()
            plt.close(fig)
        if tkMessageBox.askyesno("Quit", "Do you want to save the model before quitting?"):
            minesweeper.agent.save()
        if graph_window.winfo_exists():
            graph_window.destroy()
        window.destroy()

    def on_close_graph():
        ani.event_source.stop()
        plt.close(fig)
        graph_window.destroy()
        is_animation_running = False

    window.protocol("WM_DELETE_WINDOW", on_close)
    graph_window.protocol("WM_DELETE_WINDOW", on_close_graph)

    # run event loop
    window.mainloop()

if __name__ == "__main__":
    main()
