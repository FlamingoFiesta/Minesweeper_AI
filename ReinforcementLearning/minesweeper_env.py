# minesweeper_env.py
import numpy as np
import random
from collections import deque

class MinesweeperEnv:
    def __init__(self, SIZE_X, SIZE_Y, NUM_MINES):
        self.rows = SIZE_Y
        self.cols = SIZE_X
        self.mines = NUM_MINES

        self.terminated = False # A boolean that 
        self.stopped = False

        self.minefield = [] # The information of the locations of the mines and the number of adjacent mines of each tile. 
        self.playerfield = [] # The information the agent receives, at the beginning the whole matrix is covered in 9s, which means that it is uncovered.

        # based on LLM claude
        self.rewards = {
            'lose': -1,          # Penalty for uncovering a mine.
            'win': 1,            # Reward for revealing all non-mine tile.
            'progress': 0.3,        # Reward for revealing a non-mine cell, next to an already uncovered tile.
            'guess': -0.3,         # Penalty for choosing a tile, far from other uncovered tiles. 
        }

    def step(self, action):
        cell_index = action % (self.rows * self.cols) # Finding the coordinates of the uncovered tiles.
        r = cell_index // self.rows # Calculate the coordinates of the affected tile. 
        c = cell_index % self.cols

        state = self.playerfield.flatten() # This turns the playerfield matrix into a one dimensional vector.
        minefield_state = self.minefield.flatten() # This turns the minefield matrix into a one dimensional vector.
        state[action] = minefield_state[action] # This gives a value to the tile that was just cleared. The values are based on the minefield matrix, that matrix contains the information not known to the agent. 
        
        num_uncovered_tiles = np.count_nonzero(state == 9) # This calculates the total amount of uncovered tiles in the grid.

        if state[action] == -1: # A mine has been revealed: LOSE
            self.terminated = True # End of the episode. 
            reward = self.rewards["lose"]
            print(f"Hit mine at ({r}, {c}). Game over.")

        elif num_uncovered_tiles == self.mines: # The only tiles left undiscovered are the tiles with mines. Therefore: All safe tiles have been revealed: WIN!
            self.terminated = True # End of the episode.
            reward = self.rewards["win"]
            print(f"Revealed last safe tile at ({r}, {c}). Won.")

        else: # A non-mine tile is discovered but the game is not over yet.
            adj_tiles = self._adjacent_tiles(r, c)
            if all(self.playerfield[tiles] == 9 for tiles in adj_tiles):
                reward = self.rewards['guess'] # This strategy is punished since the agent is not basing its decisions on the complete mindsweeper context.
                print(f"Guesed a safe cell ({r}, {c}). Reward: {reward}")

            else: # The agent has picked a tile next to at least one uncovered tile.
                reward = self.rewards['progress']
                print(f"Revealed safe cell ({r}, {c}). Reward: {reward}")

            if state[action] == 0:
                    self.reveal_adjacent_tiles(r, c, reward, state)  # Additional logic to reveal adjacent cells if the cell is empty

        if num_uncovered_tiles == self.mines: # A second win condition check.
            self.terminated = True
            reward = self.rewards["win"]
            print(f"Revealed last safe tile at ({r}, {c}). Won.")

        self.playerfield = state.reshape(5, 5)

        return self.playerfield, reward, self.terminated
        
    def reveal_adjacent_tiles(self, r, c, reward, state): # Algorithm implemented from Minesweeper-DDQN github.
        print(state)
        print(self.playerfield)
        
        self.playerfield = state.reshape(5, 5)
        print(self.playerfield)
        visited_tiles = [] # Tiles with no mines in their surroundings whos adjacent tiles have already been revealed.
        queued_tiles = deque([(r, c)]) # Establishing a deque that will add tiles that have no mines in its.
        run = True # Start the while loop.

        while run:
            for a in range(-1, 2): # Going through the neighbouring tiles of (r, c).
                for b in range(-1, 2):
                    r = queued_tiles[0][0] + a
                    c = queued_tiles[0][1] + b

                    if r >= 0 and c >= 0: # Avoid negative coordinates.
                        try: 
                            if self.playerfield[(r, c)] == 9: # If they are not yet revealed, they are revealed now.

                                self.playerfield[(r, c)] = self.minefield[(r, c)] # The playerfield is updated.

                                # Eventhough the agent did not actively chose to reveal these adjacent tiles, it is still a form of progress. Since something like this happens based on the context of the game.
                                reward += self.rewards['progress']
                                print(f"Revealed safe cell ({r}, {c}). Reward: {reward}")

                                if self.minefield[(r, c)] == 0: # Here it is checked if the revealed tile has adjacent mines.
                                    if (r, c) not in visited_tiles: # Check if it has not yet been visited before. 
                                        if (r, c) not in queued_tiles: # Check if it is not already queued up.
                                            queued_tiles.append((r, c)) # Add to the que list. 
                        except IndexError: # If coordinates are out of index of the matrix, skip them.
                            pass

            # Pop the old coordinates to the visited list.              
            visited_tiles.append(queued_tiles.popleft())
            if len(queued_tiles) == 0: # Terminate the while loop, when there are no more coordinates in the deck.
                run = False  

        print(self.playerfield)
        state = self.playerfield.flatten()

    # Function to reset the minesweeper grid. 
    def reset(self): 
        self.terminated = False #
        self.stopped = False
        self.minefield = np.zeros((self.rows, self.cols), dtype='int') # Establishment of the minefield, however, no mines have been placed on it thus far. 
        self.playerfield = np.ones((self.rows, self.cols), dtype='int')*9 # Establishment of the playerfield, a 9 means that it is still uncovered
        self._place_mines()
        self._calculate_adjacent_counts()
        return self.playerfield
    
    # Function that places mines on the minefield.
    def _place_mines(self):
        num_mines = 0 # Amount of mines placed. 
        while num_mines < self.mines:
            a, b = random.randint(0, self.cols - 1), random.randint(0, self.rows - 1) # Get the coordinates of a random tile.
            if self.minefield[a][b] != -1: # If it is not yet a mine, turn it into a mine
                self.minefield[a, b] = -1
                num_mines += 1

    # Function that calculates the amount of mines in a non-mine's surrounding.
    def _calculate_adjacent_counts(self):
        for r in range(self.rows): # Go throught the minesweeper grid.
            for c in range(self.cols):
                if self.minefield[r][c] == -1: # If the tile is a mine, continue to the next coordinates. 
                    continue
                adj_tiles = self._adjacent_tiles(r, c) # Get the adjacent tiles of the non-mine tile (r, c).
                count = sum(1 for x, y in adj_tiles if self.minefield[x][y] == -1) # Count the amount of mines in its surroundings. 
                self.minefield[r][c] = count # Give the tile the value of the total sum of adjacent mines. 

    # Function that finds the neighbours of a tile.        
    def _adjacent_tiles(self, r, c):
        adj = [] 
        for x in range(max(0, r - 1), min(self.rows, r + 2)): # The min and max functions prevent the loop to go outside of the minesweeper grid.
            for y in range(max(0, c - 1), min(self.cols, c + 2)):
                if (x, y) != (r, c): # Append the coordinates to the adjacent list only if it is a different from the original tile. 
                    adj.append((x, y))
        return adj