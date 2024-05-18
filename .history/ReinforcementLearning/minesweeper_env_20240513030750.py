# minesweeper_env.py
import numpy as np
import random

class MinesweeperEnv:
    def __init__(self, rows, cols, bombs):
        self.rows = rows
        self.cols = cols
        self.bombs = bombs
        self.grid = np.zeros((rows, cols))
        self.explored = np.zeros((rows, cols), dtype=bool)
        self.flagged = np.zeros((rows, cols), dtype=bool)
        self.game_over = False
        self._place_bombs()
        self._calculate_adjacent_counts()

        self.rewards = {
            'lose': -5,          # Penalty for hitting a mine
            'win': 10,            # Reward for clearing all non-mine cells
            'progress': 1,        # Reward for revealing a non-mine cell
            'no_progress': -0.5,     # No reward/penalty for ineffective actions
            'guess': 0.5,         # Reward for making a necessary guess
            'flag': 0.1,          # Small reward for correctly flagging a mine
            'unflag': -0.2,       # Small penalty for unflagging
            'invalid': -0.5       # Penalty for invalid actions (e.g., flagging an already revealed cell)
        }

    def _place_bombs(self):
        bomb_coords = set()
        while len(bomb_coords) < self.bombs:
            x, y = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            bomb_coords.add((x, y))
        for x, y in bomb_coords:
            self.grid[x][y] = -1

    def _calculate_adjacent_counts(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == -1:
                    continue
                adj_cells = self._adjacent_cells(r, c)
                count = sum(1 for x, y in adj_cells if self.grid[x][y] == -1)
                self.grid[r][c] = count

    def _adjacent_cells(self, r, c):
        adj = []
        for x in range(max(0, r - 1), min(self.rows, r + 2)):
            for y in range(max(0, c - 1), min(self.cols, c + 2)):
                if (x, y) != (r, c):
                    adj.append((x, y))
        return adj

    def step_old(self, action, flag=False):
        #r, c = action
        flag = action >= self.rows * self.cols
        cell_index = action % (self.rows * self.cols)
        #r = action // self.cols  # Integer division to find the row
        #c = action % self.cols
        r = cell_index // self.cols
        c = cell_index % self.cols 
        reward = 0

        if flag:
            previously_flagged = self.flagged[r][c]
            self.flagged[r][c] = not self.flagged[r][c]
            reward = -1 if self.flagged[r][c] else 1
            return self.get_obs(), 1 if not previously_flagged else -1, False
        else:
            if self.grid[r][c] == -1:
                self.game_over = True
                #reward = -10
                return self.get_obs(), -10, True
            
            elif not self.explored[r][c]:
                #reward = 1 if self.grid[r][c] == 0 else self.grid[r][c] / 10
                self.explored[r][c] = True
                return self.get_obs(), 0, False
        #done = self.game_over or np.all(self.explored | (self.grid == -1))
        #return self._get_obs(), reward, done, self.game_over
        return self.get_obs(), 0, self.game_over

    def step_beforegh(self, action):
        # Determine if the action is a reveal or a flag based on the index
        flag = action >= self.rows * self.cols
        cell_index = action % (self.rows * self.cols)
        r = cell_index // self.cols
        c = cell_index % self.cols

        if flag:
            # Toggle the flag status and return appropriate reward
            if not self.explored[r][c]:  # Can only flag unexplored cells
                self.flagged[r][c] = not self.flagged[r][c]
                reward = 1 if self.flagged[r][c] else -1
            else:
                reward = -1  # Penalize for flagging an already revealed cell
        else:
            # Handle cell reveal
            if self.grid[r][c] == -1 and not self.flagged[r][c]:
                self.game_over = True
                reward = -10  # Penalty for hitting a mine
            elif not self.explored[r][c] and not self.flagged[r][c]:
                self.explored[r][c] = True
                reward = 1  # Reward for safely revealing a cell
            else:
                reward = -0.1  # Small penalty for redundant actions

        done = self.game_over or np.all(self.explored | (self.grid == -1))
        return self.get_obs(), reward, done, self.game_over

    def step_newold(self, action):
        flag = action >= self.rows * self.cols
        cell_index = action % (self.rows * self.cols)
        r = cell_index // self.cols
        c = cell_index % self.cols
    
        done = False
        current_state = np.array([self.get_obs()]).flatten()  # Get the current state for comparison
    
        if flag:
            # Toggle flag status
            if not self.explored[r][c] and not self.flagged[r][c]:
                self.flagged[r][c] = True
                reward = self.rewards['flag']  # Define this in your rewards dictionary
            elif self.flagged[r][c]:
                self.flagged[r][c] = False
                reward = self.rewards['unflag']  # Define this too
            else:
                reward = self.rewards['invalid']  # Penalize invalid flagging actions
        else:
            # Click action
            if self.grid[r][c] == -1:  # Mine hit
                reward = self.rewards['lose']
                done = True
                self.game_over = True
            elif not self.explored[r][c]:
                self.explored[r][c] = True
                # Check for win condition
                if np.all(self.explored | (self.grid == -1)):
                    reward = self.rewards['win']
                    done = True
                else:
                    reward = self.rewards['progress']  # Reward for revealing a safe cell
            else:
                reward = self.rewards['no_progress']  # No reward or penalty for clicking an already revealed cell

        # Update the state image (similar to your new state image)
        new_state_im = self.get_obs()
        self.state_im = new_state_im  # Update internal state

        # Additional logic to handle guessing or progress based on neighbors
        if not done and not self.explored[r][c]:
            neighbors = self._adjacent_cells(r, c)
            if all(self.grid[x][y] == -1 for x, y in neighbors):
                reward = self.rewards['guess']
            else:
                self.n_progress += 1  # Increment progress count

        return new_state_im, reward, done

    def step(self, action):
        flag = action >= self.rows * self.cols  # Determine if the action is to flag/unflag
        cell_index = action % (self.rows * self.cols)
        r = cell_index // self.cols
        c = cell_index % self.cols

        if self.game_over:
            print("Action taken while game over.")
            return self.get_obs(), 0, True, self.game_over  # No action allowed if game is over

        if flag:
            if not self.explored[r][c] and not self.flagged[r][c]:
                self.flagged[r][c] = True
                reward = self.rewards['flag']
                print(f"Flagged cell ({r}, {c}). Reward: {reward}")
            elif self.flagged[r][c]:
                self.flagged[r][c] = False
                reward = self.rewards['unflag']
                print(f"Unflagged cell ({r}, {c}). Reward: {reward}")
            else:
                reward = self.rewards['invalid']
                print(f"Invalid flag action at ({r}, {c}). Reward: {reward}")
        else:
            if self.flagged[r][c]:
                reward = self.rewards['invalid']
                print(f"Attempt to reveal flagged cell ({r}, {c}). Reward: {reward}")
            elif self.explored[r][c]:
                reward = self.rewards['no_progress']
                print(f"Attempt to reveal explored cell ({r}, {c}). Reward: {reward}")
            elif self.grid[r][c] == -1:
                self.game_over = True
                reward = self.rewards['lose']
                print(f"Hit mine at ({r}, {c}). Game over.")
            else:
                self.explored[r][c] = True
                reward = self.rewards['progress']
                print(f"Revealed safe cell ({r}, {c}). Reward: {reward}")
                if self.grid[r][c] == 0:
                    self.reveal_adjacent_cells(r, c)  # Additional logic to reveal adjacent cells if the cell is empty

        done = self.check_win_condition() or self.game_over
        return self.get_obs(), reward, done, self.game_over

    def reveal_adjacent_cells(self, r, c):
        """Recursive function to reveal all adjacent cells if they are safe (no mine and no number)"""
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if not self.explored[nr][nc] and self.grid[nr][nc] == 0 and not self.flagged[nr][nc]:
                        self.explored[nr][nc] = True
                        self.reveal_adjacent_cells(nr, nc)  # Recursive reveal

    def check_win_condition(self):
        """Check if all non-mine cells are revealed"""
        not_mined = self.grid != -1
        revealed_or_flagged = self.explored | self.flagged
        return np.all(revealed_or_flagged == not_mined)

    def _get_obs(self):
        return np.where(self.explored, self.grid, -2)

    def get_obs(self):
        """Generate an observable state of the grid for the AI"""
        obs = np.full((self.rows, self.cols), -2)  # -2 represents unrevealed cells
        for r in range(self.rows):
            for c in range(self.cols):
                if self.explored[r][c]:
                    obs[r][c] = self.grid[r][c]  # Reveal number or mine
                elif self.flagged[r][c]:
                    obs[r][c] = -3  # -3 represents flagged cells
        return obs

    def reset(self):
        self.grid.fill(0)
        self.explored.fill(False)
        self.flagged.fill(False)
        self.game_over = False
        self._place_bombs()
        self._calculate_adjacent_counts()
        return self._get_obs()
