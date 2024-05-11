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

    def step(self, action, flag=False):
        r, c = action
        if flag:
            self.flagged[r][c] = not self.flagged[r][c]
        elif self.grid[r][c] == -1:
            self.game_over = True
            reward = -10
        else:
            self.explored[r][c] = True
            reward = 1 if self.grid[r][c] == 0 else self.grid[r][c] / 10
        done = self.game_over or np.all(self.explored | (self.grid == -1))
        return self._get_obs(), reward, done

    def _get_obs(self):
        return np.where(self.explored, self.grid, -2)

    def reset(self):
        self.grid.fill(0)
        self.explored.fill(False)
        self.flagged.fill(False)
        self.game_over = False
        self._place_bombs()
        self._calculate_adjacent_counts()
        return self._get_obs()
