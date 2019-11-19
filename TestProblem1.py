import numpy as np

import maze_lab1 as mz
import pandas as pd

maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])

env = mz.Maze(maze)

# Discount Factor
gamma = 0.95;
# Accuracy treshold
epsilon = 0.0001;
V, policy = mz.value_iteration(env, gamma, epsilon)

method = 'ValIter'
start_player = (0,0)
start_minotaur = (6, 5)
path_player, path_minotaur = env.simulate(start_player, start_minotaur, policy, method)

mz.animate_solution(maze, path_player, path_minotaur)

