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

# # Discount Factor
# gamma = 0.95;
# # Accuracy treshold
# epsilon = 0.0001;
# V, policy = mz.value_iteration(env, gamma, epsilon)
#
# method = 'ValIter'
# start_player = (0,0)
# start_minotaur = (6, 5)
# path_player, path_minotaur = env.simulate(start_player, start_minotaur, policy, method)
#
# mz.animate_solution(maze, path_player, path_minotaur)

# Finite horizon
# horizon = 20
# # Solve the MDP problem with dynamic programming
# V, policy= mz.dynamic_programming(env,horizon);
#
# method = 'DynProg'
# start_player = (0,0)
# start_minotaur = (6, 5)
# path_player, path_minotaur = env.simulate(start_player, start_minotaur, policy, method)
#
# mz.animate_solution(maze, path_player, path_minotaur)

n_games = 10000
p = 1.0 / 30.0

for n in range(n_games):
    # horizon = np.random.geometric(p)
    horizon = 17
    # Solve the MDP problem with dynamic programming
    V, policy = mz.dynamic_programming(env, horizon);

    method = 'DynProg'
    start_player = (0, 0)
    start_minotaur = (6, 5)
    path_player, path_minotaur = env.simulate(start_player, start_minotaur, policy, method, horizon)
