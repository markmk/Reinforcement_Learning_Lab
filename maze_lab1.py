import random
import time

import matplotlib.pyplot as plt
import numpy as np
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED = '#FFC4CC';
LIGHT_GREEN = '#95FD99';
BLACK = '#000000';
WHITE = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';


class Maze:
    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    OBSTACLE_REWARD = -100
    MINOTAUR_REWARD = -150

    EXIT = (6, 5)

    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze = maze;
        self.actions = self.__actions();
        self.states_player, self.map_player = self.__states_player();
        self.states_minotaur, self.map_minotaur = self.__states_minotaur();
        self.n_actions = len(self.actions);  # maybe need multiply human and minotaur ??
        self.n_states_player = len(self.states_player);  # need varying ??
        self.n_states_minotaur = len(self.states_minotaur);  # need varying ??
        self.transition_probabilities_player = self.__transitions_player();
        self.transition_probabilities_minotaur = self.__transitions_minotaur();
        # self.path_minotaur = self.__get_minotaur_path()
        self.rewards = self.__get_rewards()


    def __actions(self):
        actions = dict();
        actions[self.STAY] = (0, 0);
        actions[self.MOVE_LEFT] = (0, -1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP] = (-1, 0);
        actions[self.MOVE_DOWN] = (1, 0);
        return actions;

    def __states_player(self):
        states = dict()
        map = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                states[s] = (i, j)
                map[(i, j)] = s
                s += 1
        return states, map

    def __states_minotaur(self):
        states_minotaur = dict()
        map_minotaur = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                states_minotaur[s] = (i, j)
                map_minotaur[(i, j)] = s
                s += 1
        return states_minotaur, map_minotaur

    def __move_player(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        row = self.states_player[state][0] + self.actions[action][0];  # state_player
        col = self.states_player[state][1] + self.actions[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                             (col == -1) or (col == self.maze.shape[1]) or \
                             (self.maze[row, col] == 1);
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return state;
        else:
            return self.map_player[(row, col)];

    def __move_minotaur(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        row = self.states_minotaur[state][0] + self.actions[action][0];  # state_minotaur
        col = self.states_minotaur[state][1] + self.actions[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                             (col == -1) or (col == self.maze.shape[1])
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return state;
        else:
            return self.map_minotaur[(row, col)];

    def __transitions_player(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states_player, self.n_states_player, self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states_player):
            for a in range(self.n_actions):
                next_s = self.__move_player(s, a);
                transition_probabilities[next_s, s, a] = 1;
        return transition_probabilities;

    def __transitions_minotaur(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states_minotaur, self.n_states_minotaur, self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states_minotaur):
            for a in range(self.n_actions):
                next_s = self.__move_minotaur(s, a);
                transition_probabilities[next_s, s, a] = 1;
        return transition_probabilities;

    def get_minotaur_path(self, horizon):
        start_minotaur = (6, 5)
        s_minotaur = self.map_minotaur[start_minotaur]
        T = horizon
        path_minotaur = list()
        path_minotaur.append(start_minotaur)
        for t in range(T):
            next_s_minotaur = self.__move_minotaur(s_minotaur, random.randint(0, 4))
            path_minotaur.append(self.states_minotaur[next_s_minotaur])
            s_minotaur = next_s_minotaur

        return path_minotaur

    def __get_rewards(self):
        rewards = np.zeros((self.n_states_player, self.n_actions));

        for s in range(self.n_states_player):
            for a in range(self.n_actions):
                next_s = self.__move_player(s, a)
                # Rewrd for hitting a wall
                if s == next_s and a != self.STAY:
                    rewards[s, a] = self.OBSTACLE_REWARD
                # Reward for reaching the exit
                elif s == next_s and self.maze[self.states_player[next_s]] == 2:
                    rewards[s, a] = self.GOAL_REWARD
                # Reward for taking a step to an empty cell that is not the exit
                else:
                    rewards[s, a] = self.STEP_REWARD
        return rewards

    def simulate(self, start_player, start_minotaur, policy, method, horizon):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path_player = list();
        path_minotaur = self.get_minotaur_path(horizon)

        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s_player = self.map_player[start_player];
            xy_minotaur = path_minotaur[t]
            # Add the starting position in the maze to the path
            path_player.append(start_player);

            while t < horizon - 1 and self.states_player[s_player] != xy_minotaur and self.states_player[s_player] != self.EXIT:
                t += 1
                # Move to next state given the policy and the current state
                next_s_player = self.__move_player(s_player, policy[s_player, t]);

                # Add the position in the maze corresponding to the next state
                # to the path
                path_player.append(self.states_player[next_s_player])
                xy_minotaur = path_minotaur[t]
                # Update time and state for next iteration
                s_player = next_s_player

            if self.states_player[s_player] == xy_minotaur:
                print('Player was caight at ', t)
            else:
                print('Player exited ', t)
            print(path_player)
            print(path_minotaur)

        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s_player = self.map_player[start_player];
            s_minotaur = self.map_minotaur[start_minotaur]
            # Add the starting position in the maze to the path
            path_player.append(start_player);
            path_minotaur.append(start_minotaur);
            # Move to next state given the policy and the current state
            next_s_player = self.__move_player(s_player, policy[s_player]);
            next_s_minotaur = self.__move_minotaur(s_minotaur, random.randint(0, 4))
            # Add the position in the maze corresponding to the next state
            # to the path
            path_player.append(self.states_player[next_s_player])
            path_minotaur.append(self.states_minotaur[next_s_minotaur])
            # Loop while state is not the goal state
            while ((s_player != s_minotaur or s_player != next_s_player) and t < 100):
                # Update state
                s_player = next_s_player;
                s_minotaur = next_s_minotaur
                # Move to next state given the policy and the current state
                next_s_player = self.__move_player(s_player, policy[s_player]);
                next_s_minotaur = self.__move_minotaur(s_minotaur, random.randint(0, 4))

                # Add the position in the maze corresponding to the next state
                # to the path
                path_player.append(self.states_player[next_s_player])
                path_minotaur.append(self.states_minotaur[next_s_minotaur])
                # Update time and state for next iteration
                t += 1;
                print(t)

        return path_player, path_minotaur

    def show(self):
        print('The player states are :')
        print(self.states_player)
        print('The minotaur states are :')
        print(self.states_minotaur)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the player states:')
        print(self.map_player)
        print('The mapping of the minotaur states:')
        print(self.map_minotaur)
        print('Trans Matrix Player:')
        print(self.transition_probabilities_player)
        print('Trans Matrix Minotaur:')
        print(self.transition_probabilities_minotaur)
        # print('The rewards:')
        # print(self.rewards)


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    path_minotaur = env.get_minotaur_path(horizon)
    r = env.rewards
    p_player = env.transition_probabilities_player
    n_states_player = env.n_states_player
    states_player = env.states_player


    n_actions = env.n_actions

    T = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states_player, T + 1));
    policy = np.zeros((n_states_player, T + 1));
    Q = np.zeros((n_states_player, n_actions));

    # Initialization
    Q = np.copy(r);
    V[:, T] = np.max(Q, 1);
    policy[:, T] = np.argmax(Q, 1);

    # The dynamic programming bakwards recursion
    for t in range(T - 1, -1, -1):
        # Update the value function acccording to the bellman equation
        for s_p in range(n_states_player):
            for a_p in range(n_actions):
                if states_player[s_p] == path_minotaur[t]:
                    Q[s_p, a_p] = r[s_p, a_p] + np.dot(p_player[:, s_p, a_p], V[:, t + 1]) + env.MINOTAUR_REWARD
                else:
                    Q[s_p, a_p] = r[s_p,a_p] + np.dot(p_player[:, s_p, a_p], V[:, t + 1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1);
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1);
    return V, policy;


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p_player = env.transition_probabilities_player
    n_states_player = env.n_states_player
    states_player = env.states_player

    p_minotaur = env.transition_probabilities_minotaur
    n_states_minotaur = env.n_states_minotaur
    states_minotaur = env.states_minotaur

    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V = np.zeros(n_states_player);
    Q = np.zeros((n_states_player, n_actions));
    BV = np.zeros(n_states_player);
    # Iteration counter
    n = 0;
    # Tolerance error
    tol = (1 - gamma) * epsilon / gamma;

    for s_p in range(n_states_player):
        for s_m in range(n_states_minotaur):
            for a_p in range(n_actions):
                for a_m in range(n_actions):
                    Q[s_p, a_p] = env.get_rewards(s_p, s_m, a_p, a_m) + gamma * np.dot(p_player[:, s_p, a_p], V)
    BV = np.max(Q, 1)

    while np.linalg.norm(V - BV) >= tol and n < 200:
        n += 1
        V = np.copy(BV)
        for s_p in range(n_states_player):
            for s_m in range(n_states_minotaur):
                for a_p in range(n_actions):
                    for a_m in range(n_actions):
                        Q[s_p, a_p] = env.get_rewards(s_p, s_m, a_p,
                                                      a_m) + gamma * np.dot(p_player[:, s_p, a_p], V)
        BV = np.max(Q, 1)
        # print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q, 1);
    # Return the obtained policy
    return V, policy;


def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows, cols = maze.shape;
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows, cols = maze.shape;
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0 / rows);
        cell.set_width(1.0 / cols);


def animate_solution(maze, path_player, path_minotaur):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows, cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0 / rows);
        cell.set_width(1.0 / cols);

    # Update the color at each frame
    for i in range(len(path_player)):
        grid.get_celld()[(path_player[i])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path_player[i])].get_text().set_text('Player')

        grid.get_celld()[(path_minotaur[i])].set_facecolor(LIGHT_RED)
        grid.get_celld()[(path_minotaur[i])].get_text().set_text('Minotaur')
        if i > 0:
            if path_player[i] == path_player[i - 1]:
                grid.get_celld()[(path_player[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path_player[i])].get_text().set_text('Player is out')
            elif path_player[i] == path_minotaur[i]:
                grid.get_celld()[(path_player[i])].set_facecolor(LIGHT_PURPLE)
                grid.get_celld()[(path_player[i])].get_text().set_text('Player is caught')
            # elif path_minotaur[i] == path_minotaur[i - 1]:
            #     grid.get_celld()[(path_minotaur[i])].set_facecolor(LIGHT_RED)
            #     grid.get_celld()[(path_minotaur[i])].get_text().set_text('Minotaur')
            else:
                grid.get_celld()[(path_player[i - 1])].set_facecolor(col_map[maze[path_player[i - 1]]])
                grid.get_celld()[(path_player[i - 1])].get_text().set_text('')
                grid.get_celld()[(path_minotaur[i - 1])].set_facecolor(col_map[maze[path_minotaur[i - 1]]])
                grid.get_celld()[(path_minotaur[i - 1])].get_text().set_text('')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
