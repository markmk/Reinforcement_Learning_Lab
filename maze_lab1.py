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
    MINOTAUR_REWARD = -200

    EXIT = (5, 5)

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
        # self.rewards = self.__rewards(weights=weights,
        #                               random_rewards=random_rewards);

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

    def get_rewards(self, state_player, state_minotaur, action_player, action_minotaur):
        next_state_player = self.__move_player(state_player, action_player)
        next_state_minotaur = self.__move_minotaur(state_minotaur, action_minotaur)
        if next_state_player == next_state_minotaur:
            return self.MINOTAUR_REWARD
        elif next_state_player == state_player:
            return self.OBSTACLE_REWARD
        elif next_state_player == self.EXIT:
            return self.GOAL_REWARD
        else:
            return self.STEP_REWARD

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon - 1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s, t]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1;
                s = next_s;
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s, policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1;
        return path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


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
    p = env.transition_probabilities;
    r = env.rewards;
    n_states = env.n_states;
    n_actions = env.n_actions;
    T = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states, T + 1));
    policy = np.zeros((n_states, T + 1));
    Q = np.zeros((n_states, n_actions));

    # Initialization
    Q = np.copy(r);
    V[:, T] = np.max(Q, 1);
    policy[:, T] = np.argmax(Q, 1);

    # The dynamic programming bakwards recursion
    for t in range(T - 1, -1, -1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t + 1])
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
                    Q[s_p, a_p] = env.get_rewards(states_player[s_p], states_minotaur[s_m], a_p, a_m) + gamma * np.dot(p_player[:, s_p, a_p], V)
    BV = np.max(Q, 1)

    while np.linalg.norm(V - BV) >= tol and n < 200:
        n += 1
        V = np.copy(BV)
        for s_p in range(n_states_player):
            for s_m in range(n_states_minotaur):
                for a_p in range(n_actions):
                    for a_m in range(n_actions):
                        Q[s_p, a_p] = env.get_rewards(states_player[s_p], states_minotaur[s_m], a_p,
                                                      a_m) + gamma * np.dot(p_player[:, s_p, a_p], V)
        BV = np.max(Q, 1)
        print(np.linalg.norm(V - BV))

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


def animate_solution(maze, path):
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
    for i in range(len(path)):
        grid.get_celld()[(path[i])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i])].get_text().set_text('Player')
        if i > 0:
            if path[i] == path[i - 1]:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i])].get_text().set_text('Player is out')
            else:
                grid.get_celld()[(path[i - 1])].set_facecolor(col_map[maze[path[i - 1]]])
                grid.get_celld()[(path[i - 1])].get_text().set_text('')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
