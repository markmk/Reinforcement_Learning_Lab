import numpy as np
import random
import matplotlib.pyplot as plt

def move_robber(state_id, action_id):
    robber_action = actions[action_id]
    robber_state = id_state_map[state_id]
    return (robber_state[0] + robber_action[0], robber_state[1] + robber_action[1])


def move_police(police_location, action_id):
    police_action = actions[action_id]
    return (police_location[0] + police_action[0], police_location[1] + police_action[1])


def get_police_actions(police_location):
    police_action = actions.copy()
    if police_location[1] == 0:
        police_action.pop(MOVE_LEFT)
    if police_location[1] == 3:
        police_action.pop(MOVE_RIGHT)
    if police_location[0] == 0:
        police_action.pop(MOVE_UP)
    if police_location[0] == 3:
        police_action.pop(MOVE_DOWN)
    police_action.pop(STAY)
    return police_action


def get_reward(state_id, police_location):
    my_reward = 0
    my_location = id_state_map[state_id]
    if my_location == police_location:
        my_reward = my_reward - 10
    elif my_location == bank_location:
        my_reward = my_reward + 1
    return my_reward


def q_learning(n, state_id, police_location):
    lr = 1.0 /((n+1)**(2/3))
    reward = get_reward(state_id, police_location)
    if random.uniform(0, 1) < epsilon:
        random_action = random.choice(list(state_action[state_id].keys()))
        next_state = move_robber(state_id, random_action)
        next_state_id = state_id_map[next_state]
        q_table[state_id, random_action] = q_table[state_id, random_action] + lr * (reward + gamma * np.max(q_table[next_state_id, :]) - q_table[state_id, random_action])
    else:
        allowed_actions_id = np.array(list(state_action[state_id].keys()))
        # print(id_state_map[state_id])
        # print(allowed_actions_id)
        action = allowed_actions_id[np.argmax(q_table[state_id, allowed_actions_id])]
        next_state = move_robber(state_id, action)
        next_state_id = state_id_map[next_state]
        q_table[state_id, action] = q_table[state_id, action] + lr * (
                    reward + gamma * np.max(q_table[next_state_id, :]) - q_table[state_id, action])
    return next_state_id, reward


if __name__ == '__main__':
    robber_start = (0, 0)
    police_start = (3, 3)
    bank_location = (1, 1)

    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    actions = dict()
    actions[STAY] = (0, 0)
    actions[MOVE_LEFT] = (0, -1)
    actions[MOVE_RIGHT] = (0, 1)
    actions[MOVE_UP] = (-1, 0)
    actions[MOVE_DOWN] = (1, 0)

    q_table = np.zeros((16, 5))
    epsilon = 0.2
    gamma = 0.8

    id_state_map = dict()
    state_id_map = dict()
    s = 0
    for i in range(4):
        for j in range(4):
            id_state_map[s] = (i, j)
            state_id_map[(i, j)] = s
            s += 1


    state_action = dict()
    for s in list(id_state_map.keys()):
        my_action = actions.copy()
        my_location = id_state_map[s]
        if my_location[1] == 0:
            my_action.pop(MOVE_LEFT)
        if my_location[1] == 3:
            my_action.pop(MOVE_RIGHT)
        if my_location[0] == 0:
            my_action.pop(MOVE_UP)
        if my_location[0] == 3:
            my_action.pop(MOVE_DOWN)
        state_action[s] = my_action

    police_location = police_start
    state_id = 0

    episode_rewards = []
    for episode in range(1000):
        episode_reward = 0
        for step in range(100):
            state_id, reward = q_learning(step, state_id, police_location)
            police_actions = get_police_actions(police_location)
            police_random_action = random.choice(list(police_actions.keys()))
            police_location = move_police(police_location, police_random_action)
            episode_reward += reward
        episode_rewards.append(episode_reward)
        print("eposide number: {}".format(episode))

    idx = np.arange(start=0, stop=len(episode_rewards), step=100)
    # print(idx)
    plot_episode = np.array(episode_rewards)[idx]
    plt.plot(idx, plot_episode)
    plt.ylabel("Reward")
    plt.xlabel("episode #")
    plt.show()


