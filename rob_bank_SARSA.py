import numpy as np
import random
import matplotlib.pyplot as plt

def move_robber(state_id, action_id):
    robber_action = actions[action_id]
    robber_state = id_state_map[state_id][0]
    return (robber_state[0] + robber_action[0], robber_state[1] + robber_action[1])


def move_police(state_id, action_id):
    police_location = id_state_map[state_id][1]
    police_action = actions[action_id]
    return (police_location[0] + police_action[0], police_location[1] + police_action[1])


# def get_police_actions(police_location):
#     police_action = actions.copy()
#     if police_location[1] == 0:
#         police_action.pop(MOVE_LEFT)
#     if police_location[1] == 3:
#         police_action.pop(MOVE_RIGHT)
#     if police_location[0] == 0:
#         police_action.pop(MOVE_UP)
#     if police_location[0] == 3:
#         police_action.pop(MOVE_DOWN)
#     police_action.pop(STAY)
#     return police_action


def get_reward(state_id):
    my_reward = 0
    my_location = id_state_map[state_id][0]
    police_location = id_state_map[state_id][1]
    if my_location == police_location:
        my_reward = my_reward - 10
    elif my_location == bank_location:
        my_reward = my_reward + 1
    return my_reward


def q_learning(state_id, action):
    reward = get_reward(state_id)
    lr = get_lr(state_id, action)
    police_actions = state_action[state_id][1]
    police_random_action = random.choice(list(police_actions.keys()))
    police_next_state = move_police(state_id, police_random_action)
    allowed_actions_id = np.array(list(state_action[state_id][0].keys()))
    new_action_prob = epsilon_greedy(state_id, allowed_actions_id, epsilon)
    new_action_idx = np.random.choice(np.arange(len(new_action_prob)), p=new_action_prob)
    new_action = allowed_actions_id[new_action_idx]
    next_state = move_robber(state_id, new_action)
    next_state_id = state_id_map[(next_state, police_next_state)]
    q_table[state_id, action] = q_table[state_id, action] + lr * (
                reward + gamma * q_table[next_state_id, new_action] - q_table[state_id, action])
    return next_state_id, new_action


def epsilon_greedy(state_id, actions_id, epsilon=0.1):
    num_actions = len(actions_id)
    action_prob = np.ones(num_actions) * (epsilon / num_actions)
    action = np.argmax(q_table[state_id, actions_id])
    action_prob[action] = action_prob[action] + (1 - epsilon)
    return action_prob


def get_lr(state_id, action_id):
    if (state_id, action_id) in state_action_update_map:
        n = state_action_update_map[(state_id, action_id)] + 1
        state_action_update_map[(state_id, action_id)] = n
        return 1.0 /((n+1)**(2/3))
    else:
        state_action_update_map[(state_id, action_id)] = 1
        return 1.0


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

    state_action_update_map = dict()

    q_table = np.zeros((256, 5))
    epsilon = 0.4
    gamma = 0.8

    id_state_map = dict()
    state_id_map = dict()
    s = 0
    for robber_x in range(4):
        for robber_y in range(4):
            for police_x in range(4):
                for police_y in range(4):
                    id_state_map[s] = ((robber_x, robber_y), (police_x, police_y))
                    state_id_map[((robber_x, robber_y), (police_x, police_y))] = s
                    s += 1

    state_action = dict()
    for s in list(id_state_map.keys()):
        my_action = actions.copy()
        my_location = id_state_map[s][0]
        if my_location[1] == 0:
            my_action.pop(MOVE_LEFT)
        if my_location[1] == 3:
            my_action.pop(MOVE_RIGHT)
        if my_location[0] == 0:
            my_action.pop(MOVE_UP)
        if my_location[0] == 3:
            my_action.pop(MOVE_DOWN)
        police_action = actions.copy()
        police_location = id_state_map[s][1]
        if police_location[1] == 0:
            police_action.pop(MOVE_LEFT)
        if police_location[1] == 3:
            police_action.pop(MOVE_RIGHT)
        if police_location[0] == 0:
            police_action.pop(MOVE_UP)
        if police_location[0] == 3:
            police_action.pop(MOVE_DOWN)
        police_action.pop(STAY)

        state_action[s] = (my_action, police_action)

    state_id = state_id_map[(robber_start, police_start)]
    episode_q_values = []
    allowed_actions_id = np.array(list(state_action[state_id][0].keys()))
    action_prob = epsilon_greedy(state_id, allowed_actions_id, epsilon)
    action = np.random.choice(np.arange(len(action_prob)), p=action_prob)

    for episode in range(10000000):
        state_id, action = q_learning(state_id, action)
        # episode_reward += reward
        # if episode_reward < -100:
        #     break
        # episode_q_values.append(np.max(q_table[1, :]))
        episode_q_values.append(np.linalg.norm(q_table))
        print("eposide number: {}".format(episode))

    # idx = np.arange(start=0, stop=len(episode_rewards), step=1000)
    # print(idx)
    # plot_episode = np.array(episode_rewards)[idx]
    # plt.plot(idx, plot_episode)
    plt.plot(episode_q_values)
    plt.ylabel("q_value")
    plt.xlabel("episode #")
    plt.show()


