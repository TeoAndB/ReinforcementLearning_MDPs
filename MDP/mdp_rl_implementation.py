from copy import deepcopy
import random
import numpy as np


# return positions/states given a board (2d array)
def get_states(board):
    positions = []
    for x in range(len(board)):
        for y in range(len(board[x])):
            positions.append((x, y))
    return positions

# get reward matrix
def get_rewards(mdp, board):
    R = np.zeros(np.array(board).shape)
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            val = board[r][c]
            if board[r][c] == 'WALL':
                R[r,c] = 0.0
            else:
                R[r,c] = float(val)
    return R

# function to calculate expectation value per action
def expecation_per_action(action, s, mdp, U, R, gamma):

    # get next states for all actions
    next_states = [mdp.step(s, "UP"), mdp.step(s, "DOWN"), mdp.step(s, "RIGHT"), mdp.step(s, "LEFT")]

    probabilities = mdp.transition_function[action]
    # get utilities of the next states
    utility_next_states = [U[next_states[i][0],next_states[i][1]] for i in range(len(next_states))]
    # get rewards of the next states

    # get value of each next state multiplied with its probability
    # print(f'utilities next states: {utility_next_states}')
    #
    # print(f'probabilities next states: {probabilities}')

    values_w_prob_next_states = [(gamma * probabilities[i] * utility_next_states[i]) for i in range(len(next_states))]
    # print(f'values with prob next states {values_w_prob_next_states}')
    # sum up to get expectation value:
    expectation_action_value = sum(values_w_prob_next_states)

    return expectation_action_value


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # getting the variables we need
    U_prime = U_init[:]

    gamma = mdp.gamma
    #states = get_states(mdp.board)
    states = [(0, 3), (0,2), (0,1), (0,0), (1,3), (1,2), (1,1), (1,0), (2,3), (2,2), (2,1), (2,0)]

    delta = 0.0
    print(epsilon*(1-gamma)/gamma)

    U_prime = np.array(U_prime, dtype=float)

    # test with 0 reward:
    R = np.zeros(U_prime.shape)
    R[0,3] = 1
    R[1,3] = -1

    print('Rewards')
    print(R)
    #R = get_rewards(mdp, mdp.board)

    iterations = 0
    while True:
        print(f'\nITERATION: {iterations}')


        U = U_prime.copy()
        delta = 0.0

        for s in states:
            #print(f'state: {s}')
            actions = ["UP", "DOWN", "RIGHT", "LEFT"]

            expectations_per_action = [expecation_per_action(actions[i], s, mdp, U, R, gamma) for i in range(len(actions))]


            max_val = max(expectations_per_action)
            action_best = expectations_per_action.index(max_val)

            #print(f' max value for best action {actions[action_best]} in state {s} is {max_val}')

            U_prime[s[0],s[1]] = max_val + R[s[0],s[1]]

            if np.abs(U_prime[s[0],s[1]] - U[s[0],s[1]]) > delta:
                delta = np.abs(U_prime[s[0],s[1]] - U[s[0],s[1]])
                print(f'delta = {delta}')
                if delta < (epsilon*(1-gamma)/gamma):
                    break


            # special case:
            if s in mdp.terminal_states:
                U_prime[s[0], s[1]] = R[s[0], s[1]]
            if mdp.board[s[0]][s[1]] == "WALL":
                U_prime[s[0], s[1]] = R[s[0], s[1]]




        print("Utility: ")
        mdp.print_utility(U)

        print("U_prime: ")
        print(U_prime)

        # testing
        iterations +=1
        if iterations == 10:
            break


    return U
    # ========================

# def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
#     # Given the mdp, the initial utility of each state - U_init,
#     #   and the upper limit - epsilon.
#     # run the value iteration algorithm and
#     # return: the U obtained at the end of the algorithms' run.
#     #
#
#     # getting the variables we need
#     U_prime = U_init[:]
#
#     gamma = mdp.gamma
#     #states = get_states(mdp.board)
#     states = [(0, 3), (0,2), (0,1), (0,0), (1,3), (1,2), (1,1), (1,0), (2,3), (2,2), (2,1), (2,0)]
#
#     actions = ["UP", "DOWN", "RIGHT", "LEFT"]
#
#     raise NotImplementedError


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def q_learning(mdp, init_state, total_episodes=10000, max_steps=999, learning_rate=0.7, epsilon=1.0,
                      max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.8):
    # TODO:
    # Given the mdp and the Qlearning parameters:
    # total_episodes - number of episodes to run for the learning algorithm
    # max_steps - for each episode, the limit for the number of steps
    # learning_rate - the "learning rate" (alpha) for updating the table values
    # epsilon - the starting value of epsilon for the exploration-exploitation choosing of an action
    # max_epsilon - max value of the epsilon parameter
    # min_epsilon - min value of the epsilon parameter
    # decay_rate - exponential decay rate for exploration prob
    # init_state - the initial state to start each episode from
    # return: the Qtable learned by the algorithm
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def q_table_policy_extraction(mdp, qtable):
    # TODO:
    # Given the mdp and the Qtable:
    # return: the policy corresponding to the Qtable
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


# BONUS

def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
