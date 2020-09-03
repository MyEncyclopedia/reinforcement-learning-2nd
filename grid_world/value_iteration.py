import math

import numpy as np

from GridWorldEnv import GridWorldEnv


def state_action_value(env, state, V, gamma=1.0):
    """
    Helper function to calculate the value for all action in a given state.

    Args:
        state: The state to consider (int)
        V: The value to use as an estimator, Vector of length env.nS

    Returns:
        A vector of length env.nA containing the expected value of each action.
    """
    action_values = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[state][a]:
            action_values[a] += prob * (reward + gamma * V[next_state])
    return action_values

def value_iteration(env, gamma=1.0, theta=0.0001):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            action_values = state_action_value(env, s, V, gamma=gamma)
            best_action_value = np.max(action_values)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        if delta < theta:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        action_values = state_action_value(env, s, V, gamma=gamma)
        best_action = np.argmax(action_values)
        # Always take the best action
        policy[s, best_action] = 1.0

    return policy, V

if __name__ == '__main__':
    env = GridWorldEnv()
    policy, v = value_iteration(env)

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Value Function:")
    print(v)
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")