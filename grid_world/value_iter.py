import math
from typing import Tuple

import numpy as np

from GridWorldEnv import GridWorldEnv, Policy, StateValue, ActionValue
from plot import matplot_bar3d_ex
from policy_iter import action_value


def value_iteration(env:GridWorldEnv, gamma=1.0, theta=0.0001) -> Tuple[Policy, StateValue]:
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            action_values = action_value(env, s, V, gamma=gamma)
            best_action_value = np.max(action_values)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        if delta < theta:
            break

    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        action_values = action_value(env, s, V, gamma=gamma)
        best_action = np.argmax(action_values)
        policy[s, best_action] = 1.0

    return policy, V

if __name__ == '__main__':
    env = GridWorldEnv()
    policy, v = value_iteration(env)
    matplot_bar3d_ex(v, f'Values')

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