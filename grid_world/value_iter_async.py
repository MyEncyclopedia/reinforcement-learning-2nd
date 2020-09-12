import math
from typing import Tuple, Dict, Set

import numpy as np

from GridWorldEnv import GridWorldEnv, Policy, StateValue, ActionValue, Action, State
from plot import matplot_bar3d_ex
from policy_iter import action_value


def build_reverse_mapping(env:GridWorldEnv) -> Dict[State, Set[State]]:
    MAX_R, MAX_C = env.shape[0], env.shape[1]
    mapping = {s: set() for s in range(0, MAX_R * MAX_C)}
    action_delta = {Action.UP: (-1, 0), Action.DOWN: (1, 0), Action.LEFT: (0, -1), Action.RIGHT: (0, 1)}
    for s in range(0, MAX_R * MAX_C):
        r = s // MAX_R
        c = s % MAX_R
        for a in list(Action):
            neighbor_r = min(MAX_R - 1, max(0, r + action_delta[a][0]))
            neighbor_c = min(MAX_C - 1, max(0, c + action_delta[a][1]))
            s_ = neighbor_r * MAX_R + neighbor_c
            mapping[s_].add(s)
    return mapping


def value_iteration_async(env:GridWorldEnv, gamma=1.0, theta=0.0001) -> Tuple[Policy, StateValue]:
    mapping = build_reverse_mapping(env)

    V = np.zeros(env.nS)
    changed_state_set = set(s for s in range(env.nS))

    iter = 0
    while len(changed_state_set) > 0:
        changed_state_set_ = set()
        for s in changed_state_set:
            action_values = action_value(env, s, V, gamma=gamma)
            best_action_value = np.max(action_values)
            v_diff = np.abs(best_action_value - V[s])
            if v_diff > theta:
                changed_state_set_.update(mapping[s])
                V[s] = best_action_value
        changed_state_set = changed_state_set_
        iter += 1
        print(f'iter {iter}')

    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        action_values = action_value(env, s, V, gamma=gamma)
        best_action = np.argmax(action_values)
        policy[s, best_action] = 1.0

    return policy, V

if __name__ == '__main__':
    env = GridWorldEnv()
    policy, v = value_iteration_async(env)
    # matplot_bar3d_ex(v, f'Values')

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