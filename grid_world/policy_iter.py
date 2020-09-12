from typing import Tuple

from GridWorldEnv import GridWorldEnv, Policy, StateValue, ActionValue, State
from plot import matplot_bar3d_ex
from policy_eval import policy_evaluate
import numpy as np

def action_value(env: GridWorldEnv, state: State, V: StateValue, gamma=1.0) -> ActionValue:
    q = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[state][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q


def policy_improvement(env: GridWorldEnv, policy: Policy, V: StateValue, gamma=1.0) -> bool:
    policy_stable = True

    for s in range(env.nS):
        old_action = np.argmax(policy[s])
        Q_s = action_value(env, s, V)
        best_action = np.argmax(Q_s)
        policy[s] = np.eye(env.nA)[best_action]

        if old_action != best_action:
            policy_stable = False
    return policy_stable


def policy_iteration(env: GridWorldEnv, policy: Policy, gamma=1.0) -> Tuple[Policy, StateValue]:
    iter = 0
    while True:
        V = policy_evaluate(policy, env, gamma)
        policy_stable = policy_improvement(env, policy, V)
        matplot_bar3d_ex(V, f'Iteration {iter}')
        iter += 1

        if policy_stable:
            return policy, V


if __name__ == '__main__':
    env = GridWorldEnv()

    policy_random = np.ones([env.nS, env.nA]) / env.nA

    policy, v = policy_iteration(env, policy_random)
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



