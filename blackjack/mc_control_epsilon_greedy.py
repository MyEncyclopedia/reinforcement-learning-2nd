from collections import defaultdict
from itertools import product
from typing import Tuple

import numpy as np
from gym.envs.toy_text import BlackjackEnv

from blackjack.common import gen_stochastic_episode, ActionValue, Policy, State
from blackjack.plotting import plot_value_function


def mc_control_epsilon_greedy(env: BlackjackEnv, num_episodes, discount_factor=1.0, epsilon=0.1) \
        -> Tuple[ActionValue, Policy]:
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    states = list(product(range(10, 22), range(1, 11), (True, False)))
    policy = {s: np.ones(env.action_space.n) * 1.0 / env.action_space.n for s in states}
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def update_epsilon_greedy_policy(policy: Policy, Q: ActionValue, s: State):
        policy[s] = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n
        best_action = np.argmax(Q[s])
        policy[s][best_action] += (1.0 - epsilon)

    for episode_i in range(1, num_episodes + 1):
        episode_history = gen_stochastic_episode(policy, env)

        G = 0
        for t in range(len(episode_history) - 1, -1, -1):
            s, a, r = episode_history[t]
            G = discount_factor * G + r
            if not any(s_a_r[0] == s and s_a_r[1] == a for s_a_r in episode_history[0: t]):
                returns_sum[s, a] += G
                returns_count[s, a] += 1.0
                Q[s][a] = returns_sum[s, a] / returns_count[s, a]
                update_epsilon_greedy_policy(policy, Q, s)

    return Q, policy

if __name__ == "__main__":
    env = BlackjackEnv()

    Q, policy = mc_control_epsilon_greedy(env, num_episodes=12000, epsilon=0.1)
    print(policy((17, 5, True)))

    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value
    # plot_value_function(V, title="Optimal Value Function")

    import matplotlib.pyplot as plt
    actions = {True: [], False: []}
    for usable in (True, False):
        for player in range(11, 22):
            row = []
            for opponent in range(1, 11):
                s = (player, opponent, usable)
                best_a = np.argmax(policy(s))
                row.append(best_a)
                print(f'{s} {best_a}')
            actions[usable].append(row)

    x, y = np.meshgrid(range(1, 11), range(11, 22))

    intensity = np.array(actions[True])

    plt.pcolormesh(x, y, intensity)
    plt.colorbar() #need a colorbar to show the intensity scale
    plt.show() #boom

    intensity = np.array(actions[False])

    plt.pcolormesh(x, y, intensity)
    plt.colorbar() #need a colorbar to show the intensity scale
    plt.show() #boom
